// ======================================================================
// \title  WasmSequencerHarness.cpp
// \brief  Runs wasm sequences on a real Svc::WasmSequencer
// ======================================================================

#include "WasmSequencerHarness.hpp"

#include <climits>
#include <unistd.h>

#include "Fw/Com/ComBuffer.hpp"
#include "Svc/Seq/SeqArgsSerializableAc.hpp"
#include "Svc/WasmSequencer/WasmSequencerComponentAc.hpp"

namespace Svc {

namespace {

//! RUN's opcode, relative to the component's id base.
const FwOpcodeType RELATIVE_OPCODE_RUN = 0;

//! The opcode out of a dispatched command packet, which leads with a packet
//! descriptor.
FwOpcodeType readOpcode(const Fw::ComBuffer& command) {
    Fw::ComBuffer copy = command;
    copy.resetDeser();
    FwPacketDescriptorType descriptor = 0;
    Fw::SerializeStatus status = copy.deserialize(descriptor);
    FW_ASSERT(status == Fw::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
    FwOpcodeType opcode = 0;
    status = copy.deserialize(opcode);
    FW_ASSERT(status == Fw::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
    return opcode;
}

}  // namespace

WasmSequencerTester::WasmSequencerTester()
    : Fw::PassiveComponentBase("WasmSequencerTester"),
      m_component("WasmSequencer"),
      m_request(nullptr),
      m_result(nullptr) {
    this->init(INSTANCE_ID);
    this->m_component.init(QUEUE_DEPTH, INSTANCE_ID);
}

WasmSequencerTester::~WasmSequencerTester() {
    this->m_component.deinit();
}

void WasmSequencerTester::connectPorts(const HarnessRequest& request) {
    this->m_cmdOut.init();
    this->m_cmdOut.addCallComp(this, cmdOutThunk);
    this->m_component.set_cmdOut_OutputPort(0, &this->m_cmdOut);

    this->m_timeCaller.init();
    this->m_timeCaller.addCallComp(this, timeThunk);
    this->m_component.set_timeCaller_OutputPort(0, &this->m_timeCaller);

    this->m_getTlmChan.init();
    this->m_getTlmChan.addCallComp(this, getTlmChanThunk);
    this->m_component.set_getTlmChan_OutputPort(0, &this->m_getTlmChan);

    this->m_getParam.init();
    this->m_getParam.addCallComp(this, getParamThunk);
    this->m_component.set_getParam_OutputPort(0, &this->m_getParam);

    this->m_prmGet.init();
    this->m_prmGet.addCallComp(this, getParamThunk);
    this->m_component.set_prmGet_OutputPort(0, &this->m_prmGet);

    this->m_prmSet.init();
    this->m_prmSet.addCallComp(this, prmSetThunk);
    this->m_component.set_prmSet_OutputPort(0, &this->m_prmSet);

    this->m_logOut.init();
    this->m_logOut.addCallComp(this, logThunk);
    this->m_component.set_logOut_OutputPort(0, &this->m_logOut);

#if FW_ENABLE_TEXT_LOGGING == 1
    this->m_logTextOut.init();
    this->m_logTextOut.addCallComp(this, logTextThunk);
    this->m_component.set_logTextOut_OutputPort(0, &this->m_logTextOut);
#endif

    this->m_tlmOut.init();
    this->m_tlmOut.addCallComp(this, tlmOutThunk);
    this->m_component.set_tlmOut_OutputPort(0, &this->m_tlmOut);

    this->m_cmdRegOut.init();
    this->m_cmdRegOut.addCallComp(this, cmdRegThunk);
    this->m_component.set_cmdRegOut_OutputPort(0, &this->m_cmdRegOut);

    this->m_cmdResponseOut.init();
    this->m_cmdResponseOut.addCallComp(this, cmdResponseThunk);
    this->m_component.set_cmdResponseOut_OutputPort(0, &this->m_cmdResponseOut);

    // The component has both a synchronous and an asynchronous serial port
    // array; both land in the same recorded history.
    for (FwIndexType i = 0; i < SERIAL_PORTS; i++) {
        if (request.disconnectedSerialPorts.count(i) > 0) {
            continue;
        }
        this->m_serialSyncOut[i].init();
        this->m_serialSyncOut[i].addCallComp(this, serialOutThunk);
        this->m_serialSyncOut[i].setPortNum(i);
        this->m_component.set_serialSyncOut_OutputPort(i, &this->m_serialSyncOut[i]);

        this->m_serialAsyncOut[i].init();
        this->m_serialAsyncOut[i].addCallComp(this, serialOutThunk);
        this->m_serialAsyncOut[i].setPortNum(i);
        this->m_component.set_serialAsyncOut_OutputPort(i, &this->m_serialAsyncOut[i]);
    }
}

HarnessResult WasmSequencerTester::run(const HarnessRequest& request) {
    HarnessResult result;
    this->m_request = &request;
    this->m_result = &result;

    this->m_now = Fw::Time(static_cast<TimeBase::T>(request.timeBase), request.timeContext,
                           static_cast<U32>(request.initialTimeUs / 1000000),
                           static_cast<U32>(request.initialTimeUs % 1000000));

    this->connectPorts(request);
    this->m_component.regCommands();
    // A deployment loads parameters at init. Without it INSTRUCTION_FUEL reads
    // as zero, so each interpreter slice runs no instructions and the module
    // spins out of fuel forever.
    this->m_component.loadParameters();

    std::string previousDir;
    if (!request.cwd.empty()) {
        char buf[PATH_MAX];
        if (getcwd(buf, sizeof(buf)) != nullptr) {
            previousDir = buf;
        }
        if (chdir(request.cwd.c_str()) != 0) {
            result.error = "bad_cwd";
            return result;
        }
    }

    // RUN always carries a Svc::SeqArgs, unlike the bytecode sequencer's
    // separate RUN/RUN_ARGS pair.
    Fw::CmdArgBuffer args;
    Fw::CmdStringArg path(request.seqPath.c_str());
    Fw::SerializeStatus status = args.serialize(path);
    FW_ASSERT(status == Fw::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
    status = args.serialize(Svc::BlockState(Svc::BlockState::BLOCK));
    FW_ASSERT(status == Fw::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
    Svc::SeqArgs seqArgs;
    seqArgs.set_size(static_cast<FwSizeType>(request.args.size()));
    U8* buffer = seqArgs.get_buffer();
    FW_ASSERT(request.args.size() <= SequenceArgumentsMaxSize,
              static_cast<FwAssertArgType>(request.args.size()));
    for (size_t i = 0; i < request.args.size(); i++) {
        buffer[i] = request.args[i];
    }
    status = args.serialize(seqArgs);
    FW_ASSERT(status == Fw::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));

    const FwOpcodeType opcode = RELATIVE_OPCODE_RUN + this->m_component.getIdBase();
    this->m_component.get_cmdIn_InputPort(0)->invoke(opcode, 0, args);

    this->pump(request, result);

    result.simTimeUs =
        static_cast<U64>(this->m_now.getSeconds()) * 1000000 + this->m_now.getUSeconds();
    result.ok = result.error.empty();

    if (!previousDir.empty()) {
        const int rc = chdir(previousDir.c_str());
        FW_ASSERT(rc == 0, rc);
    }

    this->m_request = nullptr;
    this->m_result = nullptr;
    return result;
}

void WasmSequencerTester::pump(const HarnessRequest& request, HarnessResult& result) {
    using State = WasmSequencer_SequencerStateMachineStateMachineBase::State;

    U32 dispatches = 0;
    while (dispatches < request.maxDispatch) {
        if (this->m_component.m_queue.getMessagesAvailable() > 0) {
            this->m_component.doDispatch();
            dispatches++;
            continue;
        }

        const State state = this->m_component.sequencer_getState();
        if (state == State::IDLE || state == State::READY) {
            break;
        }
        // The module runs in slices bounded by INSTRUCTION_FUEL, yielding
        // between them; a timer tick is what resumes it.
        this->m_component.get_checkTimers_InputPort(0)->invoke(0);
        dispatches++;
    }
    if (dispatches >= request.maxDispatch) {
        result.error = "dispatch_limit";
    }
    if (result.finalState.empty()) {
        switch (this->m_component.sequencer_getState()) {
            case State::IDLE: result.finalState = "IDLE"; break;
            case State::LOADING: result.finalState = "LOADING"; break;
            case State::READY: result.finalState = "READY"; break;
            case State::RUNNING_AWAITING_RESPONSE:
                result.finalState = "RUNNING_AWAITING_RESPONSE";
                break;
            case State::RUNNING_PAUSED: result.finalState = "RUNNING_PAUSED"; break;
            case State::RUNNING_SPINNING: result.finalState = "RUNNING_SPINNING"; break;
            case State::STARTING: result.finalState = "STARTING"; break;
            default: result.finalState = "UNINITIALIZED"; break;
        }
    }
}

Fw::CmdResponse WasmSequencerTester::respondTo(FwOpcodeType opcode,
                                               const HarnessRequest& request) {
    if (request.failOpcodes.count(opcode) > 0) {
        return Fw::CmdResponse(Fw::CmdResponse::EXECUTION_ERROR);
    }
    return Fw::CmdResponse(static_cast<Fw::CmdResponse::T>(request.cmdResponse));
}

void WasmSequencerTester::handleCmdOut(Fw::ComBuffer& data, U32 context) {
    FW_ASSERT(this->m_request != nullptr);
    FW_ASSERT(this->m_result != nullptr);

    const FwOpcodeType opcode = readOpcode(data);
    const FwSizeType prefix = sizeof(FwPacketDescriptorType);
    FW_ASSERT(data.getBuffLength() >= prefix, static_cast<FwAssertArgType>(data.getBuffLength()));
    this->m_result->cmds.emplace_back(data.getBuffAddr() + prefix,
                                      data.getBuffAddr() + data.getBuffLength());

    const Fw::CmdResponse response = this->respondTo(opcode, *this->m_request);
    this->m_component.get_cmdResponseIn_InputPort(0)->invoke(opcode, context, response);
}

void WasmSequencerTester::handleTime(Fw::Time& time) {
    time = this->m_now;
}

Fw::TlmValid WasmSequencerTester::handleGetTlmChan(FwChanIdType id,
                                                   Fw::Time& timeTag,
                                                   Fw::TlmBuffer& val) {
    FW_ASSERT(this->m_request != nullptr);
    const auto it = this->m_request->tlm.find(id);
    if (it == this->m_request->tlm.end()) {
        val.setBuffLen(0);
        return Fw::TlmValid::INVALID;
    }
    timeTag = this->m_now;
    const Fw::SerializeStatus status =
        val.setBuff(it->second.data(), static_cast<FwSizeType>(it->second.size()));
    FW_ASSERT(status == Fw::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
    return Fw::TlmValid::VALID;
}

Fw::ParamValid WasmSequencerTester::handleGetParam(FwPrmIdType id, Fw::ParamBuffer& val) {
    FW_ASSERT(this->m_request != nullptr);
    const auto it = this->m_request->prm.find(id);
    if (it == this->m_request->prm.end()) {
        val.setBuffLen(0);
        return Fw::ParamValid::INVALID;
    }
    const Fw::SerializeStatus status =
        val.setBuff(it->second.data(), static_cast<FwSizeType>(it->second.size()));
    FW_ASSERT(status == Fw::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
    return Fw::ParamValid::VALID;
}

void WasmSequencerTester::handleLogText(const Fw::LogSeverity& severity, Fw::TextLogString& text) {
    FW_ASSERT(this->m_result != nullptr);
    this->m_result->events.emplace_back(static_cast<U32>(severity.e), text.toChar());
}

void WasmSequencerTester::handleLog(FwEventIdType id, Fw::LogBuffer& args) {
    FW_ASSERT(this->m_result != nullptr);
    // The module's outcome is only reported through events: an explicit exit or
    // a panic carries its code, and a plain failure carries none.
    const FwEventIdType relativeId = id - this->m_component.getIdBase();
    const bool isExit = relativeId == WasmSequencerComponentBase::EVENTID_PROGRAMEXITED;
    const bool isPanic = relativeId == WasmSequencerComponentBase::EVENTID_PANICOCCURRED;
    if (isExit || isPanic) {
        Fw::LogBuffer copy = args;
        copy.resetDeser();
        I32 code = 0;
        if (copy.deserialize(code) == Fw::FW_SERIALIZE_OK) {
            this->m_result->hasExitCode = true;
            this->m_result->exitCode = code;
            this->m_result->errorCode = static_cast<U8>(code == 0 ? 0 : 1);
        }
        return;
    }
    if (relativeId == WasmSequencerComponentBase::EVENTID_SEQUENCEFAILED) {
        this->m_result->errorCode = 1;
        return;
    }
    // A module that never loads runs nothing, so it would otherwise leave a
    // clean result behind and read as success.
    if (relativeId == WasmSequencerComponentBase::EVENTID_MODULELOADFAILED) {
        this->m_result->error = "module_load_failed";
    } else if (relativeId == WasmSequencerComponentBase::EVENTID_MODULEINVOKEFAILED) {
        this->m_result->error = "module_invoke_failed";
    }
}

void WasmSequencerTester::handleSerialOut(FwIndexType portNum, Fw::LinearBufferBase& buffer) {
    FW_ASSERT(this->m_result != nullptr);
    this->m_result->serialWrites.emplace_back(
        portNum, std::vector<uint8_t>(buffer.getBuffAddr(),
                                      buffer.getBuffAddr() + buffer.getBuffLength()));
}

// ----------------------------------------------------------------------
// Static thunks
// ----------------------------------------------------------------------

void WasmSequencerTester::cmdOutThunk(Fw::PassiveComponentBase* comp,
                                      FwIndexType portNum,
                                      Fw::ComBuffer& data,
                                      U32 context) {
    static_cast<WasmSequencerTester*>(comp)->handleCmdOut(data, context);
}

void WasmSequencerTester::timeThunk(Fw::PassiveComponentBase* comp,
                                    FwIndexType portNum,
                                    Fw::Time& time) {
    static_cast<WasmSequencerTester*>(comp)->handleTime(time);
}

Fw::TlmValid WasmSequencerTester::getTlmChanThunk(Fw::PassiveComponentBase* comp,
                                                  FwIndexType portNum,
                                                  FwChanIdType id,
                                                  Fw::Time& timeTag,
                                                  Fw::TlmBuffer& val) {
    return static_cast<WasmSequencerTester*>(comp)->handleGetTlmChan(id, timeTag, val);
}

Fw::ParamValid WasmSequencerTester::getParamThunk(Fw::PassiveComponentBase* comp,
                                                  FwIndexType portNum,
                                                  FwPrmIdType id,
                                                  Fw::ParamBuffer& val) {
    return static_cast<WasmSequencerTester*>(comp)->handleGetParam(id, val);
}

void WasmSequencerTester::prmSetThunk(Fw::PassiveComponentBase* comp,
                                      FwIndexType portNum,
                                      FwPrmIdType id,
                                      Fw::ParamBuffer& val) {}

void WasmSequencerTester::logTextThunk(Fw::PassiveComponentBase* comp,
                                       FwIndexType portNum,
                                       FwEventIdType id,
                                       Fw::Time& timeTag,
                                       const Fw::LogSeverity& severity,
                                       Fw::TextLogString& text) {
    static_cast<WasmSequencerTester*>(comp)->handleLogText(severity, text);
}

void WasmSequencerTester::logThunk(Fw::PassiveComponentBase* comp,
                                   FwIndexType portNum,
                                   FwEventIdType id,
                                   Fw::Time& timeTag,
                                   const Fw::LogSeverity& severity,
                                   Fw::LogBuffer& args) {
    static_cast<WasmSequencerTester*>(comp)->handleLog(id, args);
}

void WasmSequencerTester::tlmOutThunk(Fw::PassiveComponentBase* comp,
                                      FwIndexType portNum,
                                      FwChanIdType id,
                                      Fw::Time& timeTag,
                                      Fw::TlmBuffer& val) {}

void WasmSequencerTester::cmdRegThunk(Fw::PassiveComponentBase* comp,
                                      FwIndexType portNum,
                                      FwOpcodeType opCode) {}

void WasmSequencerTester::cmdResponseThunk(Fw::PassiveComponentBase* comp,
                                           FwIndexType portNum,
                                           FwOpcodeType opCode,
                                           U32 cmdSeq,
                                           const Fw::CmdResponse& response) {}

void WasmSequencerTester::serialOutThunk(Fw::PassiveComponentBase* comp,
                                         FwIndexType portNum,
                                         Fw::LinearBufferBase& buffer) {
    static_cast<WasmSequencerTester*>(comp)->handleSerialOut(portNum, buffer);
}

}  // namespace Svc
