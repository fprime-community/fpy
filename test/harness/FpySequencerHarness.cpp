// ======================================================================
// \title  FpySequencerHarness.cpp
// \brief  Runs sequences on a real Svc::FpySequencer
// ======================================================================

#include "FpySequencerHarness.hpp"

#include <climits>
#include <cstdio>
#include <unistd.h>

#include "Fw/Com/ComBuffer.hpp"
#include "Fw/Types/SerialBuffer.hpp"
#include "Svc/FpySequencer/FppConstantsAc.hpp"
#include "Svc/Seq/SeqArgsSerializableAc.hpp"

namespace Svc {

namespace {

//! Command opcodes, relative to the component's id base.
const FwOpcodeType RELATIVE_OPCODE_RUN = 0;
const FwOpcodeType RELATIVE_OPCODE_RUN_ARGS = 1;

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

//! Whether the state machine is in one of the states a sequence reaches only
//! after it has validated and begun executing.
bool isRunning(State state) {
    return state == State::RUNNING_AWAITING_STATEMENT_RESPONSE ||
           state == State::RUNNING_DISPATCH_STATEMENT || state == State::RUNNING_PAUSED ||
           state == State::RUNNING_SLEEPING;
}

const char* stateName(State state) {
    switch (state) {
        case State::AWAITING_CMD_RUN_VALIDATED: return "AWAITING_CMD_RUN_VALIDATED";
        case State::IDLE: return "IDLE";
        case State::RUNNING_AWAITING_STATEMENT_RESPONSE: return "RUNNING_AWAITING_STATEMENT_RESPONSE";
        case State::RUNNING_DISPATCH_STATEMENT: return "RUNNING_DISPATCH_STATEMENT";
        case State::RUNNING_PAUSED: return "RUNNING_PAUSED";
        case State::RUNNING_SLEEPING: return "RUNNING_SLEEPING";
        case State::VALIDATING: return "VALIDATING";
        default: return "UNINITIALIZED";
    }
}

//! STATEMENT_TIMEOUT_SECS is answered INVALID so the component falls back to
//! its own default of "never time out". Fast-forwarding simulated time would
//! otherwise trip a statement timeout on any long sleep.
const Fw::ParamValid PARAM_ANSWER_WHEN_UNSET = Fw::ParamValid::INVALID;

}  // namespace

FpySequencerTester::FpySequencerTester()
    : Fw::PassiveComponentBase("FpySequencerTester"),
      m_component("FpySequencer"),
      m_request(nullptr),
      m_result(nullptr) {
    this->init(INSTANCE_ID);
    this->m_component.init(QUEUE_DEPTH, INSTANCE_ID);
    this->m_component.allocateBuffer(0, this->m_allocator, Fpy::Sequence::SERIALIZED_SIZE);
}

FpySequencerTester::~FpySequencerTester() {
    this->m_component.deallocateBuffer(this->m_allocator);
    this->m_component.deinit();
}

// ----------------------------------------------------------------------
// Port wiring
// ----------------------------------------------------------------------

void FpySequencerTester::connectPorts(const HarnessRequest& request) {
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

    this->m_seqDoneOut.init();
    this->m_seqDoneOut.addCallComp(this, cmdResponseThunk);
    this->m_component.set_seqDoneOut_OutputPort(0, &this->m_seqDoneOut);

    this->m_pingOut.init();
    this->m_pingOut.addCallComp(this, pingThunk);
    this->m_component.set_pingOut_OutputPort(0, &this->m_pingOut);

    this->m_seqStartOut.init();
    this->m_seqStartOut.addCallComp(this, seqStartThunk);
    this->m_component.set_seqStartOut_OutputPort(0, &this->m_seqStartOut);

    for (FwIndexType i = 0; i < static_cast<FwIndexType>(Fpy::SerialPortIndex::MAX_SERIAL_PORTS); i++) {
        if (request.disconnectedSerialPorts.count(i) > 0) {
            continue;
        }
        this->m_serialOut[i].init();
        this->m_serialOut[i].addCallComp(this, serialOutThunk);
        this->m_serialOut[i].setPortNum(i);
        this->m_component.set_serialOut_OutputPort(i, &this->m_serialOut[i]);
    }
}

// ----------------------------------------------------------------------
// Running
// ----------------------------------------------------------------------

Fw::Time FpySequencerTester::now() const {
    return this->m_now;
}

HarnessResult FpySequencerTester::run(const HarnessRequest& request) {
    HarnessResult result;
    this->m_request = &request;
    this->m_result = &result;

    this->m_now = Fw::Time(static_cast<TimeBase::T>(request.timeBase), request.timeContext,
                           static_cast<U32>(request.initialTimeUs / 1000000),
                           static_cast<U32>(request.initialTimeUs % 1000000));

    this->connectPorts(request);
    this->m_component.regCommands();

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

    // RUN's fileName arrives as an Fw::CmdStringArg, which holds
    // FW_CMD_STRING_MAX_SIZE characters regardless of the width the command
    // declares. Longer paths do not survive the round trip, so the caller
    // supplies a working directory and a short path relative to it.
    Fw::CmdArgBuffer args;
    Fw::CmdStringArg path(request.seqPath.c_str());
    Fw::SerializeStatus status = args.serialize(path);
    FW_ASSERT(status == Fw::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
    status = args.serialize(Svc::BlockState(Svc::BlockState::BLOCK));
    FW_ASSERT(status == Fw::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));

    // A sequence that takes arguments is started through RUN_ARGS, which
    // carries them in a Svc::SeqArgs alongside the path.
    FwOpcodeType relativeOpcode = RELATIVE_OPCODE_RUN;
    if (!request.args.empty()) {
        relativeOpcode = RELATIVE_OPCODE_RUN_ARGS;
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
    }

    const FwOpcodeType opcode = relativeOpcode + this->m_component.getIdBase();
    this->m_component.get_cmdIn_InputPort(0)->invoke(opcode, 0, args);

    this->pump(request, result);

    result.errorCode = static_cast<U8>(this->m_component.m_tlm.lastDirectiveError.e);
    result.stackSize = this->m_component.m_runtime.stack.size;
    result.statementsDispatched = this->m_component.m_runtime.nextStatementIndex;
    result.simTimeUs = static_cast<U64>(this->m_now.getSeconds()) * 1000000 + this->m_now.getUSeconds();
    result.ok = result.error.empty();

    if (!previousDir.empty()) {
        const int rc = chdir(previousDir.c_str());
        FW_ASSERT(rc == 0, rc);
    }

    this->m_request = nullptr;
    this->m_result = nullptr;
    return result;
}

void FpySequencerTester::pump(const HarnessRequest& request, HarnessResult& result) {
    bool reachedRunning = false;
    U32 dispatches = 0;

    while (dispatches < request.maxDispatch) {
        if (this->m_component.m_queue.getMessagesAvailable() > 0) {
            this->m_component.doDispatch();
            dispatches++;
            reachedRunning = reachedRunning || isRunning(this->m_component.sequencer_getState());
            continue;
        }

        const State state = this->m_component.sequencer_getState();
        if (state == State::IDLE) {
            break;
        }
        if (state == State::RUNNING_SLEEPING) {
            // Nothing is queued and the component is waiting on the clock:
            // jump to its wakeup time so the sleep resolves on the next tick.
            const Fw::Time wakeup = this->m_component.m_runtime.wakeupTime;
            if (this->m_now < wakeup) {
                this->m_now = wakeup;
            }
            this->m_component.get_checkTimers_InputPort(0)->invoke(0);
            dispatches++;
            continue;
        }
        // Any other non-idle state with an empty queue is waiting on a timer
        // tick (a statement timeout or a `check` period).
        this->m_component.get_checkTimers_InputPort(0)->invoke(0);
        dispatches++;
        if (this->m_component.m_queue.getMessagesAvailable() == 0) {
            result.error = "stalled";
            break;
        }
    }

    if (dispatches >= request.maxDispatch) {
        result.error = "dispatch_limit";
    }

    result.validationFailed = !reachedRunning;
    result.finalState = stateName(this->m_component.sequencer_getState());
}

// ----------------------------------------------------------------------
// Command completion
// ----------------------------------------------------------------------

Fw::CmdResponse FpySequencerTester::respondTo(FwOpcodeType opcode,
                                              const Fw::ComBuffer& command,
                                              const HarnessRequest& request) {
    if (request.seqRunOpcodes.count(opcode) > 0) {
        return this->runChild(command, request);
    }
    if (request.failOpcodes.count(opcode) > 0) {
        return Fw::CmdResponse(Fw::CmdResponse::EXECUTION_ERROR);
    }
    return Fw::CmdResponse(static_cast<Fw::CmdResponse::T>(request.cmdResponse));
}

Fw::CmdResponse FpySequencerTester::runChild(const Fw::ComBuffer& command,
                                             const HarnessRequest& request) {
    // A sequence-run command carries (fileName, blockState, SeqArgs) after the
    // opcode. The deployment would hand this to a sequencer; here a nested
    // instance runs it and its outcome becomes the command's response.
    if (request.depth >= 8) {
        return Fw::CmdResponse(Fw::CmdResponse::EXECUTION_ERROR);
    }

    Fw::ComBuffer copy = command;
    copy.resetDeser();
    FwPacketDescriptorType descriptor = 0;
    Fw::SerializeStatus status = copy.deserialize(descriptor);
    if (status != Fw::FW_SERIALIZE_OK) {
        return Fw::CmdResponse(Fw::CmdResponse::FORMAT_ERROR);
    }
    FwOpcodeType opcode = 0;
    status = copy.deserialize(opcode);
    if (status != Fw::FW_SERIALIZE_OK) {
        return Fw::CmdResponse(Fw::CmdResponse::FORMAT_ERROR);
    }

    Fw::CmdStringArg fileName;
    status = copy.deserialize(fileName);
    if (status != Fw::FW_SERIALIZE_OK) {
        return Fw::CmdResponse(Fw::CmdResponse::FORMAT_ERROR);
    }
    Svc::BlockState block;
    status = copy.deserialize(block);
    if (status != Fw::FW_SERIALIZE_OK) {
        return Fw::CmdResponse(Fw::CmdResponse::FORMAT_ERROR);
    }

    HarnessRequest child = request;
    child.seqPath = fileName.toChar();
    child.depth = request.depth + 1;
    child.args.clear();

    Svc::SeqArgs seqArgs;
    if (copy.deserialize(seqArgs) == Fw::FW_SERIALIZE_OK) {
        const FwSizeType size = seqArgs.get_size();
        const U8* buffer = seqArgs.get_buffer();
        child.args.assign(buffer, buffer + size);
    }

    // The child inherits the clock so its now() continues from the parent's.
    child.initialTimeUs =
        static_cast<U64>(this->m_now.getSeconds()) * 1000000 + this->m_now.getUSeconds();

    FpySequencerTester childTester;
    const HarnessResult childResult = childTester.run(child);

    // Record what the child did, so a test can assert on it.
    if (this->m_result != nullptr) {
        for (const auto& cmd : childResult.cmds) {
            this->m_result->cmds.push_back(cmd);
        }
        for (const auto& event : childResult.events) {
            this->m_result->events.push_back(event);
        }
    }

    const bool childSucceeded = childResult.ok && !childResult.validationFailed &&
                                childResult.errorCode == 0;
    return Fw::CmdResponse(childSucceeded ? Fw::CmdResponse::OK : Fw::CmdResponse::EXECUTION_ERROR);
}

// ----------------------------------------------------------------------
// Port handlers
// ----------------------------------------------------------------------

void FpySequencerTester::handleCmdOut(Fw::ComBuffer& data, U32 context) {
    FW_ASSERT(this->m_request != nullptr);
    FW_ASSERT(this->m_result != nullptr);

    // The dispatched buffer is a command packet: a packet descriptor, then the
    // opcode, then the arguments. Only the opcode onwards is the command.
    const FwOpcodeType opcode = readOpcode(data);
    const FwSizeType prefix = sizeof(FwPacketDescriptorType);
    FW_ASSERT(data.getBuffLength() >= prefix, static_cast<FwAssertArgType>(data.getBuffLength()));
    this->m_result->cmds.emplace_back(data.getBuffAddr() + prefix,
                                      data.getBuffAddr() + data.getBuffLength());

    const Fw::CmdResponse response = this->respondTo(opcode, data, *this->m_request);
    this->m_component.get_cmdResponseIn_InputPort(0)->invoke(opcode, context, response);
}

void FpySequencerTester::handleTime(Fw::Time& time) {
    time = this->m_now;
}

Fw::TlmValid FpySequencerTester::handleGetTlmChan(FwChanIdType id,
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

Fw::ParamValid FpySequencerTester::handleGetParam(FwPrmIdType id, Fw::ParamBuffer& val) {
    FW_ASSERT(this->m_request != nullptr);
    const auto it = this->m_request->prm.find(id);
    if (it == this->m_request->prm.end()) {
        val.setBuffLen(0);
        return PARAM_ANSWER_WHEN_UNSET;
    }
    const Fw::SerializeStatus status =
        val.setBuff(it->second.data(), static_cast<FwSizeType>(it->second.size()));
    FW_ASSERT(status == Fw::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
    return Fw::ParamValid::VALID;
}

void FpySequencerTester::handleLogText(FwEventIdType id,
                                       Fw::Time& timeTag,
                                       const Fw::LogSeverity& severity,
                                       Fw::TextLogString& text) {
    FW_ASSERT(this->m_result != nullptr);
    this->m_result->events.emplace_back(static_cast<U32>(severity.e), text.toChar());
}

void FpySequencerTester::handleLog(FwEventIdType id,
                                   Fw::Time& timeTag,
                                   const Fw::LogSeverity& severity,
                                   Fw::LogBuffer& args) {
    FW_ASSERT(this->m_result != nullptr);
    // exit() does not carry its code out through the component's error state;
    // the code only appears in this event, so it is read back here.
    const FwEventIdType relativeId = id - this->m_component.getIdBase();
    if (relativeId != FpySequencerComponentBase::EVENTID_SEQUENCEEXITEDWITHERROR) {
        return;
    }
    Fw::LogBuffer copy = args;
    copy.resetDeser();
    Fw::LogStringArg filePath;
    if (copy.deserialize(filePath) != Fw::FW_SERIALIZE_OK) {
        return;
    }
    I32 code = 0;
    if (copy.deserialize(code) != Fw::FW_SERIALIZE_OK) {
        return;
    }
    this->m_result->hasExitCode = true;
    this->m_result->exitCode = code;
}

void FpySequencerTester::handleSerialOut(FwIndexType portNum, Fw::LinearBufferBase& buffer) {
    FW_ASSERT(this->m_result != nullptr);
    this->m_result->serialWrites.emplace_back(
        portNum, std::vector<uint8_t>(buffer.getBuffAddr(),
                                      buffer.getBuffAddr() + buffer.getBuffLength()));
}

// ----------------------------------------------------------------------
// Static thunks
// ----------------------------------------------------------------------

void FpySequencerTester::cmdOutThunk(Fw::PassiveComponentBase* comp,
                                     FwIndexType portNum,
                                     Fw::ComBuffer& data,
                                     U32 context) {
    static_cast<FpySequencerTester*>(comp)->handleCmdOut(data, context);
}

void FpySequencerTester::timeThunk(Fw::PassiveComponentBase* comp, FwIndexType portNum, Fw::Time& time) {
    static_cast<FpySequencerTester*>(comp)->handleTime(time);
}

Fw::TlmValid FpySequencerTester::getTlmChanThunk(Fw::PassiveComponentBase* comp,
                                                 FwIndexType portNum,
                                                 FwChanIdType id,
                                                 Fw::Time& timeTag,
                                                 Fw::TlmBuffer& val) {
    return static_cast<FpySequencerTester*>(comp)->handleGetTlmChan(id, timeTag, val);
}

Fw::ParamValid FpySequencerTester::getParamThunk(Fw::PassiveComponentBase* comp,
                                                 FwIndexType portNum,
                                                 FwPrmIdType id,
                                                 Fw::ParamBuffer& val) {
    return static_cast<FpySequencerTester*>(comp)->handleGetParam(id, val);
}

void FpySequencerTester::prmSetThunk(Fw::PassiveComponentBase* comp,
                                     FwIndexType portNum,
                                     FwPrmIdType id,
                                     Fw::ParamBuffer& val) {}

void FpySequencerTester::logTextThunk(Fw::PassiveComponentBase* comp,
                                      FwIndexType portNum,
                                      FwEventIdType id,
                                      Fw::Time& timeTag,
                                      const Fw::LogSeverity& severity,
                                      Fw::TextLogString& text) {
    static_cast<FpySequencerTester*>(comp)->handleLogText(id, timeTag, severity, text);
}

void FpySequencerTester::logThunk(Fw::PassiveComponentBase* comp,
                                  FwIndexType portNum,
                                  FwEventIdType id,
                                  Fw::Time& timeTag,
                                  const Fw::LogSeverity& severity,
                                  Fw::LogBuffer& args) {
    static_cast<FpySequencerTester*>(comp)->handleLog(id, timeTag, severity, args);
}

void FpySequencerTester::tlmOutThunk(Fw::PassiveComponentBase* comp,
                                     FwIndexType portNum,
                                     FwChanIdType id,
                                     Fw::Time& timeTag,
                                     Fw::TlmBuffer& val) {}

void FpySequencerTester::cmdRegThunk(Fw::PassiveComponentBase* comp,
                                     FwIndexType portNum,
                                     FwOpcodeType opCode) {}

void FpySequencerTester::cmdResponseThunk(Fw::PassiveComponentBase* comp,
                                          FwIndexType portNum,
                                          FwOpcodeType opCode,
                                          U32 cmdSeq,
                                          const Fw::CmdResponse& response) {}

void FpySequencerTester::pingThunk(Fw::PassiveComponentBase* comp, FwIndexType portNum, U32 key) {}

void FpySequencerTester::seqStartThunk(Fw::PassiveComponentBase* comp,
                                       FwIndexType portNum,
                                       const Fw::StringBase& fileName,
                                       const Svc::SeqArgs& args) {}

void FpySequencerTester::serialOutThunk(Fw::PassiveComponentBase* comp,
                                                       FwIndexType portNum,
                                                       Fw::LinearBufferBase& buffer) {
    static_cast<FpySequencerTester*>(comp)->handleSerialOut(portNum, buffer);
}

}  // namespace Svc
