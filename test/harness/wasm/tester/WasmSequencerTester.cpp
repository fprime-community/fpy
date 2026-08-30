#include "test/harness/wasm/tester/WasmSequencerTester.hpp"

#include <cstring>

#include "Fw/Com/ComPacket.hpp"
#include "config/SerialPortIndexEnumAc.hpp"

namespace Svc {

using WasmState = WasmSequencer_InterpreterStateMachine_State;

WasmSequencerTester::WasmSequencerTester()
    : WasmSequencerTesterComponentBase("WasmSequencerTester"), m_sequencer("WasmSequencer") {
    this->init(0);
    this->m_sequencer.init(QUEUE_DEPTH, 0);
    this->connectPorts();
    this->m_sequencer.loadParameters();
}

void WasmSequencerTester::connectPorts() {
    WasmSequencer& seq = this->m_sequencer;

    // Sequencer outputs, wired back to this tester.
    seq.set_cmdOut_OutputPort(0, this->get_comCmdIn_InputPort(0));
    seq.set_cmdResponseOut_OutputPort(0, this->get_cmdResponseIn_InputPort(0));
    seq.set_cmdRegOut_OutputPort(0, this->get_cmdRegIn_InputPort(0));
    seq.set_logOut_OutputPort(0, this->get_logIn_InputPort(0));
#if FW_ENABLE_TEXT_LOGGING
    seq.set_logTextOut_OutputPort(0, this->get_logTextIn_InputPort(0));
#endif
    seq.set_tlmOut_OutputPort(0, this->get_tlmIn_InputPort(0));
    seq.set_getTlmChan_OutputPort(0, this->get_tlmGetIn_InputPort(0));
    seq.set_getParam_OutputPort(0, this->get_prmGetIn_InputPort(0));
    seq.set_prmGet_OutputPort(0, this->get_prmGetIn_InputPort(1));
    seq.set_prmSet_OutputPort(0, this->get_prmSetIn_InputPort(0));
    seq.set_timeCaller_OutputPort(0, this->get_timeGetIn_InputPort(0));
    // serialReply stays unconnected: the sequencer requires that for ports
    // the guest writes synchronously, the only way the compiler writes them.
    for (FwIndexType i = 0; i < Fpy::SerialPortIndex::MAX_SERIAL_PORTS; i++) {
        seq.set_serialOut_OutputPort(i, this->get_serialIn_InputPort(i));
    }

    // Tester outputs, into the sequencer.
    this->set_cmdSend_OutputPort(0, seq.get_cmdIn_InputPort(0));
    this->set_cmdResponseSend_OutputPort(0, seq.get_cmdResponseIn_InputPort(0));
    this->set_schedSend_OutputPort(0, seq.get_checkTimers_InputPort(0));
}

harness::HarnessResult WasmSequencerTester::run(const harness::HarnessRequest& request) {
    this->m_request = &request;
    this->m_result = harness::HarnessResult();
    this->m_now = Fw::Time(static_cast<TimeBase::T>(request.timeBase), request.timeContext, request.seconds,
                           request.useconds);

    this->sendRunCommand();
    if (this->m_result.error.empty()) {
        this->pump();
    }

    WasmSequencer& seq = this->m_sequencer;
    this->m_result.state = static_cast<I32>(seq.interpreter_getState());
    this->m_result.sequencesSucceeded = seq.m_tlm.sequencesSucceeded;
    this->m_result.statementsDispatched = seq.m_tlm.commandsDispatched;

    this->m_request = nullptr;
    return this->m_result;
}

void WasmSequencerTester::sendRunCommand() {
    const harness::HarnessRequest& request = *this->m_request;

    // The file path travels as a command string argument, exactly as it
    // would from ground control; F Prime silently caps those at
    // FW_CMD_STRING_MAX_SIZE, so fail loudly instead.
    if (request.seqFile.size() >= FW_CMD_STRING_MAX_SIZE) {
        this->m_result.error = "wasm module path is too long for a command string argument: " + request.seqFile;
        return;
    }

    Svc::SeqArgs seqArgs;
    if (request.args.size() > sizeof seqArgs.get_buffer()) {
        this->m_result.error = "sequence arguments do not fit in Svc::SeqArgs";
        return;
    }
    seqArgs.set_size(request.args.size());
    if (!request.args.empty()) {
        (void)std::memcpy(seqArgs.get_buffer(), request.args.data(), request.args.size());
    }

    Fw::CmdArgBuffer args;
    Fw::SerializeStatus status = args.serializeFrom(Fw::CmdStringArg(request.seqFile.c_str()));
    FW_ASSERT(status == Fw::SerializeStatus::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
    status = args.serializeFrom(Svc::BlockState(Svc::BlockState::BLOCK));
    FW_ASSERT(status == Fw::SerializeStatus::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
    status = args.serializeFrom(seqArgs);
    FW_ASSERT(status == Fw::SerializeStatus::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));

    this->cmdSend_out(0, WasmSequencer::OPCODE_RUN, 0, args);
}

void WasmSequencerTester::pump() {
    WasmSequencer& seq = this->m_sequencer;

    for (U32 dispatches = 0; dispatches < MAX_DISPATCHES; dispatches++) {
        if (seq.m_queue.getMessagesAvailable() == 0) {
            if (this->m_result.gotCmdResponse) {
                // The RUN command has its answer and the sequencer is quiet:
                // the run is over.
                return;
            }
            WasmState state = seq.interpreter_getState();
            if (state == WasmState::RUNNING_AWAITING_RESPONSE_SLEEPING && seq.m_hasPendingTimer) {
                // The sequence is sleeping. Jump the clock to the wake-up
                // time instead of waiting, then let the sequencer check its
                // timers. On a time base mismatch the clock is left alone;
                // the sequencer fails the sequence itself when it compares
                // the times.
                const Fw::Time& wakeup = seq.m_pendingTimer;
                if (wakeup.getTimeBase() == this->m_now.getTimeBase() && this->m_now < wakeup) {
                    this->m_now = wakeup;
                }
                this->schedSend_out(0, 0);
                continue;
            }
            this->m_result.error =
                "sequencer went quiet in state " + std::to_string(static_cast<I32>(state)) + " without responding";
            return;
        }

        (void)seq.doDispatch();
        WasmState state = seq.interpreter_getState();
        this->m_result.reachedRunning = this->m_result.reachedRunning || state == WasmState::RUNNING_SPINNING;
    }
    this->m_result.error = "dispatch cap hit; the sequence appears to loop forever";
}

void WasmSequencerTester::comCmdIn_handler(FwIndexType portNum, Fw::ComBuffer& data, U32 context) {
    const harness::HarnessRequest& request = *this->m_request;

    FwPacketDescriptorType descriptor = 0;
    FwOpcodeType opcode = 0;
    Fw::SerializeStatus status = data.deserializeTo(descriptor);
    if (status == Fw::SerializeStatus::FW_SERIALIZE_OK) {
        status = data.deserializeTo(opcode);
    }
    if (status != Fw::SerializeStatus::FW_SERIALIZE_OK ||
        descriptor != static_cast<FwPacketDescriptorType>(Fw::ComPacketType::FW_PACKET_COMMAND)) {
        this->m_result.error = "sequence dispatched a malformed command packet";
        return;
    }

    // Record the command as the wire bytes past the packet descriptor:
    // serialized opcode followed by the argument bytes.
    const U8* packet = data.getBuffAddr();
    FwSizeType packetSize = data.getSize();
    const U8* cmd = packet + sizeof(FwPacketDescriptorType);
    FwSizeType cmdSize = packetSize - sizeof(FwPacketDescriptorType);
    this->m_result.cmds.emplace_back(cmd, cmd + cmdSize);

    Fw::CmdResponse response(static_cast<Fw::CmdResponse::T>(request.cmdResponse));
    if (request.seqRunOpcodes.count(opcode) > 0) {
        const U8* args = cmd + sizeof(FwOpcodeType);
        FwSizeType argsSize = cmdSize - sizeof(FwOpcodeType);
        response = this->runChildSequence(args, argsSize);
    } else if (request.failOpcodes.count(opcode) > 0) {
        response = Fw::CmdResponse::EXECUTION_ERROR;
    }

    // Answer right away, echoing the context back as the command sequence
    // value.
    this->cmdResponseSend_out(0, opcode, context, response);
}

Fw::CmdResponse WasmSequencerTester::runChildSequence(const U8* args, FwSizeType argsSize) {
    const harness::HarnessRequest& request = *this->m_request;

    // Parse (fileName, blockState, seqArgs) following the dictionary layout
    // the compiler serialized: the SeqArgs buffer length comes from the
    // request because the dictionary's length can differ from this build's.
    Fw::ExternalSerializeBuffer buffer(const_cast<U8*>(args), argsSize);
    Fw::SerializeStatus status = buffer.setBuffLen(argsSize);
    FW_ASSERT(status == Fw::SerializeStatus::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));

    Fw::CmdStringArg fileName;
    Svc::BlockState blockState;
    FwSizeType childArgsSize = 0;
    if (buffer.deserializeTo(fileName) != Fw::SerializeStatus::FW_SERIALIZE_OK ||
        buffer.deserializeTo(blockState) != Fw::SerializeStatus::FW_SERIALIZE_OK ||
        buffer.deserializeTo(childArgsSize) != Fw::SerializeStatus::FW_SERIALIZE_OK) {
        this->m_result.error = "could not parse the arguments of a seq-run command";
        return Fw::CmdResponse::EXECUTION_ERROR;
    }
    // The size field counts only the used bytes of the fixed-capacity SeqArgs
    // buffer, so any value up to the capacity is valid; a larger value would
    // point past the end of the buffer.
    if (buffer.getBuffLeft() != request.seqArgsBufferSize || childArgsSize > request.seqArgsBufferSize) {
        this->m_result.error = "seq-run command arguments do not match the dictionary's SeqArgs layout";
        return Fw::CmdResponse::EXECUTION_ERROR;
    }
    const U8* childArgs = buffer.getBuffAddr() + (argsSize - buffer.getBuffLeft());

    harness::HarnessRequest childRequest = request;
    childRequest.seqFile = fileName.toChar();
    childRequest.hasArgs = true;
    childRequest.args.assign(childArgs, childArgs + childArgsSize);
    // The child starts at the parent's current clock; the parent's clock does
    // not move while the child runs.
    childRequest.timeBase = static_cast<U16>(this->m_now.getTimeBase());
    childRequest.timeContext = this->m_now.getContext();
    childRequest.seconds = this->m_now.getSeconds();
    childRequest.useconds = this->m_now.getUSeconds();

    WasmSequencerTester child;
    harness::HarnessResult childResult = child.run(childRequest);
    if (!childResult.error.empty()) {
        this->m_result.error = "child sequence " + childRequest.seqFile + ": " + childResult.error;
        return Fw::CmdResponse::EXECUTION_ERROR;
    }
    if (childResult.gotCmdResponse && childResult.cmdResponse == Fw::CmdResponse::OK) {
        return Fw::CmdResponse::OK;
    }
    return Fw::CmdResponse::EXECUTION_ERROR;
}

void WasmSequencerTester::cmdResponseIn_handler(FwIndexType portNum,
                                                FwOpcodeType opCode,
                                                U32 cmdSeq,
                                                const Fw::CmdResponse& response) {
    // The only command the tester ever sends to the sequencer is RUN, so
    // this is its answer.
    this->m_result.gotCmdResponse = true;
    this->m_result.cmdResponse = static_cast<I32>(response.e);
}

void WasmSequencerTester::cmdRegIn_handler(FwIndexType portNum, FwOpcodeType opCode) {}

bool WasmSequencerTester::isGuestLogEvent(FwEventIdType id) {
    switch (id) {
        case WasmSequencer::EVENTID_LOGFATAL:
        case WasmSequencer::EVENTID_LOGWARNINGHI:
        case WasmSequencer::EVENTID_LOGWARNINGLO:
        case WasmSequencer::EVENTID_LOGCOMMAND:
        case WasmSequencer::EVENTID_LOGACTIVITYHI:
        case WasmSequencer::EVENTID_LOGACTIVITYLO:
        case WasmSequencer::EVENTID_LOGDIAGNOSTIC:
            return true;
        default:
            return false;
    }
}

void WasmSequencerTester::logIn_handler(FwIndexType portNum,
                                        FwEventIdType id,
                                        Fw::Time& timeTag,
                                        const Fw::LogSeverity& severity,
                                        Fw::LogBuffer& args) {
    if (id == WasmSequencer::EVENTID_SEQUENCEEXITED || id == WasmSequencer::EVENTID_SEQUENCEPANIC) {
        // The code the guest passed to the exit or panic host function. Only
        // a nonzero code raises these events (exit(0) finishes cleanly). The
        // code follows the module index and execution phase.
        args.resetDeser();
        WasmSequencer_ModuleIdx index = 0;
        WasmSequencer_SequencePhase phase;
        I32 exitCode = 0;
        if (args.deserializeTo(index) == Fw::SerializeStatus::FW_SERIALIZE_OK &&
            args.deserializeTo(phase) == Fw::SerializeStatus::FW_SERIALIZE_OK &&
            args.deserializeTo(exitCode) == Fw::SerializeStatus::FW_SERIALIZE_OK) {
            this->m_result.exited = true;
            this->m_result.exitCode = exitCode;
        }
        return;
    }
    if (isGuestLogEvent(id)) {
        // A message the guest program logged: report the raw message, not
        // the formatted event text.
        args.resetDeser();
        Fw::LogStringArg message;
        if (args.deserializeTo(message) == Fw::SerializeStatus::FW_SERIALIZE_OK) {
            harness::HarnessEvent event;
            event.id = id;
            event.severity = static_cast<I32>(severity.e);
            event.text = message.toChar();
            event.guest = true;
            this->m_result.events.push_back(event);
        }
    }
}

void WasmSequencerTester::logTextIn_handler(FwIndexType portNum,
                                            FwEventIdType id,
                                            Fw::Time& timeTag,
                                            const Fw::LogSeverity& severity,
                                            Fw::TextLogString& text) {
    if (isGuestLogEvent(id)) {
        // Reported through logIn with the raw message instead.
        return;
    }
    harness::HarnessEvent event;
    event.id = id;
    event.severity = static_cast<I32>(severity.e);
    event.text = text.toChar();
    this->m_result.events.push_back(event);
}

void WasmSequencerTester::tlmIn_handler(FwIndexType portNum, FwChanIdType id, Fw::Time& timeTag, Fw::TlmBuffer& val) {}

Fw::TlmValid WasmSequencerTester::tlmGetIn_handler(FwIndexType portNum,
                                                   FwChanIdType id,
                                                   Fw::Time& timeTag,
                                                   Fw::TlmBuffer& val) {
    const auto entry = this->m_request->tlm.find(id);
    if (entry == this->m_request->tlm.end()) {
        val.resetSer();
        return Fw::TlmValid::INVALID;
    }
    val.resetSer();
    Fw::SerializeStatus status =
        val.serializeFrom(entry->second.data(), entry->second.size(), Fw::Serialization::OMIT_LENGTH);
    FW_ASSERT(status == Fw::SerializeStatus::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
    timeTag = this->m_now;
    return Fw::TlmValid::VALID;
}

Fw::ParamValid WasmSequencerTester::prmGetIn_handler(FwIndexType portNum, FwPrmIdType id, Fw::ParamBuffer& val) {
    // Also called with no request active, when the constructor loads the
    // sequencer's parameters.
    if (this->m_request != nullptr) {
        const auto found = this->m_request->prms.find(id);
        if (found != this->m_request->prms.end()) {
            val.resetSer();
            Fw::SerializeStatus status =
                val.serializeFrom(found->second.data(), found->second.size(), Fw::Serialization::OMIT_LENGTH);
            FW_ASSERT(status == Fw::SerializeStatus::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
            return Fw::ParamValid::VALID;
        }
    }
    // No such parameter: the component falls back to the default value.
    return Fw::ParamValid::INVALID;
}

void WasmSequencerTester::prmSetIn_handler(FwIndexType portNum, FwPrmIdType id, Fw::ParamBuffer& val) {}

void WasmSequencerTester::timeGetIn_handler(FwIndexType portNum, Fw::Time& time) {
    time = this->m_now;
}

void WasmSequencerTester::pingIn_handler(FwIndexType portNum, U32 key) {}

void WasmSequencerTester::serialIn_handler(FwIndexType portNum, Fw::LinearBufferBase& buffer) {
    harness::HarnessSerialWrite write;
    write.port = portNum;
    write.data.assign(buffer.getBuffAddr(), buffer.getBuffAddr() + buffer.getSize());
    this->m_result.serial.push_back(write);
}

}  // namespace Svc
