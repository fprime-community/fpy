#include "test/harness/FpySequencerTester.hpp"

#include <cstring>

#include "Fw/Com/ComPacket.hpp"
#include "config/SerialPortIndexEnumAc.hpp"

namespace Svc {

FpySequencerTester::FpySequencerTester()
    : FpySequencerTesterComponentBase("FpySequencerTester"), m_sequencer("FpySequencer") {
    this->init(0);
    this->m_sequencer.init(QUEUE_DEPTH, 0);
    this->connectPorts();
    // Give the sequencer its file buffer, sized for the largest possible
    // sequence, and let it load its parameters (they all fall back to their
    // defaults; see prmGetIn_handler).
    this->m_sequencer.allocateBuffer(0, this->m_allocator, Fpy::Sequence::SERIALIZED_SIZE);
    this->m_sequencer.loadParameters();
}

FpySequencerTester::~FpySequencerTester() {
    this->m_sequencer.deallocateBuffer(this->m_allocator);
}

void FpySequencerTester::connectPorts() {
    FpySequencer& seq = this->m_sequencer;

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
    seq.set_seqStartOut_OutputPort(0, this->get_seqStartIn_InputPort(0));
    seq.set_seqDoneOut_OutputPort(0, this->get_seqDoneIn_InputPort(0));
    seq.set_pingOut_OutputPort(0, this->get_pingIn_InputPort(0));
    for (FwIndexType i = 0; i < Fpy::SerialPortIndex::MAX_SERIAL_PORTS; i++) {
        seq.set_serialOut_OutputPort(i, this->get_serialIn_InputPort(i));
    }

    // Tester outputs, into the sequencer.
    this->set_cmdSend_OutputPort(0, seq.get_cmdIn_InputPort(0));
    this->set_cmdResponseSend_OutputPort(0, seq.get_cmdResponseIn_InputPort(0));
    this->set_schedSend_OutputPort(0, seq.get_checkTimers_InputPort(0));
}

harness::HarnessResult FpySequencerTester::run(const harness::HarnessRequest& request) {
    this->m_request = &request;
    this->m_result = harness::HarnessResult();
    this->m_now = Fw::Time(static_cast<TimeBase::T>(request.timeBase), request.timeContext, request.seconds,
                           request.useconds);

    this->sendRunCommand();
    if (this->m_result.error.empty()) {
        this->pump();
    }

    // Report the sequencer's final internal state alongside the command
    // response, so the Python side can cross-check the two.
    FpySequencer& seq = this->m_sequencer;
    this->m_result.state = static_cast<I32>(seq.sequencer_getState());
    this->m_result.statementsDispatched = seq.m_statementsDispatched;
    this->m_result.lastDirectiveError = static_cast<I32>(seq.m_tlm.lastDirectiveError);
    this->m_result.sequencesSucceeded = seq.m_tlm.sequencesSucceeded;
    this->m_result.frameStart = seq.m_runtime.stack.currentFrameStart;
    this->m_result.stack.assign(seq.m_runtime.stack.bytes, seq.m_runtime.stack.bytes + seq.m_runtime.stack.size);

    this->m_request = nullptr;
    return this->m_result;
}

void FpySequencerTester::sendRunCommand() {
    const harness::HarnessRequest& request = *this->m_request;

    // The file path travels as a command string argument, exactly as it
    // would from ground control. Fail loudly instead of letting the command
    // layer truncate it (FW_CMD_STRING_MAX_SIZE includes room it reserves
    // for the terminator).
    if (request.seqFile.size() >= FW_CMD_STRING_MAX_SIZE) {
        this->m_result.error = "sequence file path is too long for a command string argument: " + request.seqFile;
        return;
    }

    Fw::CmdArgBuffer args;
    Fw::SerializeStatus status = args.serializeFrom(Fw::CmdStringArg(request.seqFile.c_str()));
    FW_ASSERT(status == Fw::SerializeStatus::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
    status = args.serializeFrom(Svc::BlockState(Svc::BlockState::BLOCK));
    FW_ASSERT(status == Fw::SerializeStatus::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));

    FwOpcodeType opcode = FpySequencer::OPCODE_RUN;
    if (request.hasArgs) {
        Svc::SeqArgs seqArgs;
        if (request.args.size() > sizeof seqArgs.get_buffer()) {
            this->m_result.error = "sequence arguments do not fit in Svc::SeqArgs";
            return;
        }
        seqArgs.set_size(request.args.size());
        (void)std::memcpy(seqArgs.get_buffer(), request.args.data(), request.args.size());
        status = args.serializeFrom(seqArgs);
        FW_ASSERT(status == Fw::SerializeStatus::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
        opcode = FpySequencer::OPCODE_RUN_ARGS;
    }

    this->cmdSend_out(0, opcode, 0, args);
}

void FpySequencerTester::pump() {
    FpySequencer& seq = this->m_sequencer;

    for (U32 dispatches = 0; dispatches < MAX_DISPATCHES; dispatches++) {
        if (seq.m_queue.getMessagesAvailable() == 0) {
            if (this->m_result.gotCmdResponse) {
                // The RUN command has its answer and the sequencer is quiet:
                // the run is over.
                return;
            }
            State state = seq.sequencer_getState();
            if (state == State::RUNNING_SLEEPING) {
                // Jump the clock to the wake-up time instead of waiting, then
                // let the sequencer check its timers. On a time base mismatch
                // the clock is left alone; the sequencer will fail the
                // sequence itself when it compares the times.
                const Fw::Time& wakeup = seq.m_runtime.wakeupTime;
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
        this->m_result.reachedRunning =
            this->m_result.reachedRunning || seq.isRunningState(seq.sequencer_getState());
    }
    this->m_result.error = "dispatch cap hit; the sequence appears to loop forever";
}

void FpySequencerTester::comCmdIn_handler(FwIndexType portNum, Fw::ComBuffer& data, U32 context) {
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
    // value: that is where the sequencer's command UID travels.
    this->cmdResponseSend_out(0, opcode, context, response);
}

Fw::CmdResponse FpySequencerTester::runChildSequence(const U8* args, FwSizeType argsSize) {
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

    FpySequencerTester child;
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

void FpySequencerTester::cmdResponseIn_handler(FwIndexType portNum,
                                               FwOpcodeType opCode,
                                               U32 cmdSeq,
                                               const Fw::CmdResponse& response) {
    // The only command the tester ever sends to the sequencer is RUN (or
    // RUN_ARGS), so this is its answer.
    this->m_result.gotCmdResponse = true;
    this->m_result.cmdResponse = static_cast<I32>(response.e);
}

void FpySequencerTester::cmdRegIn_handler(FwIndexType portNum, FwOpcodeType opCode) {}

void FpySequencerTester::logIn_handler(FwIndexType portNum,
                                       FwEventIdType id,
                                       Fw::Time& timeTag,
                                       const Fw::LogSeverity& severity,
                                       Fw::LogBuffer& args) {
    if (id == FpySequencer::EVENTID_SEQUENCEEXITEDWITHERROR) {
        // The event's arguments are the file path and the exit code.
        args.resetDeser();
        Fw::LogStringArg filePath;
        I32 exitCode = 0;
        if (args.deserializeTo(filePath) == Fw::SerializeStatus::FW_SERIALIZE_OK &&
            args.deserializeTo(exitCode) == Fw::SerializeStatus::FW_SERIALIZE_OK) {
            this->m_result.exited = true;
            this->m_result.exitCode = exitCode;
        }
    }
}

void FpySequencerTester::logTextIn_handler(FwIndexType portNum,
                                           FwEventIdType id,
                                           Fw::Time& timeTag,
                                           const Fw::LogSeverity& severity,
                                           Fw::TextLogString& text) {
    harness::HarnessEvent event;
    event.id = id;
    event.severity = static_cast<I32>(severity.e);
    event.text = text.toChar();
    this->m_result.events.push_back(event);
}

void FpySequencerTester::tlmIn_handler(FwIndexType portNum, FwChanIdType id, Fw::Time& timeTag, Fw::TlmBuffer& val) {}

Fw::TlmValid FpySequencerTester::tlmGetIn_handler(FwIndexType portNum,
                                                  FwChanIdType id,
                                                  Fw::Time& timeTag,
                                                  Fw::TlmBuffer& val) {
    const auto entry = this->m_request->tlm.find(id);
    if (entry == this->m_request->tlm.end()) {
        val.resetSer();
        return Fw::TlmValid::INVALID;
    }
    val.resetSer();
    Fw::SerializeStatus status = val.serializeFrom(entry->second.data(), entry->second.size(), Fw::Serialization::OMIT_LENGTH);
    FW_ASSERT(status == Fw::SerializeStatus::FW_SERIALIZE_OK, static_cast<FwAssertArgType>(status));
    timeTag = this->m_now;
    return Fw::TlmValid::VALID;
}

Fw::ParamValid FpySequencerTester::prmGetIn_handler(FwIndexType portNum, FwPrmIdType id, Fw::ParamBuffer& val) {
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

void FpySequencerTester::prmSetIn_handler(FwIndexType portNum, FwPrmIdType id, Fw::ParamBuffer& val) {}

void FpySequencerTester::timeGetIn_handler(FwIndexType portNum, Fw::Time& time) {
    time = this->m_now;
}

void FpySequencerTester::seqStartIn_handler(FwIndexType portNum,
                                            const Fw::StringBase& filename,
                                            const Svc::SeqArgs& args) {}

void FpySequencerTester::seqDoneIn_handler(FwIndexType portNum,
                                           FwOpcodeType opCode,
                                           U32 cmdSeq,
                                           const Fw::CmdResponse& response) {}

void FpySequencerTester::pingIn_handler(FwIndexType portNum, U32 key) {}

void FpySequencerTester::serialIn_handler(FwIndexType portNum, Fw::LinearBufferBase& buffer) {
    harness::HarnessSerialWrite write;
    write.port = portNum;
    write.data.assign(buffer.getBuffAddr(), buffer.getBuffAddr() + buffer.getSize());
    this->m_result.serial.push_back(write);
}

}  // namespace Svc
