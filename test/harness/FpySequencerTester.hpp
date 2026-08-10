// The harness side of a test run: one FpySequencerTester owns one real
// Svc::FpySequencer, wires every sequencer port to itself, and plays the rest
// of the spacecraft. It sends the same RUN command ground control would send,
// then dispatches the sequencer's queued messages until the sequencer answers
// that command. While the sequencer sleeps, the tester jumps its simulated
// clock forward to the wake-up time, so waits cost nothing.
//
// The class must be named Svc::FpySequencerTester: FpySequencer declares that
// class a friend, which is what lets the harness dispatch the message queue
// and read internal state (state machine, stack, telemetry) directly.

#ifndef TEST_HARNESS_FPYSEQUENCERTESTER_HPP
#define TEST_HARNESS_FPYSEQUENCERTESTER_HPP

#include "Fw/Types/MallocAllocator.hpp"
#include "Svc/FpySequencer/FpySequencer.hpp"
#include "test/harness/FpySequencerTesterComponentAc.hpp"
#include "test/harness/Harness.hpp"

namespace Svc {

class FpySequencerTester : public FpySequencerTesterComponentBase {
  public:
    FpySequencerTester();
    ~FpySequencerTester();

    // Runs one sequence on the owned sequencer and reports what happened.
    // Sequence arguments, when present, make the tester start the sequence
    // with RUN_ARGS instead of RUN.
    harness::HarnessResult run(const harness::HarnessRequest& request);

  private:
    // Upper bound on queue dispatches per run: turns a sequence that loops
    // forever into a clean harness error. Sleeps don't count against it in
    // real time because the clock jumps instead of waiting.
    static constexpr U32 MAX_DISPATCHES = 1000 * 1000;
    static constexpr FwSizeType QUEUE_DEPTH = 32;

    void connectPorts();
    // Sends the RUN (or RUN_ARGS) command that starts the sequence.
    void sendRunCommand();
    // Dispatches queued messages until the sequencer answers the RUN
    // command, jumping the clock forward whenever the sequencer sleeps.
    void pump();
    // Runs a seq-run command's child sequence on a nested tester and returns
    // the command response the child's outcome maps to.
    Fw::CmdResponse runChildSequence(const U8* args, FwSizeType argsSize);

    void comCmdIn_handler(FwIndexType portNum, Fw::ComBuffer& data, U32 context) override;
    void cmdResponseIn_handler(FwIndexType portNum,
                               FwOpcodeType opCode,
                               U32 cmdSeq,
                               const Fw::CmdResponse& response) override;
    void cmdRegIn_handler(FwIndexType portNum, FwOpcodeType opCode) override;
    void logIn_handler(FwIndexType portNum,
                       FwEventIdType id,
                       Fw::Time& timeTag,
                       const Fw::LogSeverity& severity,
                       Fw::LogBuffer& args) override;
    void logTextIn_handler(FwIndexType portNum,
                           FwEventIdType id,
                           Fw::Time& timeTag,
                           const Fw::LogSeverity& severity,
                           Fw::TextLogString& text) override;
    void tlmIn_handler(FwIndexType portNum, FwChanIdType id, Fw::Time& timeTag, Fw::TlmBuffer& val) override;
    Fw::TlmValid tlmGetIn_handler(FwIndexType portNum, FwChanIdType id, Fw::Time& timeTag, Fw::TlmBuffer& val) override;
    Fw::ParamValid prmGetIn_handler(FwIndexType portNum, FwPrmIdType id, Fw::ParamBuffer& val) override;
    void prmSetIn_handler(FwIndexType portNum, FwPrmIdType id, Fw::ParamBuffer& val) override;
    void timeGetIn_handler(FwIndexType portNum, Fw::Time& time) override;
    void seqStartIn_handler(FwIndexType portNum, const Fw::StringBase& filename, const Svc::SeqArgs& args) override;
    void seqDoneIn_handler(FwIndexType portNum,
                           FwOpcodeType opCode,
                           U32 cmdSeq,
                           const Fw::CmdResponse& response) override;
    void pingIn_handler(FwIndexType portNum, U32 key) override;
    void serialIn_handler(FwIndexType portNum, Fw::LinearBufferBase& buffer) override;

    FpySequencer m_sequencer;
    Fw::MallocAllocator m_allocator;
    const harness::HarnessRequest* m_request = nullptr;
    harness::HarnessResult m_result;
    // The simulated clock, served to the sequencer through its time port.
    Fw::Time m_now;
};

}  // namespace Svc

#endif
