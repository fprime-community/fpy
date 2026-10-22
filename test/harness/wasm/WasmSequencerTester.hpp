// The harness side of a wasm test run: one WasmSequencerTester owns one real
// Svc::WasmSequencer, wires every sequencer port to itself, and plays the
// rest of the spacecraft. It sends the same RUN command ground control would
// send, then dispatches the sequencer's queued messages until the sequencer
// answers that command. The interpreter runs in fuel-limited slices that feed
// themselves through the queue; when the sequencer sleeps, the tester jumps
// its simulated clock to the wake-up time and nudges the timer check.
//
// The class must be named Svc::WasmSequencerTester: WasmSequencer declares
// that class a friend, which is what lets the harness dispatch the message
// queue and read internal state.

#ifndef TEST_HARNESS_WASM_WASMSEQUENCERTESTER_HPP
#define TEST_HARNESS_WASM_WASMSEQUENCERTESTER_HPP

#include "Fw/Types/MallocAllocator.hpp"
#include "Svc/WasmSequencer/WasmSequencer.hpp"
#include "test/harness/Harness.hpp"
#include "test/harness/wasm/tester/WasmSequencerTesterComponentAc.hpp"

namespace Svc {

class WasmSequencerTester : public WasmSequencerTesterComponentBase {
  public:
    WasmSequencerTester();

    // Runs one wasm module on the owned sequencer and reports what happened.
    harness::HarnessResult run(const harness::HarnessRequest& request);

  private:
    // Upper bound on queue dispatches per run: turns a module that loops
    // forever into a clean harness error. Each dispatch executes up to
    // INSTRUCTION_FUEL (default 1000) wasm instructions.
    static constexpr U32 MAX_DISPATCHES = 1000 * 1000;
    static constexpr FwSizeType QUEUE_DEPTH = 32;
    // Simulated period of the checkTimers port: how far the clock advances
    // when the sequencer sleeps until a time that has already come.
    static constexpr U32 CHECK_TIMERS_PERIOD_USEC = 100 * 1000;

    void connectPorts();
    // Sends the RUN command that loads and runs the module.
    void sendRunCommand();
    // Dispatches queued messages until the sequencer answers the RUN
    // command, jumping the clock forward whenever the sequencer sleeps.
    void pump();
    // Runs the child module a dispatched seq-run command names on a fresh
    // tester, returning the response the command completes with.
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
    void pingIn_handler(FwIndexType portNum, U32 key);
    void serialIn_handler(FwIndexType portNum, Fw::LinearBufferBase& buffer) override;

    // True when the event is one the guest program logged (its log builtin).
    static bool isGuestLogEvent(FwEventIdType id);

    // Backs the sequencer's configure() pools; declared before the sequencer,
    // whose destructor deallocates through it.
    Fw::MallocAllocator m_allocator;
    WasmSequencer m_sequencer;
    const harness::HarnessRequest* m_request = nullptr;
    harness::HarnessResult m_result;
    // The simulated clock, served to the sequencer through its time port.
    Fw::Time m_now;
};

}  // namespace Svc

#endif
