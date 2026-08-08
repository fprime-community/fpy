// ======================================================================
// \title  WasmSequencerHarness.hpp
// \brief  Runs wasm sequences on a real Svc::WasmSequencer
// ======================================================================

#ifndef WASM_SEQUENCER_HARNESS_HPP
#define WASM_SEQUENCER_HARNESS_HPP

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "Fw/Cmd/CmdRegPortAc.hpp"
#include "Fw/Cmd/CmdResponsePortAc.hpp"
#include "Fw/Com/ComPortAc.hpp"
#include "Fw/Comp/PassiveComponentBase.hpp"
#include "Fw/Log/LogPortAc.hpp"
#include "Fw/Log/LogTextPortAc.hpp"
#include "Fw/Port/InputSerializePort.hpp"
#include "Fw/Prm/PrmGetPortAc.hpp"
#include "Fw/Prm/PrmSetPortAc.hpp"
#include "Fw/Time/TimePortAc.hpp"
#include "Fw/Tlm/TlmGetPortAc.hpp"
#include "Fw/Tlm/TlmPortAc.hpp"
#include "Fw/Types/Serializable.hpp"
#include "HarnessRequest.hpp"
#include "Svc/Sched/SchedPortAc.hpp"
#include "Svc/WasmSequencer/WasmSequencer.hpp"
#include "config/SerialPortIndexEnumAc.hpp"

namespace Svc {

//! Drives a real Svc::WasmSequencer with no topology and no wall clock.
//!
//! Named WasmSequencerTester because Svc::WasmSequencer declares that class a
//! friend, which is what grants access to the component's internals.
class WasmSequencerTester final : public Fw::PassiveComponentBase {
  public:
    static const FwSizeType QUEUE_DEPTH = 100;
    static const FwSizeType INSTANCE_ID = 0;
    static const FwIndexType SERIAL_PORTS =
        static_cast<FwIndexType>(Fpy::SerialPortIndex::MAX_SERIAL_PORTS);

    WasmSequencerTester();
    ~WasmSequencerTester();

    //! Runs one wasm module to completion.
    HarnessResult run(const HarnessRequest& request);

  private:
    void connectPorts(const HarnessRequest& request);
    void pump(const HarnessRequest& request, HarnessResult& result);
    Fw::CmdResponse respondTo(FwOpcodeType opcode, const HarnessRequest& request);

    void handleCmdOut(Fw::ComBuffer& data, U32 context);
    void handleTime(Fw::Time& time);
    Fw::TlmValid handleGetTlmChan(FwChanIdType id, Fw::Time& timeTag, Fw::TlmBuffer& val);
    Fw::ParamValid handleGetParam(FwPrmIdType id, Fw::ParamBuffer& val);
    void handleLogText(const Fw::LogSeverity& severity, Fw::TextLogString& text);
    void handleLog(FwEventIdType id, Fw::LogBuffer& args);
    void handleSerialOut(FwIndexType portNum, Fw::LinearBufferBase& buffer);

    static void cmdOutThunk(Fw::PassiveComponentBase* comp,
                            FwIndexType portNum,
                            Fw::ComBuffer& data,
                            U32 context);
    static void timeThunk(Fw::PassiveComponentBase* comp, FwIndexType portNum, Fw::Time& time);
    static Fw::TlmValid getTlmChanThunk(Fw::PassiveComponentBase* comp,
                                        FwIndexType portNum,
                                        FwChanIdType id,
                                        Fw::Time& timeTag,
                                        Fw::TlmBuffer& val);
    static Fw::ParamValid getParamThunk(Fw::PassiveComponentBase* comp,
                                        FwIndexType portNum,
                                        FwPrmIdType id,
                                        Fw::ParamBuffer& val);
    static void prmSetThunk(Fw::PassiveComponentBase* comp,
                            FwIndexType portNum,
                            FwPrmIdType id,
                            Fw::ParamBuffer& val);
    static void logTextThunk(Fw::PassiveComponentBase* comp,
                             FwIndexType portNum,
                             FwEventIdType id,
                             Fw::Time& timeTag,
                             const Fw::LogSeverity& severity,
                             Fw::TextLogString& text);
    static void logThunk(Fw::PassiveComponentBase* comp,
                         FwIndexType portNum,
                         FwEventIdType id,
                         Fw::Time& timeTag,
                         const Fw::LogSeverity& severity,
                         Fw::LogBuffer& args);
    static void tlmOutThunk(Fw::PassiveComponentBase* comp,
                            FwIndexType portNum,
                            FwChanIdType id,
                            Fw::Time& timeTag,
                            Fw::TlmBuffer& val);
    static void cmdRegThunk(Fw::PassiveComponentBase* comp,
                            FwIndexType portNum,
                            FwOpcodeType opCode);
    static void cmdResponseThunk(Fw::PassiveComponentBase* comp,
                                 FwIndexType portNum,
                                 FwOpcodeType opCode,
                                 U32 cmdSeq,
                                 const Fw::CmdResponse& response);
    static void serialOutThunk(Fw::PassiveComponentBase* comp,
                               FwIndexType portNum,
                               Fw::LinearBufferBase& buffer);

    WasmSequencer m_component;

    Fw::InputComPort m_cmdOut;
    Fw::InputTimePort m_timeCaller;
    Fw::InputTlmGetPort m_getTlmChan;
    Fw::InputPrmGetPort m_getParam;
    Fw::InputPrmGetPort m_prmGet;
    Fw::InputPrmSetPort m_prmSet;
    Fw::InputLogPort m_logOut;
    Fw::InputLogTextPort m_logTextOut;
    Fw::InputTlmPort m_tlmOut;
    Fw::InputCmdRegPort m_cmdRegOut;
    Fw::InputCmdResponsePort m_cmdResponseOut;
    Fw::InputSerializePort m_serialSyncOut[SERIAL_PORTS];
    Fw::InputSerializePort m_serialAsyncOut[SERIAL_PORTS];

    const HarnessRequest* m_request;
    HarnessResult* m_result;
    Fw::Time m_now;
};

}  // namespace Svc

#endif
