// ======================================================================
// \title  FpySequencerHarness.hpp
// \brief  Runs sequences on a real Svc::FpySequencer
// ======================================================================

#ifndef FPY_SEQUENCER_HARNESS_HPP
#define FPY_SEQUENCER_HARNESS_HPP

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
#include "Fw/Types/MallocAllocator.hpp"
#include "Fw/Types/Serializable.hpp"
#include "HarnessRequest.hpp"
#include "Fw/Prm/PrmGetPortAc.hpp"
#include "Fw/Prm/PrmSetPortAc.hpp"
#include "Fw/Time/TimePortAc.hpp"
#include "Fw/Tlm/TlmGetPortAc.hpp"
#include "Fw/Tlm/TlmPortAc.hpp"
#include "Svc/FpySequencer/FpySequencer.hpp"
#include "Svc/Ping/PingPortAc.hpp"
#include "Svc/Seq/CmdSeqInPortAc.hpp"
#include "config/SerialPortIndexEnumAc.hpp"

namespace Svc {

//! Drives a real Svc::FpySequencer with no topology and no wall clock.
//!
//! Named FpySequencerTester because Svc::FpySequencer declares that class a
//! friend, which is what grants access to the component's runtime state.
class FpySequencerTester final : public Fw::PassiveComponentBase {
  public:
    static const FwSizeType QUEUE_DEPTH = 100;
    static const FwSizeType INSTANCE_ID = 0;

    FpySequencerTester();
    ~FpySequencerTester();

    //! Runs one sequence to completion.
    HarnessResult run(const HarnessRequest& request);

  private:
    void connectPorts(const HarnessRequest& request);

    //! Advances the run until the component goes idle, blocks, or hits the
    //! dispatch bound.
    void pump(const HarnessRequest& request, HarnessResult& result);

    //! Completes *opcode*, running a child sequence first when the request
    //! says the opcode names one.
    Fw::CmdResponse respondTo(FwOpcodeType opcode,
                              const Fw::ComBuffer& command,
                              const HarnessRequest& request);

    //! Runs the child sequence named by a seq-run command's arguments.
    Fw::CmdResponse runChild(const Fw::ComBuffer& command, const HarnessRequest& request);

    Fw::Time now() const;

    // Port handlers, reached through the static thunks below.
    void handleCmdOut(Fw::ComBuffer& data, U32 context);
    void handleTime(Fw::Time& time);
    Fw::TlmValid handleGetTlmChan(FwChanIdType id, Fw::Time& timeTag, Fw::TlmBuffer& val);
    Fw::ParamValid handleGetParam(FwPrmIdType id, Fw::ParamBuffer& val);
    void handleLogText(FwEventIdType id,
                       Fw::Time& timeTag,
                       const Fw::LogSeverity& severity,
                       Fw::TextLogString& text);
    void handleLog(FwEventIdType id,
                   Fw::Time& timeTag,
                   const Fw::LogSeverity& severity,
                   Fw::LogBuffer& args);
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
    static void cmdRegThunk(Fw::PassiveComponentBase* comp, FwIndexType portNum, FwOpcodeType opCode);
    static void cmdResponseThunk(Fw::PassiveComponentBase* comp,
                                 FwIndexType portNum,
                                 FwOpcodeType opCode,
                                 U32 cmdSeq,
                                 const Fw::CmdResponse& response);
    static void pingThunk(Fw::PassiveComponentBase* comp, FwIndexType portNum, U32 key);
    static void seqStartThunk(Fw::PassiveComponentBase* comp,
                              FwIndexType portNum,
                              const Fw::StringBase& fileName,
                              const Svc::SeqArgs& args);
    static void serialOutThunk(Fw::PassiveComponentBase* comp,
                                              FwIndexType portNum,
                                              Fw::LinearBufferBase& buffer);

    FpySequencer m_component;
    Fw::MallocAllocator m_allocator;

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
    Fw::InputCmdResponsePort m_seqDoneOut;
    Svc::InputPingPort m_pingOut;
    Svc::InputCmdSeqInPort m_seqStartOut;
    Fw::InputSerializePort m_serialOut[Fpy::SerialPortIndex::MAX_SERIAL_PORTS];

    //! The request and result in flight, so port handlers can reach them.
    const HarnessRequest* m_request;
    HarnessResult* m_result;
    Fw::Time m_now;
};

}  // namespace Svc

#endif
