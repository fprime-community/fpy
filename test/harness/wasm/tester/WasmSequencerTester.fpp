module Svc {
    @ The mirror side of every WasmSequencer port, so one instance of this
    @ component can stand in for the rest of the topology around a real
    @ WasmSequencer. The implementation class is named WasmSequencerTester
    @ because WasmSequencer declares a friend class of that name, which gives
    @ the harness access to the sequencer's internal state (message queue,
    @ state machine, sleep timer).
    passive component WasmSequencerTester {

        @ receives commands the sequence dispatches (from cmdOut)
        sync input port comCmdIn: Fw.Com

        @ receives responses to commands sent to the sequencer (from cmdResponseOut)
        sync input port cmdResponseIn: Fw.CmdResponse

        @ receives command registrations (from cmdRegOut)
        sync input port cmdRegIn: Fw.CmdReg

        @ receives events (from logOut)
        sync input port logIn: Fw.Log

        @ receives text events (from logTextOut)
        sync input port logTextIn: Fw.LogText

        @ receives telemetry writes (from tlmOut)
        sync input port tlmIn: Fw.Tlm

        @ serves telemetry channel reads (for getTlmChan)
        sync input port tlmGetIn: Fw.TlmGet

        @ serves parameter reads (for getParam and prmGet)
        sync input port prmGetIn: [2] Fw.PrmGet

        @ receives parameter writes (from prmSet)
        sync input port prmSetIn: Fw.PrmSet

        @ serves the simulated clock (for timeCaller)
        sync input port timeGetIn: Fw.Time

        @ receives data the sequence writes to serial ports (from serialSyncOut)
        sync input port serialIn: [Svc.Fpy.SerialPortIndex.MAX_SERIAL_PORTS] serial

        @ sends commands to the sequencer (to cmdIn)
        output port cmdSend: Fw.Cmd

        @ sends command responses to the sequencer (to cmdResponseIn)
        output port cmdResponseSend: Fw.CmdResponse

        @ drives the sequencer's timer check (to checkTimers)
        output port schedSend: Svc.Sched
    }
}
