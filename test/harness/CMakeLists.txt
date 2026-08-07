####
# FpyHarness: drives the real Svc::FpySequencer over a stdin/stdout protocol.
#
# The component is linked as an ordinary library. No unit-test autocode is
# involved: Svc::FpySequencer declares `friend class FpySequencerTester`, so the
# harness declares that class itself and wires the ports by hand.
####

register_fprime_deployment(
    FpyHarness
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/FpyHarnessMain.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/FpySequencerHarness.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/Json.cpp"
  HEADERS
    "${CMAKE_CURRENT_LIST_DIR}/FpySequencerHarness.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/Json.hpp"
  DEPENDS
    Svc_FpySequencer
    Svc_Ping
    Svc_Sched
    Svc_Seq
    Fw_Comp
    Fw_Cmd
    Fw_Log
    Fw_Tlm
    Fw_Prm
    Fw_Time
    Fw_Com
    Fw_Types
    Os
)

# fprime's flight warning set (-Wold-style-cast, -Wconversion, -Wshadow, -Werror)
# is aimed at flight code; the harness is host-only test scaffolding.
target_compile_options(FpyHarness PRIVATE
    -Wno-old-style-cast -Wno-conversion -Wno-shadow -Wno-error)
