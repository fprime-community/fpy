####
# FpySeqHarness:
#
# Runs compiled fpy sequences on a real Svc::FpySequencer. The
# FpySequencerTester component stands in for the rest of the topology around
# it.
####

register_fprime_module(
  AUTOCODER_INPUTS
    "${CMAKE_CURRENT_LIST_DIR}/FpySequencerTester.fpp"
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/FpySequencerTester.cpp"
  HEADERS
    "${CMAKE_CURRENT_LIST_DIR}/FpySequencerTester.hpp"
  DEPENDS
    Svc_FpySequencer
    test_harness
)

register_fprime_deployment(
  FpySeqHarness
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/FpySeqHarnessMain.cpp"
  DEPENDS
    test_harness_fpy
)
