####
# test_harness module:
#
# The FpySequencerTester component that stands in for the rest of the topology
# around a real Svc::FpySequencer, plus the small JSON layer the harness uses
# to talk to pytest.
####

register_fprime_module(
  AUTOCODER_INPUTS
    "${CMAKE_CURRENT_LIST_DIR}/FpySequencerTester.fpp"
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/FpySequencerTester.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/Harness.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/Json.cpp"
  HEADERS
    "${CMAKE_CURRENT_LIST_DIR}/FpySequencerTester.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/Harness.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/Json.hpp"
  DEPENDS
    Svc_FpySequencer
)
