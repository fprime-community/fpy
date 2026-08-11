####
# The WasmSequencerTester component that stands in for the rest of the
# topology around a real Svc::WasmSequencer. The JSON protocol layer is
# shared with the FpySequencer harness (../../).
####

register_fprime_module(
  AUTOCODER_INPUTS
    "${CMAKE_CURRENT_LIST_DIR}/WasmSequencerTester.fpp"
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/WasmSequencerTester.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../../Harness.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../../Json.cpp"
  HEADERS
    "${CMAKE_CURRENT_LIST_DIR}/WasmSequencerTester.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/../../Harness.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/../../Json.hpp"
  DEPENDS
    Svc_WasmSequencer
)
