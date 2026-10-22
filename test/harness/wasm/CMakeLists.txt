####
# WasmSeqHarness:
#
# Runs compiled wasm sequences on a real Svc::WasmSequencer. The
# WasmSequencerTester component stands in for the rest of the topology around
# it.
####

# Svc_WasmSequencer is disabled without a usable Rust toolchain; FpySeqHarness
# still builds then.
if (NOT TARGET Svc_WasmSequencer)
    fprime_cmake_status("[harness] WasmSeqHarness disabled: Svc_WasmSequencer is not available")
    return()
endif()

register_fprime_module(
  AUTOCODER_INPUTS
    "${CMAKE_CURRENT_LIST_DIR}/WasmSequencerTester.fpp"
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/WasmSequencerTester.cpp"
  HEADERS
    "${CMAKE_CURRENT_LIST_DIR}/WasmSequencerTester.hpp"
  DEPENDS
    Svc_WasmSequencer
    test_harness
)

register_fprime_deployment(
  WasmSeqHarness
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/WasmSeqHarnessMain.cpp"
  DEPENDS
    test_harness_wasm
)
