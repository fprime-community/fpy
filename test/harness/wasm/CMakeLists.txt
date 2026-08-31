####
# WasmSeqHarness:
#
# The test harness executable that runs compiled wasm sequences on the real
# Svc::WasmSequencer, from the fprime-wasm submodule (with the local patch in
# ../patches applied so exit codes get reported). Same stdin/stdout protocol
# as FpyHarness.
#
# Build with:
#   uv run fprime-util generate -d test/harness/wasm
#   uv run fprime-util build -d test/harness/wasm
####

cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0048 NEW)
project(WasmSeqHarness VERSION 1.0.0 LANGUAGES C CXX)

# The harness wires ports together at runtime, which direct port calls
# (the default outside unit-test builds) compile away.
add_compile_definitions(FW_DIRECT_PORT_CALLS=0)

include("${FPRIME_FRAMEWORK_PATH}/cmake/FPrime.cmake")
include("${FPRIME_FRAMEWORK_PATH}/cmake/FPrime-Code.cmake")

add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/tester")

register_fprime_deployment(
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/WasmSeqHarnessMain.cpp"
  DEPENDS
    test_harness_wasm_tester
)
