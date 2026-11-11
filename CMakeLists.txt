####
# Fpy test harnesses:
#
# The test harness executables that run compiled sequences on the real
# sequencers of the fprime submodule: FpySeqHarness runs fpy bytecode on
# Svc::FpySequencer, WasmSeqHarness runs wasm modules on Svc::WasmSequencer.
# The pytest suite starts them and talks to them over stdin/stdout; see
# test/harness/Harness.hpp.
#
# Build with the tools from the `harness` dependency group:
#   uv run fprime-util generate
#   uv run fprime-util build --target FpySeqHarness
#   uv run fprime-util build --target WasmSeqHarness
####

cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0048 NEW)
project(FpyHarnesses VERSION 1.0.0 LANGUAGES C CXX)

# The harnesses wire ports together at runtime, which direct port calls
# (the default outside unit-test builds) compile away.
add_compile_definitions(FW_DIRECT_PORT_CALLS=0)

include("${FPRIME_FRAMEWORK_PATH}/cmake/FPrime.cmake")
include("${FPRIME_FRAMEWORK_PATH}/cmake/FPrime-Code.cmake")

add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/test/harness/config")
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/test/harness")
