####
# FpyHarness:
#
# The test harness executable that runs compiled fpy sequences on the real
# Svc::FpySequencer. The pytest suite starts this program and talks to it over
# stdin/stdout; see test/harness/README.md.
#
# Build with the tools from the `harness` dependency group:
#   uv run fprime-util generate
#   uv run fprime-util build
####

cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0048 NEW)
project(FpyHarness VERSION 1.0.0 LANGUAGES C CXX)

# The harness wires ports together at runtime, which direct port calls
# (the default outside unit-test builds) compile away.
add_compile_definitions(FW_DIRECT_PORT_CALLS=0)

include("${FPRIME_FRAMEWORK_PATH}/cmake/FPrime.cmake")
include("${FPRIME_FRAMEWORK_PATH}/cmake/FPrime-Code.cmake")

add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/test/harness")

register_fprime_deployment(
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/test/harness/FpyHarnessMain.cpp"
  DEPENDS
    test_harness
)
