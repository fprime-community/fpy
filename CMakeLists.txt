####
# CMakeLists.txt:
#
# Builds the sequencer harness that fpy's test suite runs compiled sequences on.
# The harness links the real Svc::FpySequencer out of the test/fprime submodule.
####
cmake_minimum_required(VERSION 3.18)
project(FpyHarnessProject C CXX)

set(FPRIME_FRAMEWORK_PATH "${CMAKE_CURRENT_LIST_DIR}/test/fprime" CACHE PATH "Location of F prime framework" FORCE)
set(FPRIME_PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Root path of F prime project" FORCE)

# Port objects, rather than direct calls between components. F Prime compiles
# the Fw::Input*Port classes out when direct calls are on, and connecting a
# component to a handler of our own needs them. This is the mode the framework
# selects for its own test builds.
add_compile_definitions(FW_DIRECT_PORT_CALLS=0)

include("${FPRIME_FRAMEWORK_PATH}/cmake/FPrime.cmake")
include("${FPRIME_FRAMEWORK_PATH}/cmake/FPrime-Code.cmake")

add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/test/harness")
