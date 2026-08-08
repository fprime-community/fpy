####
# CMakeLists.txt:
#
# Builds the sequencer harnesses that fpy's test suite runs compiled sequences
# on. Each links a real flight sequencer component:
#
#   FpyHarness      Svc::FpySequencer, from the test/fprime submodule
#   WasmSeqHarness  Svc::WasmSequencer, from FPY_WASM_SEQUENCER_DIR
#
# Svc::WasmSequencer is not upstream yet, so it comes from a second submodule
# tracking the branch it lives on. Nothing is copied out of that submodule:
# the component's sources, its fpp, and its config header are all compiled
# where they sit, so pulling in new work is a submodule bump and nothing else.
# When it merges, point FPY_WASM_SEQUENCER_DIR at test/fprime and delete the
# test/fprime-wasm submodule -- that is the whole migration.
####
cmake_minimum_required(VERSION 3.19)
project(FpyHarnessProject C CXX)

set(FPRIME_FRAMEWORK_PATH "${CMAKE_CURRENT_LIST_DIR}/test/fprime" CACHE PATH "Location of F prime framework" FORCE)
set(FPRIME_PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Root path of F prime project" FORCE)

set(FPY_WASM_SEQUENCER_ROOT "${CMAKE_CURRENT_LIST_DIR}/test/fprime-wasm"
    CACHE PATH "F Prime checkout holding Svc/WasmSequencer")

# Port objects, rather than direct calls between components. F Prime compiles
# the Fw::Input*Port classes out when direct calls are on, and connecting a
# component to a handler of our own needs them. This is the mode the framework
# selects for its own test builds.
add_compile_definitions(FW_DIRECT_PORT_CALLS=0)

# The WasmSequencer's interpreter is a Rust static library, so its build needs
# cargo. Without it the component is skipped rather than failing the build.
find_program(CARGO NAMES cargo)
if (CARGO)
    message(STATUS "[rust] cargo found at: ${CARGO}")
endif()

include("${FPRIME_FRAMEWORK_PATH}/cmake/FPrime.cmake")
include("${FPRIME_FRAMEWORK_PATH}/cmake/FPrime-Code.cmake")

# Register the WasmSequencer checkout as a source root so its sources resolve
# their own "Svc/WasmSequencer/..." includes and its autocode lands under the
# matching path in the build tree -- the same treatment the framework and the
# project root get.
if (CARGO AND EXISTS "${FPY_WASM_SEQUENCER_ROOT}/Svc/WasmSequencer/WasmSequencer.fpp")
    fprime_add_location_pair("${FPY_WASM_SEQUENCER_ROOT}" "${CMAKE_BINARY_DIR}")
    # The component's config header lives with its checkout, not with ours.
    include_directories("${FPY_WASM_SEQUENCER_ROOT}/default")
    # Its own CMakeLists builds the Rust interpreter and sets its include
    # paths, so it is used as-is rather than restated here: updating the
    # component stays a submodule bump.
    add_fprime_subdirectory("${FPY_WASM_SEQUENCER_ROOT}/Svc/WasmSequencer")
endif()

add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/test/harness")
