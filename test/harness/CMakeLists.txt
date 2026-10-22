####
# test_harness module:
#
# The protocol both harness executables speak with pytest, plus the small
# JSON layer it is built on. Each harness executable lives in its own
# subdirectory, with the tester component that stands in for the rest of the
# topology around its sequencer.
####

register_fprime_module(
  SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/Harness.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/Json.cpp"
  HEADERS
    "${CMAKE_CURRENT_LIST_DIR}/Harness.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/Json.hpp"
  DEPENDS
    Fw_Types
)

add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/fpy")
add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/wasm")
