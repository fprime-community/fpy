####
# Harness configuration overrides for the fprime-wasm framework.
####

register_fprime_config(
        harness_config
    CONFIGURATION_OVERRIDES
        "${CMAKE_CURRENT_LIST_DIR}/WasmSequencerConfig.hpp"
    INTERFACE
)
