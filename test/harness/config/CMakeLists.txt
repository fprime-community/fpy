####
# test_harness_config:
#
# The F Prime configuration of the deployment the test dictionary
# (test/fpy/RefTopologyDictionary.json) describes, where it differs from the
# framework defaults. Each override is a copy of the framework's file with
# only the differing definitions changed.
####

register_fprime_config(
  CONFIGURATION_OVERRIDES
    "${CMAKE_CURRENT_LIST_DIR}/AcConstants.fpp"
    "${CMAKE_CURRENT_LIST_DIR}/FpConfig.fpp"
  INTERFACE
)
