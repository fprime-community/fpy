"""
Tests for FW type aliases (FwOpcodeType, FwPrmIdType, FwChanIdType, FwSizeType).

These tests verify that:
1. Type aliases are correctly loaded from the dictionary
2. The ConfigManager is populated with the correct types
3. Module-level type variables are updated after dictionary loading
4. Directive serialization uses the correct types from the dictionary
5. Different dictionary configurations produce different binary output
"""

import pytest
from unittest.mock import patch
from fprime_gds.common.models.serialize.numerical_types import (
    U8Type as U8Value,
    U16Type as U16Value,
    U32Type as U32Value,
    U64Type as U64Value,
)
from fprime_gds.common.utils.config_manager import ConfigManager

from fpy.bytecode.directives import (
    ConstCmdDirective,
    PushTlmValDirective,
    PushPrmDirective,
    get_fw_opcode_type,
    get_fw_prm_id_type,
    get_fw_chan_id_type,
    get_fw_size_type,
    update_fw_types_from_config,
    _DEFAULT_FW_OPCODE_TYPE,
    _DEFAULT_FW_PRM_ID_TYPE,
    _DEFAULT_FW_CHAN_ID_TYPE,
    _DEFAULT_FW_SIZE_TYPE,
)
from fpy.compiler import _load_dictionary


# Path to the test dictionary
TEST_DICTIONARY = "test/fpy/RefTopologyDictionary.json"


class TestFwTypeGetters:
    """Test that the getter functions return correct types."""

    def test_get_fw_opcode_type_returns_from_config(self):
        """After loading dictionary, getter should return type from ConfigManager."""
        _load_dictionary(TEST_DICTIONARY)
        opcode_type = get_fw_opcode_type()
        # The RefTopologyDictionary defines FwOpcodeType as U32
        assert opcode_type == U32Value

    def test_get_fw_prm_id_type_returns_from_config(self):
        """After loading dictionary, getter should return type from ConfigManager."""
        _load_dictionary(TEST_DICTIONARY)
        prm_id_type = get_fw_prm_id_type()
        # The RefTopologyDictionary defines FwPrmIdType as U32
        assert prm_id_type == U32Value

    def test_get_fw_chan_id_type_returns_from_config(self):
        """After loading dictionary, getter should return type from ConfigManager."""
        _load_dictionary(TEST_DICTIONARY)
        chan_id_type = get_fw_chan_id_type()
        # The RefTopologyDictionary defines FwChanIdType as U32
        assert chan_id_type == U32Value

    def test_get_fw_size_type_returns_from_config(self):
        """After loading dictionary, getter should return type from ConfigManager."""
        _load_dictionary(TEST_DICTIONARY)
        size_type = get_fw_size_type()
        # The RefTopologyDictionary defines FwSizeType as U64
        assert size_type == U64Value


class TestModuleLevelTypeUpdates:
    """Test that module-level type variables are updated after dictionary loading."""

    def test_module_level_types_updated_after_dictionary_load(self):
        """Module-level type variables should be updated after loading dictionary."""
        import fpy.bytecode.directives as directives

        _load_dictionary(TEST_DICTIONARY)

        # After loading, module-level variables should match ConfigManager
        assert directives.FwOpcodeType == get_fw_opcode_type()
        assert directives.FwPrmIdType == get_fw_prm_id_type()
        assert directives.FwChanIdType == get_fw_chan_id_type()
        assert directives.FwSizeType == get_fw_size_type()


class TestDirectiveSerialization:
    """Test that directives serialize using correct types from dictionary."""

    def test_const_cmd_directive_serializes_opcode_correctly(self):
        """ConstCmdDirective should serialize cmd_opcode using FwOpcodeType from dict."""
        _load_dictionary(TEST_DICTIONARY)

        # Create directive with int opcode
        opcode = 0x12345678
        directive = ConstCmdDirective(cmd_opcode=opcode, args=b"")

        # Serialize
        serialized = directive.serialize_args()

        # With U32 FwOpcodeType, opcode should be 4 bytes big-endian
        expected = U32Value(opcode).serialize()
        assert serialized == expected
        assert len(serialized) == 4  # U32 is 4 bytes

    def test_push_tlm_val_directive_serializes_chan_id_correctly(self):
        """PushTlmValDirective should serialize chan_id using FwChanIdType from dict."""
        _load_dictionary(TEST_DICTIONARY)

        # Create directive with int chan_id
        chan_id = 0xABCD
        directive = PushTlmValDirective(chan_id=chan_id)

        # Serialize
        serialized = directive.serialize_args()

        # With U32 FwChanIdType, chan_id should be 4 bytes big-endian
        expected = U32Value(chan_id).serialize()
        assert serialized == expected
        assert len(serialized) == 4  # U32 is 4 bytes

    def test_push_prm_directive_serializes_prm_id_correctly(self):
        """PushPrmDirective should serialize prm_id using FwPrmIdType from dict."""
        _load_dictionary(TEST_DICTIONARY)

        # Create directive with int prm_id
        prm_id = 0x1234
        directive = PushPrmDirective(prm_id=prm_id)

        # Serialize
        serialized = directive.serialize_args()

        # With U32 FwPrmIdType, prm_id should be 4 bytes big-endian
        expected = U32Value(prm_id).serialize()
        assert serialized == expected
        assert len(serialized) == 4  # U32 is 4 bytes


class TestDifferentTypeConfigurations:
    """
    Test that different type configurations produce different binary output.
    
    This simulates what would happen if a dictionary specified different sizes
    for the type aliases (e.g., U16 instead of U32 for FwOpcodeType).
    """

    def test_opcode_serialization_differs_with_different_types(self):
        """
        Verify that changing FwOpcodeType changes the serialized output.
        
        This test manually sets ConfigManager types to simulate a dictionary
        with different type configurations.
        """
        import fpy.bytecode.directives as directives

        opcode = 0x1234

        # First, configure with U32 (default)
        config = ConfigManager.get_instance()
        config.set_type("FwOpcodeType", U32Value)
        update_fw_types_from_config()

        directive_u32 = ConstCmdDirective(cmd_opcode=opcode, args=b"")
        serialized_u32 = directive_u32.serialize_args()

        # U32 serialization should be 4 bytes
        assert len(serialized_u32) == 4
        assert serialized_u32 == bytes([0x00, 0x00, 0x12, 0x34])

        # Now configure with U16
        config.set_type("FwOpcodeType", U16Value)
        update_fw_types_from_config()

        directive_u16 = ConstCmdDirective(cmd_opcode=opcode, args=b"")
        serialized_u16 = directive_u16.serialize_args()

        # U16 serialization should be 2 bytes
        assert len(serialized_u16) == 2
        assert serialized_u16 == bytes([0x12, 0x34])

        # The outputs should be different
        assert serialized_u32 != serialized_u16

        # Restore to default for other tests
        config.set_type("FwOpcodeType", U32Value)
        update_fw_types_from_config()

    def test_chan_id_serialization_differs_with_different_types(self):
        """
        Verify that changing FwChanIdType changes the serialized output.
        """
        import fpy.bytecode.directives as directives

        chan_id = 0x5678

        config = ConfigManager.get_instance()

        # Configure with U32
        config.set_type("FwChanIdType", U32Value)
        update_fw_types_from_config()

        directive_u32 = PushTlmValDirective(chan_id=chan_id)
        serialized_u32 = directive_u32.serialize_args()
        assert len(serialized_u32) == 4

        # Configure with U16
        config.set_type("FwChanIdType", U16Value)
        update_fw_types_from_config()

        directive_u16 = PushTlmValDirective(chan_id=chan_id)
        serialized_u16 = directive_u16.serialize_args()
        assert len(serialized_u16) == 2

        # Outputs should differ
        assert serialized_u32 != serialized_u16

        # Restore
        config.set_type("FwChanIdType", U32Value)
        update_fw_types_from_config()

    def test_prm_id_serialization_differs_with_different_types(self):
        """
        Verify that changing FwPrmIdType changes the serialized output.
        """
        prm_id = 0x9ABC

        config = ConfigManager.get_instance()

        # Configure with U32
        config.set_type("FwPrmIdType", U32Value)
        update_fw_types_from_config()

        directive_u32 = PushPrmDirective(prm_id=prm_id)
        serialized_u32 = directive_u32.serialize_args()
        assert len(serialized_u32) == 4

        # Configure with U16
        config.set_type("FwPrmIdType", U16Value)
        update_fw_types_from_config()

        directive_u16 = PushPrmDirective(prm_id=prm_id)
        serialized_u16 = directive_u16.serialize_args()
        assert len(serialized_u16) == 2

        # Outputs should differ
        assert serialized_u32 != serialized_u16

        # Restore
        config.set_type("FwPrmIdType", U32Value)
        update_fw_types_from_config()


class TestFullCycleWithDifferentConfigs:
    """
    Full cycle tests that verify the complete flow from ConfigManager
    configuration through directive creation and serialization.
    """

    def test_full_cycle_u32_opcode(self):
        """Full cycle test with U32 FwOpcodeType."""
        config = ConfigManager.get_instance()
        config.set_type("FwOpcodeType", U32Value)
        update_fw_types_from_config()

        # Verify getter returns correct type
        assert get_fw_opcode_type() == U32Value

        # Create and serialize directive
        opcode = 0xDEADBEEF
        directive = ConstCmdDirective(cmd_opcode=opcode, args=b"test")
        serialized = directive.serialize_args()

        # Verify serialization
        expected_opcode = bytes([0xDE, 0xAD, 0xBE, 0xEF])
        expected_args = b"test"
        assert serialized == expected_opcode + expected_args

    def test_full_cycle_u16_opcode(self):
        """Full cycle test with U16 FwOpcodeType."""
        config = ConfigManager.get_instance()
        config.set_type("FwOpcodeType", U16Value)
        update_fw_types_from_config()

        # Verify getter returns correct type
        assert get_fw_opcode_type() == U16Value

        # Create and serialize directive
        opcode = 0xBEEF  # Use value that fits in U16
        directive = ConstCmdDirective(cmd_opcode=opcode, args=b"test")
        serialized = directive.serialize_args()

        # Verify serialization - U16 should be 2 bytes
        expected_opcode = bytes([0xBE, 0xEF])
        expected_args = b"test"
        assert serialized == expected_opcode + expected_args

        # Restore for other tests
        config.set_type("FwOpcodeType", U32Value)
        update_fw_types_from_config()

    def test_full_cycle_u64_opcode(self):
        """Full cycle test with U64 FwOpcodeType (larger than default)."""
        config = ConfigManager.get_instance()
        config.set_type("FwOpcodeType", U64Value)
        update_fw_types_from_config()

        # Verify getter returns correct type
        assert get_fw_opcode_type() == U64Value

        # Create and serialize directive
        opcode = 0x123456789ABCDEF0
        directive = ConstCmdDirective(cmd_opcode=opcode, args=b"")
        serialized = directive.serialize_args()

        # Verify serialization - U64 should be 8 bytes
        expected = bytes([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0])
        assert serialized == expected
        assert len(serialized) == 8

        # Restore for other tests
        config.set_type("FwOpcodeType", U32Value)
        update_fw_types_from_config()


class TestDefaultFallback:
    """Test that defaults are used when ConfigManager doesn't have a type."""

    def test_defaults_are_correct(self):
        """Verify the default type constants are what we expect."""
        assert _DEFAULT_FW_OPCODE_TYPE == U32Value
        assert _DEFAULT_FW_PRM_ID_TYPE == U32Value
        assert _DEFAULT_FW_CHAN_ID_TYPE == U32Value
        assert _DEFAULT_FW_SIZE_TYPE == U64Value
