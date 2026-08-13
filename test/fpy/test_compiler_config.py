"""
Tests for loading sequence configuration from dictionary constants.

These tests verify that MAX_DIRECTIVE_SIZE and MAX_SEQUENCE_STATEMENT_COUNT
are correctly read from the dictionary's constants section.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import fpy.error
import fpy.types
from fpy.compiler import (
    text_to_ast,
    analyze_ast,
    analysis_to_fpybc_directives,
)
from fpy.state import (
    get_base_compile_state,
    _build_global_scopes,
)
from fpy.dictionary import load_dictionary
from fpy.types import (
    DEFAULT_MAX_DIRECTIVES_COUNT,
    DEFAULT_MAX_DIRECTIVE_SIZE,
    DEFAULT_MAX_SEQ_ARG_COUNT,
    DEFAULT_MAX_STACK_SIZE,
)

# Path to the test dictionary
DEFAULT_DICTIONARY = str(Path(__file__).parent / "RefTopologyDictionary.json")


def _clear_caches():
    """Clear all relevant caches so tests get fresh loads."""
    load_dictionary.cache_clear()
    _build_global_scopes.cache_clear()


def test_load_sequence_config_from_default_dictionary():
    """Test that sequence config is loaded from the standard test dictionary."""
    _clear_caches()

    state = get_base_compile_state(DEFAULT_DICTIONARY, {})

    # The RefTopologyDictionary.json has these values:
    # Svc.Fpy.MAX_SEQUENCE_STATEMENT_COUNT = 2048
    # Svc.Fpy.MAX_DIRECTIVE_SIZE = 2048
    assert state.max_directives_count == 2048
    assert state.max_directive_size == 2048


def test_compile_state_has_sequence_config():
    """Test that CompileState is populated with sequence config from dictionary."""
    _clear_caches()

    state = get_base_compile_state(DEFAULT_DICTIONARY, {})

    assert state.max_directives_count == 2048
    assert state.max_directive_size == 2048


def create_test_dictionary(constants: list[dict]) -> str:
    """
    Create a minimal test dictionary JSON file with specified constants.
    Returns the path to the temporary file.
    """
    # Load the real dictionary to get the structure
    with open(DEFAULT_DICTIONARY, "r") as f:
        base_dict = json.load(f)

    # Replace constants with our test constants, keeping the original ones
    # that aren't being overridden
    test_constant_names = {c["qualifiedName"] for c in constants}
    filtered_constants = [
        c
        for c in base_dict.get("constants", [])
        if c.get("qualifiedName") not in test_constant_names
    ]
    base_dict["constants"] = filtered_constants + constants

    # Write to temp file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(base_dict, temp_file)
    temp_file.close()

    return temp_file.name


def test_custom_max_directives_count():
    """Test that a custom MAX_SEQUENCE_STATEMENT_COUNT is loaded from dictionary."""
    _clear_caches()

    custom_count = 500
    dict_path = create_test_dictionary(
        [
            {
                "kind": "constant",
                "qualifiedName": "Svc.Fpy.MAX_SEQUENCE_STATEMENT_COUNT",
                "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
                "value": custom_count,
                "annotation": "Custom max sequence statement count",
            }
        ]
    )

    try:
        state = get_base_compile_state(dict_path, {})
        assert state.max_directives_count == custom_count
        # max_directive_size should still come from the base dictionary
        assert state.max_directive_size == 2048
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_custom_max_directive_size():
    """Test that a custom MAX_DIRECTIVE_SIZE is loaded from dictionary."""
    _clear_caches()

    custom_size = 4096
    dict_path = create_test_dictionary(
        [
            {
                "kind": "constant",
                "qualifiedName": "Svc.Fpy.MAX_DIRECTIVE_SIZE",
                "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
                "value": custom_size,
                "annotation": "Custom max directive size",
            }
        ]
    )

    try:
        state = get_base_compile_state(dict_path, {})
        assert state.max_directive_size == custom_size
        # max_directives_count should still come from the base dictionary
        assert state.max_directives_count == 2048
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_custom_both_limits():
    """Test that both custom limits can be set together."""
    _clear_caches()

    custom_count = 256
    custom_size = 1024
    dict_path = create_test_dictionary(
        [
            {
                "kind": "constant",
                "qualifiedName": "Svc.Fpy.MAX_SEQUENCE_STATEMENT_COUNT",
                "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
                "value": custom_count,
                "annotation": "Custom max sequence statement count",
            },
            {
                "kind": "constant",
                "qualifiedName": "Svc.Fpy.MAX_DIRECTIVE_SIZE",
                "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
                "value": custom_size,
                "annotation": "Custom max directive size",
            },
        ]
    )

    try:
        state = get_base_compile_state(dict_path, {})
        assert state.max_directives_count == custom_count
        assert state.max_directive_size == custom_size
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_missing_constants_use_defaults():
    """Test that missing constants fall back to default values."""
    _clear_caches()

    # Create a dictionary with no Svc.Fpy constants
    dict_path = create_test_dictionary([])

    # Manually remove the Svc.Fpy constants
    with open(dict_path, "r") as f:
        dict_json = json.load(f)

    dict_json["constants"] = [
        c
        for c in dict_json.get("constants", [])
        if not c.get("qualifiedName", "").startswith("Svc.Fpy.")
    ]

    with open(dict_path, "w") as f:
        json.dump(dict_json, f)

    try:
        state = get_base_compile_state(dict_path, {})
        assert state.max_directives_count == DEFAULT_MAX_DIRECTIVES_COUNT
        assert state.max_directive_size == DEFAULT_MAX_DIRECTIVE_SIZE
        assert state.max_seq_arg_count == DEFAULT_MAX_SEQ_ARG_COUNT
        assert state.max_stack_size == DEFAULT_MAX_STACK_SIZE
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_too_many_directives_with_custom_limit():
    """Test that the custom limit is enforced during compilation."""
    _clear_caches()

    # Set a very low limit
    custom_count = 5
    dict_path = create_test_dictionary(
        [
            {
                "kind": "constant",
                "qualifiedName": "Svc.Fpy.MAX_SEQUENCE_STATEMENT_COUNT",
                "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
                "value": custom_count,
                "annotation": "Very low limit for testing",
            }
        ]
    )

    try:
        # This sequence has more than 5 directives when compiled
        seq = "CdhCore.cmdDisp.CMD_NO_OP()\n" * (custom_count + 1)

        fpy.error.file_name = "<test>"
        state = get_base_compile_state(dict_path)
        body = text_to_ast(seq)
        assert body is not None

        # Should fail because we exceed the custom limit
        with pytest.raises(fpy.error.BackendError) as exc_info:
            state = analyze_ast(body, state)
            analysis_to_fpybc_directives(state)
        assert "Too many directives" in str(exc_info.value)
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_within_custom_limit_succeeds():
    """Test that compilation succeeds when within the custom limit."""
    _clear_caches()

    # Set a reasonable limit
    custom_count = 100
    dict_path = create_test_dictionary(
        [
            {
                "kind": "constant",
                "qualifiedName": "Svc.Fpy.MAX_SEQUENCE_STATEMENT_COUNT",
                "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
                "value": custom_count,
                "annotation": "Reasonable limit for testing",
            }
        ]
    )

    try:
        # This sequence should be within the limit
        seq = "CdhCore.cmdDisp.CMD_NO_OP()\n" * 10

        fpy.error.file_name = "<test>"
        state = get_base_compile_state(dict_path)
        body = text_to_ast(seq)
        assert body is not None

        # Should succeed
        state = analyze_ast(body, state)
        analysis_to_fpybc_directives(state)
    finally:
        Path(dict_path).unlink()
        _clear_caches()


# ============================================================================
# FW_SERIALIZE_TRUE_VALUE / FW_SERIALIZE_FALSE_VALUE loading tests
# ============================================================================


def fw_serialize_constant(name: str, value: int) -> dict:
    """Build a dictionary constant descriptor for a FW_SERIALIZE_* value."""
    return {
        "kind": "constant",
        "qualifiedName": name,
        "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
        "value": value,
        "annotation": f"Custom {name} for testing",
    }


def test_load_fw_serialize_from_default_dictionary():
    """Test that boolean wire-format values are loaded from the standard dictionary."""
    _clear_caches()
    from fpy.types import BOOL, FpyValue

    get_base_compile_state(DEFAULT_DICTIONARY, {})

    # The RefTopologyDictionary.json has FW_SERIALIZE_TRUE_VALUE=255, FALSE=0
    assert fpy.types.FW_SERIALIZE_TRUE_VALUE == 0xFF
    assert fpy.types.FW_SERIALIZE_FALSE_VALUE == 0x00
    assert FpyValue(BOOL, True).serialize() == b"\xff"
    assert FpyValue(BOOL, False).serialize() == b"\x00"


def test_custom_fw_serialize_values():
    """Test that custom boolean wire-format values from the dictionary are used."""
    _clear_caches()
    from fpy.types import (
        BOOL,
        DEFAULT_FW_SERIALIZE_FALSE_VALUE,
        DEFAULT_FW_SERIALIZE_TRUE_VALUE,
        FpyValue,
    )

    dict_path = create_test_dictionary(
        [
            fw_serialize_constant("FW_SERIALIZE_TRUE_VALUE", 1),
            fw_serialize_constant("FW_SERIALIZE_FALSE_VALUE", 2),
        ]
    )

    try:
        get_base_compile_state(dict_path, {})
        assert fpy.types.FW_SERIALIZE_TRUE_VALUE == 1
        assert fpy.types.FW_SERIALIZE_FALSE_VALUE == 2
        # Booleans now serialize using the dictionary-provided wire values.
        assert FpyValue(BOOL, True).serialize() == b"\x01"
        assert FpyValue(BOOL, False).serialize() == b"\x02"
    finally:
        # Restore the framework defaults so we don't leak into other tests.
        fpy.types.FW_SERIALIZE_TRUE_VALUE = DEFAULT_FW_SERIALIZE_TRUE_VALUE
        fpy.types.FW_SERIALIZE_FALSE_VALUE = DEFAULT_FW_SERIALIZE_FALSE_VALUE
        Path(dict_path).unlink()
        _clear_caches()


def test_missing_fw_serialize_use_defaults():
    """Test that missing FW_SERIALIZE constants fall back to framework defaults."""
    _clear_caches()
    from fpy.types import (
        DEFAULT_FW_SERIALIZE_FALSE_VALUE,
        DEFAULT_FW_SERIALIZE_TRUE_VALUE,
    )

    dict_path = create_test_dictionary([])

    # Remove the FW_SERIALIZE constants from the dictionary.
    with open(dict_path, "r") as f:
        dict_json = json.load(f)
    dict_json["constants"] = [
        c
        for c in dict_json.get("constants", [])
        if not c.get("qualifiedName", "").startswith("FW_SERIALIZE_")
    ]
    with open(dict_path, "w") as f:
        json.dump(dict_json, f)

    try:
        get_base_compile_state(dict_path, {})
        assert fpy.types.FW_SERIALIZE_TRUE_VALUE == DEFAULT_FW_SERIALIZE_TRUE_VALUE
        assert fpy.types.FW_SERIALIZE_FALSE_VALUE == DEFAULT_FW_SERIALIZE_FALSE_VALUE
    finally:
        Path(dict_path).unlink()
        _clear_caches()


# ============================================================================
# TimeBase dictionary loading tests
# ============================================================================


def create_test_dict_with_timebase(
    enum_constants: list[dict] | None = None,
    rep_type: dict | None = None,
) -> str:
    """Create a test dictionary with a custom TimeBase enum definition."""
    with open(DEFAULT_DICTIONARY, "r") as f:
        base_dict = json.load(f)

    # Find and replace the TimeBase enum
    for i, type_def in enumerate(base_dict.get("typeDefinitions", [])):
        if type_def.get("qualifiedName") == "TimeBase":
            if enum_constants is not None:
                type_def["enumeratedConstants"] = enum_constants
            if rep_type is not None:
                type_def["representationType"] = rep_type
            break

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(base_dict, temp_file)
    temp_file.close()
    return temp_file.name


def create_test_dict_without_timebase() -> str:
    """Create a test dictionary without a TimeBase enum."""
    with open(DEFAULT_DICTIONARY, "r") as f:
        base_dict = json.load(f)

    # Remove TimeBase from typeDefinitions
    base_dict["typeDefinitions"] = [
        t
        for t in base_dict.get("typeDefinitions", [])
        if t.get("qualifiedName") != "TimeBase"
    ]

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(base_dict, temp_file)
    temp_file.close()
    return temp_file.name


def test_timebase_loaded_from_dictionary():
    """Test that TimeBase enum constants are loaded from dictionary."""
    _clear_caches()
    from fpy.types import TIME_BASE

    state = get_base_compile_state(DEFAULT_DICTIONARY, {})

    # The RefTopologyDictionary has these TimeBase constants:
    assert TIME_BASE.enum_dict == {
        "TB_NONE": 0,
        "TB_PROC_TIME": 1,
        "TB_WORKSTATION_TIME": 2,
        "TB_SC_TIME": 3,
        "TB_DONT_CARE": 65535,
    }


def test_timebase_rep_type_from_dictionary():
    """Test that TimeBase rep_type is loaded from dictionary."""
    _clear_caches()
    from fpy.types import TIME_BASE, U16

    state = get_base_compile_state(DEFAULT_DICTIONARY, {})

    # RefTopologyDictionary has representationType U16
    assert TIME_BASE.rep_type == U16


def test_timebase_custom_rep_type():
    """Test that a custom TimeBase rep_type from dictionary is used."""
    _clear_caches()
    from fpy.types import TIME_BASE, U32

    dict_path = create_test_dict_with_timebase(
        rep_type={
            "name": "U32",
            "kind": "integer",
            "size": 32,
            "signed": False,
        }
    )

    try:
        state = get_base_compile_state(dict_path, {})
        assert TIME_BASE.rep_type == U32
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_timebase_missing_raises_error():
    """Test that missing TimeBase enum raises an error."""
    _clear_caches()

    dict_path = create_test_dict_without_timebase()

    try:
        import pytest

        # Dictionary parsing fails because Fw.TimeValue depends on TimeBase
        with pytest.raises(AssertionError, match="Could not resolve types"):
            get_base_compile_state(dict_path, {})
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_timebase_missing_tb_none_raises_error():
    """Test that TimeBase without TB_NONE raises an error."""
    _clear_caches()

    dict_path = create_test_dict_with_timebase(
        enum_constants=[
            {"name": "TB_PROC_TIME", "value": 1},
            {"name": "TB_SC_TIME", "value": 3},
        ]
    )

    try:
        import pytest
        from fpy.error import DictionaryError

        with pytest.raises(DictionaryError, match="must include TB_NONE"):
            get_base_compile_state(dict_path, {})
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_timebase_tb_none_wrong_value_raises_error():
    """Test that TB_NONE with wrong value raises an error."""
    _clear_caches()

    dict_path = create_test_dict_with_timebase(
        enum_constants=[
            {"name": "TB_NONE", "value": 42},  # wrong value
            {"name": "TB_PROC_TIME", "value": 1},
        ]
    )

    try:
        import pytest
        from fpy.error import DictionaryError

        with pytest.raises(DictionaryError, match="TB_NONE constant must have value 0"):
            get_base_compile_state(dict_path, {})
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_timebase_additional_constants_available():
    """Test that additional TimeBase constants from dict are usable in code."""
    _clear_caches()

    # Using the default dictionary which has all 5 constants
    seq = """
t: Fw.Time = Fw.Time(TimeBase.TB_SC_TIME, 0, 100, 0)
"""
    fpy.error.file_name = "<test>"
    fpy.error.input_text = seq
    fpy.error.input_lines = seq.splitlines()

    state = get_base_compile_state(DEFAULT_DICTIONARY)
    body = text_to_ast(seq)
    assert body is not None

    state = analyze_ast(body, state)
    analysis_to_fpybc_directives(state)


# ============================================================================
# FpySequencer limit checks: MAX_SEQUENCE_ARG_COUNT, MAX_STACK_SIZE,
# MAX_DIRECTIVE_SIZE, FW_COM_BUFFER_MAX_SIZE, FW_CMD_ARG_BUFFER_MAX_SIZE
# ============================================================================


def _compile(dict_path: str, seq: str):
    """Compile *seq* against *dict_path* all the way to fpybc directives."""
    fpy.error.file_name = "<test>"
    fpy.error.input_text = seq
    fpy.error.input_lines = seq.splitlines()
    state = get_base_compile_state(dict_path)
    body = text_to_ast(seq)
    assert body is not None
    state = analyze_ast(body, state)
    return analysis_to_fpybc_directives(state)


def svc_fpy_constant(name: str, value: int) -> dict:
    """Build a dictionary constant descriptor for an integer limit."""
    return {
        "kind": "constant",
        "qualifiedName": name,
        "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
        "value": value,
        "annotation": f"Custom {name} for testing",
    }


def test_new_limits_loaded_from_default_dictionary():
    """The Ref dictionary's limit constants all land in CompileState."""
    _clear_caches()

    state = get_base_compile_state(DEFAULT_DICTIONARY, {})

    assert state.max_seq_arg_count == 16
    assert state.max_stack_size == 65535
    assert state.com_buffer_max_size == 512
    assert state.cmd_arg_buffer_max_size == 506
    assert state.cmd_names_by_opcode  # populated from the dictionary


def test_seq_arg_count_exceeding_custom_limit_fails():
    """More sequence args than Svc.Fpy.MAX_SEQUENCE_ARG_COUNT is a compile error."""
    _clear_caches()
    dict_path = create_test_dictionary(
        [svc_fpy_constant("Svc.Fpy.MAX_SEQUENCE_ARG_COUNT", 2)]
    )
    try:
        with pytest.raises(fpy.error.CompileError) as exc_info:
            _compile(dict_path, "sequence(a: U8, b: U8, c: U8)\n")
        assert "MAX_SEQUENCE_ARG_COUNT" in str(exc_info.value)

        # at the limit is fine
        _compile(dict_path, "sequence(a: U8, b: U8)\n")
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_seq_arg_count_exceeding_default_limit_fails():
    """17 sequence args exceeds the Ref dictionary's MAX_SEQUENCE_ARG_COUNT of 16."""
    _clear_caches()
    args = ", ".join(f"a{i}: U8" for i in range(17))
    with pytest.raises(fpy.error.CompileError) as exc_info:
        _compile(DEFAULT_DICTIONARY, f"sequence({args})\n")
    assert "MAX_SEQUENCE_ARG_COUNT" in str(exc_info.value)
    _clear_caches()


def test_seq_arg_total_size_exceeding_stack_size_fails():
    """Total sequence arg bytes above Svc.Fpy.MAX_STACK_SIZE is a compile error."""
    _clear_caches()
    dict_path = create_test_dictionary([svc_fpy_constant("Svc.Fpy.MAX_STACK_SIZE", 8)])
    try:
        with pytest.raises(fpy.error.CompileError) as exc_info:
            _compile(dict_path, "sequence(a: U64, b: U64)\n")
        assert "MAX_STACK_SIZE" in str(exc_info.value)
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_seq_arg_total_size_exceeding_seq_args_buffer_fails():
    """Sequence args that cannot fit in the Svc.SeqArgs buffer are a compile error."""
    _clear_caches()
    # 13 x Ref.DpDemo.U32Array (20 bytes each) = 260 > the 255-byte SeqArgs buffer,
    # while the arg count (13) is within MAX_SEQUENCE_ARG_COUNT (16)
    args = ", ".join(f"a{i}: Ref.DpDemo.U32Array" for i in range(13))
    with pytest.raises(fpy.error.CompileError) as exc_info:
        _compile(DEFAULT_DICTIONARY, f"sequence({args})\n")
    assert "SeqArgs buffer capacity" in str(exc_info.value)
    _clear_caches()


def test_directive_size_exceeding_custom_limit_fails():
    """A directive larger than the dictionary's MAX_DIRECTIVE_SIZE is a compile error."""
    _clear_caches()
    dict_path = create_test_dictionary(
        [svc_fpy_constant("Svc.Fpy.MAX_DIRECTIVE_SIZE", 16)]
    )
    try:
        # CONST_CMD with a 39-char string arg serializes to ~48 bytes > 16
        arg = "a" * 39
        with pytest.raises(fpy.error.BackendError) as exc_info:
            _compile(dict_path, f'CdhCore.cmdDisp.CMD_NO_OP_STRING("{arg}")\n')
        assert "too large" in str(exc_info.value)
        # the error must point at the source line that generated the directive
        assert "<test>:1" in str(exc_info.value)
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_cmd_args_exceeding_cmd_arg_buffer_fails():
    """Command args above FW_CMD_ARG_BUFFER_MAX_SIZE are a compile error."""
    _clear_caches()
    dict_path = create_test_dictionary(
        [svc_fpy_constant("FW_CMD_ARG_BUFFER_MAX_SIZE", 8)]
    )
    try:
        # string args serialize as a 2-byte length + the bytes: 2 + 39 > 8
        arg = "a" * 39
        with pytest.raises(fpy.error.BackendError) as exc_info:
            _compile(dict_path, f'CdhCore.cmdDisp.CMD_NO_OP_STRING("{arg}")\n')
        assert "FW_CMD_ARG_BUFFER_MAX_SIZE" in str(exc_info.value)
        assert "CdhCore.cmdDisp.CMD_NO_OP_STRING" in str(exc_info.value)
        assert "<test>:1" in str(exc_info.value)

        # a command with no args is fine
        _compile(dict_path, "CdhCore.cmdDisp.CMD_NO_OP()\n")

        # a non-const arg forces the STACK_CMD path; the error must still
        # name the command (arg bytes: I32 + F32 + U8 = 9 > 8)
        seq = "x: I32 = 1\nCdhCore.cmdDisp.CMD_TEST_CMD_1(x, 2.0, 3)\n"
        with pytest.raises(fpy.error.BackendError) as exc_info:
            _compile(dict_path, seq)
        assert "FW_CMD_ARG_BUFFER_MAX_SIZE" in str(exc_info.value)
        assert "CdhCore.cmdDisp.CMD_TEST_CMD_1" in str(exc_info.value)
        assert "<test>:2" in str(exc_info.value)
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_cmd_packet_exceeding_com_buffer_fails():
    """A command packet above FW_COM_BUFFER_MAX_SIZE is a compile error."""
    _clear_caches()
    # keep FW_CMD_ARG_BUFFER_MAX_SIZE permissive so the com buffer check fires
    dict_path = create_test_dictionary(
        [
            svc_fpy_constant("FW_COM_BUFFER_MAX_SIZE", 16),
            svc_fpy_constant("FW_CMD_ARG_BUFFER_MAX_SIZE", 506),
        ]
    )
    try:
        # packet = descriptor (2, Ref FwPacketDescriptorType is U16) + opcode (4)
        # + args (2 + 39) = 47 > 16
        arg = "a" * 39
        with pytest.raises(fpy.error.BackendError) as exc_info:
            _compile(dict_path, f'CdhCore.cmdDisp.CMD_NO_OP_STRING("{arg}")\n')
        assert "FW_COM_BUFFER_MAX_SIZE" in str(exc_info.value)
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_cmd_size_checks_skipped_without_dictionary_constants():
    """Without the FW_* buffer constants in the dictionary, no cmd size check runs."""
    _clear_caches()
    dict_path = create_test_dictionary([])
    with open(dict_path, "r") as f:
        dict_json = json.load(f)
    dict_json["constants"] = [
        c
        for c in dict_json.get("constants", [])
        if c.get("qualifiedName")
        not in ("FW_COM_BUFFER_MAX_SIZE", "FW_CMD_ARG_BUFFER_MAX_SIZE")
    ]
    with open(dict_path, "w") as f:
        json.dump(dict_json, f)

    try:
        state = get_base_compile_state(dict_path, {})
        assert state.com_buffer_max_size is None
        assert state.cmd_arg_buffer_max_size is None
        _clear_caches()

        arg = "a" * 39
        _compile(dict_path, f'CdhCore.cmdDisp.CMD_NO_OP_STRING("{arg}")\n')
    finally:
        Path(dict_path).unlink()
        _clear_caches()


# ============================================================================
# Validity enum validation (Fw.TlmValid / Fw.ParamValid)
# ============================================================================


def _create_test_dict_with_modified_type(qualified_name: str, mutate) -> str:
    """Create a test dictionary with one type definition passed through
    *mutate* (a function of the JSON type def), or removed when *mutate* is
    None."""
    with open(DEFAULT_DICTIONARY, "r") as f:
        base_dict = json.load(f)

    defs = base_dict["typeDefinitions"]
    if mutate is None:
        base_dict["typeDefinitions"] = [
            t for t in defs if t.get("qualifiedName") != qualified_name
        ]
    else:
        for type_def in defs:
            if type_def.get("qualifiedName") == qualified_name:
                mutate(type_def)
                break

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(base_dict, temp_file)
    temp_file.close()
    return temp_file.name


def test_param_valid_mismatch_raises_error():
    """A dictionary whose Fw.ParamValid disagrees with the canonical enum is
    rejected: the compiled code bakes the canonical VALID value into every
    parameter read's validity check."""
    _clear_caches()

    def swap_valid(type_def):
        for const in type_def["enumeratedConstants"]:
            if const["name"] == "VALID":
                const["value"] = 9

    dict_path = _create_test_dict_with_modified_type("Fw.ParamValid", swap_valid)

    try:
        import pytest
        from fpy.error import DictionaryError

        with pytest.raises(DictionaryError, match="Fw.ParamValid"):
            get_base_compile_state(dict_path, {})
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_param_valid_absent_is_tolerated():
    """The validity enums are port argument types most dictionaries never
    define; their absence leaves the canonical definitions standing."""
    _clear_caches()

    dict_path = _create_test_dict_with_modified_type("Fw.ParamValid", None)

    try:
        state = get_base_compile_state(dict_path, {})
        assert state is not None
    finally:
        Path(dict_path).unlink()
        _clear_caches()
