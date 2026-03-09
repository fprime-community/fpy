"""
Tests for loading sequence configuration from dictionary constants.

These tests verify that MAX_DIRECTIVE_SIZE and MAX_SEQUENCE_STATEMENT_COUNT
are correctly read from the dictionary's constants section.
"""

import json
import tempfile
from pathlib import Path

import fpy.error
from fpy.compiler import (
    text_to_ast,
    ast_to_directives,
    get_base_compile_state,
    _build_global_scopes,
)
from fpy.dictionary import load_dictionary
from fpy.types import DEFAULT_MAX_DIRECTIVES_COUNT, DEFAULT_MAX_DIRECTIVE_SIZE


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
        c for c in base_dict.get("constants", [])
        if c.get("qualifiedName") not in test_constant_names
    ]
    base_dict["constants"] = filtered_constants + constants

    # Write to temp file
    temp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(base_dict, temp_file)
    temp_file.close()

    return temp_file.name


def test_custom_max_directives_count():
    """Test that a custom MAX_SEQUENCE_STATEMENT_COUNT is loaded from dictionary."""
    _clear_caches()

    custom_count = 500
    dict_path = create_test_dictionary([
        {
            "kind": "constant",
            "qualifiedName": "Svc.Fpy.MAX_SEQUENCE_STATEMENT_COUNT",
            "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
            "value": custom_count,
            "annotation": "Custom max sequence statement count"
        }
    ])

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
    dict_path = create_test_dictionary([
        {
            "kind": "constant",
            "qualifiedName": "Svc.Fpy.MAX_DIRECTIVE_SIZE",
            "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
            "value": custom_size,
            "annotation": "Custom max directive size"
        }
    ])

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
    dict_path = create_test_dictionary([
        {
            "kind": "constant",
            "qualifiedName": "Svc.Fpy.MAX_SEQUENCE_STATEMENT_COUNT",
            "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
            "value": custom_count,
            "annotation": "Custom max sequence statement count"
        },
        {
            "kind": "constant",
            "qualifiedName": "Svc.Fpy.MAX_DIRECTIVE_SIZE",
            "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
            "value": custom_size,
            "annotation": "Custom max directive size"
        }
    ])

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
        c for c in dict_json.get("constants", [])
        if not c.get("qualifiedName", "").startswith("Svc.Fpy.")
    ]

    with open(dict_path, "w") as f:
        json.dump(dict_json, f)

    try:
        state = get_base_compile_state(dict_path, {})
        assert state.max_directives_count == DEFAULT_MAX_DIRECTIVES_COUNT
        assert state.max_directive_size == DEFAULT_MAX_DIRECTIVE_SIZE
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_too_many_directives_with_custom_limit():
    """Test that the custom limit is enforced during compilation."""
    _clear_caches()

    # Set a very low limit
    custom_count = 5
    dict_path = create_test_dictionary([
        {
            "kind": "constant",
            "qualifiedName": "Svc.Fpy.MAX_SEQUENCE_STATEMENT_COUNT",
            "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
            "value": custom_count,
            "annotation": "Very low limit for testing"
        }
    ])

    try:
        # This sequence has more than 5 directives when compiled
        seq = "CdhCore.cmdDisp.CMD_NO_OP()\n" * (custom_count + 1)

        fpy.error.file_name = "<test>"
        body = text_to_ast(seq)
        assert body is not None

        result = ast_to_directives(body, dict_path)

        # Should fail because we exceed the custom limit
        assert isinstance(result, fpy.error.BackendError)
        assert "Too many directives" in str(result)
    finally:
        Path(dict_path).unlink()
        _clear_caches()


def test_within_custom_limit_succeeds():
    """Test that compilation succeeds when within the custom limit."""
    _clear_caches()

    # Set a reasonable limit
    custom_count = 100
    dict_path = create_test_dictionary([
        {
            "kind": "constant",
            "qualifiedName": "Svc.Fpy.MAX_SEQUENCE_STATEMENT_COUNT",
            "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
            "value": custom_count,
            "annotation": "Reasonable limit for testing"
        }
    ])

    try:
        # This sequence should be within the limit
        seq = "CdhCore.cmdDisp.CMD_NO_OP()\n" * 10

        fpy.error.file_name = "<test>"
        body = text_to_ast(seq)
        assert body is not None

        result = ast_to_directives(body, dict_path)

        # Should succeed
        assert not isinstance(result, (fpy.error.CompileError, fpy.error.BackendError)), \
            f"Compilation failed unexpectedly: {result}"
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

    temp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(base_dict, temp_file)
    temp_file.close()
    return temp_file.name


def create_test_dict_without_timebase() -> str:
    """Create a test dictionary without a TimeBase enum."""
    with open(DEFAULT_DICTIONARY, "r") as f:
        base_dict = json.load(f)

    # Remove TimeBase from typeDefinitions
    base_dict["typeDefinitions"] = [
        t for t in base_dict.get("typeDefinitions", [])
        if t.get("qualifiedName") != "TimeBase"
    ]

    temp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
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
        with pytest.raises(ValueError, match="must contain TB_NONE"):
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
        with pytest.raises(ValueError, match="TB_NONE must have value 0"):
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

    body = text_to_ast(seq)
    assert body is not None

    result = ast_to_directives(body, DEFAULT_DICTIONARY)
    assert not isinstance(result, (fpy.error.CompileError, fpy.error.BackendError)), \
        f"Compilation failed: {result}"
