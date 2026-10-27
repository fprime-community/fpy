"""Regression tests for dictionary-configured time contexts."""

import json
from pathlib import Path

import pytest

from fpy.compiler import (
    analysis_to_fpybc_directives,
    analysis_to_wasm,
    analyze_ast,
    text_to_ast,
)
from fpy.dictionary import load_dictionary
from fpy.error import CompileError
from fpy.macros import TIME_MACRO
from fpy.state import _build_global_scopes, get_base_compile_state
from fpy.test_helpers import FPYBC, assert_run_success, default_dictionary
from fpy.types import TIME


@pytest.fixture(autouse=True)
def clear_dictionary_caches():
    """Each test loads one configuration without reusing another test's scopes."""
    load_dictionary.cache_clear()
    _build_global_scopes.cache_clear()
    # TIME also keeps derived defaults outside the cached scopes.
    TIME.member_defaults = None
    yield
    load_dictionary.cache_clear()
    _build_global_scopes.cache_clear()
    TIME.member_defaults = None
    get_base_compile_state(default_dictionary)


@pytest.fixture(
    params=[
        {"name": "U8", "kind": "integer", "size": 8, "signed": False},
        {"name": "U16", "kind": "integer", "size": 16, "signed": False},
        {"name": "U32", "kind": "integer", "size": 32, "signed": False},
        {"name": "U64", "kind": "integer", "size": 64, "signed": False},
    ],
    ids=["U8", "U16", "U32", "U64"],
)
def context_dictionary(request, tmp_path):
    """Use the reference dictionary with one explicit time-context type."""
    data = json.loads(Path(default_dictionary).read_text(encoding="utf-8"))
    data["typeDefinitions"] = [
        definition
        for definition in data["typeDefinitions"]
        if definition["qualifiedName"] != "FwTimeContextStoreType"
    ] + [
        {
            "kind": "alias",
            "qualifiedName": "FwTimeContextStoreType",
            "type": request.param,
            "underlyingType": request.param,
        }
    ]
    path = tmp_path / "dictionary.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _time_value(dictionary, context):
    suffix = "" if context is None else f", timeContext={context}"
    body = text_to_ast(f'time("2023-11-14T22:13:20.5Z"{suffix})\n')
    call = body.stmts[0]
    state = analyze_ast(body, get_base_compile_state(dictionary))
    return state.const_expr_values[call]


def test_time_builtin_context_parameter_type(context_dictionary):
    get_base_compile_state(context_dictionary)
    configured = load_dictionary(context_dictionary)["type_defs"][
        "FwTimeContextStoreType"
    ]
    assert TIME.members[1].type == configured
    assert TIME.member_defaults["timeContext"].type == configured
    assert TIME_MACRO.args[2][1] == configured
    assert TIME_MACRO.args[2][2].type == configured


@pytest.mark.parametrize("context", [None, 9], ids=["default-context", "context-nine"])
def test_time_constant_context_layout(context_dictionary, context):
    value = _time_value(context_dictionary, context)
    types = load_dictionary(context_dictionary)["type_defs"]
    configured = types["FwTimeContextStoreType"]
    expected = (
        types["TimeBase"]
        .enum_dict["TB_WORKSTATION_TIME"]
        .to_bytes(types["TimeBase"].max_size, "big")
        + (0 if context is None else context).to_bytes(configured.max_size, "big")
        + (1_700_000_000).to_bytes(4, "big")
        + (500_000).to_bytes(4, "big")
    )
    serialized = value.serialize()
    assert len(serialized) == len(expected)
    assert serialized == expected
    assert value.val["timeContext"].type == configured
    assert len(serialized) == value.type.max_size


def test_time_context_300_matches_configured_range(context_dictionary):
    configured = load_dictionary(context_dictionary)["type_defs"][
        "FwTimeContextStoreType"
    ]
    if configured.bits == 8:
        with pytest.raises(CompileError, match="out of range for type U8"):
            _time_value(context_dictionary, 300)
    else:
        value = _time_value(context_dictionary, 300)
        assert value.val["timeContext"].val == 300
        assert value.val["timeContext"].type == configured


def test_time_assignment_compiles_for_configured_context(
    context_dictionary, single_backend
):
    body = text_to_ast(
        't: Fw.Time = time("2023-11-14T22:13:20.5Z", timeContext=9)\n'
        "assert t.timeContext == 9\n"
    )
    state = analyze_ast(body, get_base_compile_state(context_dictionary))
    if single_backend == FPYBC:
        analysis_to_fpybc_directives(state)
    else:
        analysis_to_wasm(state)


def test_time_context_range_boundaries(context_dictionary):
    configured = load_dictionary(context_dictionary)["type_defs"][
        "FwTimeContextStoreType"
    ]
    maximum = (1 << configured.bits) - 1
    assert _time_value(context_dictionary, maximum).val["timeContext"].val == maximum
    for invalid in (-1, maximum + 1):
        with pytest.raises(
            CompileError, match=f"out of range for type {configured.name}"
        ):
            _time_value(context_dictionary, invalid)


@pytest.mark.parametrize(
    "context", [None, 9, "maximum"], ids=["default", "nine", "maximum"]
)
def test_time_context_runs_on_sequencer(fprime_test_api, single_backend, context):
    configured = load_dictionary(default_dictionary)["type_defs"][
        "FwTimeContextStoreType"
    ]
    if context == "maximum":
        context = (1 << configured.bits) - 1
    suffix = "" if context is None else f", timeContext={context}"
    expected = 0 if context is None else context
    seq = (
        f't: Fw.Time = time("2023-11-14T22:13:20.5Z"{suffix})\n'
        f"assert t.timeContext == {expected}\n"
        "assert t.timeBase == TimeBase.TB_WORKSTATION_TIME\n"
        "assert t.seconds == 1700000000\n"
        "assert t.useconds == 500000\n"
        "assert (t + Fw.TimeIntervalValue(1, 0)).timeContext == t.timeContext\n"
    )
    assert_run_success(fprime_test_api, seq)
