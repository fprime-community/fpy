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
from fpy.types import TIME, U8


@pytest.fixture
def context_dictionary(tmp_path):
    paths = {}

    def make(width):
        if width in paths:
            return paths[width]
        data = json.loads(Path(default_dictionary).read_text(encoding="utf-8"))
        integer = {
            "name": f"U{width or 8}",
            "kind": "integer",
            "size": width or 8,
            "signed": False,
        }
        for definition in data["typeDefinitions"]:
            if definition["qualifiedName"] == "FwTimeContextStoreType":
                definition["type"] = integer
                definition["underlyingType"] = dict(integer)
                break
        else:
            raise AssertionError("Reference dictionary has no time-context alias")
        if width is None:
            data["typeDefinitions"].remove(definition)
            for definition in data["typeDefinitions"]:
                if definition["qualifiedName"] == "Fw.TimeValue":
                    definition["members"]["timeContext"]["type"] = integer
        path = tmp_path / f"time-context-u{width}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        paths[width] = str(path)
        return paths[width]

    yield make
    load_dictionary.cache_clear()
    _build_global_scopes.cache_clear()
    get_base_compile_state(default_dictionary)


def _time_value(dictionary, context):
    suffix = "" if context is None else f", timeContext={context}"
    body = text_to_ast(f'time("2023-11-14T22:13:20.5Z"{suffix})\n')
    call = body.stmts[0]
    state = analyze_ast(body, get_base_compile_state(dictionary))
    return state.const_expr_values[call]


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_time_builtin_context_parameter_type(context_dictionary, width):
    dictionary = context_dictionary(width)
    get_base_compile_state(dictionary)
    configured = load_dictionary(dictionary)["type_defs"]["FwTimeContextStoreType"]
    assert TIME_MACRO.args[2][1] == configured
    assert TIME_MACRO.args[2][2].type == configured


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("context", [None, 9], ids=["default-context", "context-nine"])
def test_time_constant_context_layout(context_dictionary, width, context):
    dictionary = context_dictionary(width)
    value = _time_value(dictionary, context)
    types = load_dictionary(dictionary)["type_defs"]
    configured = types["FwTimeContextStoreType"]
    expected = (
        types["TimeBase"]
        .enum_dict["TB_WORKSTATION_TIME"]
        .to_bytes(types["TimeBase"].max_size, "big")
        + (0 if context is None else context).to_bytes(width // 8, "big")
        + (1_700_000_000).to_bytes(4, "big")
        + (500_000).to_bytes(4, "big")
    )
    serialized = value.serialize()
    assert len(serialized) == len(expected)
    assert serialized == expected
    assert value.val["timeContext"].type == configured
    assert len(serialized) == value.type.max_size


@pytest.mark.parametrize("width", [16, 32, 64])
def test_wide_time_context_accepts_300(context_dictionary, width):
    value = _time_value(context_dictionary(width), 300)
    assert value.val["timeContext"].val == 300
    assert value.val["timeContext"].type.bits == width


def test_time_assignment_compiles_for_configured_context(
    context_dictionary, single_backend
):
    body = text_to_ast(
        't: Fw.Time = time("2023-11-14T22:13:20.5Z", timeContext=9)\n'
        "assert t.timeContext == 9\n"
    )
    state = analyze_ast(body, get_base_compile_state(context_dictionary(16)))
    if single_backend == FPYBC:
        analysis_to_fpybc_directives(state)
    else:
        analysis_to_wasm(state)


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_time_context_range_boundaries(context_dictionary, width):
    dictionary = context_dictionary(width)
    maximum = (1 << width) - 1
    assert _time_value(dictionary, maximum).val["timeContext"].val == maximum
    for invalid in (-1, maximum + 1):
        with pytest.raises(CompileError, match=f"out of range for type U{width}"):
            _time_value(dictionary, invalid)


@pytest.mark.parametrize(
    "widths", [(8, 16, 8), (16, 32, 16), (64, 8, 64), (16, None, 16)]
)
def test_time_context_reapplied_on_scope_cache_hit(context_dictionary, widths):
    _build_global_scopes.cache_clear()
    for index, width in enumerate(widths):
        dictionary = context_dictionary(width)
        before = _build_global_scopes.cache_info()
        value = _time_value(dictionary, None)
        if index == 2:
            assert _build_global_scopes.cache_info().hits == before.hits + 1
        configured = load_dictionary(dictionary)["type_defs"].get(
            "FwTimeContextStoreType", U8
        )
        assert TIME.members[1].type == configured
        assert TIME_MACRO.args[2][1] == configured
        assert TIME_MACRO.args[2][2].type == configured
        assert TIME.member_defaults["timeContext"].type == configured
        assert value.val["timeContext"].type == configured
        assert len(value.serialize()) == value.type.max_size
        body = text_to_ast("Fw.TimeValue()\n")
        state = analyze_ast(body, get_base_compile_state(dictionary))
        default = state.const_expr_values[body.stmts[0]]
        assert default.val["timeContext"].type == configured
        assert len(default.serialize()) == default.type.max_size


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
