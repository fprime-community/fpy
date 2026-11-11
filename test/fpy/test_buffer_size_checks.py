"""Compiler diagnostics for command and bytecode statement buffer limits."""

import json
from pathlib import Path

import pytest

import fpy.error
from fpy.compiler import (
    analysis_to_fpybc_directives,
    analysis_to_llvm_module,
    analyze_ast,
    text_to_ast,
)
from fpy.dictionary import load_dictionary
from fpy.state import _build_global_scopes, get_base_compile_state
from fpy.buffer_limits import serialize_value
from fpy.types import FpyType, FpyValue, StructMember, TypeKind

DEFAULT_DICTIONARY = Path(__file__).with_name("RefTopologyDictionary.json")


@pytest.fixture
def dictionary(tmp_path):
    def make(
        *,
        size_type="U16",
        string_capacity=None,
        aliases=None,
        drop_constants=(),
        **limits,
    ):
        data = json.loads(DEFAULT_DICTIONARY.read_text())
        names = {
            "cmd": "FW_CMD_ARG_BUFFER_MAX_SIZE",
            "com": "FW_COM_BUFFER_MAX_SIZE",
            "statement": "FW_STATEMENT_ARG_BUFFER_MAX_SIZE",
        }
        replacements = {names[name]: value for name, value in limits.items()}
        data["constants"] = [
            item
            for item in data["constants"]
            if item["qualifiedName"] not in replacements
            and item["qualifiedName"] not in drop_constants
        ] + [
            {
                "kind": "constant",
                "qualifiedName": name,
                "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
                "value": value,
            }
            for name, value in replacements.items()
        ]
        configured_aliases = {"FwSizeStoreType": size_type, **(aliases or {})}
        for item in data["typeDefinitions"]:
            if item["qualifiedName"] in configured_aliases:
                name = configured_aliases[item["qualifiedName"]]
                item["type"] = {
                    "name": name,
                    "kind": "integer",
                    "size": int(name[1:]),
                    "signed": False,
                }
                item["underlyingType"] = dict(item["type"])
        if string_capacity is not None:
            for command in data["commands"]:
                if command["name"] == "CdhCore.cmdDisp.CMD_NO_OP_STRING":
                    command["formalParams"][0]["type"]["size"] = string_capacity
        path = tmp_path / f"dictionary-{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps(data))
        return str(path)

    yield make
    load_dictionary.cache_clear()
    _build_global_scopes.cache_clear()
    get_base_compile_state(str(DEFAULT_DICTIONARY))


def compile_source(source, dictionary, backend="fpybc"):
    fpy.error.file_name = "buffer-limits.fpy"
    state = get_base_compile_state(dictionary)
    state = analyze_ast(text_to_ast(source), state)
    if backend == "fpybc":
        return analysis_to_fpybc_directives(state)
    return analysis_to_llvm_module(state)


@pytest.mark.parametrize("backend", ["fpybc", "wasm"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "limits, diagnostic",
    [
        ({"cmd": 8, "statement": 64}, "FW_CMD_ARG_BUFFER_MAX_SIZE"),
        ({"com": 14, "statement": 64}, "FW_COM_BUFFER_MAX_SIZE"),
    ],
)
def test_command_overflow_is_a_source_diagnostic(
    dictionary, backend, dynamic, limits, diagnostic
):
    # Arguments are I32+F32+U8=9 bytes; descriptor+opcode adds2+4=6.
    source = (
        "value: I32 = 1\nCdhCore.cmdDisp.CMD_TEST_CMD_1(value, 1.0, 1)\n"
        if dynamic
        else "CdhCore.cmdDisp.CMD_TEST_CMD_1(1, 1.0, 1)\n"
    )
    with pytest.raises(
        (fpy.error.CompileError, fpy.error.BackendError), match=diagnostic
    ) as result:
        compile_source(source, dictionary(**limits), backend)
    assert result.value.node is not None
    assert "CdhCore.cmdDisp.CMD_TEST_CMD_1" in result.value.msg


@pytest.mark.parametrize("backend", ["fpybc", "wasm"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_command_exact_transport_limits_compile(dictionary, backend, dynamic):
    source = (
        "value: I32 = 1\nCdhCore.cmdDisp.CMD_TEST_CMD_1(value, 1.0, 1)\n"
        if dynamic
        else "CdhCore.cmdDisp.CMD_TEST_CMD_1(1, 1.0, 1)\n"
    )
    compile_source(source, dictionary(cmd=9, com=15, statement=64), backend)


@pytest.mark.parametrize("length", [256, 300])
def test_u8_string_or_statement_length_has_a_diagnostic(dictionary, length):
    source = 'log("' + "x" * length + '")\n'
    with pytest.raises((fpy.error.CompileError, fpy.error.BackendError)) as result:
        compile_source(source, dictionary(size_type="U8", statement=506))
    assert result.value.node is not None
    assert "FwSizeStoreType" in result.value.msg


def test_u8_exact_representable_statement_compiles(dictionary):
    # log() emits raw bytes, not a serialized F Prime string prefix.
    directives, _ = compile_source(
        'log("' + "x" * 255 + '")\n', dictionary(size_type="U8", statement=506)
    )
    assert max(len(item.serialize_args()) for item in directives) == 255


@pytest.mark.parametrize("length", [507, 508])
def test_default_statement_capacity_is_not_directive_limit(dictionary, length):
    with pytest.raises((fpy.error.CompileError, fpy.error.BackendError)) as result:
        compile_source('log("' + "x" * length + '")\n', dictionary(statement=506))
    assert "FW_STATEMENT_ARG_BUFFER_MAX_SIZE" in result.value.msg
    assert result.value.node is not None


def test_default_statement_exact_capacity_compiles(dictionary):
    directives, _ = compile_source(
        'log("' + "x" * 506 + '")\n', dictionary(statement=506)
    )
    assert max(len(item.serialize_args()) for item in directives) == 506


@pytest.mark.parametrize("backend", ["fpybc", "wasm"])
def test_serialized_command_string_prefix_overflow(dictionary, backend):
    source = 'CdhCore.cmdDisp.CMD_NO_OP_STRING("' + "x" * 256 + '")\n'
    with pytest.raises((fpy.error.CompileError, fpy.error.BackendError)) as result:
        compile_source(
            source,
            dictionary(size_type="U8", string_capacity=300, statement=506),
            backend,
        )
    assert "FwSizeStoreType" in result.value.msg
    assert result.value.node is not None


def test_wasm_string_prefix_exact_range_compiles(dictionary):
    source = 'CdhCore.cmdDisp.CMD_NO_OP_STRING("' + "x" * 255 + '")\n'
    compile_source(source, dictionary(size_type="U8", string_capacity=300), "wasm")


def test_statement_limit_counts_utf8_bytes(dictionary):
    with pytest.raises((fpy.error.CompileError, fpy.error.BackendError)) as result:
        compile_source('log("' + "é" * 128 + '")\n', dictionary(size_type="U8"))
    assert "256" in result.value.msg


def test_const_command_statement_includes_opcode(dictionary):
    with pytest.raises((fpy.error.CompileError, fpy.error.BackendError)) as result:
        compile_source(
            "CdhCore.cmdDisp.CMD_TEST_CMD_1(1, 1.0, 1)\n", dictionary(statement=12)
        )
    assert "FW_STATEMENT_ARG_BUFFER_MAX_SIZE" in result.value.msg


def test_const_command_exact_statement_capacity_compiles(dictionary):
    compile_source(
        "CdhCore.cmdDisp.CMD_TEST_CMD_1(1, 1.0, 1)\n", dictionary(statement=13)
    )


@pytest.mark.parametrize("backend", ["fpybc", "wasm"])
def test_command_transport_checks_use_configured_header_widths(dictionary, backend):
    source = "CdhCore.cmdDisp.CMD_TEST_CMD_1(1, 1.0, 1)\n"
    aliases = {"FwOpcodeType": "U64", "FwPacketDescriptorType": "U32"}
    compile_source(source, dictionary(aliases=aliases, com=21, statement=64), backend)
    with pytest.raises((fpy.error.CompileError, fpy.error.BackendError)) as result:
        compile_source(
            source, dictionary(aliases=aliases, com=20, statement=64), backend
        )
    assert "21 byte packet" in result.value.msg


def test_statement_capacity_fallback_override_and_cached_dictionary_loads(dictionary):
    missing = ("FW_STATEMENT_ARG_BUFFER_MAX_SIZE",)
    fallback = dictionary(cmd=8, drop_constants=missing)
    override = dictionary(cmd=8, statement=13)
    for path, expected in ((fallback, 8), (override, 13), (fallback, 8)):
        assert get_base_compile_state(path).statement_arg_buffer_max_size == expected
    no_limit = dictionary(drop_constants=missing + ("FW_CMD_ARG_BUFFER_MAX_SIZE",))
    assert get_base_compile_state(no_limit).statement_arg_buffer_max_size is None


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("length", [255, 256])
def test_nested_string_prefix_validation(dictionary, wrapped, length):
    get_base_compile_state(dictionary(size_type="U8"))
    string = FpyType(TypeKind.STRING, "LongString", max_length=300)
    array = FpyType(TypeKind.ARRAY, "Strings", elem_type=string, length=1)
    struct = FpyType(
        TypeKind.STRUCT, "Container", members=(StructMember("items", array),)
    )
    value = "x" * length
    items = FpyValue(array, [FpyValue(string, value)]) if wrapped else [value]
    container = FpyValue(struct, {"items": items})
    if length == 256:
        with pytest.raises(fpy.error.BackendError, match="256 UTF-8 bytes"):
            serialize_value(container)
    else:
        assert serialize_value(container) == b"\xff" + b"x" * 255
