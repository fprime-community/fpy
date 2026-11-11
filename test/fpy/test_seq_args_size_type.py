"""Regression tests for dictionary-configured Svc.SeqArgs sizes."""

import json
from pathlib import Path

import pytest

from fpy.compiler import (
    analysis_to_fpybc_directives,
    analysis_to_llvm_module,
    analyze_ast,
    text_to_ast,
)
from fpy.dictionary import load_dictionary
from fpy.error import CompileError, DictionaryError
from fpy.state import _build_global_scopes, get_base_compile_state
from fpy.types import DEFAULT_SEQ_ARGS_BUFFER_SIZE, FpyValue, SEQ_ARGS, U64

DEFAULT_DICTIONARY = Path(__file__).with_name("RefTopologyDictionary.json")


@pytest.fixture(autouse=True)
def isolate_dictionary_configurations():
    """Rebuild scopes and SeqArgs defaults independently for each test."""
    load_dictionary.cache_clear()
    _build_global_scopes.cache_clear()
    SEQ_ARGS.member_defaults = None
    yield
    load_dictionary.cache_clear()
    _build_global_scopes.cache_clear()
    SEQ_ARGS.member_defaults = None
    get_base_compile_state(str(DEFAULT_DICTIONARY))


def _dictionary(tmp_path, *, width=32, capacity=8, default_size=0, signed=False):
    """Copy the reference dictionary with a configured size field and buffer."""
    data = json.loads(DEFAULT_DICTIONARY.read_text(encoding="utf-8"))
    for definition in data["typeDefinitions"]:
        if definition["qualifiedName"] == "FwSizeType":
            definition["underlyingType"] = {
                "name": f"{'I' if signed else 'U'}{width}",
                "kind": "integer",
                "size": width,
                "signed": signed,
            }
        elif definition["qualifiedName"] == "Svc.SeqArgs":
            definition["members"]["buffer"]["size"] = capacity
            definition["default"] = {
                "size": default_size,
                "buffer": [0] * capacity,
            }
    path = tmp_path / "dictionary.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_size_field_and_exact_serialized_bytes(tmp_path, width):
    get_base_compile_state(_dictionary(tmp_path, width=width))
    size, buffer = SEQ_ARGS.members
    assert size.type.name == f"U{width}"
    assert buffer.type.length == 8
    assert SEQ_ARGS.max_size == width // 8 + 8
    value = FpyValue(SEQ_ARGS, {"size": 3, "buffer": list(range(8))})
    assert value.serialize() == (3).to_bytes(width // 8, "big") + bytes(range(8))


@pytest.mark.parametrize("backend", ["fpybc", "llvm"])
@pytest.mark.parametrize("width", [32, 64])
def test_constructor_default_and_codegen(tmp_path, backend, width):
    state = get_base_compile_state(_dictionary(tmp_path, width=width, default_size=3))
    state = analyze_ast(
        text_to_ast("args: Svc.SeqArgs = Svc.SeqArgs()\nassert args.size == 3\n"), state
    )
    if backend == "fpybc":
        directives, _ = analysis_to_fpybc_directives(state)
        expected = (3).to_bytes(width // 8, "big") + bytes(8)
        assert any(getattr(d, "val", None) == expected for d in directives)
    else:
        import llvmlite.binding as llvm

        module, _ = analysis_to_llvm_module(state)
        llvm.parse_assembly(str(module)).verify()

    assert SEQ_ARGS.member_defaults["size"].type.name == f"U{width}"
    assert SEQ_ARGS.member_defaults["size"].val == 3
    ctor = state.type_ctors[SEQ_ARGS]
    assert ctor.args[0][1].name == f"U{width}"
    assert ctor.args[0][2].type.name == f"U{width}"
    assert ctor.args[0][2].val == 3
    assert len(ctor.args[1][2].val) == 8


def test_dictionary_without_seq_args_uses_builtin_fallback(tmp_path):
    data = json.loads(DEFAULT_DICTIONARY.read_text(encoding="utf-8"))
    data["typeDefinitions"] = [
        definition
        for definition in data["typeDefinitions"]
        if definition["qualifiedName"] != "Svc.SeqArgs"
    ]
    data["commands"] = [
        command
        for command in data["commands"]
        if not any(
            arg["type"].get("name") == "Svc.SeqArgs"
            for arg in command["formalParams"]
        )
    ]
    path = tmp_path / "no-seq-args.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    state = get_base_compile_state(str(path))
    assert SEQ_ARGS.members[0].type is U64
    assert SEQ_ARGS.members[1].type.length == DEFAULT_SEQ_ARGS_BUFFER_SIZE
    assert state.type_ctors[SEQ_ARGS].args[0][2].type is U64


def test_signed_size_member_is_rejected(tmp_path):
    with pytest.raises(DictionaryError, match="Svc.SeqArgs"):
        get_base_compile_state(_dictionary(tmp_path, signed=True))


@pytest.mark.parametrize(
    "capacity, message",
    [(0, "positive length"), (256, "length 256 exceeds.*U8")],
)
def test_invalid_buffer_capacities_are_rejected(tmp_path, capacity, message):
    with pytest.raises(DictionaryError, match=message):
        get_base_compile_state(_dictionary(tmp_path, width=8, capacity=capacity))


def test_u8_capacity_limit_is_accepted(tmp_path):
    get_base_compile_state(_dictionary(tmp_path, width=8, capacity=255))
    assert SEQ_ARGS.members[1].type.length == 255
    assert SEQ_ARGS.max_size == 256


@pytest.mark.parametrize("backend", ["fpybc", "llvm"])
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("dynamic", [False, True])
def test_seq_run_exact_capacity_compiles(tmp_path, backend, width, dynamic):
    (tmp_path / "child.fpy").write_text(
        "sequence(x: U32, y: U32)\nassert x == 42\nassert y == 7\n"
    )
    source = (
        'x: U32 = 42\nRef.seqDisp.RUN_ARGS("child.bin", Svc.BlockState.BLOCK, x, 7)\n'
        if dynamic
        else 'Ref.seqDisp.RUN_ARGS("child.bin", Svc.BlockState.BLOCK, 42, 7)\n'
    )
    state = get_base_compile_state(
        _dictionary(tmp_path, width=width), seq_maps=[("", str(tmp_path) + "/")]
    )
    state = analyze_ast(text_to_ast(source), state)
    if backend == "fpybc":
        analysis_to_fpybc_directives(state)
    else:
        import llvmlite.binding as llvm

        module, _ = analysis_to_llvm_module(state)
        llvm.parse_assembly(str(module)).verify()


@pytest.mark.parametrize("width", [32, 64])
def test_seq_run_over_capacity_is_a_source_diagnostic(tmp_path, width):
    (tmp_path / "child.fpy").write_text("sequence(x: U32, y: U32, z: U8)\n")
    state = get_base_compile_state(
        _dictionary(tmp_path, width=width), seq_maps=[("", str(tmp_path) + "/")]
    )
    with pytest.raises(CompileError, match="SeqArgs buffer capacity"):
        analyze_ast(
            text_to_ast(
                'Ref.seqDisp.RUN_ARGS("child.bin", Svc.BlockState.BLOCK, 1, 2, 3)\n'
            ),
            state,
        )
