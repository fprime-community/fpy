"""Configured sequence argument size fields and dictionary-load isolation."""

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
from fpy.error import DictionaryError
from fpy.error import CompileError
from fpy.state import _build_global_scopes, get_base_compile_state
from fpy.types import DEFAULT_SEQ_ARGS_BUFFER_SIZE, FpyValue, SEQ_ARGS, U64

DEFAULT_DICTIONARY = Path(__file__).with_name("RefTopologyDictionary.json")


@pytest.fixture
def dictionary(tmp_path):
    def make(width=32, capacity=8, default_size=0, omit=False, signed=False):
        data = json.loads(DEFAULT_DICTIONARY.read_text())
        for item in data["typeDefinitions"]:
            if item["qualifiedName"] == "FwSizeType":
                item["underlyingType"] = {
                    "name": f"{'I' if signed else 'U'}{width}",
                    "kind": "integer",
                    "size": width,
                    "signed": signed,
                }
            elif item["qualifiedName"] == "Svc.SeqArgs":
                item["members"]["buffer"]["size"] = capacity
                item["default"] = {"size": default_size, "buffer": [0] * capacity}
        if omit:
            data["typeDefinitions"] = [
                item
                for item in data["typeDefinitions"]
                if item["qualifiedName"] != "Svc.SeqArgs"
            ]
            data["commands"] = [
                item
                for item in data["commands"]
                if not any(
                    p["type"].get("name") == "Svc.SeqArgs" for p in item["formalParams"]
                )
            ]
        path = tmp_path / f"dictionary-{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps(data))
        return str(path)

    yield make
    load_dictionary.cache_clear()
    _build_global_scopes.cache_clear()
    get_base_compile_state(str(DEFAULT_DICTIONARY))


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_seq_args_adopts_unsigned_size_type_and_exact_wire_layout(dictionary, width):
    get_base_compile_state(dictionary(width=width))
    size, buffer = SEQ_ARGS.members
    assert size.type.name == f"U{width}"
    assert SEQ_ARGS.max_size == width // 8 + 8
    value = FpyValue(SEQ_ARGS, {"size": 3, "buffer": list(range(8))})
    assert value.serialize() == (3).to_bytes(width // 8, "big") + bytes(range(8))


@pytest.mark.parametrize("backend", ["fpybc", "wasm"])
@pytest.mark.parametrize("width", [32, 64])
def test_default_constructor_tracks_dictionary_width_and_default(
    dictionary, backend, width
):
    state = get_base_compile_state(dictionary(width=width, default_size=3))
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
    ctor = state.type_ctors[SEQ_ARGS]
    assert ctor.args[0][1].name == f"U{width}"
    assert ctor.args[0][2].type.name == f"U{width}"
    assert ctor.args[0][2].val == 3


def test_cached_loads_restore_size_capacity_and_constructor_defaults(dictionary):
    first = dictionary(width=32, capacity=8, default_size=3)
    second = dictionary(width=64, capacity=16, default_size=5)
    for path, width, capacity, default in (
        (first, 32, 8, 3),
        (second, 64, 16, 5),
        (first, 32, 8, 3),
        (second, 64, 16, 5),
    ):
        state = get_base_compile_state(path)
        assert SEQ_ARGS.members[0].type.name == f"U{width}"
        assert SEQ_ARGS.members[1].type.length == capacity
        assert SEQ_ARGS.member_defaults["size"].type.name == f"U{width}"
        assert SEQ_ARGS.member_defaults["size"].val == default
        assert len(SEQ_ARGS.member_defaults["buffer"].val) == capacity
        ctor = state.type_ctors[SEQ_ARGS]
        assert ctor.args[0][1].name == f"U{width}"
        assert ctor.args[0][2].val == default
        assert len(ctor.args[1][2].val) == capacity
        raw = load_dictionary(path)["type_defs"]["Svc.SeqArgs"]
        assert raw is not SEQ_ARGS
        assert raw.members[0].type.name == f"U{width}"


def test_missing_seq_args_restores_default_size_and_capacity(dictionary):
    missing = dictionary(omit=True)
    get_base_compile_state(missing)
    get_base_compile_state(dictionary(width=32))
    get_base_compile_state(missing)
    assert SEQ_ARGS.members[0].type is U64
    assert SEQ_ARGS.members[1].type.length == DEFAULT_SEQ_ARGS_BUFFER_SIZE
    assert SEQ_ARGS.member_defaults["size"].type is U64


def test_signed_size_member_is_rejected(dictionary):
    with pytest.raises(DictionaryError, match="Svc.SeqArgs"):
        get_base_compile_state(dictionary(signed=True))


def test_capacity_must_be_representable_by_size_member(dictionary):
    with pytest.raises(DictionaryError, match="length 256 exceeds.*U8"):
        get_base_compile_state(dictionary(width=8, capacity=256))


@pytest.mark.parametrize("backend", ["fpybc", "wasm"])
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("dynamic", [False, True])
def test_seq_run_exact_capacity_compiles(dictionary, tmp_path, backend, width, dynamic):
    (tmp_path / "child.fpy").write_text(
        "sequence(x: U32, y: U32)\nassert x == 42\nassert y == 7\n"
    )
    source = (
        'x: U32 = 42\nRef.seqDisp.RUN_ARGS("child.bin", Svc.BlockState.BLOCK, x, 7)\n'
        if dynamic
        else 'Ref.seqDisp.RUN_ARGS("child.bin", Svc.BlockState.BLOCK, 42, 7)\n'
    )
    state = get_base_compile_state(
        dictionary(width=width), seq_maps=[("", str(tmp_path) + "/")]
    )
    state = analyze_ast(text_to_ast(source), state)
    if backend == "fpybc":
        analysis_to_fpybc_directives(state)
    else:
        import llvmlite.binding as llvm

        module, _ = analysis_to_llvm_module(state)
        llvm.parse_assembly(str(module)).verify()


@pytest.mark.parametrize("width", [32, 64])
def test_seq_run_over_capacity_is_a_source_diagnostic(dictionary, tmp_path, width):
    (tmp_path / "child.fpy").write_text("sequence(x: U32, y: U32, z: U8)\n")
    state = get_base_compile_state(
        dictionary(width=width), seq_maps=[("", str(tmp_path) + "/")]
    )
    with pytest.raises(CompileError, match="SeqArgs buffer capacity"):
        analyze_ast(
            text_to_ast(
                'Ref.seqDisp.RUN_ARGS("child.bin", Svc.BlockState.BLOCK, 1, 2, 3)\n'
            ),
            state,
        )
