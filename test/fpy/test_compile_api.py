from pathlib import Path

import pytest

import fpy.error
from fpy.bytecode.assembler import deserialize_directives, serialize_directives
from fpy.compiler import (
    analysis_to_fpybc_directives,
    analyze_ast,
    compile_to_fpybin,
    text_to_ast,
)
from fpy.state import get_base_compile_state

DICT = str(Path(__file__).parent / "RefTopologyDictionary.json")
GOLDEN = Path(__file__).parent / "golden"


def _manual(source: str) -> tuple[bytes, int]:
    state = get_base_compile_state(DICT)
    body = text_to_ast(source)
    state = analyze_ast(body, state)
    directives, seq_arg_types = analysis_to_fpybc_directives(state)
    arg_specs = [(name, t.name, t.max_size) for name, t in seq_arg_types]
    return serialize_directives(
        directives, arg_specs, max_directive_size=state.max_directive_size
    )


def test_matches_manual_pipeline():
    source = (GOLDEN / "func_used.fpy").read_text()
    assert compile_to_fpybin(source, DICT) == _manual(source)


def test_deserialize_bytes():
    source = (GOLDEN / "func_used.fpy").read_text()
    data, crc = compile_to_fpybin(source, DICT)
    assert isinstance(data, bytes) and len(data) > 0
    assert isinstance(crc, int)
    directives, _arg_type_names = deserialize_directives(data)
    assert len(directives) > 0


def test_compile_error_propagates():
    with pytest.raises(fpy.error.CompileError):
        compile_to_fpybin("not_a_real_callable(1)\n", DICT)


def test_node_meta_records_source_file(monkeypatch):
    monkeypatch.setattr(fpy.error, "file_name", "/some/driver.fpy")
    body = text_to_ast("x: U32 = 1\n")
    assert body.stmts[0].meta.file == "/some/driver.fpy"


def test_directives_record_originating_file(tmp_path, monkeypatch):
    (tmp_path / "lib.fpy").write_text(
        "def bump(x: U64) -> U64:\n"
        "    y: U64 = x\n"
        "    y += 1\n"
        "    return y\n"
    )
    monkeypatch.setattr(fpy.error, "file_name", "<driver>.fpy")
    state = get_base_compile_state(
        DICT,
        import_directories=[str(tmp_path)],
        main_file_dir=str(tmp_path),
        main_file_path="<driver>.fpy",
    )
    body = text_to_ast("from lib import bump\n\nassert bump(1) == 2\n")
    state = analyze_ast(body, state)
    directives, _ = analysis_to_fpybc_directives(state)

    located = set()
    for d in directives:
        node = d.source_node
        if node is None:
            continue
        meta = getattr(node, "meta", node)
        if getattr(meta, "line", None) is None:
            continue
        located.add((getattr(meta, "file", None), meta.line))

    lib = str(tmp_path / "lib.fpy")
    # directives originate in both files, each tagged with its own source
    assert any(f == "<driver>.fpy" for f, _ in located)
    assert any(f == lib for f, _ in located)
    # the desugared `y += 1` (lib.fpy line 3) is attributed to the library it was
    # written in, not the driver that called it
    assert (lib, 3) in located
