from pathlib import Path

import pytest

import fpy.error
from fpy.bytecode.assembler import deserialize_directives, serialize_directives
from fpy.compiler import (
    analysis_to_fpybc_directives,
    analyze_ast,
    compile_to_fpybin,
    fpybin_source_map,
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
    result = compile_to_fpybin(source, DICT)
    assert (result.data, result.crc) == _manual(source)


def test_deserialize_bytes():
    source = (GOLDEN / "func_used.fpy").read_text()
    result = compile_to_fpybin(source, DICT)
    assert isinstance(result.data, bytes) and len(result.data) > 0
    assert isinstance(result.crc, int)
    directives, _arg_type_names = deserialize_directives(result.data)
    assert len(directives) > 0


def test_compile_error_propagates():
    with pytest.raises(fpy.error.CompileError):
        compile_to_fpybin("not_a_real_callable(1)\n", DICT)


def test_node_meta_records_source_file(monkeypatch):
    monkeypatch.setattr(fpy.error, "file_name", "/some/driver.fpy")
    body = text_to_ast("x: U32 = 1\n")
    assert body.stmts[0].meta.file_name == "/some/driver.fpy"


def test_directives_record_originating_file(tmp_path, monkeypatch):
    """
    Functions get compiled into their calling sites, which may be in different files.
    Ensure that the file attribution remains correct.
    """
    driver = "driver.fpy"
    lib = str(tmp_path / "lib.fpy")
    Path(lib).write_text(
        "def bump(x: U64) -> U64:\n"
        "    y: U64 = x\n"
        "    y += 1\n"
        "    return y\n"
    )
    monkeypatch.setattr(fpy.error, "file_name", driver)
    state = get_base_compile_state(
        DICT,
        import_directories=[tmp_path],
        main_file_dir=tmp_path,
        main_file_path=driver,
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
        located.add((getattr(meta, "file_name", None), meta.line))

    # Every emitting statement maps to exactly its own file and line: the driver's
    # import and assert to the driver, the def body (including the desugared
    # `y += 1`) to the library. The def signature (lib:1) and the blank driver
    # line emit no directives.
    assert located == {
        (driver, 1),  # from lib import bump
        (driver, 3),  # assert bump(1) == 2
        (lib, 2),  # y: U64 = x
        (lib, 3),  # y += 1  (desugared)
        (lib, 4),  # return y
    }


def test_source_map_locates_each_directive(tmp_path):
    """The source map has one entry per directive, each naming the file and
    line it came from, including lines in an imported file."""
    lib = tmp_path / "lib.fpy"
    lib.write_text(
        "\n".join(
            [
                "def bump(x: U64) -> U64:",
                "    y: U64 = x",
                "    y += 1",
                "    return y",
                "",
            ]
        )
    )
    driver = str(tmp_path / "driver.fpy")
    source = "from .lib import bump\n\nassert bump(1) == 2\n"

    result = compile_to_fpybin(
        source, DICT, main_file_dir=str(tmp_path), main_file_path=driver
    )

    smap = result.source_map
    assert smap["crc"] == result.crc
    directives, _ = deserialize_directives(result.data)
    assert len(smap["directives"]) == len(directives)
    located = {(e["file"], e["line"]) for e in smap["directives"] if e is not None}
    assert located == {
        (driver, 3),  # assert bump(1) == 2
        (str(lib.resolve()), 2),  # y: U64 = x
        (str(lib.resolve()), 3),  # y += 1
        (str(lib.resolve()), 4),  # return y
    }


def test_source_map_omits_block_directives(tmp_path):
    """Directives a block generates for itself, such as the main sequence's
    frame allocation and exit, have None entries rather than the block's first
    line, here a comment."""
    driver = str(tmp_path / "driver.fpy")
    source = "# a comment\nx: U32 = 1\nif x == 1:\n    CdhCore.cmdDisp.CMD_NO_OP()\n"

    result = compile_to_fpybin(
        source, DICT, main_file_dir=str(tmp_path), main_file_path=driver
    )

    entries = result.source_map["directives"]
    assert {e["line"] for e in entries if e is not None} == {2, 3, 4}
    assert None in entries


def test_source_map_entry_without_source_is_none():
    """A directive with no originating statement has a None entry."""

    class Bare:
        source_node = None

    assert fpybin_source_map([Bare()], 7) == {"crc": 7, "directives": [None]}
