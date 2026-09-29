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
