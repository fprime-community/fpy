"""
Golden tests for the fpy compiler.

Each golden case is a .fpy source file in test/fpy/golden/ with golden
artifacts alongside it:
- <name>.fpybc: the bytecode backend's assembly output
- <name>.wat: the LLVM backend's WebAssembly text output, or the backend
  error it raises for constructs it does not support
- <name>.json: the raw JSON replies from running the compiled sequence on
  the FpySequencer ("fpybc") and WasmSequencer ("wasm") test harnesses --
  commands dispatched, events, final state, stack bytes. The fields both
  replies agree on are stored once under "common"; each backend's key
  holds only the fields where it diverges from the other

A sequence declares the inputs its harness runs need in "# harness:"
comments, one directive per line:
    # harness: tlm <channel> <hex>            answer to a telemetry read
    # harness: prm <parameter> <hex>          answer to a parameter read
    # harness: arg <hex>                      appended to the sequence arguments
    # harness: time <base> <context> <us>     the sequencer's start time
    # harness: cmd_response <command> <int>   that command's Fw.CmdResponse

Regenerate the artifacts with:
    uv run pytest test/fpy/test_golden.py --update-goldens
"""

import json
import re
from pathlib import Path

import pytest

import fpy.error
from fpy.bytecode.assembler import fpybc_directives_to_fpyasm
from fpy.compiler import (
    analysis_to_fpybc_directives,
    analysis_to_wasm,
    analysis_to_wat,
    analyze_ast,
    text_to_ast,
)
from fpy.dictionary import load_dictionary
from fpy.error import BackendError
from fpy.state import get_base_compile_state
from fpy.test_helpers import run_seq_raw, run_wasm_raw

GOLDEN_DIR = Path(__file__).parent / "golden"

# The backends a golden run file records, one harness each.
BACKENDS = ("fpybc", "wasm")

# Path to the test dictionary
DEFAULT_DICTIONARY = str(Path(__file__).parent / "RefTopologyDictionary.json")

_HARNESS_INPUT = re.compile(r"^#\s*harness:\s*(.+?)\s*$", re.MULTILINE)


def parse_harness_inputs(source: str) -> dict:
    """The harness run inputs declared in the sequence's "# harness:"
    comments (see the module docstring for the directives), as keyword
    arguments for run_seq_raw / run_wasm_raw."""
    d = load_dictionary(DEFAULT_DICTIONARY)
    inputs = {"tlm": {}, "prms": {}, "cmd_responses": {}}
    args = b""
    for match in _HARNESS_INPUT.finditer(source):
        directive, *operands = match.group(1).split()
        if directive == "tlm":
            name, value = operands
            inputs["tlm"][name] = bytes.fromhex(value)
        elif directive == "prm":
            name, value = operands
            inputs["prms"][name] = bytes.fromhex(value)
        elif directive == "arg":
            (value,) = operands
            args += bytes.fromhex(value)
        elif directive == "time":
            base, context, microseconds = operands
            inputs["time_base"] = int(base)
            inputs["time_context"] = int(context)
            inputs["initial_time_us"] = int(microseconds)
        elif directive == "cmd_response":
            name, value = operands
            opcode = d["cmd_name_dict"][name].opcode
            inputs["cmd_responses"][opcode] = int(value)
        else:
            raise ValueError(f"unknown harness input directive: {match.group(1)!r}")
    if args:
        inputs["args"] = args
    return inputs


def _analyze(source: str):
    """Parse and semantically analyze fpy source, returning the compile
    state ready for a backend."""
    fpy.error.file_name = "<golden-test>"
    fpy.error.input_text = source
    fpy.error.input_lines = source.splitlines()

    state = get_base_compile_state(DEFAULT_DICTIONARY)

    body = text_to_ast(source)
    assert body is not None, "Parsing failed"

    return analyze_ast(body, state)


def compile_to_fpybc(source: str) -> str:
    """Compile fpy source code to fpybc bytecode text."""
    directives, _ = analysis_to_fpybc_directives(_analyze(source))
    return fpybc_directives_to_fpyasm(directives)


def compile_to_wat(source: str) -> str:
    """Compile fpy source code to WebAssembly text, or the error message the
    LLVM backend raises for sequences it does not support."""
    state = _analyze(source)
    try:
        wat, _ = analysis_to_wat(state)
        return wat
    except (BackendError, NotImplementedError) as e:
        return f"compile error: {type(e).__name__}: {e}\n"


def run_on_fpybc_harness(name: str, source: str) -> dict:
    """Compile fpy source to bytecode and run it on the FpySequencer harness,
    returning the raw JSON reply."""
    directives, arg_types = analysis_to_fpybc_directives(_analyze(source))
    inputs = parse_harness_inputs(source)
    reply = run_seq_raw(directives, arg_types=arg_types, **inputs)
    assert "error" not in reply, f"harness failed to run {name}: {reply}"
    return reply


def run_on_wasm_harness(name: str, source: str) -> dict:
    """Compile fpy source to wasm and run it on the WasmSequencer harness,
    returning the raw JSON reply, or the error message the LLVM backend
    raises for sequences it does not support."""
    state = _analyze(source)
    try:
        wasm, _ = analysis_to_wasm(state)
    except (BackendError, NotImplementedError) as e:
        return {"compileError": f"{type(e).__name__}: {e}"}
    reply = run_wasm_raw(wasm, **parse_harness_inputs(source))
    assert "error" not in reply, f"harness failed to run {name}: {reply}"
    return reply


def get_golden_test_cases():
    """Find all golden test cases (.fpy files)."""
    return sorted(f.stem for f in GOLDEN_DIR.glob("*.fpy"))


def _check_or_update_text(path: Path, actual: str, update: bool):
    """Compare *actual* against the golden file, or rewrite it with
    --update-goldens."""
    if update:
        if not path.exists() or path.read_text() != actual:
            path.write_text(actual)
        return
    assert path.exists(), (
        f"golden file {path.name} is missing; generate it with "
        "'pytest test/fpy/test_golden.py --update-goldens'"
    )
    expected = path.read_text()
    assert actual == expected, (
        f"Golden test '{path.name}' failed.\n"
        f"Expected:\n{expected}\n"
        f"Actual:\n{actual}\n"
    )


def _merged_run(recorded: dict, backend: str) -> dict | None:
    """The full reply recorded for *backend*: the "common" fields plus the
    backend's own section. None when the backend has no recorded run."""
    if backend not in recorded:
        return None
    return {**recorded.get("common", {}), **recorded[backend]}


def _split_runs(replies: dict[str, dict]) -> dict:
    """The golden file layout for the backends' full replies: the fields
    every reply agrees on once under "common", the rest under each backend's
    own key."""
    first, *rest = replies.values()
    common = {}
    if rest:
        common = {
            k: v for k, v in first.items() if all(k in r and r[k] == v for r in rest)
        }
    split = {"common": common} if common else {}
    for backend, reply in replies.items():
        split[backend] = {k: v for k, v in reply.items() if k not in common}
    return split


def _check_or_update_run(path: Path, backend: str, actual: dict, update: bool):
    """Compare *actual* against the *backend* run recorded in the golden run
    file, or rewrite that run with --update-goldens."""
    recorded = json.loads(path.read_text()) if path.exists() else {}
    replies = {b: _merged_run(recorded, b) for b in BACKENDS}
    replies = {b: r for b, r in replies.items() if r is not None}
    if update:
        replies[backend] = actual
        split = _split_runs(replies)
        if split != recorded:
            path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
        return
    assert backend in replies, (
        f"golden file {path.name} has no '{backend}' run; generate it with "
        "'pytest test/fpy/test_golden.py --update-goldens'"
    )
    assert actual == replies[backend], (
        f"Golden run '{path.name}' ({backend}) failed.\n"
        f"Expected:\n{json.dumps(replies[backend], indent=2, sort_keys=True)}\n"
        f"Actual:\n{json.dumps(actual, indent=2, sort_keys=True)}\n"
    )


@pytest.mark.parametrize("test_name", get_golden_test_cases())
def test_golden_fpybc(test_name: str, update_goldens: bool):
    """Compile the .fpy file and compare against the .fpybc file."""
    source = (GOLDEN_DIR / f"{test_name}.fpy").read_text()
    actual = compile_to_fpybc(source)
    _check_or_update_text(GOLDEN_DIR / f"{test_name}.fpybc", actual, update_goldens)


@pytest.mark.parametrize("test_name", get_golden_test_cases())
def test_golden_wat(test_name: str, update_goldens: bool):
    """Compile the .fpy file with the LLVM backend and compare against the
    .wat file."""
    source = (GOLDEN_DIR / f"{test_name}.fpy").read_text()
    actual = compile_to_wat(source)
    _check_or_update_text(GOLDEN_DIR / f"{test_name}.wat", actual, update_goldens)


@pytest.mark.parametrize("test_name", get_golden_test_cases())
def test_golden_run_fpybc(test_name: str, update_goldens: bool):
    """Run the compiled sequence on the FpySequencer harness and compare the
    raw reply against the .json file's "fpybc" entry."""
    source = (GOLDEN_DIR / f"{test_name}.fpy").read_text()
    actual = run_on_fpybc_harness(test_name, source)
    _check_or_update_run(
        GOLDEN_DIR / f"{test_name}.json", "fpybc", actual, update_goldens
    )


@pytest.mark.wasm
@pytest.mark.parametrize("test_name", get_golden_test_cases())
def test_golden_run_wasm(test_name: str, update_goldens: bool):
    """Run the compiled sequence on the WasmSequencer harness and compare the
    raw reply against the .json file's "wasm" entry."""
    source = (GOLDEN_DIR / f"{test_name}.fpy").read_text()
    actual = run_on_wasm_harness(test_name, source)
    _check_or_update_run(
        GOLDEN_DIR / f"{test_name}.json", "wasm", actual, update_goldens
    )
