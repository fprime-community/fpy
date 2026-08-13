"""Helpers for fpy tests: compile sequences, run them on the test harnesses
(or a live GDS deployment with --use-gds), and assert success or failure."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import fpy.error
from fpy.bytecode.assembler import serialize_directives
from fpy.bytecode.directives import (
    AllocateDirective,
    Directive,
    DirectiveErrorCode,
    FwOpcodeType,
    GotoDirective,
    PushValDirective,
)
from fpy.compiler import (
    analysis_to_fpybc_directives,
    analysis_to_wasm,
    analyze_ast,
    text_to_ast,
)
from fpy.dictionary import load_dictionary
from fpy.error import WarningType
from fpy.harness import HarnessError, fpybc_harness, wasm_harness
from fpy.state import CompileState, get_base_compile_state
from fpy.types import CmdDef, FpyType, FpyValue, TypeKind

default_dictionary = str(
    Path(__file__).parent.parent.parent / "test" / "fpy" / "RefTopologyDictionary.json"
)

# Flipped to True by conftest's pytest_configure when --wasm is passed, routing
# the assert_* helpers through the LLVM/wasm backend (run on the real
# Svc::WasmSequencer through the wasm harness) instead of the bytecode VM.
USE_WASM = False

# Fw.CmdResponse enum values.
CMD_RESPONSE_OK = 0
CMD_RESPONSE_EXECUTION_ERROR = 4


class CompilationFailed(Exception):
    """Raised when compilation fails expectedly (parse error or semantic error)."""


class ValidationError(Exception):
    """Raised when the sequencer rejects a sequence during validation (bad
    file, bad CRC, argument size mismatch, ...), before running anything."""


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------

# Every known warning type. Tests fail on ANY warning by default: the compile
# helpers promote every warning to a hard error unless the caller declares it in
# `expected_warnings` (kept as a collected warning) or `ignored_warnings`
# (dropped). This surfaces stray warnings -- e.g. an accidental shadow -- that a
# test did not mean to trigger.
ALL_WARNINGS = frozenset(WarningType)


def _default_error_warnings(error_warnings, ignored_warnings, expected_warnings):
    """The set of warnings to promote to errors. An explicit *error_warnings*
    wins; otherwise it is every warning except those expected or ignored."""
    if error_warnings is not None:
        return error_warnings
    return ALL_WARNINGS - set(expected_warnings or ()) - set(ignored_warnings or ())


def _assert_expected_emitted(state, expected_warnings):
    """A warning in *expected_warnings* must actually be emitted, not merely
    allowed -- so declaring it both permits it and asserts it. (Unexpected
    warnings already fail via promotion to errors.)"""
    if not expected_warnings:
        return
    emitted = {w.type for w in state.warnings}
    missing = set(expected_warnings) - emitted
    assert not missing, f"expected warnings not emitted: {missing} (got {emitted})"


def _compile(
    seq: str,
    to_wasm: bool,
    ground_binary_dir: str = None,
    ignored_warnings=None,
    error_warnings=None,
    expected_warnings=None,
    import_directories: list[str] | None = None,
    main_file_dir: str | None = None,
    main_file_path: str | None = None,
):
    """Compile a sequence string and return (state, backend output): the wasm
    binary bytes when *to_wasm*, else (directives, arg_types).

    By default every warning is a hard error; pass *expected_warnings* to allow
    (and still collect) specific ones."""
    fpy.error.file_name = "<test>"

    state = get_base_compile_state(
        default_dictionary,
        ground_binary_dir,
        ignored_warnings=ignored_warnings,
        error_warnings=_default_error_warnings(
            error_warnings, ignored_warnings, expected_warnings
        ),
        import_directories=import_directories,
        main_file_dir=main_file_dir,
        main_file_path=main_file_path,
    )

    try:
        body = text_to_ast(seq)
        state = analyze_ast(body, state)
        if to_wasm:
            output, _ = analysis_to_wasm(state)
        else:
            output = analysis_to_fpybc_directives(state)
    except (fpy.error.CompileError, fpy.error.BackendError) as e:
        raise CompilationFailed(f"Compilation failed:\n{e}")

    _assert_expected_emitted(state, expected_warnings)
    return state, output


def compile_seq(
    seq: str, **kwargs
) -> tuple[CompileState, list[Directive], list[tuple[str, FpyType]]]:
    """Compile a sequence string to fpy bytecode. Returns
    (state, directives, arg_types). See _compile for the keyword args."""
    state, (directives, arg_types) = _compile(seq, to_wasm=False, **kwargs)
    return state, directives, arg_types


def compile_seq_wasm(seq: str, **kwargs) -> bytes:
    """Compile a sequence string to a runnable wasm binary (the LLVM backend).
    See _compile for the keyword args."""
    _, wasm = _compile(seq, to_wasm=True, **kwargs)
    return wasm


# ---------------------------------------------------------------------------
# Compiled sequence files
# ---------------------------------------------------------------------------

# One scratch directory per test session for compiled sequence files. The
# harness runs with this as its working directory and gets the short relative
# file name, because the RUN command's file path argument is a command string,
# which F Prime silently caps at FW_CMD_STRING_MAX_SIZE (40) characters.
_scratch_dir: tempfile.TemporaryDirectory | None = None


def _write_for_harness(
    data: bytes, name: str, directory: str = None
) -> tuple[str, str]:
    """Write *data* to <directory>/<name> (the per-session scratch directory by
    default) and return (directory, name)."""
    global _scratch_dir
    if directory is None:
        if _scratch_dir is None:
            _scratch_dir = tempfile.TemporaryDirectory(prefix="fpy-harness-")
        directory = _scratch_dir.name
    Path(directory, name).write_bytes(data)
    return directory, name


def _write_tmpfile(data: bytes, suffix: str) -> str:
    """Write *data* to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(data)
    f.close()
    return f.name


def _serialize_seq(
    directives: list[Directive], arg_types: list[tuple[str, FpyType]]
) -> bytes:
    """Serialize directives (with the sequence arg specs) to .bin file bytes."""
    arg_specs = [(name, t.name, t.max_size) for name, t in (arg_types or [])]
    return serialize_directives(directives, arg_specs=arg_specs)[0]


def _serialize_args(args: list[FpyValue] | None) -> bytes | None:
    """Serialize a list of sequence argument values to bytes."""
    if args is None:
        return None
    return b"".join(v.serialize() for v in args)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _run_request(
    seq_file: str,
    seq_dir: str,
    tlm: dict[str, bytes] = None,
    prms: dict[str, bytes] = None,
    time_base: int = 0,
    time_context: int = 0,
    initial_time_us: int = 0,
    args: bytes = None,
    cmd_responses: dict[int, int] = None,
) -> dict:
    """The run request fields common to both sequencer harnesses. *tlm* and
    *prms* map channel/parameter names to the serialized values the harness
    answers reads with; every command completes OK unless *cmd_responses*
    maps its opcode to another Fw.CmdResponse value."""
    d = load_dictionary(default_dictionary)
    responses = cmd_responses or {}
    request = {
        "seqFile": seq_file,
        "cwd": seq_dir,
        "time": {
            "base": time_base,
            "context": time_context,
            "seconds": initial_time_us // 1_000_000,
            "useconds": initial_time_us % 1_000_000,
        },
        "tlm": {
            str(d["ch_name_dict"][chan_name].ch_id): bytes(val).hex()
            for chan_name, val in (tlm or {}).items()
        },
        "prms": {
            str(d["prm_name_dict"][prm_name].prm_id): bytes(val).hex()
            for prm_name, val in (prms or {}).items()
        },
        "cmdResponses": {
            str(opcode): response for opcode, response in sorted(responses.items())
        },
    }
    if args is not None:
        request["args"] = args.hex()
    return request


def _seq_args_buffer_len(d: dict) -> int:
    """The dictionary's Svc.SeqArgs buffer length. The harness needs it to
    parse seq-run commands, and it can differ from the flight build's own
    Svc::SeqArgs size."""
    (buffer_member,) = [
        m for m in d["type_defs"]["Svc.SeqArgs"].members if m.name == "buffer"
    ]
    return buffer_member.type.max_size


def _expected_stack_bytes(directives: list[Directive], args: bytes | None) -> int:
    """The exact stack size a successful run must end with: the sequence
    arguments plus the frame setup (PushVal for the flags default, then
    optionally Allocate for the remaining locals). If functions are present
    the first directive is a Goto that jumps past them; the setup starts at
    its target."""
    setup_start = 0
    if directives and isinstance(directives[0], GotoDirective):
        setup_start = directives[0].dir_idx
    setup_size = 0
    if setup_start < len(directives) and isinstance(
        directives[setup_start], PushValDirective
    ):
        setup_size += len(directives[setup_start].val)
        if setup_start + 1 < len(directives) and isinstance(
            directives[setup_start + 1], AllocateDirective
        ):
            setup_size += directives[setup_start + 1].size
    return len(args or b"") + setup_size


def _as_int(v) -> int:
    """An error code as a plain int, whether it is a DirectiveErrorCode or
    already an int."""
    return v.value if isinstance(v, DirectiveErrorCode) else v


def run_seq_raw(
    directives: list[Directive],
    tlm: dict[str, bytes] = None,
    time_base: int = 0,
    time_context: int = 0,
    initial_time_us: int = 0,
    args: bytes = None,
    arg_types: list[tuple[str, FpyType]] = None,
    seq_run_opcodes: set[int] = None,
    ground_binary_dir: str = None,
    prms: dict[str, bytes] = None,
    cmd_responses: dict[int, int] = None,
) -> dict:
    """Run a list of directives on a real Svc::FpySequencer through the test
    harness (test/harness) and return the harness's raw JSON reply. See
    _run_request for the inputs."""
    d = load_dictionary(default_dictionary)

    # When the test provides a ground_binary_dir, that directory doubles as
    # the harness's working directory so child sequence files resolve against
    # it, like they did against the compiler's ground_binary_dir.
    seq_dir, seq_file = _write_for_harness(
        _serialize_seq(directives, arg_types), "s0.bin", directory=ground_binary_dir
    )
    if seq_run_opcodes is None and ground_binary_dir is not None:
        seq_run_opcodes = {d["cmd_name_dict"]["Ref.seqDisp.RUN_ARGS"].opcode}

    request = _run_request(
        seq_file,
        seq_dir,
        tlm=tlm,
        prms=prms,
        time_base=time_base,
        time_context=time_context,
        initial_time_us=initial_time_us,
        args=args,
        cmd_responses=cmd_responses,
    )
    if seq_run_opcodes:
        request["seqRunOpcodes"] = sorted(seq_run_opcodes)
        request["seqArgsBufferSize"] = _seq_args_buffer_len(d)

    return fpybc_harness().run(request)


def run_seq(directives: list[Directive], **run_kwargs) -> list[bytes]:
    """Run a list of directives on a real Svc::FpySequencer through the test
    harness (test/harness). Returns the command buffers the sequence
    dispatched (the big-endian serialized FwOpcodeType + arguments), in call
    order. See run_seq_raw for the keyword args.

    Raises ValidationError when the sequencer rejects the sequence before
    running it, and RuntimeError when the sequence fails: with the
    DirectiveErrorCode for a trap, or the raw error code int for a nonzero
    exit.
    """
    result = run_seq_raw(directives, **run_kwargs)

    if "error" in result:
        raise HarnessError(result["error"])
    if "cmdResponse" not in result:
        raise HarnessError(f"harness gave no command response: {result}")

    response = result["cmdResponse"]
    if response == CMD_RESPONSE_OK:
        # Success is judged by the sequencer's own answer to the RUN command,
        # the same signal a ground station sees. Cross-check it against the
        # sequencer's internal state, so a disagreement fails loudly instead
        # of passing silently.
        if result["sequencesSucceeded"] != 1:
            raise HarnessError(
                f"sequencer responded OK but did not count a success: {result}"
            )
        if result["lastDirectiveError"] != DirectiveErrorCode.NO_ERROR.value:
            raise HarnessError(
                f"sequencer responded OK but recorded a directive error: {result}"
            )
        # A finished run must leave exactly the stack bytes the compiler
        # expected; a leak of even one byte is a failure.
        expected_stack = _expected_stack_bytes(directives, run_kwargs.get("args"))
        actual_stack = len(bytes.fromhex(result["stack"]))
        if actual_stack != expected_stack:
            raise RuntimeError(f"Sequence leaked {actual_stack - expected_stack} bytes")
        return [bytes.fromhex(c) for c in result["cmds"]]

    if response != CMD_RESPONSE_EXECUTION_ERROR:
        raise HarnessError(f"unexpected response {response} to the RUN command")
    if result["sequencesSucceeded"] != 0:
        raise HarnessError(
            f"sequencer responded EXECUTION_ERROR but counted a success: {result}"
        )
    if not result["reachedRunning"]:
        # The sequencer never started running the sequence: validation
        # rejected it. The events say why.
        raise ValidationError("; ".join(e["text"] for e in result["events"]))
    # A nonzero exit surfaces as the raw error code int (reported through the
    # SequenceExitedWithError event); a trap surfaces as its
    # DirectiveErrorCode.
    if "exitCode" in result:
        raise RuntimeError(result["exitCode"])
    raise RuntimeError(DirectiveErrorCode(result["lastDirectiveError"]))


def run_wasm_raw(
    wasm: bytes,
    tlm: dict[str, bytes] = None,
    prms: dict[str, bytes] = None,
    time_base: int = 0,
    time_context: int = 0,
    initial_time_us: int = 0,
    args: bytes = None,
    cmd_responses: dict[int, int] = None,
) -> dict:
    """Run an already-linked wasm module on a real Svc::WasmSequencer through
    the wasm harness and return the harness's raw JSON reply. See _run_request
    for the inputs."""
    seq_dir, seq_file = _write_for_harness(wasm, "m0.wasm")
    request = _run_request(
        seq_file,
        seq_dir,
        tlm=tlm,
        prms=prms,
        time_base=time_base,
        time_context=time_context,
        initial_time_us=initial_time_us,
        args=args,
        cmd_responses=cmd_responses,
    )
    return wasm_harness().run(request)


def run_wasm(
    wasm: bytes, **run_kwargs
) -> tuple[int, list[tuple[int, str]], list[bytes]]:
    """Run an already-linked wasm module on a real Svc::WasmSequencer through
    the wasm harness and return (error code, reported events, dispatched
    command buffers). See run_wasm_raw for the keyword args."""
    result = run_wasm_raw(wasm, **run_kwargs)

    if "error" in result:
        raise HarnessError(result["error"])
    if "cmdResponse" not in result:
        raise HarnessError(f"wasm harness gave no command response: {result}")

    # The guest-flagged events are the ones the sequence itself logged; the
    # rest are the sequencer's own reporting.
    events = [(e["severity"], e["text"]) for e in result["events"] if e.get("guest")]
    cmds = [bytes.fromhex(c) for c in result["cmds"]]

    if result["cmdResponse"] == CMD_RESPONSE_OK:
        return 0, events, cmds
    if "exitCode" in result:
        # The code the sequence passed to the exit or panic host import,
        # reported through the SequenceExitedWithError event.
        return result["exitCode"], events, cmds
    raise HarnessError(
        "wasm sequence failed without an exit code (interpreter trap): "
        + "; ".join(e["text"] for e in result["events"])
    )


def _run_seq_wasm(
    seq: str,
    cmd_responses: dict[int, int] = None,
    **kwargs,
) -> tuple[int, list[tuple[int, str]], list[bytes]]:
    """Compile *seq* to wasm and run it through the wasm harness. Returns
    (error code, reported events, dispatched command buffers). See _compile
    for the remaining keyword args."""
    wasm = compile_seq_wasm(seq, **kwargs)
    return run_wasm(wasm, cmd_responses=cmd_responses)


def run_seq_wasm(seq: str, **kwargs) -> int:
    """Compile *seq* to wasm and run it, returning the sequence's error code
    (reported via the exit/panic host imports; 0 when the void entrypoint
    falls off its end without failing)."""
    code, _, _ = _run_seq_wasm(seq, **kwargs)
    return code


def run_seq_wasm_with_events(seq: str, **kwargs) -> tuple[int, list[tuple[int, str]]]:
    """Like run_seq_wasm, but also returns the events the sequence reported
    through the event host import (the log() builtin) as (severity, message)
    pairs, in call order."""
    code, events, _ = _run_seq_wasm(seq, **kwargs)
    return code, events


def run_seq_wasm_with_cmds(seq: str, **kwargs) -> tuple[int, list[bytes]]:
    """Like run_seq_wasm, but also returns the command buffers the sequence
    dispatched through the cmd host import (the big-endian serialized
    FwOpcodeType + arguments), in call order."""
    code, _, cmds = _run_seq_wasm(seq, **kwargs)
    return code, cmds


# ---------------------------------------------------------------------------
# Running on a live GDS deployment (--use-gds)
# ---------------------------------------------------------------------------


def _build_seq_args_json(args: bytes) -> str:
    """Build a JSON string for the Svc.SeqArgs struct expected by RUN_ARGS."""
    buf = list(args) + [0] * (255 - len(args))
    return json.dumps({"size": len(args), "buffer": buf})


def _run_gds(fprime_test_api, file_path, args, wasm, expect_ok, timeout_s=4):
    """Send a compiled sequence file to a live GDS deployment. With
    *expect_ok*, assert the RUN command succeeds; otherwise assert it fails
    with an OpCodeError event."""
    if wasm:
        cmd, cmd_args = "Ref.wasmSeq.RUN", [file_path, "BLOCK"]
    elif args:
        cmd, cmd_args = (
            "Ref.seqDisp.RUN_ARGS",
            [file_path, "BLOCK", _build_seq_args_json(args)],
        )
    else:
        cmd, cmd_args = "Ref.seqDisp.RUN", [file_path, "BLOCK"]
    if expect_ok:
        fprime_test_api.send_and_assert_command(cmd, cmd_args, timeout=timeout_s)
    else:
        fprime_test_api.send_and_assert_event(
            cmd, cmd_args, events="CdhCore.cmdDisp.OpCodeError", timeout=timeout_s
        )


# ---------------------------------------------------------------------------
# Asserts
# ---------------------------------------------------------------------------


def lookup_type(type_name: str) -> FpyType:
    """Look up a type from the test dictionary by name."""
    return load_dictionary(default_dictionary)["type_defs"][type_name]


def assert_compile_success(fprime_test_api, seq: str, **kwargs):
    """Compile *seq* on the current backend. See _compile for the keyword
    args."""
    if USE_WASM:
        compile_seq_wasm(seq, **kwargs)
    else:
        compile_seq(seq, **kwargs)


def assert_compile_failure(fprime_test_api, seq: str, match: str = None, **kwargs):
    """Compile *seq* on the current backend and assert it fails, optionally
    matching the error message against the *match* regex."""
    try:
        assert_compile_success(fprime_test_api, seq, **kwargs)
    except (SystemExit, CompilationFailed) as e:
        if match is not None:
            assert re.search(match, str(e)), f"Expected match {match!r} in {e!r}"
        return
    raise RuntimeError("compile_seq succeeded")


def assert_run_success(
    fprime_test_api,
    seq: str,
    timeout_s: int = 4,
    args: list[FpyValue] = None,
    ground_binary_dir: str = None,
    import_directories: list[str] | None = None,
    expected_warnings=None,
    main_file_dir: str | None = None,
    **run_kwargs,
) -> list[bytes] | None:
    """Compile *seq* on the current backend, run it, and assert it succeeds.
    Returns the command buffers the sequence dispatched, or None when running
    against a live GDS deployment. The remaining keyword args (tlm, prms,
    time, cmd_responses, ...) are the current backend's run_*_raw inputs.

    Runs on the test harness by default, or against a live GDS deployment
    when fprime_test_api is not None (--use-gds)."""
    compile_kwargs = dict(
        ground_binary_dir=ground_binary_dir,
        import_directories=import_directories,
        expected_warnings=expected_warnings,
        main_file_dir=main_file_dir,
    )
    if USE_WASM:
        wasm = compile_seq_wasm(seq, **compile_kwargs)
        if fprime_test_api is not None:
            _run_gds(
                fprime_test_api,
                _write_tmpfile(wasm, ".wasm"),
                None,
                wasm=True,
                expect_ok=True,
                timeout_s=timeout_s,
            )
            return
        code, _, cmds = run_wasm(wasm, **run_kwargs)
        if code != DirectiveErrorCode.NO_ERROR.value:
            raise RuntimeError(f"wasm sequence returned error code {code}")
        return cmds

    _, directives, arg_types = compile_seq(seq, **compile_kwargs)
    args_bytes = _serialize_args(args)
    if fprime_test_api is not None:
        _run_gds(
            fprime_test_api,
            _write_tmpfile(_serialize_seq(directives, arg_types), ".bin"),
            args_bytes,
            wasm=False,
            expect_ok=True,
            timeout_s=timeout_s,
        )
        return None
    return run_seq(
        directives,
        args=args_bytes,
        arg_types=arg_types,
        ground_binary_dir=ground_binary_dir,
        **run_kwargs,
    )


def assert_run_failure(
    fprime_test_api,
    seq: str,
    error_code: DirectiveErrorCode | int = None,
    validation_error: bool = False,
    args: list[FpyValue] = None,
    ground_binary_dir: str = None,
    import_directories: list[str] | None = None,
    **run_kwargs,
):
    """Compile *seq* on the current backend, run it, and assert it fails:
    with *error_code* (a DirectiveErrorCode trap or a raw exit code int), or
    with *validation_error* when the sequencer must reject the sequence
    before running it. The remaining keyword args (tlm, prms, time,
    cmd_responses, ...) are the current backend's run_*_raw inputs."""
    assert not (
        error_code is not None and validation_error
    ), "Cannot specify both error_code and validation_error"
    assert (
        error_code is not None or validation_error
    ), "Must specify either error_code or validation_error"

    compile_kwargs = dict(
        ground_binary_dir=ground_binary_dir, import_directories=import_directories
    )
    if USE_WASM:
        wasm = compile_seq_wasm(seq, **compile_kwargs)
        if fprime_test_api is not None:
            _run_gds(
                fprime_test_api,
                _write_tmpfile(wasm, ".wasm"),
                None,
                wasm=True,
                expect_ok=False,
            )
            return
        # The wasm backend has no separate validation step or VM-internal
        # faults: a failed sequence is one that reports a nonzero code
        # through the exit/fault host imports.
        code, _, _ = run_wasm(wasm, **run_kwargs)
        if code == DirectiveErrorCode.NO_ERROR.value:
            raise RuntimeError("wasm sequence succeeded")
        if error_code is not None and code != _as_int(error_code):
            raise RuntimeError(f"wasm sequence returned {code}, expected {error_code}")
        return

    _, directives, arg_types = compile_seq(seq, **compile_kwargs)
    args_bytes = _serialize_args(args)
    if fprime_test_api is not None:
        _run_gds(
            fprime_test_api,
            _write_tmpfile(_serialize_seq(directives, arg_types), ".bin"),
            args_bytes,
            wasm=False,
            expect_ok=False,
        )
        return

    try:
        run_seq(
            directives,
            args=args_bytes,
            arg_types=arg_types,
            ground_binary_dir=ground_binary_dir,
            **run_kwargs,
        )
    except ValidationError as e:
        if not validation_error:
            raise
        print(e)
        return
    except RuntimeError as e:
        if validation_error:
            raise RuntimeError("Expected ValidationError, got", type(e).__name__, e)
        # The failure surfaces as either a DirectiveErrorCode trap or a raw
        # exit code int; the expected value may likewise be either. Compare by
        # integer value so e.g. an exit code of 7 matches
        # DirectiveErrorCode.EXIT_WITH_ERROR.
        if len(e.args) == 1 and _as_int(e.args[0]) != _as_int(error_code):
            raise RuntimeError(
                "run_seq failed with error", e.args[0], "expected", error_code
            )
        print(e)
        return

    raise RuntimeError("run_seq succeeded")
