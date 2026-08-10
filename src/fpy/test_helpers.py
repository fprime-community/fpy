from __future__ import annotations
from pathlib import Path
import tempfile
import fpy.error
from fpy.harness import HarnessError, fpy_harness
from fpy.bytecode.directives import (
    AllocateDirective,
    Directive,
    DirectiveErrorCode,
    GotoDirective,
    PushValDirective,
)
from fpy.compiler import (
    text_to_ast,
    analyze_ast,
    analysis_to_fpybc_directives,
    analysis_to_wasm,
)
from fpy.state import CompileState, get_base_compile_state
from fpy.bytecode.assembler import serialize_directives
from fpy.dictionary import load_dictionary
from fpy.error import WarningType
from fpy.types import FpyType, FpyValue

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


default_dictionary = str(
    Path(__file__).parent.parent.parent / "test" / "fpy" / "RefTopologyDictionary.json"
)


class CompilationFailed(Exception):
    """Raised when compilation fails expectedly (parse error or semantic error)."""

    pass


class ValidationError(Exception):
    """Raised when the sequencer rejects a sequence during validation (bad
    file, bad CRC, argument size mismatch, ...), before running anything."""

    pass


# Flipped to True by conftest's pytest_configure when --wasm is passed, routing
# the assert_* helpers through the LLVM/wasm backend (run on the real
# Svc::WasmSequencer through the wasm harness) instead of the bytecode VM.
USE_WASM = False


def compile_seq(
    seq: str,
    ground_binary_dir: str = None,
    ignored_warnings=None,
    error_warnings=None,
    expected_warnings=None,
    import_directories: list[str] | None = None,
    main_file_dir: str | None = None,
    main_file_path: str | None = None,
) -> tuple[CompileState, list[Directive], list[tuple[str, FpyType]]]:
    """Compile a sequence string and return (state, directives, arg_types).

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
        directives, arg_types = analysis_to_fpybc_directives(state)
    except (fpy.error.CompileError, fpy.error.BackendError) as e:
        raise CompilationFailed(f"Compilation failed:\n{e}")

    _assert_expected_emitted(state, expected_warnings)
    return state, directives, arg_types


def compile_seq_wasm(
    seq: str,
    ground_binary_dir: str = None,
    import_directories: list[str] | None = None,
    ignored_warnings=None,
    error_warnings=None,
    expected_warnings=None,
    main_file_dir: str | None = None,
) -> bytes:
    """Compile a sequence string to a runnable wasm binary (the LLVM backend).

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
    )

    try:
        body = text_to_ast(seq)
        state = analyze_ast(body, state)
        wasm, _ = analysis_to_wasm(state)
    except (fpy.error.CompileError, fpy.error.BackendError) as e:
        raise CompilationFailed(f"Compilation failed:\n{e}")

    _assert_expected_emitted(state, expected_warnings)
    return wasm


def run_seq_wasm(
    seq: str,
    ground_binary_dir: str = None,
    import_directories: list[str] | None = None,
    expected_warnings=None,
    main_file_dir: str | None = None,
    failing_opcodes: set[int] = None,
) -> int:
    """Compile *seq* to wasm and run it, returning the sequence's error code
    (reported via the exit/panic host imports; 0 when the void entrypoint
    falls off its end without failing).

    Runs the compiled module on a real Svc::WasmSequencer through the wasm
    harness built by conftest."""
    code, _, _ = _run_seq_wasm(
        seq,
        ground_binary_dir,
        import_directories=import_directories,
        expected_warnings=expected_warnings,
        main_file_dir=main_file_dir,
        failing_opcodes=failing_opcodes,
    )
    return code


def run_seq_wasm_with_events(
    seq: str,
    ground_binary_dir: str = None,
    import_directories: list[str] | None = None,
    expected_warnings=None,
    main_file_dir: str | None = None,
) -> tuple[int, list[tuple[int, str]]]:
    """Like run_seq_wasm, but also returns the events the sequence reported
    through the event host import (the log() builtin) as (severity, message)
    pairs, in call order."""
    code, events, _ = _run_seq_wasm(
        seq,
        ground_binary_dir,
        import_directories=import_directories,
        expected_warnings=expected_warnings,
        main_file_dir=main_file_dir,
    )
    return code, events


def run_seq_wasm_with_cmds(
    seq: str,
    ground_binary_dir: str = None,
    import_directories: list[str] | None = None,
    expected_warnings=None,
    main_file_dir: str | None = None,
    failing_opcodes: set[int] = None,
    cmd_response: int = None,
) -> tuple[int, list[bytes]]:
    """Like run_seq_wasm, but also returns the command buffers the sequence
    dispatched through the cmd host import (the big-endian serialized
    FwOpcodeType + arguments), in call order. Every command completes with
    *cmd_response* (an Fw.CmdResponse value, default OK) unless its opcode is
    in *failing_opcodes*, which makes it complete with EXECUTION_ERROR."""
    code, _, cmds = _run_seq_wasm(
        seq,
        ground_binary_dir,
        import_directories=import_directories,
        expected_warnings=expected_warnings,
        main_file_dir=main_file_dir,
        failing_opcodes=failing_opcodes,
        cmd_response=cmd_response,
    )
    return code, cmds


def _run_seq_wasm(
    seq: str,
    ground_binary_dir: str = None,
    import_directories: list[str] | None = None,
    expected_warnings=None,
    main_file_dir: str | None = None,
    failing_opcodes: set[int] = None,
    cmd_response: int = None,
) -> tuple[int, list[tuple[int, str]], list[bytes]]:
    """Compile *seq* to wasm, run it through the spacewasm runner harness, and
    return (error code, reported events, dispatched command buffers).

    The commands that fail are *failing_opcodes* plus the RUN commands that
    always fail when called from within a running sequence on the same
    sequencer instance -- the same set the bytecode reference model uses."""
    wasm = compile_seq_wasm(
        seq,
        ground_binary_dir,
        import_directories=import_directories,
        expected_warnings=expected_warnings,
        main_file_dir=main_file_dir,
    )
    return run_wasm(wasm, failing_opcodes=failing_opcodes, cmd_response=cmd_response)


def run_wasm(
    wasm: bytes,
    failing_opcodes: set[int] = None,
    cmd_response: int = None,
) -> tuple[int, list[tuple[int, str]], list[bytes]]:
    """Run an already-linked wasm module on a real Svc::WasmSequencer through
    the wasm harness and return (error code, reported events, dispatched
    command buffers).

    The commands that fail are *failing_opcodes* plus the RUN commands that
    always fail when called from within a running sequence on the same
    sequencer instance."""
    from fpy.harness import wasm_harness

    d = load_dictionary(default_dictionary)
    always_failing = {d["cmd_name_dict"]["Ref.cmdSeq0.RUN"].opcode}

    seq_dir, seq_file = _write_wasm_for_harness(wasm)
    request = {
        "seqFile": seq_file,
        "cwd": seq_dir,
        "time": {"base": 0, "context": 0, "seconds": 0, "useconds": 0},
        "failOpcodes": sorted(always_failing | set(failing_opcodes or ())),
    }
    if cmd_response is not None:
        request["cmdResponse"] = cmd_response

    result = wasm_harness().run(request)

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


def lookup_type(fprime_test_api, type_name: str):
    d = load_dictionary(default_dictionary)
    return d["type_defs"][type_name]


def _write_wasm_to_tmpfile(wasm: bytes) -> str:
    """Write a compiled wasm module to a temp .wasm file and return its path."""
    wasm_file = tempfile.NamedTemporaryFile(suffix=".wasm", delete=False)
    wasm_file.write(wasm)
    wasm_file.close()
    return wasm_file.name


def _write_wasm_for_harness(wasm: bytes) -> tuple[str, str]:
    """Write a compiled wasm module to <scratch>/m0.wasm and return
    (directory, file name); like sequence files, the module travels to the
    sequencer as a short relative name because the RUN command's file path
    argument is capped at FW_CMD_STRING_MAX_SIZE characters."""
    global _seq_scratch_dir
    if _seq_scratch_dir is None:
        _seq_scratch_dir = tempfile.TemporaryDirectory(prefix="fpy-harness-")
    name = "m0.wasm"
    Path(_seq_scratch_dir.name, name).write_bytes(wasm)
    return _seq_scratch_dir.name, name


def _write_seq_to_tmpfile(
    directives: list[Directive], arg_types: list[tuple[str, FpyType]] = None
) -> str:
    """Serialize directives to a temp .bin file and return its path."""
    arg_specs = [(name, t.name, t.max_size) for name, t in (arg_types or [])]
    seq_file = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    Path(seq_file.name).write_bytes(
        serialize_directives(directives, arg_specs=arg_specs)[0]
    )
    return seq_file.name


def _build_seq_args_json(args: bytes) -> str:
    """Build a JSON string for the Svc.SeqArgs struct expected by RUN_ARGS."""
    import json

    buf = list(args) + [0] * (255 - len(args))
    return json.dumps({"size": len(args), "buffer": buf})


# Fw.CmdResponse enum values.
CMD_RESPONSE_OK = 0
CMD_RESPONSE_EXECUTION_ERROR = 4

# One scratch directory per test session for compiled sequence files. The
# harness runs with this as its working directory and gets the short relative
# file name, because the RUN command's file path argument is a command string,
# which F Prime silently caps at FW_CMD_STRING_MAX_SIZE (40) characters.
_seq_scratch_dir: tempfile.TemporaryDirectory | None = None


# FIXME again should be fpybc. it's all either wasm or fpybc
def _write_seq_for_harness(
    directives: list[Directive],
    arg_types: list[tuple[str, FpyType]] = None,
    directory: str = None,
) -> tuple[str, str]:
    """Serialize directives to <directory>/s0.bin (a per-session scratch
    directory by default) and return (directory, file name)."""
    global _seq_scratch_dir
    if directory is None:
        if _seq_scratch_dir is None:
            _seq_scratch_dir = tempfile.TemporaryDirectory(prefix="fpy-harness-")
        directory = _seq_scratch_dir.name
    arg_specs = [(name, t.name, t.max_size) for name, t in (arg_types or [])]
    name = "s0.bin"
    Path(directory, name).write_bytes(
        serialize_directives(directives, arg_specs=arg_specs)[0]
    )
    return directory, name


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


def run_seq(
    fprime_test_api,
    directives: list[Directive],
    tlm: dict[str, bytes] = None,
    time_base: int = 0,
    time_context: int = 0,
    initial_time_us: int = 0,
    timeout_s: int = 4,
    failing_opcodes: set[int] = None,
    args: bytes = None,
    seq_run_opcodes: set[int] = None,
    arg_name_types: list[tuple[str, FpyType]] = None,
    ground_binary_dir: str = None,
):
    """Run a list of directives.

    When fprime_test_api is None (the default), runs against a real
    Svc::FpySequencer through the test harness (test/harness). When
    fprime_test_api is a live IntegrationTestAPI (i.e. --use-gds was passed
    to pytest), serializes the directives to a temp file and sends them to
    the running GDS deployment.

    Raises ValidationError when the sequencer rejects the sequence before
    running it, and RuntimeError when the sequence fails: with the
    DirectiveErrorCode for a trap, or the raw error code int for a nonzero
    exit.
    """
    if tlm is None:
        tlm = {}

    if fprime_test_api is not None:
        seq_path = _write_seq_to_tmpfile(directives, arg_name_types)
        if args:
            seq_args = _build_seq_args_json(args)
            fprime_test_api.send_and_assert_command(
                "Ref.seqDisp.RUN_ARGS", [seq_path, "BLOCK", seq_args], timeout=timeout_s
            )
        else:
            fprime_test_api.send_and_assert_command(
                "Ref.seqDisp.RUN", [seq_path, "BLOCK"], timeout=timeout_s
            )
        return

    d = load_dictionary(default_dictionary)
    ch_name_dict = d["ch_name_dict"]
    # These RUN commands always fail when called from within a running sequence
    # on the same sequencer instance; the harness completes them with
    # EXECUTION_ERROR.
    always_failing = {
        d["cmd_name_dict"]["Ref.cmdSeq0.RUN"].opcode,
    }
    if failing_opcodes:
        always_failing |= failing_opcodes

    # The sequence file always sits in the harness's working directory and is
    # named by its short relative name: the RUN command's file path argument
    # is a command string, which F Prime silently caps at
    # FW_CMD_STRING_MAX_SIZE (40) characters. When the test provides a
    # ground_binary_dir, that directory doubles as the working directory so
    # child sequence files resolve against it, like they did against the
    # model's cwd.
    seq_dir, seq_file = _write_seq_for_harness(
        directives, arg_name_types, directory=ground_binary_dir
    )
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
            str(ch_name_dict[chan_name].ch_id): bytes(val).hex()
            for chan_name, val in tlm.items()
        },
        "failOpcodes": sorted(always_failing),
    }
    if args is not None:
        request["args"] = args.hex()
    if seq_run_opcodes:
        request["seqRunOpcodes"] = sorted(seq_run_opcodes)
        request["seqArgsBufferSize"] = _seq_args_buffer_len(d)

    result = fpy_harness().run(request)

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
        expected_stack = _expected_stack_bytes(directives, args)
        actual_stack = len(bytes.fromhex(result["stack"]))
        if actual_stack != expected_stack:
            raise RuntimeError(f"Sequence leaked {actual_stack - expected_stack} bytes")
        return

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


def assert_compile_success(
    fprime_test_api,
    seq: str,
    import_directories: list[str] | None = None,
    expected_warnings=None,
):
    if USE_WASM:
        compile_seq_wasm(
            seq,
            import_directories=import_directories,
            expected_warnings=expected_warnings,
        )
        return
    compile_seq(
        seq,
        import_directories=import_directories,
        expected_warnings=expected_warnings,
    )


def assert_run_success(
    fprime_test_api,
    seq: str,
    tlm: dict[str, bytes] = None,
    time_base: int = 0,
    time_context: int = 0,
    initial_time_us: int = 0,
    timeout_s: int = 4,
    failing_opcodes: set[int] = None,
    args: list[FpyValue] = None,
    ground_binary_dir: str = None,
    seq_run_opcodes: set[int] = None,
    import_directories: list[str] | None = None,
    expected_warnings=None,
    main_file_dir: str | None = None,
):
    if USE_WASM:
        if fprime_test_api is not None:
            wasm = compile_seq_wasm(
                seq,
                ground_binary_dir=ground_binary_dir,
                import_directories=import_directories,
                expected_warnings=expected_warnings,
                main_file_dir=main_file_dir,
            )
            wasm_path = _write_wasm_to_tmpfile(wasm)
            fprime_test_api.send_and_assert_command(
                "Ref.wasmSeq.RUN", [wasm_path, "BLOCK"], timeout=timeout_s
            )
            return
        code = run_seq_wasm(
            seq,
            ground_binary_dir=ground_binary_dir,
            import_directories=import_directories,
            expected_warnings=expected_warnings,
            main_file_dir=main_file_dir,
            failing_opcodes=failing_opcodes,
        )
        if code != DirectiveErrorCode.NO_ERROR.value:
            raise RuntimeError(f"wasm sequence returned error code {code}")
        return
    _, directives, arg_name_types = compile_seq(
        seq,
        ground_binary_dir=ground_binary_dir,
        import_directories=import_directories,
        expected_warnings=expected_warnings,
        main_file_dir=main_file_dir,
    )
    args_bytes = None
    if args is not None:
        args_bytes = b"".join(v.serialize() for v in args)
    if seq_run_opcodes is None and ground_binary_dir is not None:
        d = load_dictionary(default_dictionary)
        seq_run_opcodes = {d["cmd_name_dict"]["Ref.seqDisp.RUN_ARGS"].opcode}
    run_seq(
        fprime_test_api,
        directives,
        tlm,
        time_base,
        time_context,
        initial_time_us,
        timeout_s,
        failing_opcodes,
        args=args_bytes,
        arg_name_types=arg_name_types,
        seq_run_opcodes=seq_run_opcodes,
        ground_binary_dir=ground_binary_dir,
    )


def assert_compile_failure(
    fprime_test_api,
    seq: str,
    match: str = None,
    ground_binary_dir: str = None,
    import_directories: list[str] | None = None,
    ignored_warnings=None,
    error_warnings=None,
    expected_warnings=None,
    main_file_dir: str | None = None,
):
    try:
        if USE_WASM:
            compile_seq_wasm(
                seq,
                ground_binary_dir=ground_binary_dir,
                import_directories=import_directories,
                ignored_warnings=ignored_warnings,
                error_warnings=error_warnings,
                expected_warnings=expected_warnings,
                main_file_dir=main_file_dir,
            )
        else:
            compile_seq(
                seq,
                ground_binary_dir=ground_binary_dir,
                import_directories=import_directories,
                ignored_warnings=ignored_warnings,
                error_warnings=error_warnings,
                expected_warnings=expected_warnings,
                main_file_dir=main_file_dir,
            )
    except (SystemExit, CompilationFailed) as e:
        if match is not None:
            import re

            assert re.search(match, str(e)), f"Expected match {match!r} in {e!r}"
        return

    # no error was generated
    raise RuntimeError("compile_seq succeeded")


def assert_run_failure(
    fprime_test_api,
    seq: str,
    error_code: DirectiveErrorCode | int = None,
    validation_error: bool = False,
    initial_time_us: int = 0,
    failing_opcodes: set[int] = None,
    args: list[FpyValue] = None,
    ground_binary_dir: str = None,
    seq_run_opcodes: set[int] = None,
    import_directories: list[str] | None = None,
):
    assert not (
        error_code is not None and validation_error
    ), "Cannot specify both error_code and validation_error"
    assert (
        error_code is not None or validation_error
    ), "Must specify either error_code or validation_error"

    if USE_WASM:
        if fprime_test_api is not None:
            # GDS mode: send the wasm module and assert that it fails via
            # OpCodeError event, mirroring the bytecode GDS failure path.
            wasm = compile_seq_wasm(
                seq,
                ground_binary_dir=ground_binary_dir,
                import_directories=import_directories,
            )
            wasm_path = _write_wasm_to_tmpfile(wasm)
            fprime_test_api.send_and_assert_event(
                "Ref.wasmSeq.RUN",
                [wasm_path, "BLOCK"],
                events="CdhCore.cmdDisp.OpCodeError",
                timeout=4,
            )
            return
        # The wasm backend has no separate validation step or VM-internal
        # faults: a failed sequence is one that reports a nonzero code
        # through the exit/fault host imports.
        code = run_seq_wasm(
            seq,
            ground_binary_dir=ground_binary_dir,
            import_directories=import_directories,
            failing_opcodes=failing_opcodes,
        )
        if code == DirectiveErrorCode.NO_ERROR.value:
            raise RuntimeError("wasm sequence succeeded")
        if error_code is not None:
            if (
                isinstance(error_code, DirectiveErrorCode) and code != error_code.value
            ) or (isinstance(error_code, int) and code != error_code):
                raise RuntimeError(
                    f"wasm sequence returned {code}, expected {error_code}"
                )
        return

    _, directives, arg_name_types = compile_seq(
        seq, ground_binary_dir=ground_binary_dir, import_directories=import_directories
    )
    args_bytes = None
    if args is not None:
        args_bytes = b"".join(v.serialize() for v in args)
    if seq_run_opcodes is None and ground_binary_dir is not None:
        d = load_dictionary(default_dictionary)
        seq_run_opcodes = {d["cmd_name_dict"]["Ref.seqDisp.RUN_ARGS"].opcode}

    if fprime_test_api is not None:
        # GDS mode: send the sequence and assert that it fails via OpCodeError event
        seq_path = _write_seq_to_tmpfile(directives, arg_name_types)
        if args_bytes:
            seq_args = _build_seq_args_json(args_bytes)
            fprime_test_api.send_and_assert_event(
                "Ref.seqDisp.RUN_ARGS",
                [seq_path, "BLOCK", seq_args],
                events="CdhCore.cmdDisp.OpCodeError",
                timeout=4,
            )
        else:
            fprime_test_api.send_and_assert_event(
                "Ref.seqDisp.RUN",
                [seq_path, "BLOCK"],
                events="CdhCore.cmdDisp.OpCodeError",
                timeout=4,
            )
        return

    try:
        run_seq(
            fprime_test_api,
            directives,
            initial_time_us=initial_time_us,
            failing_opcodes=failing_opcodes,
            args=args_bytes,
            arg_name_types=arg_name_types,
            seq_run_opcodes=seq_run_opcodes,
            ground_binary_dir=ground_binary_dir,
        )
    except ValidationError as e:
        if not validation_error:
            raise
        print(e)
        return
    except RuntimeError as e:
        if validation_error:
            raise RuntimeError("Expected ValidationError, got", type(e).__name__, e)

        # The failure surfaces as either a DirectiveErrorCode trap or a raw exit
        # code int; the expected value may likewise be either. Compare by integer
        # value so e.g. an exit code of 7 matches DirectiveErrorCode.EXIT_WITH_ERROR.
        def _as_int(v):
            return v.value if isinstance(v, DirectiveErrorCode) else v

        if len(e.args) == 1 and _as_int(e.args[0]) != _as_int(error_code):
            raise RuntimeError(
                "run_seq failed with error", e.args[0], "expected", error_code
            )
        print(e)
        return

    raise RuntimeError("run_seq succeeded")
