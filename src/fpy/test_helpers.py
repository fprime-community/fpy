from __future__ import annotations
from pathlib import Path
import tempfile
import fpy.error
from fpy.model import DirectiveErrorCode, FpySequencerModel, ValidationError
from fpy.bytecode.directives import (
    AllocateDirective,
    Directive,
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


# Flipped to True by conftest's pytest_configure when --wasm is passed, routing
# the assert_* helpers through the LLVM/wasm backend (run via the NASA spacewasm
# interpreter, the on-board target runtime) instead of the bytecode VM.
USE_WASM = False

# Path to the built spacewasm runner harness, set by conftest's
# pytest_configure when --wasm is passed.
SPACEWASM_RUNNER: str | None = None


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

    Runs the compiled module through the NASA spacewasm interpreter (the
    on-board target runtime) via the runner harness built by conftest.

    The code alone does not say which channel raised it; use
    run_seq_wasm_outcome when an exit code and a panic code of the same value
    must not be confused."""
    _, code, _, _ = _run_seq_wasm(
        seq,
        ground_binary_dir,
        import_directories=import_directories,
        expected_warnings=expected_warnings,
        main_file_dir=main_file_dir,
        failing_opcodes=failing_opcodes,
    )
    return code


def run_seq_wasm_outcome(
    seq: str,
    ground_binary_dir: str = None,
    import_directories: list[str] | None = None,
    expected_warnings=None,
    main_file_dir: str | None = None,
    failing_opcodes: set[int] = None,
) -> tuple[str, int]:
    """Like run_seq_wasm, but returns (channel, code).

    channel is "exit" (a termination the sequence asked for: exit(), a failing
    assert, or main returning normally) or "panic" (an implicit runtime check
    fired -- zero divisor, arithmetic overflow, array index out of bounds --
    carrying a DirectiveErrorCode). The two are separate host imports, so an
    exit(10) can never be mistaken for a DOMAIN_ERROR panic."""
    channel, code, _, _ = _run_seq_wasm(
        seq,
        ground_binary_dir,
        import_directories=import_directories,
        expected_warnings=expected_warnings,
        main_file_dir=main_file_dir,
        failing_opcodes=failing_opcodes,
    )
    return channel, code


def run_seq_wasm_with_events(
    seq: str,
    ground_binary_dir: str = None,
    import_directories: list[str] | None = None,
    expected_warnings=None,
    main_file_dir: str | None = None,
) -> tuple[int, list[tuple[int, str]]]:
    """Like run_seq_wasm, but also returns the events the sequence reported
    through the event host import (the log() builtin) as (severity, message)
    pairs, in call order. Messages are Rust-escaped by the runner harness, so
    a plain ASCII message round-trips verbatim."""
    _, code, events, _ = _run_seq_wasm(
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
    _, code, _, cmds = _run_seq_wasm(
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
) -> tuple[str, int, list[tuple[int, str]], list[bytes]]:
    """Compile *seq* to wasm, run it through the spacewasm runner harness, and
    return (channel, error code, reported events, dispatched command buffers).

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
) -> tuple[str, int, list[tuple[int, str]], list[bytes]]:
    """Run an already-linked wasm module through the spacewasm runner harness
    and return (channel, error code, reported events, dispatched command
    buffers). channel is "exit" or "panic"; see run_seq_wasm_outcome.

    The commands that fail are *failing_opcodes* plus the RUN commands that
    always fail when called from within a running sequence on the same
    sequencer instance -- the same set the bytecode reference model uses."""
    import subprocess

    assert (
        SPACEWASM_RUNNER is not None
    ), "SPACEWASM_RUNNER not set; run pytest with --wasm"

    wasm_path = _write_wasm_to_tmpfile(wasm)

    d = load_dictionary(default_dictionary)
    always_failing = {d["cmd_name_dict"]["Ref.cmdSeq0.RUN"].opcode}
    argv = [SPACEWASM_RUNNER, wasm_path]
    for opcode in sorted(always_failing | set(failing_opcodes or ())):
        argv += ["--fail-opcode", str(opcode)]
    if cmd_response is not None:
        argv += ["--cmd-response", str(cmd_response)]

    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"spacewasm runner faulted (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    # The runner prints one `event <severity> <message>` line per event host
    # call and one `cmd <hex>` line per cmd host call, then `<channel> <code>`
    # as the final line.
    *host_call_lines, outcome_line = result.stdout.strip().splitlines()
    events = []
    cmds = []
    for line in host_call_lines:
        kind, rest = line.split(" ", 1)
        if kind == "event":
            severity, message = rest.split(" ", 1)
            events.append((int(severity), message))
        elif kind == "cmd":
            cmds.append(bytes.fromhex(rest))
        else:
            assert False, f"unexpected runner output line: {line!r}"
    channel, code = outcome_line.split()
    assert channel in ("exit", "panic"), result.stdout
    return channel, int(code), events, cmds


def lookup_type(fprime_test_api, type_name: str):
    d = load_dictionary(default_dictionary)
    return d["type_defs"][type_name]


def _write_wasm_to_tmpfile(wasm: bytes) -> str:
    """Write a compiled wasm module to a temp .wasm file and return its path."""
    wasm_file = tempfile.NamedTemporaryFile(suffix=".wasm", delete=False)
    wasm_file.write(wasm)
    wasm_file.close()
    return wasm_file.name


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
    arg_types: list[FpyType] = None,
    seq_run_opcodes: set[int] = None,
    arg_name_types: list[tuple[str, FpyType]] = None,
    ground_binary_dir: str = None,
):
    """Run a list of directives.

    When fprime_test_api is None (the default), runs against the Python
    sequencer model.  When fprime_test_api is a live IntegrationTestAPI
    (i.e. --use-gds was passed to pytest), serializes the directives to a
    temp file and sends them to the running GDS deployment.
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
    cmd_id_dict = d["cmd_id_dict"]
    cmd_name_dict = d["cmd_name_dict"]
    type_defs = d["type_defs"]
    # These RUN commands always fail when called from within a running sequence
    # on the same sequencer instance; mark them as failing for the model.
    always_failing = {
        cmd_name_dict["Ref.cmdSeq0.RUN"].opcode,
    }
    if failing_opcodes:
        always_failing |= failing_opcodes
    model = FpySequencerModel(
        cmd_dict=cmd_id_dict,
        time_base=time_base,
        time_context=time_context,
        initial_time_us=initial_time_us,
        failing_opcodes=always_failing,
        seq_run_opcodes=seq_run_opcodes or set(),
        arg_type_defs=type_defs,
    )
    tlm_db = {}
    for chan_name, val in tlm.items():
        ch_template = ch_name_dict[chan_name]
        tlm_db[ch_template.ch_id] = val

    import os

    old_cwd = None
    if ground_binary_dir is not None:
        old_cwd = os.getcwd()
        os.chdir(ground_binary_dir)
    try:
        error_code, trap = model.run(directives, tlm_db, args=args, arg_types=arg_types)
    finally:
        if old_cwd is not None:
            os.chdir(old_cwd)

    # A trap (VM fault) surfaces as its DirectiveErrorCode; an exit with a nonzero
    # code surfaces as the raw error code int.
    if trap != DirectiveErrorCode.NO_ERROR:
        raise RuntimeError(trap)
    if error_code != 0:
        raise RuntimeError(error_code)
    # Compute expected frame size: args + setup directives (PushVal for flags, then Allocate)
    # If functions are present, the first directive is a Goto that jumps past them;
    # skip to the goto target to find the actual setup directives.
    args_size = sum(t.max_size for t in (arg_types or []))
    setup_start = 0
    if directives and isinstance(directives[0], GotoDirective):
        setup_start = directives[0].dir_idx
    setup_size = 0
    # The frame setup is exactly: PushVal (flags default), then optionally Allocate (remaining locals).
    if setup_start < len(directives) and isinstance(
        directives[setup_start], PushValDirective
    ):
        setup_size += len(directives[setup_start].val)
        if setup_start + 1 < len(directives) and isinstance(
            directives[setup_start + 1], AllocateDirective
        ):
            setup_size += directives[setup_start + 1].size
    expected_stack = args_size + setup_size
    if expected_stack > 0 and len(model.stack) != expected_stack:
        raise RuntimeError(f"Sequence leaked {len(model.stack) - expected_stack} bytes")


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
    arg_types = [t for _, t in arg_name_types]
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
        arg_types=arg_types,
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
    timeBase: int = 0,
    timeContext: int = 0,
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

    # The expected failure's channel: a DirectiveErrorCode other than
    # EXIT_WITH_ERROR is a runtime panic (a compiler-emitted guard);
    # EXIT_WITH_ERROR (a bare assert) and raw ints (exit(n)) come through the
    # exit channel. The channels must not cross-match: exit(10) does not
    # satisfy an expected DOMAIN_ERROR even though DOMAIN_ERROR's value is 10.
    expect_panic = (
        isinstance(error_code, DirectiveErrorCode)
        and error_code != DirectiveErrorCode.EXIT_WITH_ERROR
    )
    # ...except that the bytecode ISA has no panic-raising directive, so the
    # compiler lowers ITS OWN runtime checks (array bounds, assert_cmd_success)
    # to `PushVal(code); Exit` -- on the VM these semantically-panic codes ride
    # the exit channel by construction. The wasm backend does the same for
    # CMD_FAIL. TODO(upstream): give the FpySequencer ISA a panic directive so
    # these stop being spoofable via exit().
    COMPILED_IN_CHECK_CODES = {
        DirectiveErrorCode.ARRAY_OUT_OF_BOUNDS,
        DirectiveErrorCode.CMD_FAIL,
    }

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
        # through the exit/panic host imports.
        channel, code = run_seq_wasm_outcome(
            seq,
            ground_binary_dir=ground_binary_dir,
            import_directories=import_directories,
            failing_opcodes=failing_opcodes,
        )
        if (channel, code) == ("exit", DirectiveErrorCode.NO_ERROR.value):
            raise RuntimeError("wasm sequence succeeded")
        if error_code is not None:
            want_channel = "panic" if expect_panic else "exit"
            want_code = (
                error_code.value
                if isinstance(error_code, DirectiveErrorCode)
                else error_code
            )
            # CMD_FAIL is lowered as a plain exit on this backend too (see
            # COMPILED_IN_CHECK_CODES), so accept it on either channel.
            ok = (channel, code) == (want_channel, want_code) or (
                error_code in COMPILED_IN_CHECK_CODES and code == want_code
            )
            if not ok:
                raise RuntimeError(
                    f"wasm sequence ended with {channel} {code}, "
                    f"expected {want_channel} {want_code} ({error_code})"
                )
        return

    _, directives, arg_name_types = compile_seq(
        seq, ground_binary_dir=ground_binary_dir, import_directories=import_directories
    )
    arg_types = [t for _, t in arg_name_types]
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
            time_base=timeBase,
            time_context=timeContext,
            initial_time_us=initial_time_us,
            failing_opcodes=failing_opcodes,
            args=args_bytes,
            arg_types=arg_types,
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

        # The failure's channel is encoded in the arg type: a runtime panic
        # surfaces as a DirectiveErrorCode trap, a user exit as a raw int.
        # The channels must not cross-match (see expect_panic above), except
        # for the compiled-in checks that the bytecode ISA forces through the
        # exit channel.
        if len(e.args) == 1:
            got = e.args[0]
            if expect_panic:
                ok = isinstance(got, DirectiveErrorCode) and got == error_code
                if error_code in COMPILED_IN_CHECK_CODES:
                    ok = ok or got == error_code.value
            else:
                want = (
                    error_code.value
                    if isinstance(error_code, DirectiveErrorCode)
                    else error_code
                )
                ok = not isinstance(got, DirectiveErrorCode) and got == want
            if not ok:
                raise RuntimeError(
                    "run_seq failed with error", got, "expected", error_code
                )
        print(e)
        return

    raise RuntimeError("run_seq succeeded")
