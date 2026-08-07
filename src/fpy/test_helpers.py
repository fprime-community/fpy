from __future__ import annotations
from pathlib import Path
import tempfile
import fpy.error
from fpy.bytecode.errors import DirectiveErrorCode, ValidationError
from fpy.model import FpySequencerModel
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

# Set by conftest's pytest_configure when --harness is passed, routing the
# bytecode run helpers at the real Svc::FpySequencer instead of the model.
HARNESS = None

# Short-lived directory the harness runs sequences from. It is deliberately
# short: the sequencer receives the path through a 40-character command string.
_HARNESS_SCRATCH: str | None = None


def _harness_scratch_dir() -> str:
    global _HARNESS_SCRATCH
    if _HARNESS_SCRATCH is None:
        _HARNESS_SCRATCH = tempfile.mkdtemp(prefix="fpyh")
    return _HARNESS_SCRATCH


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
    (reported via the exit/fault host imports; 0 when the void entrypoint
    falls off its end without failing).

    Runs the compiled module through the NASA spacewasm interpreter (the
    on-board target runtime) via the runner harness built by conftest."""
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
    pairs, in call order. Messages are Rust-escaped by the runner harness, so
    a plain ASCII message round-trips verbatim."""
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
    """Run an already-linked wasm module through the spacewasm runner harness
    and return (error code, reported events, dispatched command buffers).

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
    # call and one `cmd <hex>` line per cmd host call, then the sequence's
    # error code as the final line.
    *host_call_lines, code_line = result.stdout.strip().splitlines()
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
    return int(code_line), events, cmds


def lookup_type(type_name: str):
    d = load_dictionary(default_dictionary)
    return d["type_defs"][type_name]


def _write_wasm_to_tmpfile(wasm: bytes) -> str:
    """Write a compiled wasm module to a temp .wasm file and return its path."""
    wasm_file = tempfile.NamedTemporaryFile(suffix=".wasm", delete=False)
    wasm_file.write(wasm)
    wasm_file.close()
    return wasm_file.name


def _assert_no_stack_leak(
    final_size: int,
    directives: list[Directive],
    arg_types: list[FpyType] = None,
):
    """A finished sequence must have unwound everything it pushed.

    What is left is the frame the sequence started with: its arguments, plus
    the setup the compiler emits (a PushVal of the flags default, then an
    optional Allocate for the remaining locals). When functions are present the
    first directive is a Goto past them, so setup begins at its target."""
    args_size = sum(t.max_size for t in (arg_types or []))
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
    expected = args_size + setup_size
    if expected > 0 and final_size != expected:
        raise RuntimeError(f"Sequence leaked {final_size - expected} bytes")


def _run_seq_harness(
    directives: list[Directive],
    tlm: dict[str, bytes],
    time_base: int,
    time_context: int,
    initial_time_us: int,
    failing_opcodes: set[int],
    args: bytes,
    arg_types: list[FpyType],
    seq_run_opcodes: set[int],
    ground_binary_dir: str,
    arg_name_types: list[tuple[str, FpyType]],
):
    """Run *directives* on the real sequencer and raise what the model would."""
    d = load_dictionary(default_dictionary)
    ch_name_dict = d["ch_name_dict"]
    cmd_name_dict = d["cmd_name_dict"]

    always_failing = {cmd_name_dict["Ref.cmdSeq0.RUN"].opcode}
    if failing_opcodes:
        always_failing |= failing_opcodes

    arg_specs = [(name, t.name, t.max_size) for name, t in (arg_name_types or [])]
    seq_bytes = serialize_directives(directives, arg_specs=arg_specs)[0]

    # The sequencer receives the path through a 40-character command string, so
    # the sequence is written next to where it will run from under a short name.
    run_dir = ground_binary_dir or _harness_scratch_dir()
    seq_name = "seq.bin"
    Path(run_dir, seq_name).write_bytes(seq_bytes)

    result = HARNESS.run(
        seq_path=seq_name,
        cwd=run_dir,
        args=args,
        tlm={ch_name_dict[name].ch_id: val for name, val in tlm.items()},
        time_base=time_base,
        time_context=time_context,
        initial_time_us=initial_time_us,
        fail_opcodes=always_failing,
        seq_run_opcodes=seq_run_opcodes or set(),
    )

    if result.validation_failed:
        raise ValidationError(f"sequence failed to validate: {result.events}")
    if result.error_code != DirectiveErrorCode.NO_ERROR.value:
        # An exit carries its own code; every other failure is the directive
        # error itself.
        if (
            result.error_code == DirectiveErrorCode.EXIT_WITH_ERROR.value
            and result.exit_code is not None
        ):
            raise RuntimeError(result.exit_code)
        raise RuntimeError(DirectiveErrorCode(result.error_code))
    _assert_no_stack_leak(result.stack_size, directives, arg_types)


def run_seq(
    directives: list[Directive],
    tlm: dict[str, bytes] = None,
    time_base: int = 0,
    time_context: int = 0,
    initial_time_us: int = 0,
    failing_opcodes: set[int] = None,
    args: bytes = None,
    arg_types: list[FpyType] = None,
    seq_run_opcodes: set[int] = None,
    ground_binary_dir: str = None,
    arg_name_types: list[tuple[str, FpyType]] = None,
):
    """Run a list of directives.

    Runs on the real Svc::FpySequencer when the harness is enabled (pytest
    --harness), otherwise on the Python sequencer model."""
    if tlm is None:
        tlm = {}

    d = load_dictionary(default_dictionary)
    ch_name_dict = d["ch_name_dict"]
    cmd_id_dict = d["cmd_id_dict"]
    cmd_name_dict = d["cmd_name_dict"]
    type_defs = d["type_defs"]

    if HARNESS is not None:
        _run_seq_harness(
            directives,
            tlm=tlm,
            time_base=time_base,
            time_context=time_context,
            initial_time_us=initial_time_us,
            failing_opcodes=failing_opcodes,
            args=args,
            arg_types=arg_types,
            seq_run_opcodes=seq_run_opcodes,
            ground_binary_dir=ground_binary_dir,
            arg_name_types=arg_name_types,
        )
        return

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
    _assert_no_stack_leak(len(model.stack), directives, arg_types)


def assert_compile_success(
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
        directives,
        tlm,
        time_base,
        time_context,
        initial_time_us,
        failing_opcodes,
        args=args_bytes,
        arg_types=arg_types,
        seq_run_opcodes=seq_run_opcodes,
        ground_binary_dir=ground_binary_dir,
        arg_name_types=arg_name_types,
    )


def assert_compile_failure(
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

    if USE_WASM:
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
    arg_types = [t for _, t in arg_name_types]
    args_bytes = None
    if args is not None:
        args_bytes = b"".join(v.serialize() for v in args)
    if seq_run_opcodes is None and ground_binary_dir is not None:
        d = load_dictionary(default_dictionary)
        seq_run_opcodes = {d["cmd_name_dict"]["Ref.seqDisp.RUN_ARGS"].opcode}

    try:
        run_seq(
            directives,
            time_base=timeBase,
            time_context=timeContext,
            initial_time_us=initial_time_us,
            failing_opcodes=failing_opcodes,
            args=args_bytes,
            arg_types=arg_types,
            seq_run_opcodes=seq_run_opcodes,
            ground_binary_dir=ground_binary_dir,
            arg_name_types=arg_name_types,
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
