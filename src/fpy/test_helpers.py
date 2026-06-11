from pathlib import Path
import math
import tempfile
import fpy.error
from fpy.model import DirectiveErrorCode, FpySequencerModel, ValidationError
from fpy.bytecode.directives import AllocateDirective, Directive, GotoDirective, PushValDirective
from fpy.compiler import (
    text_to_ast,
    analyze_ast,
    analysis_to_fypbc_directives,
    analysis_to_wasm,
)
from fpy.codegen_llvm import FPY_ENTRY_POINT
from fpy.state import get_base_compile_state
from fpy.bytecode.assembler import serialize_directives
from fpy.dictionary import load_dictionary
from fpy.types import FpyType, FpyValue


default_dictionary = str(
    Path(__file__).parent.parent.parent
    / "test"
    / "fpy"
    / "RefTopologyDictionary.json"
)


class CompilationFailed(Exception):
    """Raised when compilation fails expectedly (parse error or semantic error)."""
    pass


# Flipped to True by conftest's pytest_configure when --wasm is passed, routing
# the assert_* helpers through the LLVM/wasm backend (run via wasmtime) instead
# of the bytecode VM. Sequences using features the wasm backend can't lower yet
# will surface as CompilationFailed.
USE_WASM = False


def compile_seq(fprime_test_api, seq: str, ground_binary_dir: str = None) -> tuple[list[Directive], list[tuple[str, FpyType]]]:
    """Compile a sequence string to a list of directives and arg types."""
    fpy.error.file_name = "<test>"

    state = get_base_compile_state(default_dictionary, ground_binary_dir)

    try:
        body = text_to_ast(seq)
        state = analyze_ast(body, state)
        directives, arg_types = analysis_to_fypbc_directives(body, state)
    except (fpy.error.CompileError, fpy.error.BackendError) as e:
        raise CompilationFailed(f"Compilation failed:\n{e}")

    return directives, arg_types


def compile_seq_wasm(seq: str, ground_binary_dir: str = None) -> bytes:
    """Compile a sequence string to a runnable wasm binary (the LLVM backend)."""
    fpy.error.file_name = "<test>"

    state = get_base_compile_state(default_dictionary, ground_binary_dir)

    try:
        body = text_to_ast(seq)
        state = analyze_ast(body, state)
        wasm, _ = analysis_to_wasm(body, state)
    except (fpy.error.CompileError, fpy.error.BackendError) as e:
        raise CompilationFailed(f"Compilation failed:\n{e}")

    return wasm


def run_seq_wasm(seq: str, ground_binary_dir: str = None) -> int:
    """Compile *seq* to wasm and run it, returning fpy_main's error code.

    Runs in wasmtime, our interpreted wasm runtime for tests.

    Math host calls the backend emits are provided here, so any sequence runs
    without the caller wiring up imports (unused defines are ignored, so this is
    harmless for sequences that don't use them):
      * `**` lowers to llvm.pow  -> imported ``env.pow``
      * float `%` lowers to frem -> imported ``env.fmod``
    The shims mirror the bytecode VM's handlers (see model.handle_fpow /
    handle_fmod) so the two backends agree on edge cases like the 0**-1 pole.
    """
    from wasmtime import Engine, FuncType, Linker, Module, Store, ValType

    wasm = compile_seq_wasm(seq, ground_binary_dir)
    engine = Engine()
    store = Store(engine)
    module = Module(engine, wasm)

    linker = Linker(engine)
    f64 = ValType.f64()
    binary_f64 = FuncType([f64, f64], [f64])
    unary_f64 = FuncType([f64], [f64])
    linker.define_func("env", "pow", binary_f64, _host_pow)
    linker.define_func("env", "fmod", binary_f64, math.fmod)
    linker.define_func("env", "log", unary_f64, math.log)

    instance = linker.instantiate(store, module)
    entry = instance.exports(store)[FPY_ENTRY_POINT]
    return entry(store)


def _host_pow(base: float, exp: float) -> float:
    """C/IEEE pow() semantics, matching the VM's handle_fpow: a pole (0**neg)
    is +/-inf rather than an error, and domain errors yield NaN -- where Python
    would instead raise or return a complex number."""
    try:
        result = base ** exp
    except ZeroDivisionError:
        # 0**<neg> is a pole; a negative odd-integer exponent keeps the base's
        # signed zero (pow(-0.0, -1) == -inf), otherwise +inf.
        if float(exp).is_integer() and int(exp) % 2 != 0:
            return math.copysign(math.inf, base)
        return math.inf
    except (ValueError, OverflowError):
        return math.nan
    return math.nan if isinstance(result, complex) else result


def lookup_type(fprime_test_api, type_name: str):
    d = load_dictionary(default_dictionary)
    return d["type_defs"][type_name]


def _write_seq_to_tmpfile(directives: list[Directive], arg_types: list[tuple[str, FpyType]] = None) -> str:
    """Serialize directives to a temp .bin file and return its path."""
    arg_specs = [(name, t.name, t.max_size) for name, t in (arg_types or [])]
    seq_file = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    Path(seq_file.name).write_bytes(serialize_directives(directives, arg_specs=arg_specs)[0])
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
            fprime_test_api.send_and_assert_command("Ref.seqDisp.RUN_ARGS", [seq_path, "BLOCK", seq_args], timeout=timeout_s)
        else:
            fprime_test_api.send_and_assert_command("Ref.seqDisp.RUN", [seq_path, "BLOCK"], timeout=timeout_s)
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
        ret = model.run(directives, tlm_db, args=args, arg_types=arg_types)
    finally:
        if old_cwd is not None:
            os.chdir(old_cwd)

    if ret != DirectiveErrorCode.NO_ERROR:
        raise RuntimeError(ret)
    # Compute expected frame size: args + setup directives (PushVal for flags, then Allocate)
    # If functions are present, the first directive is a Goto that jumps past them;
    # skip to the goto target to find the actual setup directives.
    args_size = sum(t.max_size for t in (arg_types or []))
    setup_start = 0
    if directives and isinstance(directives[0], GotoDirective):
        setup_start = directives[0].dir_idx
    setup_size = 0
    # The frame setup is exactly: PushVal (flags default), then optionally Allocate (remaining locals).
    if setup_start < len(directives) and isinstance(directives[setup_start], PushValDirective):
        setup_size += len(directives[setup_start].val)
        if setup_start + 1 < len(directives) and isinstance(directives[setup_start + 1], AllocateDirective):
            setup_size += directives[setup_start + 1].size
    expected_stack = args_size + setup_size
    if expected_stack > 0 and len(model.stack) != expected_stack:
        raise RuntimeError(f"Sequence leaked {len(model.stack) - expected_stack} bytes")


def assert_compile_success(fprime_test_api, seq: str):
    if USE_WASM:
        compile_seq_wasm(seq)
        return
    compile_seq(fprime_test_api, seq)


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
):
    if USE_WASM:
        code = run_seq_wasm(seq, ground_binary_dir=ground_binary_dir)
        if code != DirectiveErrorCode.NO_ERROR.value:
            raise RuntimeError(f"wasm sequence returned error code {code}")
        return
    directives, arg_name_types = compile_seq(fprime_test_api, seq, ground_binary_dir=ground_binary_dir)
    arg_types = [t for _, t in arg_name_types]
    args_bytes = None
    if args is not None:
        args_bytes = b"".join(v.serialize() for v in args)
    if seq_run_opcodes is None and ground_binary_dir is not None:
        d = load_dictionary(default_dictionary)
        seq_run_opcodes = {d["cmd_name_dict"]["Ref.seqDisp.RUN_ARGS"].opcode}
    run_seq(fprime_test_api, directives, tlm, time_base, time_context, initial_time_us, timeout_s, failing_opcodes, args=args_bytes, arg_types=arg_types, arg_name_types=arg_name_types, seq_run_opcodes=seq_run_opcodes, ground_binary_dir=ground_binary_dir)


def assert_compile_failure(fprime_test_api, seq: str, match: str = None, ground_binary_dir: str = None):
    try:
        if USE_WASM:
            compile_seq_wasm(seq, ground_binary_dir=ground_binary_dir)
        else:
            compile_seq(fprime_test_api, seq, ground_binary_dir=ground_binary_dir)
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
    error_code: DirectiveErrorCode = None,
    validation_error: bool = False,
    timeBase: int = 0,
    timeContext: int = 0,
    initial_time_us: int = 0,
    failing_opcodes: set[int] = None,
    args: list[FpyValue] = None,
    ground_binary_dir: str = None,
    seq_run_opcodes: set[int] = None,
):
    assert not (error_code is not None and validation_error), \
        "Cannot specify both error_code and validation_error"
    assert error_code is not None or validation_error, \
        "Must specify either error_code or validation_error"

    if USE_WASM:
        # The wasm backend has no separate validation step or VM-internal
        # faults: a failed sequence is one whose entry point returns nonzero.
        code = run_seq_wasm(seq, ground_binary_dir=ground_binary_dir)
        if code == DirectiveErrorCode.NO_ERROR.value:
            raise RuntimeError("wasm sequence succeeded")
        if error_code is not None and code != error_code.value:
            raise RuntimeError(
                f"wasm sequence returned {code}, expected {error_code}"
            )
        return

    directives, arg_name_types = compile_seq(fprime_test_api, seq, ground_binary_dir=ground_binary_dir)
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
        run_seq(fprime_test_api, directives, time_base=timeBase, time_context=timeContext, initial_time_us=initial_time_us, failing_opcodes=failing_opcodes, args=args_bytes, arg_types=arg_types, seq_run_opcodes=seq_run_opcodes, ground_binary_dir=ground_binary_dir)
    except ValidationError as e:
        if not validation_error:
            raise
        print(e)
        return
    except RuntimeError as e:
        if validation_error:
            raise RuntimeError("Expected ValidationError, got", type(e).__name__, e)
        if len(e.args) == 1 and e.args[0] != error_code:
            raise RuntimeError("run_seq failed with error", e.args[0], "expected", error_code)
        print(e)
        return

    raise RuntimeError("run_seq succeeded")
