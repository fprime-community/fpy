from pathlib import Path
import tempfile
import fpy.error
from fpy.model import DirectiveErrorCode, FpySequencerModel, ValidationError
from fpy.bytecode.directives import AllocateDirective, Directive, GotoDirective, PushValDirective
from fpy.compiler import text_to_ast, ast_to_directives
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


def compile_seq(fprime_test_api, seq: str, ground_binary_dir: str = None, flight_binary_dir: str = None) -> tuple[list[Directive], list[tuple[str, FpyType]]]:
    """Compile a sequence string to a list of directives and arg types."""
    fpy.error.file_name = "<test>"
    
    body = text_to_ast(seq)
    if body is None:
        # This shouldn't happen - text_to_ast calls exit(1) on parse errors
        raise CompilationFailed("Parsing failed")
    
    result = ast_to_directives(body, default_dictionary, ground_binary_dir=ground_binary_dir, flight_binary_dir=flight_binary_dir)
    if isinstance(result, (fpy.error.CompileError, fpy.error.BackendError)):
        raise CompilationFailed(f"Compilation failed:\n{result}")
    
    directives, arg_types = result
    return directives, arg_types


def lookup_type(fprime_test_api, type_name: str):
    d = load_dictionary(default_dictionary)
    return d["type_defs"][type_name]


def _write_seq_to_tmpfile(directives: list[Directive], arg_types: list[tuple[str, FpyType]] = None) -> str:
    """Serialize directives to a temp .bin file and return its path."""
    arg_specs = [(name, t.name, t.max_size) for name, t in (arg_types or [])]
    seq_file = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    Path(seq_file.name).write_bytes(serialize_directives(directives, arg_specs=arg_specs)[0])
    return seq_file.name


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
        fprime_test_api.send_and_assert_command("Ref.seqDisp.RUN", [seq_path, "WAIT"], timeout=timeout_s)
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
        cmd_name_dict["Ref.seqDisp.RUN"].opcode,
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
    flight_binary_dir: str = None,
    seq_run_opcodes: set[int] = None,
):
    directives, arg_name_types = compile_seq(fprime_test_api, seq, ground_binary_dir=ground_binary_dir, flight_binary_dir=flight_binary_dir)
    arg_types = [t for _, t in arg_name_types]
    args_bytes = None
    if args is not None:
        args_bytes = b"".join(v.serialize() for v in args)
    if seq_run_opcodes is None and ground_binary_dir is not None:
        d = load_dictionary(default_dictionary)
        seq_run_opcodes = {d["cmd_name_dict"]["Ref.seqDisp.RUN_ARGS"].opcode}
    run_seq(fprime_test_api, directives, tlm, time_base, time_context, initial_time_us, timeout_s, failing_opcodes, args=args_bytes, arg_types=arg_types, arg_name_types=arg_name_types, seq_run_opcodes=seq_run_opcodes, ground_binary_dir=ground_binary_dir)


def assert_compile_failure(fprime_test_api, seq: str, match: str = None, ground_binary_dir: str = None, flight_binary_dir: str = None):
    try:
        compile_seq(fprime_test_api, seq, ground_binary_dir=ground_binary_dir, flight_binary_dir=flight_binary_dir)
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
    flight_binary_dir: str = None,
    seq_run_opcodes: set[int] = None,
):
    assert not (error_code is not None and validation_error), \
        "Cannot specify both error_code and validation_error"
    assert error_code is not None or validation_error, \
        "Must specify either error_code or validation_error"

    directives, arg_name_types = compile_seq(fprime_test_api, seq, ground_binary_dir=ground_binary_dir, flight_binary_dir=flight_binary_dir)
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
        fprime_test_api.send_and_assert_event(
            "Ref.seqDisp.RUN",
            [seq_path, "WAIT"],
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
