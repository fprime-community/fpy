from pathlib import Path
import tempfile
import traceback
import fpy.error
import fpy.model
from fpy.model import DirectiveErrorCode, FpySequencerModel
from fpy.bytecode.directives import AllocateDirective, Directive
from fpy.compiler import text_to_ast, ast_to_directives
from fpy.bytecode.assembler import serialize_directives
from fpy.dictionary import load_dictionary


default_dictionary = str(
    Path(__file__).parent.parent.parent
    / "test"
    / "fpy"
    / "RefTopologyDictionary.json"
)


class CompilationFailed(Exception):
    """Raised when compilation fails expectedly (parse error or semantic error)."""
    pass


def compile_seq(fprime_test_api, seq: str, flags: list[str] = None) -> list[Directive]:
    """Compile a sequence string to a list of directives in memory."""
    fpy.error.file_name = "<test>"
    
    body = text_to_ast(seq)
    if body is None:
        # This shouldn't happen - text_to_ast calls exit(1) on parse errors
        raise CompilationFailed("Parsing failed")
    
    compile_args = {}
    for flag in flags or []:
        compile_args[flag] = True
    
    directives = ast_to_directives(body, default_dictionary, compile_args)
    if isinstance(directives, (fpy.error.CompileError, fpy.error.BackendError)):
        raise CompilationFailed(f"Compilation failed:\n{directives}")
    
    return directives


def lookup_type(fprime_test_api, type_name: str):
    d = load_dictionary(default_dictionary)
    return d["type_defs"][type_name]


def run_seq(
    fprime_test_api,
    directives: list[Directive],
    tlm: dict[str, bytes] = None,
    time_base: int = 0,
    time_context: int = 0,
    initial_time_us: int = 0,
    timeout_s: int = 4,
    failing_opcodes: set[int] = None,
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
        seq_file = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        Path(seq_file.name).write_bytes(serialize_directives(directives)[0])
        fprime_test_api.send_and_assert_command("Ref.cmdSeq.RUN", [seq_file.name, "BLOCK"], timeout=timeout_s)
        return

    d = load_dictionary(default_dictionary)
    ch_name_dict = d["ch_name_dict"]
    cmd_id_dict = d["cmd_id_dict"]
    model = FpySequencerModel(
        cmd_dict=cmd_id_dict,
        time_base=time_base,
        time_context=time_context,
        initial_time_us=initial_time_us,
        failing_opcodes=failing_opcodes,
    )
    tlm_db = {}
    for chan_name, val in tlm.items():
        ch_template = ch_name_dict[chan_name]
        tlm_db[ch_template.ch_id] = val
    ret = model.run(directives, tlm_db)
    if ret != DirectiveErrorCode.NO_ERROR:
        raise RuntimeError(ret)
    if len(directives) > 0 and isinstance(directives[0], AllocateDirective):
        # check that the start and end sizes are the same
        if len(model.stack) != directives[0].size:
            raise RuntimeError(f"Sequence leaked {len(model.stack) - directives[0].size} bytes")


def assert_compile_success(fprime_test_api, seq: str, flags: list[str] = None):
    compile_seq(fprime_test_api, seq, flags)


def assert_run_success(
    fprime_test_api,
    seq: str,
    tlm: dict[str, bytes] = None,
    flags: list[str] = None,
    time_base: int = 0,
    time_context: int = 0,
    initial_time_us: int = 0,
    timeout_s: int = 4,
    failing_opcodes: set[int] = None,
):
    directives = compile_seq(fprime_test_api, seq, flags)
    run_seq(fprime_test_api, directives, tlm, time_base, time_context, initial_time_us, timeout_s, failing_opcodes)


def assert_compile_failure(fprime_test_api, seq: str, flags: list[str] = None):
    try:
        compile_seq(fprime_test_api, seq, flags)
    except (SystemExit, CompilationFailed):
        # Compilation failed as expected
        return

    # no error was generated
    raise RuntimeError("compile_seq succeeded")


def assert_run_failure(
    fprime_test_api,
    seq: str,
    error_code: DirectiveErrorCode,
    flags: list[str] = None,
    time_base: int = 0,
    time_context: int = 0,
    initial_time_us: int = 0,
    failing_opcodes: set[int] = None,
):
    directives = compile_seq(fprime_test_api, seq, flags)
    try:
        run_seq(fprime_test_api, directives, time_base=time_base, time_context=time_context, initial_time_us=initial_time_us, failing_opcodes=failing_opcodes)
    except (RuntimeError, AssertionError) as e:
        if isinstance(e, RuntimeError) and len(e.args) == 1 and e.args[0] != error_code:
            raise RuntimeError("run_seq failed with error", e.args[0], "expected", error_code)
        print(e)
        return

    # other exceptions we will let through, such as assertions
    raise RuntimeError("run_seq succeeded")
