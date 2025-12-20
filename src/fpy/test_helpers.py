from pathlib import Path
import tempfile
import traceback
import fpy.error
from fpy.model import DirectiveErrorCode, FpySequencerModel
from fpy.bytecode.directives import AllocateDirective, Directive
from fpy.compiler import text_to_ast, ast_to_directives
from fprime_gds.common.loaders.ch_json_loader import ChJsonLoader
from fprime_gds.common.loaders.cmd_json_loader import CmdJsonLoader
from fprime_gds.common.loaders.event_json_loader import EventJsonLoader
from fprime_gds.common.loaders.prm_json_loader import PrmJsonLoader
from fprime_gds.common.testing_fw.api import IntegrationTestAPI


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
    dictionary = default_dictionary  # fprime_test_api.pipeline.dictionary_path
    cmd_json_dict_loader = CmdJsonLoader(dictionary)
    (cmd_id_dict, cmd_name_dict, versions) = cmd_json_dict_loader.construct_dicts(
        dictionary
    )

    ch_json_dict_loader = ChJsonLoader(dictionary)
    (ch_id_dict, ch_name_dict, versions) = ch_json_dict_loader.construct_dicts(
        dictionary
    )
    prm_json_dict_loader = PrmJsonLoader(dictionary)
    (prm_id_dict, prm_name_dict, versions) = prm_json_dict_loader.construct_dicts(
        dictionary
    )
    event_json_dict_loader = EventJsonLoader(dictionary)
    (event_id_dict, event_name_dict, versions) = event_json_dict_loader.construct_dicts(
        dictionary
    )
    type_name_dict = cmd_json_dict_loader.parsed_types
    type_name_dict.update(ch_json_dict_loader.parsed_types)
    type_name_dict.update(prm_json_dict_loader.parsed_types)
    type_name_dict.update(event_json_dict_loader.parsed_types)

    return type_name_dict[type_name]


def run_seq(
    fprime_test_api: IntegrationTestAPI,
    directives: list[Directive],
    tlm: dict[str, bytes] = None,
):
    """Run a list of directives using the sequencer model."""
    if tlm is None:
        tlm = {}

    # seq_file = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    # Path(seq_file.name).write_bytes(serialize_directives(directives)[0])
    # fprime_test_api.send_and_assert_command("Ref.cmdSeq.RUN", [seq_file.name, "BLOCK"], timeout=4)
    # return

    dictionary = default_dictionary

    ch_json_dict_loader = ChJsonLoader(dictionary)
    (ch_id_dict, ch_name_dict, versions) = ch_json_dict_loader.construct_dicts(
        dictionary
    )
    cmd_json_dict_loader = CmdJsonLoader(dictionary)
    (cmd_id_dict, cmd_name_dict, versions) = cmd_json_dict_loader.construct_dicts(
        dictionary
    )
    model = FpySequencerModel(cmd_dict=cmd_id_dict)
    tlm_db = {}
    for chan_name, val in tlm.items():
        ch_template = ch_name_dict[chan_name]
        tlm_db[ch_template.get_id()] = val
    ret = model.run(directives, tlm_db)
    if ret != DirectiveErrorCode.NO_ERROR:
        raise RuntimeError("Sequence returned", ret)
    if len(directives) > 0 and isinstance(directives[0], AllocateDirective):
        # check that the start and end sizes are the same
        if len(model.stack) != directives[0].size:
            raise RuntimeError(f"Sequence leaked {len(model.stack) - directives[0].size} bytes")


def assert_compile_success(fprime_test_api, seq: str, flags: list[str] = None):
    compile_seq(fprime_test_api, seq, flags)


def assert_run_success(fprime_test_api, seq: str, tlm: dict[str, bytes] = None, flags: list[str] = None):
    directives = compile_seq(fprime_test_api, seq, flags)
    run_seq(fprime_test_api, directives, tlm)


def assert_compile_failure(fprime_test_api, seq: str, flags: list[str] = None):
    try:
        compile_seq(fprime_test_api, seq, flags)
    except (SystemExit, CompilationFailed):
        # Compilation failed as expected
        return

    # no error was generated
    raise RuntimeError("compile_seq succeeded")


def assert_run_failure(fprime_test_api, seq: str, flags: list[str] = None):
    directives = compile_seq(fprime_test_api, seq, flags)
    try:
        run_seq(fprime_test_api, directives)
    except (RuntimeError, AssertionError) as e:
        print(e)
        return

    # other exceptions we will let through, such as assertions
    raise RuntimeError("run_seq succeeded")
