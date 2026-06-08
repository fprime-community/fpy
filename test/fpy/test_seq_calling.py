"""Tests for sequence calling with arguments (issue #39).

These tests compile a child sequence to a .bin file in a temp directory,
then compile+run a parent sequence that calls the child via Ref.seqDisp.RUN_ARGS.

All tests accept the ``fprime_test_api`` fixture so they can optionally run
against a live GDS deployment (``--use-gds``).  When the fixture is ``None``
(the default) the Python sequencer model is used instead.
"""
import json
import tempfile
from pathlib import Path

import pytest

import fpy.error
from fpy.bytecode.assembler import serialize_directives
from fpy.model import DirectiveErrorCode
from fpy.compiler import text_to_ast, analysis_to_fypbc_directives
from fpy.state import _build_global_scopes
from fpy.dictionary import load_dictionary
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_failure,
    assert_run_success,
    compile_seq,
    default_dictionary,
)


def _compile_to_bin(seq_text: str, out_path: Path, ground_binary_dir: str = None):
    """Compile a sequence string and write the binary to *out_path*.

    Returns (directives, arg_types) for the compiled sequence.
    """
    fpy.error.file_name = "<test-child>"
    body = text_to_ast(seq_text)
    assert body is not None, "Failed to parse child sequence"
    result = analysis_to_fypbc_directives(body, default_dictionary, ground_binary_dir=ground_binary_dir)
    assert not isinstance(result, (fpy.error.CompileError, fpy.error.BackendError)), (
        f"Compilation failed:\n{result}"
    )
    directives, arg_types = result
    arg_specs = [(name, t.name, t.max_size) for name, t in arg_types]
    data, _ = serialize_directives(directives, arg_specs=arg_specs)
    out_path.write_bytes(data)
    return directives, arg_types


class TestSeqRunDetection:
    """Test that the compiler correctly detects seq-run commands."""

    def test_run_args_detected_as_seq_run(self, fprime_test_api):
        """Ref.seqDisp.RUN_ARGS should be detected as a seq-run CommandSymbol."""
        from fpy.compiler import _build_global_scopes
        from fpy.state import CommandSymbol

        _build_global_scopes.cache_clear()
        load_dictionary.cache_clear()
        _, callable_scope, _, _ = _build_global_scopes(default_dictionary)

        # Navigate to Ref.seqDisp.RUN_ARGS
        sym = callable_scope["Ref"]["seqDisp"]["RUN_ARGS"]
        assert isinstance(sym, CommandSymbol)
        assert sym.is_seq_run_with_args
        # Fixed args should be (fileName, block) only
        assert len(sym.args) == 2
        assert sym.args[0][0] == "fileName"
        assert sym.args[1][0] == "block"

    def test_regular_run_not_seq_run(self, fprime_test_api):
        """Ref.seqDisp.RUN should NOT be detected as a seq-run command."""
        from fpy.compiler import _build_global_scopes
        from fpy.state import CommandSymbol

        _build_global_scopes.cache_clear()
        load_dictionary.cache_clear()
        _, callable_scope, _, _ = _build_global_scopes(default_dictionary)

        sym = callable_scope["Ref"]["seqDisp"]["RUN"]
        assert isinstance(sym, CommandSymbol)
        assert not sym.is_seq_run_with_args


class TestSeqCallingNoArgs:
    """Test calling a child sequence that takes no arguments."""

    def test_call_child_no_args(self, fprime_test_api):
        """Parent calls a child sequence with no arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child_path = str(Path(tmpdir).resolve() / "child.bin")
            child_seq = """\
CdhCore.cmdDisp.CMD_NO_OP()
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)


class TestSeqCallingWithArgs:
    """Test calling child sequences with various argument types."""

    def test_call_child_one_u32_arg(self, fprime_test_api):
        """Parent calls child with a single U32 argument; child asserts value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child_path = str(Path(tmpdir).resolve() / "child.bin")
            child_seq = """\
sequence(x: U32)
assert x == 42
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, 42)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_call_child_multiple_args(self, fprime_test_api):
        """Parent calls child with multiple arguments; child asserts values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child_path = str(Path(tmpdir).resolve() / "child.bin")
            child_seq = """\
sequence(x: U32, y: U8)
assert x == 100
assert y == 7
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, 100, 7)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_call_child_with_variable_args(self, fprime_test_api):
        """Parent passes variables as arguments to child; child asserts value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child_path = str(Path(tmpdir).resolve() / "child.bin")
            child_seq = """\
sequence(val: U32)
assert val == 99
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
my_val: U32 = 99
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, my_val)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_call_child_with_expression_arg(self, fprime_test_api):
        """Parent passes an arithmetic expression; child asserts the result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child_path = str(Path(tmpdir).resolve() / "child.bin")
            child_seq = """\
sequence(val: U32)
assert val == 30
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, 10 + 20)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_child_uses_arg_in_arithmetic(self, fprime_test_api):
        """Child sequence uses the arg in arithmetic and asserts the result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child_path = str(Path(tmpdir).resolve() / "child.bin")
            child_seq = """\
sequence(x: U32)
result: U32 = U32(x + 8)
assert result == 50
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, 42)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_call_child_u8_arg(self, fprime_test_api):
        """Parent passes a U8 argument; child asserts value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child_path = str(Path(tmpdir).resolve() / "child.bin")
            child_seq = """\
sequence(b: U8)
assert b == 255
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, 255)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_call_child_f32_arg(self, fprime_test_api):
        """Parent passes an F32 argument; child asserts approximate value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child_path = str(Path(tmpdir).resolve() / "child.bin")
            child_seq = """\
sequence(f: F32)
assert f > 3.13
assert f < 3.15
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, 3.14)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_wrong_value_causes_failure(self, fprime_test_api):
        """Child assert fails when the wrong value is passed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child_path = str(Path(tmpdir).resolve() / "child.bin")
            child_seq = """\
sequence(x: U32)
assert x == 999
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, 1)
"""
            assert_run_failure(fprime_test_api, parent_seq, error_code=DirectiveErrorCode.CMD_FAIL, ground_binary_dir=tmpdir)


class TestSeqCallingErrors:
    """Test error handling for malformed sequence calls."""

    def test_wrong_arg_count(self, fprime_test_api):
        """Providing wrong number of varargs should fail at compile time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child_path = str(Path(tmpdir).resolve() / "child.bin")
            child_seq = """\
sequence(x: U32, y: U32)
CdhCore.cmdDisp.CMD_NO_OP()
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, 42)
"""
            assert_compile_failure(fprime_test_api, parent_seq, match="Missing required argument 'y'", ground_binary_dir=tmpdir)

    def test_wrong_arg_type(self, fprime_test_api):
        """Providing incompatible vararg types should fail at compile time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child_path = str(Path(tmpdir).resolve() / "child.bin")
            child_seq = """\
sequence(x: U32)
CdhCore.cmdDisp.CMD_NO_OP()
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, true)
"""
            assert_compile_failure(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_missing_bin_file(self, fprime_test_api):
        """Calling a nonexistent .bin file should fail at compile time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = str(Path(tmpdir).resolve() / "nonexistent.bin")
            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{fake_path}", Fw.Wait.WAIT)
"""
            assert_compile_failure(fprime_test_api, parent_seq, match="not found", ground_binary_dir=tmpdir)

    def test_no_binary_dir(self, fprime_test_api):
        """Calling without binary_dir should fail at compile time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            child_path = str(Path(tmpdir).resolve() / "child.bin")
            child_seq = """\
CdhCore.cmdDisp.CMD_NO_OP()
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT)
"""
            # No binary_dir passed
            assert_compile_failure(fprime_test_api, parent_seq, match="binary directory")


class TestSeqCallingNested:
    """Test nested sequence execution: parent -> child -> grandchild."""

    def test_nested_two_levels(self, fprime_test_api):
        """Parent calls child which calls grandchild; all args verified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            # Grandchild: takes a U32 arg and asserts its value
            grandchild_path = str(Path(tmpdir) / "grandchild.bin")
            grandchild_seq = """\
sequence(gc_val: U32)
assert gc_val == 7
"""
            _compile_to_bin(grandchild_seq, Path(grandchild_path))

            # Child: takes a U32 arg, asserts it, calls grandchild with a different value
            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = f"""\
sequence(x: U32)
assert x == 42
Ref.seqDisp.RUN_ARGS("{grandchild_path}", Fw.Wait.WAIT, 7)
"""
            _compile_to_bin(child_seq, Path(child_path), ground_binary_dir=tmpdir)

            # Parent: calls child with an arg
            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, 42)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_nested_pass_through_arg(self, fprime_test_api):
        """Parent passes a value through two levels of sequence calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            grandchild_path = str(Path(tmpdir) / "grandchild.bin")
            grandchild_seq = """\
sequence(val: U32)
assert val == 123
"""
            _compile_to_bin(grandchild_seq, Path(grandchild_path))

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = f"""\
sequence(val: U32)
assert val == 123
Ref.seqDisp.RUN_ARGS("{grandchild_path}", Fw.Wait.WAIT, val)
"""
            _compile_to_bin(child_seq, Path(child_path), ground_binary_dir=tmpdir)

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, 123)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)


class TestSeqCallingReturnStatus:
    """Test branching on the return status of a seq-run command."""

    def test_branch_on_success(self, fprime_test_api):
        """Parent branches on OK response from a successful child."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
CdhCore.cmdDisp.CMD_NO_OP()
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
resp: Fw.CmdResponse = Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT)
if resp == Fw.CmdResponse.OK:
    exit(0)
exit(1)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_branch_on_child_failure(self, fprime_test_api):
        """Parent detects EXECUTION_ERROR when child asserts false."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
assert 1 == 0
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
resp: Fw.CmdResponse = Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT)
if resp == Fw.CmdResponse.EXECUTION_ERROR:
    exit(0)
exit(1)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_branch_on_success_with_args(self, fprime_test_api):
        """Parent branches on OK response from a child that receives args."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
sequence(x: U32)
assert x == 42
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
resp: Fw.CmdResponse = Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, 42)
if resp == Fw.CmdResponse.OK:
    exit(0)
exit(1)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_branch_on_failure_wrong_arg(self, fprime_test_api):
        """Parent detects failure when child gets wrong arg value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
sequence(x: U32)
assert x == 999
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
resp: Fw.CmdResponse = Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, 1)
if resp == Fw.CmdResponse.EXECUTION_ERROR:
    exit(0)
exit(1)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)


class TestSeqCallingNamedArgs:
    """Test calling child sequences with named arguments."""

    def test_single_named_arg(self, fprime_test_api):
        """Parent passes a single named argument to child sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
sequence(x: U32)
assert x == 42
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, x=42)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_multiple_named_args(self, fprime_test_api):
        """Parent passes multiple named args to child sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
sequence(a: U32, b: U8)
assert a == 100
assert b == 7
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, a=100, b=7)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_named_args_reordered(self, fprime_test_api):
        """Named args passed in different order than declared should be reordered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
sequence(first: U32, second: U32)
assert first == 1
assert second == 2
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, second=2, first=1)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_mixed_positional_and_named(self, fprime_test_api):
        """First arg positional, second arg named."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
sequence(a: U32, b: U32, c: U32)
assert a == 10
assert b == 20
assert c == 30
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, 10, c=30, b=20)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_named_arg_with_variable(self, fprime_test_api):
        """Pass a variable by name to child sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
sequence(val: U32)
assert val == 55
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
val: U32 = 55
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, val=val)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_named_arg_with_expression(self, fprime_test_api):
        """Pass an expression by name to child sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
sequence(result: U32)
assert result == 30
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, result=10 + 20)
"""
            assert_run_success(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)


class TestSeqCallingNamedArgErrors:
    """Test error handling for named varargs in sequence calls."""

    def test_unknown_named_arg(self, fprime_test_api):
        """Named arg that doesn't match any child parameter should fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
sequence(x: U32)
CdhCore.cmdDisp.CMD_NO_OP()
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, z=42)
"""
            assert_compile_failure(fprime_test_api, parent_seq, match="Unknown argument 'z'", ground_binary_dir=tmpdir)

    def test_duplicate_named_arg(self, fprime_test_api):
        """Same named arg specified twice should fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
sequence(x: U32)
CdhCore.cmdDisp.CMD_NO_OP()
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, x=1, x=2)
"""
            assert_compile_failure(fprime_test_api, parent_seq, match="specified multiple times", ground_binary_dir=tmpdir)

    def test_positional_and_named_conflict(self, fprime_test_api):
        """Same arg specified both by position and by name should fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
sequence(x: U32, y: U32)
CdhCore.cmdDisp.CMD_NO_OP()
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, 42, x=99)
"""
            assert_compile_failure(fprime_test_api, parent_seq, match="specified multiple times", ground_binary_dir=tmpdir)

    def test_missing_named_arg(self, fprime_test_api):
        """Missing a required arg when using named args should fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())

            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
sequence(x: U32, y: U32)
CdhCore.cmdDisp.CMD_NO_OP()
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, x=42)
"""
            assert_compile_failure(fprime_test_api, parent_seq, match="Missing required argument 'y'", ground_binary_dir=tmpdir)


class TestSeqArgLimits:
    """Test compile-time limits on sequence argument names and counts."""

    def test_arg_name_too_long(self, fprime_test_api):
        """Arg name exceeding 255 UTF-8 bytes should be a compile error."""
        long_name = "a" * 256
        seq = f"""\
sequence({long_name}: U32)
CdhCore.cmdDisp.CMD_NO_OP()
"""
        assert_compile_failure(fprime_test_api, seq, match="too long")

    def test_arg_name_exactly_255_bytes(self, fprime_test_api):
        """Arg name of exactly 255 bytes should compile fine."""
        name_255 = "a" * 255
        seq = f"""\
sequence({name_255}: U32)
CdhCore.cmdDisp.CMD_NO_OP()
"""
        compile_seq(fprime_test_api, seq)


def _dict_with_seq_args_buffer_size(buffer_size: int, tmpdir: Path) -> str:
    """Copy the default dictionary into *tmpdir* with Svc.SeqArgs buffer
    resized to *buffer_size*.  Returns the path to the new dict."""
    with open(default_dictionary, "r") as f:
        data = json.load(f)
    for type_def in data["typeDefinitions"]:
        if type_def.get("qualifiedName") == "Svc.SeqArgs":
            type_def["members"]["buffer"]["size"] = buffer_size
            type_def["default"]["buffer"] = [0] * buffer_size
            break
    else:
        raise AssertionError("Svc.SeqArgs not found in default dictionary")
    out_path = tmpdir / "ResizedDict.json"
    out_path.write_text(json.dumps(data))
    return str(out_path)


class TestSeqArgsBufferSizeFromDictionary:
    """The Svc.SeqArgs buffer length must be taken from the dictionary so that
    deployments with a non-default capacity compile correctly (issue: the
    compiler previously hardcoded 255 and crashed on any other size).
    """

    @pytest.fixture(autouse=True)
    def _restore_seq_args(self):
        """Reset the SEQ_ARGS singleton back to the default-dictionary state
        after each test, since these tests mutate a process-global."""
        yield
        _build_global_scopes.cache_clear()
        load_dictionary.cache_clear()
        _build_global_scopes(default_dictionary)

    def test_non_default_buffer_size_loads_cleanly(self):
        """A dictionary with a 1024-byte SeqArgs buffer must compile without
        crashing — this is the regression for the hardcoded-255 bug."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dict_path = _dict_with_seq_args_buffer_size(1024, Path(tmpdir))
            _build_global_scopes.cache_clear()
            load_dictionary.cache_clear()
            _, _, _, type_defs = _build_global_scopes(dict_path)
            seq_args = type_defs["Svc.SeqArgs"]
            assert seq_args.members[1].type.length == 1024
            assert seq_args.members[1].type.name == "Array_U8_1024"

    def test_oversized_args_use_dictionary_capacity(self, fprime_test_api):
        """A sequence whose args exceed 255 bytes but fit in a 1024-byte
        buffer should compile cleanly under a dictionary that defines the
        larger capacity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dict_path = _dict_with_seq_args_buffer_size(1024, Path(tmpdir))
            _build_global_scopes.cache_clear()
            load_dictionary.cache_clear()

            # 40 U64s = 320 bytes — overflows 255 but fits in 1024.
            params = ", ".join(f"x{i}: U64" for i in range(40))
            child_seq = f"sequence({params})\nCdhCore.cmdDisp.CMD_NO_OP()\n"
            body = text_to_ast(child_seq)
            assert body is not None
            result = analysis_to_fypbc_directives(body, dict_path)
            assert not isinstance(
                result, (fpy.error.CompileError, fpy.error.BackendError)
            ), f"Compilation failed:\n{result}"

    def test_args_still_bounded_by_dictionary_capacity(self, fprime_test_api):
        """Args larger than the dictionary's buffer must still be rejected,
        and the diagnostic should report the dictionary capacity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dict_path = _dict_with_seq_args_buffer_size(64, Path(tmpdir))
            _build_global_scopes.cache_clear()
            load_dictionary.cache_clear()

            child_path = str(Path(tmpdir).resolve() / "child.bin")
            # 10 U64s = 80 bytes — overflows the 64-byte buffer.
            params = ", ".join(f"x{i}: U64" for i in range(10))
            child_seq = f"sequence({params})\nCdhCore.cmdDisp.CMD_NO_OP()\n"
            body = text_to_ast(child_seq)
            assert body is not None
            result = analysis_to_fypbc_directives(body, dict_path)
            assert not isinstance(
                result, (fpy.error.CompileError, fpy.error.BackendError)
            )
            directives, arg_types = result
            arg_specs = [(name, t.name, t.max_size) for name, t in arg_types]
            data, _ = serialize_directives(directives, arg_specs=arg_specs)
            Path(child_path).write_bytes(data)

            args = ", ".join("0" for _ in range(10))
            parent_seq = f'Ref.seqDisp.RUN_ARGS("{child_path}", Fw.Wait.WAIT, {args})\n'
            body = text_to_ast(parent_seq)
            assert body is not None
            result = analysis_to_fypbc_directives(body, dict_path, ground_binary_dir=tmpdir)
            assert isinstance(result, fpy.error.CompileError), (
                f"Expected CompileError, got {type(result).__name__}: {result}"
            )
            assert "exceed" in str(result) and "64 bytes" in str(result), (
                f"Diagnostic should mention the 64-byte capacity, got: {result}"
            )

