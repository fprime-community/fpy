"""Tests for sequence calling with arguments (issue #39).

These tests compile a child sequence to a .bin file in a temp directory,
then compile+run a parent sequence that calls the child via Ref.seqDisp.RUN_ARGS.

All tests accept the ``fprime_test_api`` fixture so they can optionally run
against a live GDS deployment (``--use-gds``).  When the fixture is ``None``
(the default) the Python sequencer model is used instead.
"""
import tempfile
from pathlib import Path

import pytest

import fpy.error
from fpy.bytecode.assembler import serialize_directives
from fpy.compiler import text_to_ast, ast_to_directives, _build_global_scopes
from fpy.dictionary import load_dictionary
from fpy.test_helpers import (
    CompilationFailed,
    compile_seq,
    default_dictionary,
    run_seq,
)
from fpy.types import FpyType, FpyValue, U32, U8, F32


def _compile_to_bin(seq_text: str, out_path: Path, ground_binary_dir: str = None):
    """Compile a sequence string and write the binary to *out_path*.

    Returns (directives, arg_types) for the compiled sequence.
    """
    fpy.error.file_name = "<test-child>"
    body = text_to_ast(seq_text)
    assert body is not None, "Failed to parse child sequence"
    result = ast_to_directives(body, default_dictionary, ground_binary_dir=ground_binary_dir)
    assert not isinstance(result, (fpy.error.CompileError, fpy.error.BackendError)), (
        f"Compilation failed:\n{result}"
    )
    directives, arg_types = result
    arg_specs = [(name, t.name, t.max_size) for name, t in arg_types]
    data, _ = serialize_directives(directives, arg_specs=arg_specs)
    out_path.write_bytes(data)
    return directives, arg_types


def _get_seq_run_opcode() -> int:
    d = load_dictionary(default_dictionary)
    return d["cmd_name_dict"]["Ref.seqDisp.RUN_ARGS"].opcode


def _compile_and_run_parent(
    fprime_test_api,
    seq_text: str,
    ground_binary_dir: str,
    seq_run_opcodes: set[int] = None,
    failing_opcodes: set[int] = None,
    flight_binary_dir: str = None,
    timeout_s: int = 8,
):
    """Compile and run a parent sequence that may call child sequences."""
    fpy.error.file_name = "<test-parent>"
    directives, arg_types = compile_seq(
        fprime_test_api, seq_text,
        ground_binary_dir=ground_binary_dir,
        flight_binary_dir=flight_binary_dir,
    )
    if seq_run_opcodes is None:
        seq_run_opcodes = {_get_seq_run_opcode()}

    if fprime_test_api is not None:
        # GDS path: serialize parent to .bin and send via Ref.seqDisp.RUN
        arg_specs = [(name, t.name, t.max_size) for name, t in arg_types]
        data, _ = serialize_directives(directives, arg_specs=arg_specs)
        parent_bin = Path(ground_binary_dir) / "_test_parent.bin"
        parent_bin.write_bytes(data)
        fprime_test_api.send_and_assert_command(
            "Ref.seqDisp.RUN", [str(parent_bin), "WAIT"], timeout=timeout_s,
        )
        return

    # Run from ground_binary_dir so relative paths resolve correctly,
    # mimicking the real sequencer resolving paths relative to its cwd
    import os
    old_cwd = os.getcwd()
    os.chdir(ground_binary_dir)
    try:
        run_seq(
            fprime_test_api,
            directives,
            seq_run_opcodes=seq_run_opcodes,
            failing_opcodes=failing_opcodes,
        )
    finally:
        os.chdir(old_cwd)


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
        assert sym.is_seq_run
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
        assert not sym.is_seq_run


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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)


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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            with pytest.raises(RuntimeError):
                _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)


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
            with pytest.raises(CompilationFailed, match="Missing sequence argument 'y'"):
                compile_seq(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            with pytest.raises(CompilationFailed):
                compile_seq(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_missing_bin_file(self, fprime_test_api):
        """Calling a nonexistent .bin file should fail at compile time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = str(Path(tmpdir).resolve() / "nonexistent.bin")
            parent_seq = f"""\
Ref.seqDisp.RUN_ARGS("{fake_path}", Fw.Wait.WAIT)
"""
            with pytest.raises(CompilationFailed, match="not found"):
                compile_seq(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            with pytest.raises(CompilationFailed, match="binary directory"):
                compile_seq(fprime_test_api, parent_seq)


class TestSeqCallingNested:
    """Test nested sequence execution: parent -> child -> grandchild."""

    def test_nested_two_levels(self, fprime_test_api):
        """Parent calls child which calls grandchild; all args verified."""
        if fprime_test_api is not None:
            pytest.skip("Needs 3 sequencers; only 2 available on GDS")
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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

    def test_nested_pass_through_arg(self, fprime_test_api):
        """Parent passes a value through two levels of sequence calls."""
        if fprime_test_api is not None:
            pytest.skip("Needs 3 sequencers; only 2 available on GDS")
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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)


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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)


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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            _compile_and_run_parent(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)


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
            with pytest.raises(CompilationFailed, match="Unknown argument 'z'"):
                compile_seq(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            with pytest.raises(CompilationFailed, match="specified multiple times"):
                compile_seq(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            with pytest.raises(CompilationFailed, match="specified multiple times"):
                compile_seq(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)

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
            with pytest.raises(CompilationFailed, match="Missing sequence argument 'y'"):
                compile_seq(fprime_test_api, parent_seq, ground_binary_dir=tmpdir)


class TestSeqArgLimits:
    """Test compile-time limits on sequence argument names and counts."""

    def test_arg_name_too_long(self, fprime_test_api):
        """Arg name exceeding 255 UTF-8 bytes should be a compile error."""
        long_name = "a" * 256
        seq = f"""\
sequence({long_name}: U32)
CdhCore.cmdDisp.CMD_NO_OP()
"""
        with pytest.raises(CompilationFailed, match="too long"):
            compile_seq(fprime_test_api, seq)

    def test_arg_name_exactly_255_bytes(self, fprime_test_api):
        """Arg name of exactly 255 bytes should compile fine."""
        name_255 = "a" * 255
        seq = f"""\
sequence({name_255}: U32)
CdhCore.cmdDisp.CMD_NO_OP()
"""
        compile_seq(fprime_test_api, seq)


class TestFlightBinaryDir:
    """Test --flight-binary-dir prefix stripping for sequence calls."""

    def test_absolute_flight_path_resolved_via_prefix(self, fprime_test_api):
        """An absolute flight path with the configured prefix should compile successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())
            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
sequence(x: U32)
assert x == 42
"""
            _compile_to_bin(child_seq, Path(child_path))

            # Parent references child via an absolute flight path;
            # compiler strips the prefix to find child.bin in ground_binary_dir
            parent_seq = """\
Ref.seqDisp.RUN_ARGS("/seq/bin/child.bin", Fw.Wait.WAIT, 42)
"""
            compile_seq(
                fprime_test_api, parent_seq,
                ground_binary_dir=tmpdir,
                flight_binary_dir="/seq/bin",
            )

    def test_prefix_not_matching_uses_path_as_is(self, fprime_test_api):
        """A path that doesn't start with the prefix should resolve normally."""
        if fprime_test_api is not None:
            pytest.skip("Model-only: FSW can't resolve relative child paths from tmpdir")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())
            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
CdhCore.cmdDisp.CMD_NO_OP()
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = """\
Ref.seqDisp.RUN_ARGS("child.bin", Fw.Wait.WAIT)
"""
            _compile_and_run_parent(
                fprime_test_api, parent_seq,
                ground_binary_dir=tmpdir,
                flight_binary_dir="/some/other/prefix",
            )

    def test_prefix_with_trailing_slash(self, fprime_test_api):
        """Prefix with trailing slash should still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())
            child_path = str(Path(tmpdir) / "child.bin")
            child_seq = """\
CdhCore.cmdDisp.CMD_NO_OP()
"""
            _compile_to_bin(child_seq, Path(child_path))

            parent_seq = """\
Ref.seqDisp.RUN_ARGS("/seq/bin/child.bin", Fw.Wait.WAIT)
"""
            compile_seq(
                fprime_test_api, parent_seq,
                ground_binary_dir=tmpdir,
                flight_binary_dir="/seq/bin/",
            )
