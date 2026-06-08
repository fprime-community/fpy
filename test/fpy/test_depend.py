"""Integration tests for the fprime-fpy-depend tool (ast_to_dependencies / depend_main).

These tests exercise the real compiler pipeline. The key property under test is
that dependency discovery works even when the referenced .bin files do not exist
yet — the whole point of the tool is to let build systems determine compile order
before any binaries have been produced.
"""
import tempfile
from pathlib import Path

import pytest

import fpy.error
from fpy.compiler import ast_to_dependencies, text_to_ast
from fpy.state import get_base_compile_state
from fpy.test_helpers import default_dictionary


def _collect(seq: str, ground_binary_dir: str = None) -> list[str]:
    """Run ast_to_dependencies on a sequence string; return the dependency list."""
    fpy.error.file_name = "<test>"
    state = get_base_compile_state(default_dictionary, ground_binary_dir)
    body = text_to_ast(seq)
    assert body is not None
    return ast_to_dependencies(body, state)


class TestNoDependencies:
    def test_empty_sequence(self):
        assert _collect("") == []

    def test_regular_command_is_not_a_dep(self):
        assert _collect("CdhCore.cmdDisp.CMD_NO_OP()\n") == []

    def test_seq_run_without_args_is_not_a_dep(self):
        # RUN (no varargs) is not flagged as a seq-run-with-args command
        assert _collect('Ref.seqDisp.RUN("seq.bin", Fw.Wait.WAIT)\n') == []


class TestSingleDependency:
    def test_bin_name_resolved_with_ground_binary_dir(self):
        seq = 'Ref.seqDisp.RUN_ARGS("child.bin", Fw.Wait.WAIT)\n'
        deps = _collect(seq, ground_binary_dir="/tmp/bins")
        assert deps == ["/tmp/bins/child.bin"]

    def test_bin_does_not_need_to_exist(self):
        # The main differentiator from the compiler: missing binary is not an error
        seq = 'Ref.seqDisp.RUN_ARGS("nonexistent.bin", Fw.Wait.WAIT)\n'
        deps = _collect(seq, ground_binary_dir="/tmp/bins")
        assert deps == ["/tmp/bins/nonexistent.bin"]

    def test_seq_called_in_if_branch(self):
        seq = """\
x: I32 = 1
if x == 1:
    Ref.seqDisp.RUN_ARGS("branch.bin", Fw.Wait.WAIT)
"""
        deps = _collect(seq, ground_binary_dir="/tmp")
        assert deps == ["/tmp/branch.bin"]

    def test_seq_called_in_while_loop(self):
        seq = """\
x: I32 = 3
while x > 0:
    Ref.seqDisp.RUN_ARGS("loop.bin", Fw.Wait.NO_WAIT)
    x = x - 1
"""
        deps = _collect(seq, ground_binary_dir="/tmp")
        assert deps == ["/tmp/loop.bin"]

    def test_seq_called_inside_function(self):
        seq = """\
def run_it():
    Ref.seqDisp.RUN_ARGS("helper.bin", Fw.Wait.WAIT)
run_it()
"""
        deps = _collect(seq, ground_binary_dir="/tmp")
        assert deps == ["/tmp/helper.bin"]


class TestMultipleDependencies:
    def test_two_distinct_bins(self):
        seq = """\
Ref.seqDisp.RUN_ARGS("a.bin", Fw.Wait.WAIT)
Ref.seqDisp.RUN_ARGS("b.bin", Fw.Wait.WAIT)
"""
        deps = _collect(seq, ground_binary_dir="/tmp")
        assert deps == ["/tmp/a.bin", "/tmp/b.bin"]

    def test_order_matches_source_order(self):
        seq = """\
Ref.seqDisp.RUN_ARGS("first.bin", Fw.Wait.WAIT)
Ref.seqDisp.RUN_ARGS("second.bin", Fw.Wait.WAIT)
Ref.seqDisp.RUN_ARGS("third.bin", Fw.Wait.WAIT)
"""
        deps = _collect(seq, ground_binary_dir="/tmp")
        assert deps == ["/tmp/first.bin", "/tmp/second.bin", "/tmp/third.bin"]

    def test_duplicate_call_deduplicated(self):
        seq = """\
Ref.seqDisp.RUN_ARGS("child.bin", Fw.Wait.WAIT)
Ref.seqDisp.RUN_ARGS("child.bin", Fw.Wait.NO_WAIT)
"""
        deps = _collect(seq, ground_binary_dir="/tmp")
        assert deps == ["/tmp/child.bin"]

    def test_mix_of_seq_run_and_regular_commands(self):
        seq = """\
CdhCore.cmdDisp.CMD_NO_OP()
Ref.seqDisp.RUN_ARGS("only_this.bin", Fw.Wait.WAIT)
CdhCore.cmdDisp.CMD_NO_OP()
"""
        deps = _collect(seq, ground_binary_dir="/tmp")
        assert deps == ["/tmp/only_this.bin"]


class TestGroundBinaryDirHandling:
    def test_ground_binary_dir_prepended_to_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq = 'Ref.seqDisp.RUN_ARGS("child.bin", Fw.Wait.WAIT)\n'
            deps = _collect(seq, ground_binary_dir=tmpdir)
            assert deps == [str(Path(tmpdir) / "child.bin")]

    def test_no_ground_binary_dir_returns_bare_names(self):
        # Without a ground_binary_dir the raw filename is returned as-is
        seq = 'Ref.seqDisp.RUN_ARGS("child.bin", Fw.Wait.WAIT)\n'
        deps = _collect(seq, ground_binary_dir=None)
        assert deps == ["child.bin"]


class TestErrorCases:
    def test_non_string_literal_filename_is_compile_error(self):
        seq = """\
name: I32 = 0
Ref.seqDisp.RUN_ARGS(name, Fw.Wait.WAIT)
"""
        fpy.error.file_name = "<test>"
        state = get_base_compile_state(default_dictionary, "/tmp")
        body = text_to_ast(seq)
        with pytest.raises(fpy.error.CompileError) as exc_info:
            ast_to_dependencies(body, state)
        assert "string literal" in str(exc_info.value)


class TestDependMainCLI:
    """End-to-end CLI tests that write real .fpy files and call depend_main."""

    def test_no_deps_produces_no_output(self, tmp_path, capsys):
        from fpy.main import depend_main

        fpy_path = tmp_path / "seq.fpy"
        fpy_path.write_text("")
        depend_main([str(fpy_path), "-d", default_dictionary, "-g", str(tmp_path)])
        assert capsys.readouterr().out == ""

    def test_deps_printed_one_per_line(self, tmp_path, capsys):
        from fpy.main import depend_main

        fpy_path = tmp_path / "seq.fpy"
        fpy_path.write_text(
            'Ref.seqDisp.RUN_ARGS("a.bin", Fw.Wait.WAIT)\n'
            'Ref.seqDisp.RUN_ARGS("b.bin", Fw.Wait.WAIT)\n'
        )
        depend_main([str(fpy_path), "-d", default_dictionary, "-g", str(tmp_path)])
        lines = capsys.readouterr().out.splitlines()
        assert lines == [str(tmp_path / "a.bin"), str(tmp_path / "b.bin")]

    def test_bins_need_not_exist(self, tmp_path, capsys):
        from fpy.main import depend_main

        fpy_path = tmp_path / "seq.fpy"
        fpy_path.write_text('Ref.seqDisp.RUN_ARGS("ghost.bin", Fw.Wait.WAIT)\n')
        # ghost.bin is never created — should still succeed
        depend_main([str(fpy_path), "-d", default_dictionary, "-g", str(tmp_path)])
        assert capsys.readouterr().out.strip() == str(tmp_path / "ghost.bin")

    def test_default_ground_binary_dir_is_input_parent(self, tmp_path, capsys):
        from fpy.main import depend_main

        fpy_path = tmp_path / "seq.fpy"
        fpy_path.write_text('Ref.seqDisp.RUN_ARGS("child.bin", Fw.Wait.WAIT)\n')
        depend_main([str(fpy_path), "-d", default_dictionary])
        # Without -g, ground_binary_dir defaults to the input file's directory
        assert capsys.readouterr().out.strip() == str(tmp_path / "child.bin")
