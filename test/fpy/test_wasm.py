"""End-to-end tests for the LLVM/wasm backend.

These compile a sequence all the way to a runnable wasm module, run it in
wasmtime, and assert on the error code that ``fpy_main`` returns.

The backend currently only supports ``assert`` over compile-time-constant
conditions (all-literal expressions fold at compile time). Testing *runtime*
arithmetic needs a runtime operand -- i.e. variables -- which the backend
doesn't have yet, so that's deferred. What's meaningfully exercised here is the
wasm round-trip and the assert/exit-code semantics: a failed assert returns its
exit code verbatim, or EXIT_WITH_ERROR by default.
"""

import pytest

from fpy.model import DirectiveErrorCode
from fpy.test_helpers import run_seq_wasm


NO_ERROR = DirectiveErrorCode.NO_ERROR.value
EXIT_WITH_ERROR = DirectiveErrorCode.EXIT_WITH_ERROR.value


class TestWasmAssert:
    def test_passing_assert_succeeds(self):
        assert run_seq_wasm("assert 1 == 1\n") == NO_ERROR

    def test_empty_sequence_succeeds(self):
        assert run_seq_wasm("") == NO_ERROR

    @pytest.mark.parametrize(
        "exit_code, expected",
        [
            (None, EXIT_WITH_ERROR),  # no code written -> default
            (42, 42),                 # written code returned verbatim
            (123, 123),
        ],
    )
    def test_failing_assert_returns_written_code(self, exit_code, expected):
        # A false assert returns its exit code. The condition is constant-false
        # so the failure branch is taken.
        suffix = "" if exit_code is None else f", {exit_code}"
        assert run_seq_wasm(f"assert 1 == 2{suffix}\n") == expected


class TestWasmVariables:
    """Variables give us genuine *runtime* computation: reading a variable is not
    const-foldable, so these exercise the load/store/convert/arithmetic emitters
    rather than just constant folding."""

    def test_read_variable(self):
        assert run_seq_wasm("x: U32 = 5\nassert x == 5\n") == NO_ERROR
        assert run_seq_wasm("x: U32 = 5\nassert x == 6\n") == EXIT_WITH_ERROR

    def test_runtime_arithmetic(self):
        # x is a variable, so x + 1 is computed at runtime (not folded).
        assert run_seq_wasm("x: U64 = 5\ny: U64 = x + 1\nassert y == 6\n") == NO_ERROR

    def test_reassignment(self):
        assert run_seq_wasm("x: U64 = 5\nx = x + 10\nassert x == 15\n") == NO_ERROR

    def test_unsigned_widening(self):
        # U32 var read in a U64 context -> zero-extend.
        assert run_seq_wasm("x: U32 = 5\ny: U64 = x + 1\nassert y == 6\n") == NO_ERROR

    def test_signed_widening(self):
        # I32 var read in a wider context -> sign-extend.
        assert run_seq_wasm(
            "x: I32 = 0 - 5\ny: I64 = x + 1\nassert y == 0 - 4\n"
        ) == NO_ERROR

    def test_float_variable(self):
        assert run_seq_wasm("a: F64 = 2.5\nb: F64 = a + 1.5\nassert b == 4.0\n") == NO_ERROR

    def test_bool_variable(self):
        assert run_seq_wasm("ok: bool = True\nassert ok\n") == NO_ERROR
        assert run_seq_wasm("ok: bool = False\nassert ok\n") == EXIT_WITH_ERROR

    def test_enum_variable(self):
        assert run_seq_wasm(
            "c: Ref.DpDemo.ColorEnum = Ref.DpDemo.ColorEnum.RED\nassert True\n"
        ) == NO_ERROR

    def test_struct_variable(self):
        # Aggregate alloca + store of a struct constant.
        assert run_seq_wasm(
            "p: Ref.SignalPair = Ref.SignalPair(3, 4)\nassert True\n"
        ) == NO_ERROR

    def test_array_variable(self):
        assert run_seq_wasm(
            "a: Ref.DpDemo.U32Array = [1, 2, 3]\nassert True\n"
        ) == NO_ERROR

    def test_aggregate_copy(self):
        # Reading an aggregate variable (load of a struct) and storing it.
        assert run_seq_wasm(
            "p: Ref.SignalPair = Ref.SignalPair(3, 4)\n"
            "q: Ref.SignalPair = p\nassert True\n"
        ) == NO_ERROR


class TestWasmExit:
    """The exit() builtin returns its code from the sequence entry point."""

    def test_exit_returns_code_verbatim(self):
        assert run_seq_wasm("exit(42)\n") == 42
        assert run_seq_wasm("exit(7)\n") == EXIT_WITH_ERROR

    def test_exit_zero_succeeds(self):
        assert run_seq_wasm("exit(0)\n") == NO_ERROR

    def test_exit_short_circuits_rest_of_sequence(self):
        # exit() returns immediately, so the failing assert after it never runs.
        assert run_seq_wasm("exit(0)\nassert False\n") == NO_ERROR

    def test_exit_with_runtime_code(self):
        # The exit code comes from a variable (read at runtime), not a literal.
        assert run_seq_wasm("code: U8 = 9\nexit(code)\n") == 9


class TestWasmIf:
    """if / elif / else over runtime conditions (variable reads aren't folded)."""

    def test_if_taken(self):
        assert run_seq_wasm("x: U32 = 7\nif x == 7:\n    exit(5)\n") == 5

    def test_if_not_taken_falls_through(self):
        # Condition false, body skipped; sequence falls off the end -> success.
        assert run_seq_wasm("x: U32 = 7\nif x == 1:\n    exit(5)\n") == NO_ERROR

    def test_if_else(self):
        seq = "x: U32 = 3\nif x == 1:\n    exit(11)\nelse:\n    exit(33)\n"
        assert run_seq_wasm(seq) == 33

    def test_if_elif_else_chain(self):
        template = (
            "x: U32 = {v}\n"
            "if x == 1:\n    exit(11)\n"
            "elif x == 2:\n    exit(22)\n"
            "else:\n    exit(33)\n"
        )
        assert run_seq_wasm(template.format(v=1)) == 11
        assert run_seq_wasm(template.format(v=2)) == 22
        assert run_seq_wasm(template.format(v=9)) == 33

    def test_assignment_inside_if_visible_after(self):
        # The variable's slot is allocated in the entry block (frame-scoped), so
        # a store inside the taken branch is visible to a later read.
        seq = "y: U64 = 0\nif True:\n    y = 5\nassert y == 5\n"
        assert run_seq_wasm(seq) == NO_ERROR

    def test_assert_inside_if_body(self):
        assert run_seq_wasm("x: U32 = 7\nif x == 7:\n    assert False\n") == EXIT_WITH_ERROR

    def test_variable_declared_in_if_block(self):
        # A var declared in a top-level if block is block-scoped (a local, not a
        # global) and must still get storage (regression: it used to be dropped).
        assert run_seq_wasm("if True:\n    a: U32 = 5\n    assert a == 5\n") == NO_ERROR

    def test_same_name_in_separate_blocks_are_distinct(self):
        # Fpy is block-scoped: each block's `a` is a distinct variable, so they
        # must not collide.
        seq = (
            "if True:\n    a: U32 = 1\n    assert a == 1\n"
            "if True:\n    a: U32 = 2\n    assert a == 2\n"
        )
        assert run_seq_wasm(seq) == NO_ERROR
