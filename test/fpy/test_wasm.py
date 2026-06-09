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
