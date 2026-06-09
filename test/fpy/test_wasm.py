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
from fpy.test_helpers import compile_seq_wasm, run_seq_wasm


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


class TestWasmArithmetic:
    """Runtime arithmetic, comparison, and boolean ops. Each uses a variable so
    the expression isn't constant-folded and actually exercises the emitter."""

    def test_add(self):
        assert run_seq_wasm("x: U64 = 5\nassert x + 1 == 6\n") == NO_ERROR

    def test_subtract(self):
        assert run_seq_wasm("x: I64 = 10\nassert x - 3 == 7\n") == NO_ERROR

    def test_multiply(self):
        assert run_seq_wasm("x: U64 = 6\nassert x * 7 == 42\n") == NO_ERROR

    def test_divide_is_float(self):
        # `/` always computes over floats, even for integer operands.
        assert run_seq_wasm("x: F64 = 7.0\nassert x / 2.0 == 3.5\n") == NO_ERROR

    def test_modulus_unsigned(self):
        assert run_seq_wasm("x: U64 = 17\nassert x % 5 == 2\n") == NO_ERROR

    def test_modulus_signed(self):
        # Modulo is floored (Python `%` / the VM): the result takes the sign of
        # the divisor, not the dividend. So -17 % 5 == 3 (not -2, which is what
        # truncated srem alone would give).
        assert run_seq_wasm("x: I64 = 0 - 17\nassert x % 5 == 3\n") == NO_ERROR
        # Negative divisor: 17 % -5 == -3 (sign of the divisor).
        assert run_seq_wasm("x: I64 = 17\nassert x % (0 - 5) == (0 - 3)\n") == NO_ERROR

    def test_modulus_float(self):
        assert run_seq_wasm("x: F64 = 5.5\nassert x % 2.0 == 1.5\n") == NO_ERROR
        # Floored, like the integer case: -5.5 % 2.0 == 0.5 (sign of divisor).
        assert run_seq_wasm("x: F64 = 0.0 - 5.5\nassert x % 2.0 == 0.5\n") == NO_ERROR

    def test_floor_divide_unsigned(self):
        assert run_seq_wasm("x: U64 = 17\nassert x // 5 == 3\n") == NO_ERROR

    def test_floor_divide_signed(self):
        # // floors toward -inf (Python `//`): -7 // 2 == -4, and a negative
        # divisor likewise takes the floor (7 // -2 == -4).
        assert run_seq_wasm("x: I64 = 0 - 7\nassert x // 2 == (0 - 4)\n") == NO_ERROR
        assert run_seq_wasm("x: I64 = 7\nassert x // (0 - 2) == (0 - 4)\n") == NO_ERROR

    def test_floor_divide_float(self):
        assert run_seq_wasm("x: F64 = 7.5\nassert x // 2.0 == 3.0\n") == NO_ERROR
        # Floored, not truncated: -5.5 // 2.0 == -3.0.
        assert run_seq_wasm("x: F64 = 0.0 - 5.5\nassert x // 2.0 == (0.0 - 3.0)\n") == NO_ERROR

    def test_greater_than_unsigned(self):
        assert run_seq_wasm("x: U64 = 5\nassert x > 3\n") == NO_ERROR
        assert run_seq_wasm("x: U64 = 5\nassert x > 9\n") == EXIT_WITH_ERROR

    def test_greater_than_or_equal(self):
        assert run_seq_wasm("x: U64 = 5\nassert x >= 5\n") == NO_ERROR

    def test_less_than_signed(self):
        # A signed-negative value is < 0; an unsigned comparison would get this
        # wrong, so this pins the signed icmp path.
        assert run_seq_wasm("x: I64 = 0 - 1\nassert x < 0\n") == NO_ERROR

    def test_less_than_or_equal(self):
        assert run_seq_wasm("x: U64 = 5\nassert x <= 5\n") == NO_ERROR

    def test_float_comparison(self):
        assert run_seq_wasm("x: F64 = 2.5\nassert x > 1.0\n") == NO_ERROR

    def test_and_short_circuits(self):
        # rhs (x > 10) is false, so the whole `and` is false.
        seq = "x: U64 = 5\nok: bool = (x == 5) and (x > 10)\nassert ok == False\n"
        assert run_seq_wasm(seq) == NO_ERROR
        assert run_seq_wasm("x: U64 = 5\nassert (x == 5) and (x > 0)\n") == NO_ERROR

    def test_or_short_circuits(self):
        # lhs is true, so `or` is true without evaluating the (false) rhs.
        assert run_seq_wasm("x: U64 = 5\nassert (x == 5) or (x == 99)\n") == NO_ERROR
        seq = "x: U64 = 5\nok: bool = (x == 1) or (x == 2)\nassert ok == False\n"
        assert run_seq_wasm(seq) == NO_ERROR


class TestWasmUnaryOps:
    """Runtime unary ops (`-x`, `not x`, `+x`). Each uses a variable so the
    expression isn't constant-folded and actually exercises the emitter."""

    def test_negate_int(self):
        assert run_seq_wasm("x: I64 = 5\nassert -x == (0 - 5)\n") == NO_ERROR

    def test_negate_float(self):
        assert run_seq_wasm("x: F64 = 2.5\nassert -x == (0.0 - 2.5)\n") == NO_ERROR

    def test_double_negate(self):
        assert run_seq_wasm("x: I64 = 5\nassert -(-x) == 5\n") == NO_ERROR

    def test_not_true(self):
        assert run_seq_wasm("x: bool = True\nassert (not x) == False\n") == NO_ERROR

    def test_not_false(self):
        assert run_seq_wasm("x: bool = False\nassert not x\n") == NO_ERROR

    def test_identity(self):
        assert run_seq_wasm("x: I64 = 7\nassert +x == 7\n") == NO_ERROR


class TestWasmExponent:
    """`**` always computes over floats and lowers to the llvm.pow intrinsic,
    which the wasm target leaves as an imported `env.pow` host call. run_seq_wasm
    provides that import, so the emitted call is exercised end-to-end."""

    def test_exponent(self):
        assert run_seq_wasm("x: F64 = 2.0\nassert x ** 3.0 == 8.0\n") == NO_ERROR

    def test_exponent_emits_pow_import(self):
        # Document the host-call contract: the emitted module imports env.pow.
        from wasmtime import Engine, Module

        wasm = compile_seq_wasm("x: F64 = 2.0\nassert x ** 3.0 == 8.0\n")
        imports = {(i.module, i.name) for i in Module(Engine(), wasm).imports}
        assert ("env", "pow") in imports


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
