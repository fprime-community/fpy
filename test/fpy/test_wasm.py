"""End-to-end tests for the LLVM/wasm backend.

These compile a sequence all the way to a runnable wasm module, run it through
the NASA spacewasm interpreter, and assert on the error code that ``fpy_main``
returns.

The backend currently only supports ``assert`` over compile-time-constant
conditions (all-literal expressions fold at compile time). Testing *runtime*
arithmetic needs a runtime operand -- i.e. variables -- which the backend
doesn't have yet, so that's deferred. What's meaningfully exercised here is the
wasm round-trip and the assert/exit-code semantics: a failed assert returns its
exit code verbatim, or EXIT_WITH_ERROR by default.
"""

import pytest

import llvmlite.binding as llvm

from fpy.codegen_llvm import (
    LLVM_CPU,
    LLVM_TRIPLE,
    GenerateLlvmModule,
    _ensure_llvm_targets,
)
from fpy.compiler import analyze_ast, text_to_ast
from fpy.model import DirectiveErrorCode
from fpy.state import get_base_compile_state
from fpy.test_helpers import compile_seq_wasm, default_dictionary, run_seq_wasm


NO_ERROR = DirectiveErrorCode.NO_ERROR.value
EXIT_WITH_ERROR = DirectiveErrorCode.EXIT_WITH_ERROR.value


def _seq_to_llvm_module(seq: str):
    """Lower *seq* to an llvmlite ir.Module (pre-codegen, target-independent)."""
    state = get_base_compile_state(default_dictionary, None)
    body = text_to_ast(seq)
    state = analyze_ast(body, state)
    return GenerateLlvmModule().emit(body, state)


def _emit_wasm_asm(seq: str, cpu: str) -> str:
    """Lower *seq* and emit its wasm textual assembly for the given target CPU.

    Re-parses the IR each call: emitting codegen mutates the parsed module (it
    bakes target-features attributes into the functions), so a parsed module
    can't be reused across CPUs without cross-contaminating results.
    """
    _ensure_llvm_targets()
    parsed = llvm.parse_assembly(str(_seq_to_llvm_module(seq)))
    parsed.verify()
    target = llvm.Target.from_triple(LLVM_TRIPLE)
    return target.create_target_machine(cpu=cpu).emit_assembly(parsed)


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


class TestWasmCast:
    """Explicit numeric casts -- e.g. I32(x). Unlike implicit coercion, a cast
    skips the semantic range check, so it's how a sequence narrows a float to an
    int (or an int to a smaller int). The cast itself emits no instructions: the
    operand's contextual type becomes the target type, so the conversion rides
    on the operand's normal lowering. The operand is a variable here, so the
    conversion happens at runtime rather than folding at compile time."""

    def test_float_to_int_truncates_toward_zero(self):
        # 5.9 -> 5: float->int truncates toward zero (wasm trunc / C / the VM).
        assert run_seq_wasm("x: F64 = 5.9\ny: I32 = I32(x)\nassert y == 5\n") == NO_ERROR

    def test_negative_float_to_int_truncates_toward_zero(self):
        # -5.9 -> -5 (toward zero), not -6 (toward -inf).
        assert run_seq_wasm("x: F64 = -5.9\ny: I32 = I32(x)\nassert y == -5\n") == NO_ERROR

    def test_int_to_float(self):
        assert run_seq_wasm("x: I32 = 7\ny: F64 = F64(x)\nassert y == 7.0\n") == NO_ERROR

    def test_int_narrowing_wraps(self):
        # Narrowing an int truncates the high bits: 300 & 0xff == 44.
        assert run_seq_wasm("x: I32 = 300\ny: U8 = U8(x)\nassert y == 44\n") == NO_ERROR


class TestWasmFloatToIntSaturates:
    """Out-of-range float->int casts saturate, matching Rust's `as`: a value
    above/below the target type's range clamps to its max/min, and NaN maps to
    0. (The bytecode VM instead *wraps* mod 2^n, so the backends differ on
    out-of-range inputs -- the cross-backend cast tests in
    test_types_and_constructors switch on the backend.)

    The backend lowers this with llvm.fptosi.sat / llvm.fptoui.sat. Under the
    WASM 1.0 MVP target there is no saturating trunc_sat op (that's the post-MVP
    nontrapping-fptoint feature), so the intrinsic lowers to a guarded trunc
    with explicit clamping -- which still does NOT trap."""

    @pytest.mark.parametrize(
        "seq",
        [
            "x: F64 = 1e20\nassert U8(x) == 255\n",       # above U8 max -> 255
            "x: F64 = -5.0\nassert U8(x) == 0\n",         # below U8 min -> 0
            "x: F64 = 1000.0\nassert I8(x) == 127\n",     # above I8 max -> 127
            "x: F64 = -1000.0\nassert I8(x) == -128\n",   # below I8 min -> -128
            "x: F64 = 1e20\nassert I32(x) == 2147483647\n",   # I32 max
            "x: F64 = -1e20\nassert I32(x) == -2147483648\n",  # I32 min
        ],
    )
    def test_out_of_range_saturates(self, seq):
        assert run_seq_wasm(seq) == NO_ERROR

    def test_nan_to_int_is_zero(self):
        # 0.0 / 0.0 is NaN; a NaN float->int cast saturates to 0.
        assert run_seq_wasm("x: F64 = 0.0\ny: F64 = x / x\nassert I32(y) == 0\n") == NO_ERROR

    def test_infinity_to_int_saturates(self):
        # +inf clamps to the target max rather than trapping or crashing.
        assert run_seq_wasm(
            "x: F64 = 1e308\nx = x * 10.0\nassert I32(x) == 2147483647\n"
        ) == NO_ERROR

    def test_out_of_range_does_not_trap(self):
        # Runs to completion (returns a code) rather than trapping; a wasm trap
        # would surface as a RuntimeError (runner fault) out of run_seq_wasm.
        assert run_seq_wasm("x: F64 = 1e20\ny: I32 = I32(x)\nassert True\n") == NO_ERROR

    def test_stays_mvp_no_trunc_sat(self):
        """The saturating intrinsic must not pull in the post-MVP saturating op:
        the MVP target lowers it to a guarded trunc (no trunc_sat), whereas the
        default 'generic' CPU would use trunc_sat. Guards against the backend
        dropping cpu=LLVM_CPU or LLVM changing its feature defaults."""
        seq = "x: F64 = 1e20\ny: I32 = I32(x)\nassert y == 0\n"
        assert "i32.trunc_sat_f64_s" not in _emit_wasm_asm(seq, cpu=LLVM_CPU)
        assert "i32.trunc_sat_f64_s" in _emit_wasm_asm(seq, cpu="generic")
