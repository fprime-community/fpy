"""End-to-end tests for the LLVM/wasm backend.

These compile a sequence all the way to a runnable wasm module, run it through
the NASA spacewasm interpreter, and assert on the sequence's error code (what
the exit/fault host imports report, or 0 when the void entrypoint falls off
its end without failing).

Runtime behavior is exercised through variables: an all-literal expression
folds at compile time, so tests that want the wasm to actually compute
something route one operand through a variable.
"""

import struct

import pytest

from llvmlite import ir
import llvmlite.binding as llvm

from fpy.codegen_llvm import (
    FPY_ENTRY_POINT,
    LLVM_CPU,
    LLVM_TRIPLE,
    EmitLlvmExpr,
    GenerateLlvmModule,
    llvm_module_to_wasm,
    _ensure_llvm_targets,
)
from fpy.wasm_host import (
    ERROR_CODE_TYPE,
    HOST_EXIT_FUNC_NAME,
    declare_host_imports,
)
from fpy.compiler import analyze_ast, text_to_ast
from fpy.dictionary import load_dictionary
from fpy.error import BackendError
from fpy.model import DirectiveErrorCode
from fpy.state import get_base_compile_state
from fpy.test_helpers import (
    compile_seq_wasm,
    default_dictionary,
    run_seq_wasm,
    run_seq_wasm_with_cmds,
    run_seq_wasm_with_events,
    run_wasm,
)
from fpy.types import (
    BOOL,
    F32,
    F64,
    FpyValue,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
)

# Every test in this module drives the LLVM/wasm backend end-to-end. The wasm
# marker makes conftest build the spacewasm runner on demand, so these always
# run on the wasm backend even when --wasm isn't passed.
pytestmark = pytest.mark.wasm


NO_ERROR = DirectiveErrorCode.NO_ERROR.value
EXIT_WITH_ERROR = DirectiveErrorCode.EXIT_WITH_ERROR.value
ARRAY_OOB = DirectiveErrorCode.ARRAY_OUT_OF_BOUNDS.value
DOMAIN_ERROR = DirectiveErrorCode.DOMAIN_ERROR.value


def _seq_to_llvm_module(seq: str):
    """Lower *seq* to an llvmlite ir.Module (pre-codegen, target-independent)."""
    state = get_base_compile_state(default_dictionary, None)
    body = text_to_ast(seq)
    state = analyze_ast(body, state)
    return GenerateLlvmModule().emit(state.root_block, state)


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
            (42, 42),  # written code returned verbatim
            (123, 123),
        ],
    )
    def test_failing_assert_returns_written_code(self, exit_code, expected):
        # A false assert returns its exit code. The condition is constant-false
        # so the failure branch is taken.
        suffix = "" if exit_code is None else f", {exit_code}"
        assert run_seq_wasm(f"assert 1 == 2{suffix}\n") == expected


class TestWasmExit:
    """exit() lowers to the host `exit` call rather than a `ret`, so it ends
    the whole sequence from any call depth, returning its code verbatim (code 0
    is a normal exit, nonzero a fault)."""

    @pytest.mark.parametrize("code", [0, 5, 7, 42, 123])
    def test_exit_returns_code(self, code):
        assert run_seq_wasm(f"exit({code})\n") == code

    def test_exit_with_runtime_code(self):
        # The exit code comes from a variable (read at runtime), not a literal.
        # exit()'s parameter is I32, and fpy doesn't implicitly mix signedness,
        # so a runtime code must be a signed int.
        assert run_seq_wasm("code: I32 = 9\nexit(code)\n") == 9

    def test_exit_ends_sequence_early(self):
        # The exit happens before the (would-fail) asserts, so the sequence ends
        # with exit's code and never reaches them.
        assert run_seq_wasm("exit(0)\nassert 1 == 2\n") == NO_ERROR
        assert run_seq_wasm("exit(0)\nassert False\n") == NO_ERROR
        assert run_seq_wasm("exit(9)\nassert 1 == 1\n") == 9


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
        assert (
            run_seq_wasm("x: I32 = 0 - 5\ny: I64 = x + 1\nassert y == 0 - 4\n")
            == NO_ERROR
        )

    def test_float_variable(self):
        assert (
            run_seq_wasm("a: F64 = 2.5\nb: F64 = a + 1.5\nassert b == 4.0\n")
            == NO_ERROR
        )

    def test_bool_variable(self):
        assert run_seq_wasm("ok: bool = True\nassert ok\n") == NO_ERROR
        assert run_seq_wasm("ok: bool = False\nassert ok\n") == EXIT_WITH_ERROR

    def test_enum_variable(self):
        assert (
            run_seq_wasm(
                "c: Ref.DpDemo.ColorEnum = Ref.DpDemo.ColorEnum.RED\nassert True\n"
            )
            == NO_ERROR
        )

    def test_struct_variable(self):
        # Aggregate alloca + store of a struct constant.
        assert (
            run_seq_wasm("p: Ref.SignalPair = Ref.SignalPair(3, 4)\nassert True\n")
            == NO_ERROR
        )

    def test_array_variable(self):
        assert (
            run_seq_wasm("a: Ref.DpDemo.U32Array = [1, 2, 3]\nassert True\n")
            == NO_ERROR
        )

    def test_aggregate_copy(self):
        # Reading an aggregate variable (load of a struct) and storing it.
        assert (
            run_seq_wasm(
                "p: Ref.SignalPair = Ref.SignalPair(3, 4)\n"
                "q: Ref.SignalPair = p\nassert True\n"
            )
            == NO_ERROR
        )


class TestWasmMemberAccess:
    """Struct member and array element access, on both sides of an assignment,
    including runtime (non-constant) indices and their bounds checks."""

    def test_read_struct_member(self):
        assert (
            run_seq_wasm(
                "p: Ref.SignalPair = Ref.SignalPair(3.0, 4.0)\n"
                "assert p.time == 3.0\nassert p.value == 4.0\n"
            )
            == NO_ERROR
        )
        assert (
            run_seq_wasm(
                "p: Ref.SignalPair = Ref.SignalPair(3.0, 4.0)\n"
                "assert p.value == 5.0\n"
            )
            == EXIT_WITH_ERROR
        )

    def test_write_struct_member(self):
        # An in-place store: the sibling member must keep its value.
        assert (
            run_seq_wasm(
                "p: Ref.SignalPair = Ref.SignalPair(3.0, 4.0)\n"
                "p.value = 9.5\n"
                "assert p.value == 9.5\nassert p.time == 3.0\n"
            )
            == NO_ERROR
        )

    def test_read_array_element_const_index(self):
        assert (
            run_seq_wasm(
                "a: Svc.ComQueueDepth = Svc.ComQueueDepth(7, 8)\n"
                "assert a[0] == 7\nassert a[1] == 8\n"
            )
            == NO_ERROR
        )

    def test_read_array_element_runtime_index(self):
        # An I8 index also exercises the sext to ArrayIndexType (I64).
        assert (
            run_seq_wasm(
                "a: Svc.ComQueueDepth = Svc.ComQueueDepth(456, 123)\n"
                "i: I8 = 1\n"
                "assert a[i] == 123\n"
            )
            == NO_ERROR
        )

    def test_write_array_element_runtime_index(self):
        assert (
            run_seq_wasm(
                "a: Svc.ComQueueDepth = Svc.ComQueueDepth(456, 123)\n"
                "i: I8 = 1\n"
                "a[i] = 111\n"
                "assert a[1] == 111\nassert a[0] == 456\n"
            )
            == NO_ERROR
        )

    def test_read_runtime_index_out_of_bounds(self):
        assert (
            run_seq_wasm(
                "a: Svc.ComQueueDepth = Svc.ComQueueDepth(456, 123)\n"
                "i: I8 = 2\n"
                "x: U32 = a[i]\n"
            )
            == ARRAY_OOB
        )

    def test_read_runtime_index_negative(self):
        assert (
            run_seq_wasm(
                "a: Svc.ComQueueDepth = Svc.ComQueueDepth(456, 123)\n"
                "i: I8 = -1\n"
                "x: U32 = a[i]\n"
            )
            == ARRAY_OOB
        )

    def test_write_runtime_index_out_of_bounds(self):
        assert (
            run_seq_wasm(
                "a: Svc.ComQueueDepth = Svc.ComQueueDepth(456, 123)\n"
                "i: I8 = 2\n"
                "a[i] = 111\n"
            )
            == ARRAY_OOB
        )

    def test_nested_chain_runtime_index(self):
        # a[i].member on an array of structs: a GEP chain with a runtime index
        # in the middle, read and written in place.
        pairs = (
            "pairs: Ref.SignalPairSet = Ref.SignalPairSet("
            "Ref.SignalPair(1.0, 2.0), Ref.SignalPair(3.0, 4.0), "
            "Ref.SignalPair(5.0, 6.0), Ref.SignalPair(7.0, 8.0))\n"
        )
        assert (
            run_seq_wasm(
                pairs + "i: I64 = 1\n"
                "assert pairs[i].time == 3.0\n"
                "pairs[i].value = 99.0\n"
                "assert pairs[1].value == 99.0\n"
                "assert pairs[1].time == 3.0\n"
                "assert pairs[0].value == 2.0\n"
            )
            == NO_ERROR
        )

    def test_runtime_index_into_const_array(self):
        # The parent is a constant expression, not a variable, so it has no
        # storage; it gets copied to a temporary stack slot to be indexed.
        assert (
            run_seq_wasm(
                "i: I8 = 1\n"
                "x: U32 = Svc.ComQueueDepth(10, 20)[i]\n"
                "assert x == 20\n"
            )
            == NO_ERROR
        )

    def test_assign_rhs_evaluated_before_lhs_bounds_check(self):
        # In `a[i] = rhs` the rhs is evaluated before the lhs index is
        # bounds-checked, matching the VM's evaluation order: the rhs's zero
        # divisor faults with DOMAIN_ERROR before the out-of-bounds store
        # could fault with ARRAY_OOB. (The VM-side twin lives in
        # test_types_and_constructors.py under the same name.)
        assert (
            run_seq_wasm(
                "a: Svc.ComQueueDepth = Svc.ComQueueDepth(456, 123)\n"
                "i: I8 = 2\n"
                "z: U32 = 0\n"
                "a[i] = U32(456 // z)\n"
            )
            == DOMAIN_ERROR
        )

    def test_anon_struct_member(self):
        # Member access on an anonymous struct literal emits just the member
        # expression; a runtime member value keeps it from const-folding.
        assert (
            run_seq_wasm(
                "y: F32 = 4.5\n"
                "x: F32 = {time: 1.0, value: y}.value\n"
                "assert x == 4.5\n"
            )
            == NO_ERROR
        )


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
        assert (
            run_seq_wasm("x: F64 = 0.0 - 5.5\nassert x // 2.0 == (0.0 - 3.0)\n")
            == NO_ERROR
        )

    def test_floor_divide_by_zero_faults(self):
        # A zero divisor is DOMAIN_ERROR in the VM; the wasm i64.div_s/div_u
        # would trap uncatchably instead, so the backend guards and faults.
        # (The divisor is a variable so nothing folds at compile time.)
        assert run_seq_wasm("z: U64 = 0\nx: U64 = 17 // z\n") == DOMAIN_ERROR
        assert run_seq_wasm("z: I64 = 0\nx: I64 = 17 // z\n") == DOMAIN_ERROR

    def test_modulus_by_zero_faults(self):
        # Like division -- and unlike float `/` -- a zero divisor in `%` is
        # DOMAIN_ERROR even for floats (the VM checks it; libm fmod would
        # quietly return NaN).
        assert run_seq_wasm("z: U64 = 0\nx: U64 = 17 % z\n") == DOMAIN_ERROR
        assert run_seq_wasm("z: I64 = 0\nx: I64 = 17 % z\n") == DOMAIN_ERROR
        assert run_seq_wasm("z: F64 = 0.0\nx: F64 = 5.5 % z\n") == DOMAIN_ERROR

    def test_float_divide_by_zero_is_ieee(self):
        # Float `/` (and thus float `//`) by zero is IEEE inf, not a fault,
        # matching the VM.
        assert run_seq_wasm("z: F64 = 0.0\nassert 1.0 / z > 1.0e308\n") == NO_ERROR
        assert run_seq_wasm("z: F64 = 0.0\nassert 1.0 // z > 1.0e308\n") == NO_ERROR

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


class TestWasmAbsFloor:
    """iabs/fabs lower to llvm.abs/llvm.fabs and float `//` to llvm.floor.
    These pin the SPEC edge cases: fabs only clears the sign bit, and flooring
    preserves the sign of a zero quotient. The sign of a zero is observed
    through division (1.0 / -0.0 == -inf)."""

    def test_iabs_basic(self):
        assert run_seq_wasm("x: I64 = 0 - 5\nassert iabs(x) == 5\n") == NO_ERROR

    def test_iabs_int_min_wraps(self):
        # Known divergence: SPEC says iabs(I64 min) raises ARITHMETIC_OVERFLOW,
        # but the wasm backend implements no arithmetic traps yet and llvm.abs
        # with is_int_min_poison=0 wraps. This pins the current behavior so a
        # future trap implementation consciously flips it.
        seq = "x: I64 = I64(-2**63)\nassert iabs(x) == x\n"
        assert run_seq_wasm(seq) == NO_ERROR

    def test_fabs_negative_zero(self):
        seq = "neg: F64 = -0.0\nassert 1.0 / neg < 0.0\nassert 1.0 / fabs(neg) > 0.0\n"
        assert run_seq_wasm(seq) == NO_ERROR

    def test_fabs_inf_nan(self):
        seq = (
            "zero: F64 = 0.0\n"
            "inf: F64 = 1.0 / zero\n"
            "nan: F64 = zero / zero\n"
            "assert fabs(0.0 - inf) == inf\n"
            "assert fabs(nan) != fabs(nan)\n"
        )
        assert run_seq_wasm(seq) == NO_ERROR

    def test_floor_divide_preserves_zero_sign(self):
        seq = "neg: F64 = -0.0\nq: F64 = neg // 1.0\nassert 1.0 / q < 0.0\n"
        assert run_seq_wasm(seq) == NO_ERROR

    def test_floor_divide_inf_nan_passthrough(self):
        seq = (
            "zero: F64 = 0.0\n"
            "inf: F64 = 1.0 / zero\n"
            "nan: F64 = zero / zero\n"
            "assert inf // 1.0 == inf\n"
            "q: F64 = nan // 1.0\n"
            "assert q != q\n"
        )
        assert run_seq_wasm(seq) == NO_ERROR


class TestWasmExponent:
    """`**` always computes over floats and lowers to the llvm.pow intrinsic,
    which the wasm target leaves as an imported `env.pow` host call. run_seq_wasm
    provides that import, so the emitted call is exercised end-to-end."""

    def test_exponent(self):
        assert run_seq_wasm("x: F64 = 2.0\nassert x ** 3.0 == 8.0\n") == NO_ERROR

    def test_exponent_emits_pow_import(self):
        # Document the host-call contract: the linked module imports env.pow.
        # An import-section entry encodes as <len>module <len>name <kind>, so a
        # function import of env.pow is exactly this byte run.
        wasm = compile_seq_wasm("x: F64 = 2.0\nassert x ** 3.0 == 8.0\n")
        assert b"\x03env\x03pow\x00" in wasm


class TestWasmLog:
    """log() lowers to the host `event(severity, ptr, len)` call, with the
    message bytes in a constant in linear memory. The runner harness reports
    each call back as a (severity, message) pair."""

    def test_default_severity_is_activity_hi(self):
        code, events = run_seq_wasm_with_events('log("hello world")\n')
        assert code == NO_ERROR
        assert events == [(5, "hello world")]  # ACTIVITY_HI = 5

    def test_explicit_severity(self):
        code, events = run_seq_wasm_with_events('log("oh no", Fw.LogSeverity.FATAL)\n')
        assert code == NO_ERROR
        assert events == [(1, "oh no")]  # FATAL = 1

    def test_multiple_events_in_call_order(self):
        code, events = run_seq_wasm_with_events(
            'log("first")\n'
            'log("second", Fw.LogSeverity.WARNING_HI)\n'
            'log("first")\n'
        )
        assert code == NO_ERROR
        assert events == [(5, "first"), (2, "second"), (5, "first")]

    def test_empty_message(self):
        code, events = run_seq_wasm_with_events('log("")\n')
        assert code == NO_ERROR
        assert events == [(5, "")]

    def test_log_before_exit_still_reported(self):
        # The event host call must happen before the sequence terminates.
        code, events = run_seq_wasm_with_events('log("bye")\nexit(9)\n')
        assert code == 9
        assert events == [(5, "bye")]


class TestWasmBareExpressionStatements:
    """A bare expression statement must be lowered for its side effects, even
    when its top-level node type doesn't advertise any."""

    def test_constant_bare_expr_is_noop(self):
        # A constant expression statement is pure -- it folds away and the
        # sequence runs cleanly to the end.
        assert run_seq_wasm("10.0 ** 1000\nassert 1 == 1\n") == NO_ERROR

    def test_embedded_call_is_lowered_not_dropped(self):
        # `f() == 0` is an AstBinaryOp -- not a side-effecting node type -- but
        # it embeds a call that is. Lowering it must reach the call rather than
        # silently dropping the statement. The wasm backend can't lower
        # script-function calls yet, so reaching the call surfaces as a
        # BackendError; before the fix the statement was dropped and the module
        # compiled to a (wrong) no-op with no error at all.
        seq = "def f() -> U32:\n    return 0\nf() == 0\n"
        with pytest.raises(BackendError, match="script-defined function"):
            _seq_to_llvm_module(seq)


class TestWasmSequenceParameters:
    def test_sequence_with_parameters_is_rejected(self):
        # The host interface has no way to deliver a parameter's value, so the
        # parameter would silently read as zero. Refuse the sequence instead.
        seq = "sequence(x: U32)\nassert x == 1\n"
        with pytest.raises(BackendError, match="sequences with parameters"):
            _seq_to_llvm_module(seq)


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
        assert (
            run_seq_wasm("x: U32 = 7\nif x == 7:\n    assert False\n")
            == EXIT_WITH_ERROR
        )

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
        assert (
            run_seq_wasm("x: F64 = 5.9\ny: I32 = I32(x)\nassert y == 5\n") == NO_ERROR
        )

    def test_negative_float_to_int_truncates_toward_zero(self):
        # -5.9 -> -5 (toward zero), not -6 (toward -inf).
        assert (
            run_seq_wasm("x: F64 = -5.9\ny: I32 = I32(x)\nassert y == -5\n") == NO_ERROR
        )

    def test_int_to_float(self):
        assert (
            run_seq_wasm("x: I32 = 7\ny: F64 = F64(x)\nassert y == 7.0\n") == NO_ERROR
        )

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
            "x: F64 = 1e20\nassert U8(x) == 255\n",  # above U8 max -> 255
            "x: F64 = -5.0\nassert U8(x) == 0\n",  # below U8 min -> 0
            "x: F64 = 1000.0\nassert I8(x) == 127\n",  # above I8 max -> 127
            "x: F64 = -1000.0\nassert I8(x) == -128\n",  # below I8 min -> -128
            "x: F64 = 1e20\nassert I32(x) == 2147483647\n",  # I32 max
            "x: F64 = -1e20\nassert I32(x) == -2147483648\n",  # I32 min
        ],
    )
    def test_out_of_range_saturates(self, seq):
        assert run_seq_wasm(seq) == NO_ERROR

    def test_nan_to_int_is_zero(self):
        # 0.0 / 0.0 is NaN; a NaN float->int cast saturates to 0.
        assert (
            run_seq_wasm("x: F64 = 0.0\ny: F64 = x / x\nassert I32(y) == 0\n")
            == NO_ERROR
        )

    def test_infinity_to_int_saturates(self):
        # +inf clamps to the target max rather than trapping or crashing.
        assert (
            run_seq_wasm("x: F64 = 1e308\nx = x * 10.0\nassert I32(x) == 2147483647\n")
            == NO_ERROR
        )

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


class TestWasmCommands:
    """Command calls lower to the host `cmd(buf ptr, buf len)` call, where the
    buffer holds the big-endian serialized FwOpcodeType followed by the
    fprime-serialized arguments. The runner harness reports each buffer back
    verbatim, so these assert the exact wire bytes against an independent
    struct.pack encoding. Constant arguments are baked into the buffer at
    compile time; runtime arguments are byte-swapped and stored into their
    packed offsets before each dispatch."""

    def _opcode(self, name: str) -> bytes:
        d = load_dictionary(default_dictionary)
        return struct.pack(">I", d["cmd_name_dict"][name].opcode)

    def test_cmd_emits_fprime_cmd_import(self):
        # Document the host-call contract: the linked module imports
        # fprime_v1.cmd. An import-section entry encodes as
        # <len>module <len>name <kind>, so this byte run is exactly that entry.
        wasm = compile_seq_wasm("CdhCore.cmdDisp.CMD_NO_OP()\n")
        assert b"\x09fprime_v1\x03cmd\x00" in wasm

    def test_const_no_arg_command(self):
        code, cmds = run_seq_wasm_with_cmds("CdhCore.cmdDisp.CMD_NO_OP()\n")
        assert code == NO_ERROR
        assert cmds == [self._opcode("CdhCore.cmdDisp.CMD_NO_OP")]

    def test_const_string_arg_is_compact(self):
        # A constant string serializes at its actual length (u16 big-endian
        # prefix + bytes), not its declared capacity.
        code, cmds = run_seq_wasm_with_cmds('CdhCore.cmdDisp.CMD_NO_OP_STRING("hi")\n')
        assert code == NO_ERROR
        expected = self._opcode("CdhCore.cmdDisp.CMD_NO_OP_STRING")
        expected += struct.pack(">H", 2) + b"hi"
        assert cmds == [expected]

    def test_empty_string_arg(self):
        # An empty string is just the zero length prefix.
        code, cmds = run_seq_wasm_with_cmds('CdhCore.cmdDisp.CMD_NO_OP_STRING("")\n')
        assert code == NO_ERROR
        expected = self._opcode("CdhCore.cmdDisp.CMD_NO_OP_STRING")
        expected += struct.pack(">H", 0)
        assert cmds == [expected]

    def test_utf8_string_arg_prefix_counts_bytes(self):
        # The length prefix counts encoded utf-8 bytes, not characters:
        # "héllo✓" is 6 characters but 9 bytes.
        code, cmds = run_seq_wasm_with_cmds(
            'CdhCore.cmdDisp.CMD_NO_OP_STRING("héllo✓")\n'
        )
        assert code == NO_ERROR
        data = "héllo✓".encode("utf-8")
        assert len(data) == 9
        expected = self._opcode("CdhCore.cmdDisp.CMD_NO_OP_STRING")
        expected += struct.pack(">H", len(data)) + data
        assert cmds == [expected]

    def test_runtime_scalar_args(self):
        # A negative int pins the big-endian sign bytes; the float pins the
        # bitcast-then-swap path.
        code, cmds = run_seq_wasm_with_cmds(
            "var1: I32 = -2\n"
            "var2: F32 = 1.5\n"
            "var3: U8 = 8\n"
            "CdhCore.cmdDisp.CMD_TEST_CMD_1(var1, var2, var3)\n"
        )
        assert code == NO_ERROR
        expected = self._opcode("CdhCore.cmdDisp.CMD_TEST_CMD_1")
        expected += struct.pack(">ifB", -2, 1.5, 8)
        assert cmds == [expected]

    @pytest.mark.parametrize("flag, byte", [(True, b"\xff"), (False, b"\x00")])
    def test_runtime_bool_arg(self, flag, byte):
        # Bools serialize as the FW_SERIALIZE truth bytes, not 1/0.
        code, cmds = run_seq_wasm_with_cmds(
            "idx: U32 = 3\n"
            f"flag: bool = {flag}\n"
            "Ref.cmdSeq0.SET_BREAKPOINT(idx, flag)\n"
        )
        assert code == NO_ERROR
        expected = self._opcode("Ref.cmdSeq0.SET_BREAKPOINT")
        expected += struct.pack(">I", 3) + byte
        assert cmds == [expected]

    def test_mixed_const_and_runtime_args(self):
        # A compact constant string ahead of a runtime argument pins the
        # runtime argument's offset computation.
        code, cmds = run_seq_wasm_with_cmds(
            "en: Fw.Enabled = Fw.Enabled.ENABLED\n"
            'CdhCore.health.HLTH_PING_ENABLE("task1", en)\n'
        )
        assert code == NO_ERROR
        expected = self._opcode("CdhCore.health.HLTH_PING_ENABLE")
        expected += struct.pack(">H", 5) + b"task1"  # entry: String_40
        expected += struct.pack(">B", 1)  # enable: Fw.Enabled (u8 rep), ENABLED
        assert cmds == [expected]

    def test_runtime_struct_arg_all_scalar_widths(self):
        # A runtime struct argument walks every member; ScalarStruct covers
        # every scalar width, including the 8-byte swaps.
        code, cmds = run_seq_wasm_with_cmds(
            "s: Ref.ScalarStruct = "
            "Ref.ScalarStruct(-1, -2, -3, -4, 1, 2, 3, 4, 1.5, -2.5)\n"
            "Ref.typeDemo.SEND_SCALARS(s)\n"
        )
        assert code == NO_ERROR
        expected = self._opcode("Ref.typeDemo.SEND_SCALARS")
        expected += struct.pack(">bhiqBHIQfd", -1, -2, -3, -4, 1, 2, 3, 4, 1.5, -2.5)
        assert cmds == [expected]

    def test_runtime_array_arg(self):
        # An array of i32-rep enums exercises the element walk.
        code, cmds = run_seq_wasm_with_cmds(
            "c: Ref.ManyChoices = Ref.ManyChoices(Ref.Choice.TWO, Ref.Choice.RED)\n"
            "Ref.typeDemo.CHOICES(c)\n"
        )
        assert code == NO_ERROR
        expected = self._opcode("Ref.typeDemo.CHOICES")
        expected += struct.pack(">ii", 1, 2)  # TWO = 1, RED = 2
        assert cmds == [expected]

    def test_multiple_commands_in_call_order(self):
        code, cmds = run_seq_wasm_with_cmds(
            'CdhCore.cmdDisp.CMD_NO_OP_STRING("a")\n' "CdhCore.cmdDisp.CMD_NO_OP()\n"
        )
        assert code == NO_ERROR
        assert cmds == [
            self._opcode("CdhCore.cmdDisp.CMD_NO_OP_STRING")
            + struct.pack(">H", 1)
            + b"a",
            self._opcode("CdhCore.cmdDisp.CMD_NO_OP"),
        ]

    def test_captured_response_compares_ok(self):
        code, _ = run_seq_wasm_with_cmds(
            "ret: Fw.CmdResponse = CdhCore.cmdDisp.CMD_NO_OP()\n"
            "assert ret == Fw.CmdResponse.OK\n"
        )
        assert code == NO_ERROR

    def test_captured_response_carries_host_value(self):
        code, _ = run_seq_wasm_with_cmds(
            "ret: Fw.CmdResponse = CdhCore.cmdDisp.CMD_NO_OP()\n"
            "assert ret == Fw.CmdResponse.BUSY\n",
            cmd_response=5,  # BUSY
        )
        assert code == NO_ERROR

    def test_captured_failing_response_does_not_auto_exit(self):
        # Capturing the response takes responsibility for it: a failing
        # command must not end the sequence even with assert_cmd_success set.
        code, _ = run_seq_wasm_with_cmds(
            "ret: Fw.CmdResponse = CdhCore.cmdDisp.CMD_NO_OP()\n"
            "assert ret == Fw.CmdResponse.EXECUTION_ERROR\n",
            cmd_response=4,  # EXECUTION_ERROR
        )
        assert code == NO_ERROR

    def test_bare_failing_command_exits_cmd_fail(self):
        code, cmds = run_seq_wasm_with_cmds(
            "CdhCore.cmdDisp.CMD_NO_OP()\nassert False\n",
            cmd_response=4,  # EXECUTION_ERROR
        )
        assert code == DirectiveErrorCode.CMD_FAIL.value
        # The command was dispatched; the sequence ended on its response.
        assert cmds == [self._opcode("CdhCore.cmdDisp.CMD_NO_OP")]

    def test_bare_failing_command_via_fail_opcodes(self):
        d = load_dictionary(default_dictionary)
        code, _ = run_seq_wasm_with_cmds(
            "CdhCore.cmdDisp.CMD_NO_OP()\n",
            failing_opcodes={d["cmd_name_dict"]["CdhCore.cmdDisp.CMD_NO_OP"].opcode},
        )
        assert code == DirectiveErrorCode.CMD_FAIL.value

    def test_assert_cmd_success_flag_disables_check(self):
        # With the flag cleared, failing bare commands don't end the sequence;
        # both commands are still dispatched.
        code, cmds = run_seq_wasm_with_cmds(
            "flags.assert_cmd_success = False\n"
            "CdhCore.cmdDisp.CMD_NO_OP()\n"
            "CdhCore.cmdDisp.CMD_NO_OP()\n",
            cmd_response=4,  # EXECUTION_ERROR
        )
        assert code == NO_ERROR
        assert cmds == [self._opcode("CdhCore.cmdDisp.CMD_NO_OP")] * 2

    def test_bare_command_inside_if_block(self):
        # The response check applies to bare commands in nested blocks too.
        code, _ = run_seq_wasm_with_cmds(
            "x: U8 = 1\n"
            "if x == 1:\n"
            "    CdhCore.cmdDisp.CMD_NO_OP()\n"
            "assert False\n",
            cmd_response=4,  # EXECUTION_ERROR
        )
        assert code == DirectiveErrorCode.CMD_FAIL.value


def _big_endian_cases():
    """Round-trip test values spanning every serializable runtime type: each
    scalar width, both bools, both enum rep widths, and nested aggregates."""
    types = load_dictionary(default_dictionary)["type_defs"]
    choice = types["Ref.Choice"]
    slurry = types["Ref.ChoiceSlurry"]
    slurry_members = {m.name: m.type for m in slurry.members}
    many_choices = types["Ref.ManyChoices"]

    def choice_v(name):
        return FpyValue(choice, name)

    def many(a, b):
        return FpyValue(many_choices, [choice_v(a), choice_v(b)])

    # Asymmetric byte patterns, so a wrong (little-endian) order and a
    # partial/misplaced store are both visible in the bytes.
    scalars = FpyValue(
        types["Ref.ScalarStruct"],
        {
            "i8": FpyValue(I8, -1),
            "i16": FpyValue(I16, -2),
            "i32": FpyValue(I32, -3),
            "i64": FpyValue(I64, -4),
            "u8": FpyValue(U8, 1),
            "u16": FpyValue(U16, 0x0102),
            "u32": FpyValue(U32, 0x01020304),
            "u64": FpyValue(U64, 0x0102030405060708),
            "f32": FpyValue(F32, 1.5),
            "f64": FpyValue(F64, -2.5),
        },
    )
    slurry_val = FpyValue(
        slurry,
        {
            "tooManyChoices": FpyValue(
                slurry_members["tooManyChoices"],
                [many("ONE", "TWO"), many("RED", "BLUE")],
            ),
            "separateChoice": choice_v("RED"),
            "choicePair": FpyValue(
                slurry_members["choicePair"],
                {"firstChoice": choice_v("TWO"), "secondChoice": choice_v("BLUE")},
            ),
            "choiceAsMemberArray": FpyValue(
                slurry_members["choiceAsMemberArray"],
                [FpyValue(U8, 0xAA), FpyValue(U8, 0x55)],
            ),
        },
    )
    return [
        pytest.param(FpyValue(U8, 0xAB), id="u8"),
        pytest.param(FpyValue(I8, -1), id="i8-neg"),
        pytest.param(FpyValue(U16, 0x0102), id="u16"),
        pytest.param(FpyValue(I16, -2), id="i16-neg"),
        pytest.param(FpyValue(U32, 0x01020304), id="u32"),
        pytest.param(FpyValue(I32, -123456789), id="i32-neg"),
        pytest.param(FpyValue(U64, 0x0102030405060708), id="u64"),
        pytest.param(FpyValue(I64, -3_000_000_000), id="i64-neg"),
        pytest.param(FpyValue(F32, 1.5), id="f32"),
        pytest.param(FpyValue(F64, -2.5), id="f64-neg"),
        pytest.param(FpyValue(BOOL, True), id="bool-true"),
        pytest.param(FpyValue(BOOL, False), id="bool-false"),
        pytest.param(FpyValue(types["Fw.Enabled"], "ENABLED"), id="enum-u8-rep"),
        pytest.param(choice_v("RED"), id="enum-i32-rep"),
        pytest.param(scalars, id="struct-every-scalar"),
        pytest.param(many("TWO", "RED"), id="array-of-enums"),
        pytest.param(slurry_val, id="struct-nested-arrays"),
    ]


class TestWasmBigEndianSerialization:
    """_emit_store_big_endian / _emit_load_big_endian translate between LLVM
    values and the fprime wire format in linear memory. FpyValue.serialize()
    is the format oracle: storing a value must produce exactly its bytes
    (pinning big-endianness, tight packing, and the bool truth bytes -- a
    round trip alone couldn't catch both directions agreeing on the wrong
    endianness), and a value loaded from those bytes must store back to them
    (which pins the load as the store's inverse, since serialization is
    injective)."""

    def _emit_check_bytes(self, builder, buf, expected: bytes, exit_code: int):
        """Exit with *exit_code* unless the buffer holds exactly *expected*."""
        i8 = ir.IntType(8)
        i32 = ir.IntType(32)
        ok = ir.Constant(ir.IntType(1), 1)
        for i, byte in enumerate(expected):
            ptr = builder.gep(
                buf, [ir.Constant(i32, 0), ir.Constant(i32, i)], inbounds=True
            )
            ok = builder.and_(
                ok,
                builder.icmp_unsigned(
                    "==", builder.load(ptr, align=1), ir.Constant(i8, byte)
                ),
            )
        fail_block = builder.function.append_basic_block("bytes_bad")
        ok_block = builder.function.append_basic_block("bytes_ok")
        builder.cbranch(ok, ok_block, fail_block)
        builder.position_at_end(fail_block)
        builder.call(
            builder.module.globals[HOST_EXIT_FUNC_NAME],
            [ir.Constant(ERROR_CODE_TYPE, exit_code)],
        )
        builder.unreachable()
        builder.position_at_end(ok_block)

    def _build_module(self, value: FpyValue) -> ir.Module:
        module = ir.Module(name="be_test")
        module.triple = LLVM_TRIPLE
        declare_host_imports(module)
        func = ir.Function(
            module, ir.FunctionType(ir.VoidType(), []), name=FPY_ENTRY_POINT
        )
        builder = ir.IRBuilder(func.append_basic_block("entry"))
        emitter = EmitLlvmExpr(builder)

        expected = value.serialize()
        buf_type = ir.ArrayType(ir.IntType(8), len(expected))

        def byte_buffer(name, init):
            g = ir.GlobalVariable(module, buf_type, name=module.get_unique_name(name))
            g.linkage = "internal"
            g.initializer = ir.Constant(buf_type, bytearray(init))
            return g

        # Route the value through a mutable global and a runtime load, so the
        # walks execute on the interpreter instead of LLVM folding them away.
        src = ir.GlobalVariable(module, value.type.llvm_type, name="src")
        src.linkage = "internal"
        src.initializer = value.llvm_value

        # exit(1): storing the value must produce the oracle bytes.
        buf_store = byte_buffer("buf_store", bytes(len(expected)))
        written = emitter._emit_store_big_endian(
            builder.load(src), value.type, buf_store, 0
        )
        assert written == len(expected) == value.type.max_size
        self._emit_check_bytes(builder, buf_store, expected, exit_code=1)

        # exit(2): loading from the oracle bytes must store back to them.
        buf_in = byte_buffer("buf_in", expected)
        loaded = emitter._emit_load_big_endian(value.type, buf_in, 0)
        buf_out = byte_buffer("buf_out", bytes(len(expected)))
        emitter._emit_store_big_endian(loaded, value.type, buf_out, 0)
        self._emit_check_bytes(builder, buf_out, expected, exit_code=2)

        builder.ret_void()
        return module

    @pytest.mark.parametrize("value", _big_endian_cases())
    def test_round_trip_matches_serialize(self, value):
        code, _, _ = run_wasm(llvm_module_to_wasm(self._build_module(value)))
        assert code != 1, f"stored bytes diverged from serialize() for {value}"
        assert code != 2, f"load->store did not round-trip for {value}"
        assert code == NO_ERROR

    @pytest.mark.parametrize("byte", [b"\x01", b"\x7f", b"\xfe"])
    def test_invalid_bool_byte_faults(self, byte):
        # A bool byte that is neither truth value faults with
        # DESERIALIZE_ERROR_INVALID_BOOL, matching the C++ deserializer's
        # FW_DESERIALIZE_FORMAT_ERROR (and FpyValue.deserialize raising
        # DeserializeError).
        module = ir.Module(name="be_test")
        module.triple = LLVM_TRIPLE
        declare_host_imports(module)
        func = ir.Function(
            module, ir.FunctionType(ir.VoidType(), []), name=FPY_ENTRY_POINT
        )
        builder = ir.IRBuilder(func.append_basic_block("entry"))
        emitter = EmitLlvmExpr(builder)

        buf_type = ir.ArrayType(ir.IntType(8), 1)
        buf = ir.GlobalVariable(module, buf_type, name="buf_in")
        buf.linkage = "internal"
        buf.initializer = ir.Constant(buf_type, bytearray(byte))
        emitter._emit_load_big_endian(BOOL, buf, 0)
        builder.ret_void()

        code, _, _ = run_wasm(llvm_module_to_wasm(module))
        assert code == DirectiveErrorCode.DESERIALIZE_ERROR_INVALID_BOOL.value
