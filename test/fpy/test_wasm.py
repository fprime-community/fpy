"""End-to-end tests for the LLVM/wasm backend.

These compile a sequence all the way to a runnable wasm module, run it through
the NASA spacewasm interpreter, and assert on the sequence's error code (what
the exit/fault host imports report, or 0 when the void entrypoint falls off
its end without failing).

Runtime behavior is exercised through variables: an all-literal expression
folds at compile time, so tests that want the wasm to actually compute
something route one operand through a variable.
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
from fpy.error import BackendError
from fpy.model import DirectiveErrorCode
from fpy.state import get_base_compile_state
from fpy.test_helpers import (
    compile_seq_wasm,
    default_dictionary,
    run_seq_wasm,
    run_seq_wasm_with_events,
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
        # storage; it gets spilled to a temporary stack slot to be indexed.
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


class TestWasmExit:
    """exit() lowers to the host `exit` call rather than a `ret`, so it ends
    the whole sequence with its code (0 is a normal exit, nonzero an error)."""

    def test_exit_returns_code_verbatim(self):
        assert run_seq_wasm("exit(42)\n") == 42
        assert run_seq_wasm("exit(7)\n") == EXIT_WITH_ERROR

    def test_exit_ends_sequence_early(self):
        # A nonzero exit also returns immediately; nothing after it runs.
        assert run_seq_wasm("exit(9)\nassert False\n") == 9

    def test_exit_zero_succeeds(self):
        assert run_seq_wasm("exit(0)\n") == NO_ERROR

    def test_exit_short_circuits_rest_of_sequence(self):
        # exit() returns immediately, so the failing assert after it never runs.
        assert run_seq_wasm("exit(0)\nassert False\n") == NO_ERROR

    def test_exit_with_runtime_code(self):
        # The exit code comes from a variable (read at runtime), not a literal.
        # exit()'s parameter is I32, and fpy doesn't implicitly mix signedness,
        # so a runtime code must be a signed int.
        assert run_seq_wasm("code: I32 = 9\nexit(code)\n") == 9


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
