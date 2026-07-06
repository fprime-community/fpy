#!/usr/bin/env python3
"""Z3 specification of fpy's arithmetic, comparison, and boolean operators.

This is the arithmetic counterpart of cast_properties.py. It formalizes the
two-stage semantics the compiler implements for every operator expression:

  1. TYPE LEVEL (compile time, plain Python over Ty): intermediate_type(op,
     arg_types) picks the type both operands are coerced to and in which the
     operation computes. It mirrors PickTypesAndResolveFields.
     pick_intermediate_type (src/fpy/semantics.py) restricted to runtime
     types; check_intermediate_type_matches_compiler verifies the two agree
     on every (op, types) combination, so the formal rules cannot drift from
     the implementation silently.

  2. VALUE LEVEL (run time, Z3 terms): each operator is a total function
     op(a: Value, b: Value) -> Result. A Value is a Z3 term tagged with its
     fpy type. A Result is the value of the expression plus a `halted`
     condition: True on exactly the inputs where the program must end with a
     runtime error. Operands are first coerced to the intermediate type; the
     value map of every coercion the type system admits is the cast function
     from cast_properties.py (casting is just explicit coercion, MATH.md).

Results model *pure* expression evaluation. Sequencing concerns -- whether
the operands themselves halted, and/or short-circuit evaluation of `and`/
`or` -- compose outside these functions in a future program-level spec.

The rules encoded (MATH.md, "Arithmetic on I64, U64" / "Arithmetic on F64"):

  * F64 arithmetic is IEEE-754 (RNE) and never halts: x/0.0 is +-inf,
    0.0/0.0 is NaN, overflow rounds to +-inf.
  * I64/U64 `+`, `-`, `*`, unary `-` produce the exact mathematical result;
    if that result is not representable in the intermediate type the
    program halts. Overflow is an error, not a wrap. Unary `-` on an
    unsigned operand is a compile error (as in Rust).
  * `//` and `%` on integers halt on exactly the same inputs: when the
    divisor is 0, and on I64_MIN op -1 (the quotient 2^63 is
    unrepresentable; the remainder would be 0 but halts anyway, matching
    Rust and keeping the //-% pair coherent). Both are *floored* (Python
    semantics): `//` rounds the quotient toward -inf, `%` takes the sign
    of the divisor.
  * `//` on floats is round-toward-negative of the IEEE quotient. `%` on
    floats is C fmod (the exact truncated remainder) followed by one RNE
    addition of the divisor when the sign must flip -- exactly what the
    LLVM backend emits (frem + fadd) and what CPython computes. x % 0.0 is
    NaN, not a halt (CPython raises here; fpy float ops never halt).
  * `/` and `**` always compute in F64. `**` is the platform's libm pow,
    modeled as an uninterpreted function: deterministic and type-correct,
    value otherwise unspecified.
  * Comparisons coerce to the intermediate type and yield bool. On floats
    they are the IEEE predicates: every comparison involving NaN is False,
    except `!=` which is the negation of `==` and hence True.

Open questions this spec makes visible (marked OQ-n at their encodings):

  OQ-1 Integer overflow halts here, per MATH.md's normative paragraph and
       the VM (handle_add/sub/smul return ARITHMETIC_OVERFLOW/UNDERFLOW).
       The LLVM backend emits plain add/sub/mul, which WRAP (and one
       MATH.md sentence says wrapping). Proving codegen against this spec
       will fail until overflow guards exist -- or until the spec flips.
  OQ-2 RESOLVED 2026-07-06: unary minus on an unsigned operand is a
       compile error (Rust's rule), not a halt-unless-zero. Encoded in
       pick_intermediate_type and mirrored in intermediate_type here.
  OQ-3 RESOLVED 2026-07-05: float `!=` is Not(==), so NaN != NaN is True
       (IEEE, Python, and the VM agree). The LLVM backend used to emit
       fcmp `one` (False on NaN); it now emits fcmp une.
  OQ-4 RESOLVED 2026-07-05/06: LLVM udiv/sdiv/urem/srem are UB on divisor
       0 (and sdiv/srem on MIN/-1); the backend now guards them: divisor 0
       exits with DOMAIN_ERROR (matching the VM), and MIN // -1 and
       MIN % -1 both exit with ARITHMETIC_OVERFLOW (Rust's behavior; the
       VM model does the same).
  OQ-5 RESOLVED 2026-07-06: float x % 0.0 is NaN -- float ops never halt,
       matching IEEE, Rust, and C# (both verified: no trap in any mode).
       The VM model's handle_fmod now agrees; CPython raises instead, a
       deliberate divergence. The C++ op_fmod still returns DOMAIN_ERROR
       (and computes a different formula) -- needs an upstream fix.

Run:
    uv run --with z3-solver python verify/arith_properties.py
"""

import math
import operator
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from z3 import (
    RNE,
    RTN,
    And,
    BitVecVal,
    BoolSort,
    BoolVal,
    BVAddNoOverflow,
    BVAddNoUnderflow,
    BVMulNoOverflow,
    BVMulNoUnderflow,
    BVSNegNoOverflow,
    BVSubNoOverflow,
    BVSubNoUnderflow,
    Const,
    Extract,
    FPSort,
    FPVal,
    Function,
    If,
    Implies,
    Not,
    Or,
    SignExt,
    SRem,
    UDiv,
    UGE,
    UGT,
    ULE,
    ULT,
    URem,
    Xor,
    ZeroExt,
    fpAbs,
    fpAdd,
    fpDiv,
    fpEQ,
    fpGEQ,
    fpGT,
    fpIsNaN,
    fpIsNegative,
    fpIsZero,
    fpLEQ,
    fpLT,
    fpMul,
    fpNaN,
    fpNeg,
    fpRem,
    fpRoundToIntegral,
    fpSub,
    is_false,
    is_true,
    simplify,
)

from cast_properties import (
    F64S,
    FAIL,
    PASS,
    TYPES,
    Ty,
    cast,
    prove,
    record,
    results,
)

from fpy.syntax import (
    BOOLEAN_OPERATORS,
    COMPARISON_OPS,
    NUMERIC_OPERATORS,
    BinaryStackOp,
    UnaryStackOp,
)

F64TY = TYPES["F64"]
I64TY = TYPES["I64"]
U64TY = TYPES["U64"]


# --- values and results --------------------------------------------------------


@dataclass(frozen=True)
class BoolTy:
    """The bool type. Not a Ty: it has no bit-level numeric interpretation
    here; bool values are Z3 Booleans."""

    name: str = "bool"

    @property
    def sort(self):
        return BoolSort()


BOOL = BoolTy()


@dataclass(frozen=True)
class Value:
    """A runtime value: a Z3 term `expr` of sort ty.sort, tagged with its
    fpy type."""

    ty: object  # Ty | BoolTy
    expr: object  # z3 term of sort ty.sort


@dataclass(frozen=True)
class Result:
    """The outcome of evaluating one operator on already-evaluated operands:
    the value produced, and the condition under which evaluation instead
    halts the program with a runtime error. When halted is True the value
    is unconstrained (don't-care)."""

    value: Value
    halted: object  # z3 Bool


# --- type level: the intermediate type rules -------------------------------------


def intermediate_type(op, arg_tys):
    """The type the operands of `op` are coerced to and in which the op
    computes. Returns None where the compiler rejects the program.

    Mirrors PickTypesAndResolveFields.pick_intermediate_type restricted to
    the runtime types (the 10 specific numeric types and bool). The
    arb-precision Int/Float types only occur in compile-time constant
    folding, and non-numeric ==/!= (structs, arrays, enums, time) is a
    byte-comparison with no arithmetic content; both are out of scope here.

    The rules:
      1. Boolean operators compute in bool (operand bool-ness is enforced
         by coercion, exactly as in the compiler).
      2. bool == bool and bool != bool compute in bool.
      3. Everything else requires numeric operands.
      4. Unary `-` is undefined for unsigned integers (compile error, as in
         Rust: the result is negative for every nonzero operand).
      5. `/` and `**` always compute in F64.
      6. If any operand is a float, the op computes in F64.
      7. Signed and unsigned integers never mix: compile error.
      8. Unsigned operands compute in U64; signed operands in I64.
    """
    if op in BOOLEAN_OPERATORS:
        return BOOL
    if op in (BinaryStackOp.EQUAL, BinaryStackOp.NOT_EQUAL):
        if all(isinstance(t, BoolTy) for t in arg_tys):
            return BOOL
    if not all(isinstance(t, Ty) for t in arg_tys):
        return None
    # arity distinguishes negation from subtraction: ops are str-valued
    # enums and NEGATE == SUBTRACT (both "-")
    if (
        len(arg_tys) == 1
        and op == UnaryStackOp.NEGATE
        and arg_tys[0].kind == "int"
        and not arg_tys[0].signed
    ):
        return None
    if op in (BinaryStackOp.DIVIDE, BinaryStackOp.EXPONENT):
        return F64TY
    if any(t.kind == "float" for t in arg_tys):
        return F64TY
    if len(arg_tys) == 2 and arg_tys[0].signed != arg_tys[1].signed:
        return None
    if any(not t.signed for t in arg_tys):
        return U64TY
    return I64TY


def coercible(frm, to) -> bool:
    """Whether the compiler admits an implicit conversion frm -> to: only
    identity, lossless integer widening (same signedness), float widening,
    and int->float. An operand that is not coercible to the intermediate
    type is a compile error (this is where e.g. `1 and 2` is rejected,
    since boolean operators pick bool regardless of the operand types)."""
    if isinstance(frm, BoolTy) or isinstance(to, BoolTy):
        return frm == to
    if frm == to:
        return True
    if frm.kind == "int" and to.kind == "float":
        return True
    return (
        frm.kind == to.kind
        and to.bits >= frm.bits
        and (frm.kind == "float" or frm.signed == to.signed)
    )


def coerce(v: Value, to) -> Value:
    """Coerce an operand to the intermediate type.

    The value map of every admitted coercion is the spec cast function
    (casting is merely explicit coercion). Note int->F64 genuinely rounds
    (RNE) for |x| > 2^53: `/` on two I64s divides *rounded* operands.
    """
    assert coercible(v.ty, to), f"cannot coerce {v.ty.name} to {to.name}"
    if v.ty == to:
        return v
    return Value(to, cast(v.expr, v.ty, to))


# --- value level: integer ops ----------------------------------------------------


def int_exact(ty: Ty, extra: int, fn, *args):
    """The exact mathematical result of fn over [[ty]] operands, with halt
    on unrepresentability (OQ-1).

    Encoding: embed the operands in a ring wide enough that fn cannot wrap
    (extra=1 bit for +, -, unary -; extra=ty.bits for *), apply fn there --
    which therefore equals the mathematical operation -- and take the low
    ty.bits as the value. The mathematical result is representable in
    [[ty]] iff re-extending the value gives back the wide result; halt on
    exactly the complement. check_int_overflow_encoding proves this halt
    condition equals Z3's independent no-overflow/no-underflow predicates.
    """
    ext = SignExt if ty.signed else ZeroExt
    wide = fn(*(ext(extra, a) for a in args))
    value = Extract(ty.bits - 1, 0, wide)
    halted = ext(extra, value) != wide
    return value, halted


def int_floor_div(a, b, ty: Ty):
    """Floored integer division (Python //). Halts on b == 0, and for
    signed types on MIN // -1, whose mathematical quotient 2^(bits-1) is
    unrepresentable (OQ-1, OQ-4)."""
    n = ty.bits
    zero = BitVecVal(0, n)
    if not ty.signed:
        # quotient of non-negatives is non-negative: truncation == floor
        return UDiv(a, b), b == zero
    halted = Or(
        b == zero,
        And(a == BitVecVal(ty.min, n), b == BitVecVal(-1, n)),
    )
    q = a / b  # SMT bvsdiv: truncates toward zero
    r = SRem(a, b)  # sign of the dividend
    # truncation differs from floor exactly when the exact quotient is
    # negative (operand signs differ <=> xor is negative) and inexact
    # (nonzero remainder); there truncation overshoots the floor by one
    adjust = And(r != zero, (a ^ b) < zero)
    return If(adjust, q - 1, q), halted


def int_mod(a, b, ty: Ty):
    """Floored integer modulo (Python %): the result takes the sign of the
    divisor. Halts on b == 0 and, for signed types, on MIN % -1 -- exactly
    the same condition as //, and Rust's behavior. (The mathematical
    remainder there would be 0, but the operation is UB in C++/LLVM, and
    defining `a // b` and `a % b` to halt together keeps the pair coherent:
    wherever one is defined, a == (a // b) * b + (a % b).)

    check_int_divmod_props proves this equals SMT-LIB's floored bvsmod and
    satisfies the identity with the sign/range conditions.
    """
    zero = BitVecVal(0, ty.bits)
    if not ty.signed:
        return URem(a, b), b == zero
    halted = Or(
        b == zero,
        And(a == BitVecVal(ty.min, ty.bits), b == BitVecVal(-1, ty.bits)),
    )
    r = SRem(a, b)  # truncated remainder: sign of the dividend
    # flip toward the divisor's sign when nonzero and signs differ
    flip = And(r != zero, (r ^ b) < zero)
    return If(flip, r + b, r), halted


# --- value level: float ops ------------------------------------------------------

# `**`: the platform libm's pow. Deterministic and total, but its rounding
# is implementation-defined, so the spec leaves it uninterpreted. Anything
# proved about programs holds for every possible pow; nothing about pow's
# values can be proved, by design.
FPOW = Function("fpy_pow", F64S, F64S, F64S)


def float_fmod_trunc(a, b):
    """C fmod / LLVM frem: the exact truncated remainder a - trunc(a/b)*b
    (computed as if with infinite precision; it is always representable).

    Z3 only exposes fpRem, the IEEE `remainder` whose quotient is rounded
    to *nearest* instead of truncated. The two differ by exactly
    copysign(|b|, a), precisely when fpRem's nonzero result has the
    opposite sign of a; that correction addition is exact because the true
    sum (the fmod result) is representable, so RNE does not round. Special
    cases (NaN, inf a, zero b -> NaN; fmod(a, inf) = a; zero results keep
    the sign of a) come along from fpRem unchanged.
    """
    r = fpRem(a, b)
    mag = fpAbs(b)
    correction = If(fpIsNegative(a), fpNeg(mag), mag)
    flip = And(
        Not(fpIsNaN(r)),
        Not(fpIsZero(r)),
        Xor(fpIsNegative(r), fpIsNegative(a)),
    )
    return If(flip, fpAdd(RNE(), r, correction), r)


def float_mod(a, b):
    """Floored float modulo (Python %): truncated fmod, then one RNE
    addition of the divisor when the sign must flip toward it.

    This is definitionally what the LLVM backend emits (frem + fadd) and
    what CPython computes; unlike the fmod correction, this addition CAN
    round (e.g. -1e-300 % 1e300: the exact answer 1e300 - 1e-300 is not
    representable), so the rounding is part of the spec. Two deliberate
    divergences from CPython: x % 0.0 is NaN, not an error, and an exactly
    zero result keeps fmod's zero sign rather than copysign(0, b).
    """
    m = float_fmod_trunc(a, b)
    flip = And(
        Not(fpIsNaN(m)),
        Not(fpIsZero(m)),
        Xor(fpIsNegative(m), fpIsNegative(b)),
    )
    return If(flip, fpAdd(RNE(), m, b), m)


# --- value level: the operators ---------------------------------------------------

NEVER = BoolVal(False)


def _compare(op, a, b, ty: Ty):
    if ty.kind == "float":
        if op == BinaryStackOp.EQUAL:
            return fpEQ(a, b)
        if op == BinaryStackOp.NOT_EQUAL:
            return Not(fpEQ(a, b))  # OQ-3: True when either side is NaN
        fp_table = {
            BinaryStackOp.LESS_THAN: fpLT,
            BinaryStackOp.LESS_THAN_OR_EQUAL: fpLEQ,
            BinaryStackOp.GREATER_THAN: fpGT,
            BinaryStackOp.GREATER_THAN_OR_EQUAL: fpGEQ,
        }
        return fp_table[op](a, b)
    if op == BinaryStackOp.EQUAL:
        return a == b
    if op == BinaryStackOp.NOT_EQUAL:
        return a != b
    int_table = (
        {
            BinaryStackOp.LESS_THAN: operator.lt,
            BinaryStackOp.LESS_THAN_OR_EQUAL: operator.le,
            BinaryStackOp.GREATER_THAN: operator.gt,
            BinaryStackOp.GREATER_THAN_OR_EQUAL: operator.ge,
        }
        if ty.signed
        else {
            BinaryStackOp.LESS_THAN: ULT,
            BinaryStackOp.LESS_THAN_OR_EQUAL: ULE,
            BinaryStackOp.GREATER_THAN: UGT,
            BinaryStackOp.GREATER_THAN_OR_EQUAL: UGE,
        }
    )
    return int_table[op](a, b)


def _numeric_binop(op, a, b, ty: Ty):
    if ty.kind == "float":
        rm = RNE()
        if op == BinaryStackOp.ADD:
            return fpAdd(rm, a, b), NEVER
        if op == BinaryStackOp.SUBTRACT:
            return fpSub(rm, a, b), NEVER
        if op == BinaryStackOp.MULTIPLY:
            return fpMul(rm, a, b), NEVER
        if op == BinaryStackOp.DIVIDE:
            return fpDiv(rm, a, b), NEVER
        if op == BinaryStackOp.FLOOR_DIVIDE:
            # divide, then floor the quotient (round toward negative)
            return fpRoundToIntegral(RTN(), fpDiv(rm, a, b)), NEVER
        if op == BinaryStackOp.MODULUS:
            return float_mod(a, b), NEVER
        assert op == BinaryStackOp.EXPONENT, op
        return FPOW(a, b), NEVER
    if op == BinaryStackOp.ADD:
        return int_exact(ty, 1, operator.add, a, b)
    if op == BinaryStackOp.SUBTRACT:
        return int_exact(ty, 1, operator.sub, a, b)
    if op == BinaryStackOp.MULTIPLY:
        return int_exact(ty, ty.bits, operator.mul, a, b)
    if op == BinaryStackOp.FLOOR_DIVIDE:
        return int_floor_div(a, b, ty)
    assert op == BinaryStackOp.MODULUS, op  # / and ** are float-only
    return int_mod(a, b, ty)


def apply_binary(op: BinaryStackOp, a: Value, b: Value) -> Result:
    """The value-level semantics of `a op b` on already-evaluated operands."""
    ty = intermediate_type(op, [a.ty, b.ty])
    assert (
        ty is not None
    ), f"{op} undefined for {a.ty.name}, {b.ty.name} (compile error)"

    if op in BOOLEAN_OPERATORS:
        av, bv = coerce(a, BOOL).expr, coerce(b, BOOL).expr
        fn = And if op == BinaryStackOp.AND else Or
        return Result(Value(BOOL, fn(av, bv)), NEVER)

    if isinstance(ty, BoolTy):
        # bool == bool / bool != bool
        eq = a.expr == b.expr
        val = eq if op == BinaryStackOp.EQUAL else Not(eq)
        return Result(Value(BOOL, val), NEVER)

    av, bv = coerce(a, ty).expr, coerce(b, ty).expr

    if op in COMPARISON_OPS:
        return Result(Value(BOOL, _compare(op, av, bv, ty)), NEVER)

    assert op in NUMERIC_OPERATORS, op
    value, halted = _numeric_binop(op, av, bv, ty)
    return Result(Value(ty, value), halted)


def apply_unary(op: UnaryStackOp, a: Value) -> Result:
    """The value-level semantics of `op a` on an already-evaluated operand."""
    ty = intermediate_type(op, [a.ty])
    assert ty is not None, f"{op} undefined for {a.ty.name} (compile error)"

    if op == UnaryStackOp.NOT:
        return Result(Value(BOOL, Not(coerce(a, BOOL).expr)), NEVER)

    av = coerce(a, ty).expr
    if op == UnaryStackOp.IDENTITY:
        return Result(Value(ty, av), NEVER)

    assert op == UnaryStackOp.NEGATE, op
    if ty.kind == "float":
        return Result(Value(ty, fpNeg(av)), NEVER)
    # unsigned operands were rejected by intermediate_type, so ty is a
    # signed type here and the only unrepresentable result is -MIN
    assert ty.signed
    value, halted = int_exact(ty, 1, operator.neg, av)
    return Result(Value(ty, value), halted)


def add(a: Value, b: Value) -> Result:
    return apply_binary(BinaryStackOp.ADD, a, b)


def sub(a: Value, b: Value) -> Result:
    return apply_binary(BinaryStackOp.SUBTRACT, a, b)


def mul(a: Value, b: Value) -> Result:
    return apply_binary(BinaryStackOp.MULTIPLY, a, b)


def div(a: Value, b: Value) -> Result:
    return apply_binary(BinaryStackOp.DIVIDE, a, b)


def floor_div(a: Value, b: Value) -> Result:
    return apply_binary(BinaryStackOp.FLOOR_DIVIDE, a, b)


def mod(a: Value, b: Value) -> Result:
    return apply_binary(BinaryStackOp.MODULUS, a, b)


def exponent(a: Value, b: Value) -> Result:
    return apply_binary(BinaryStackOp.EXPONENT, a, b)


def neg(a: Value) -> Result:
    return apply_unary(UnaryStackOp.NEGATE, a)


# --- checks ----------------------------------------------------------------------

ALL_SPEC_TYS = list(TYPES.values()) + [BOOL]


def check_intermediate_type_matches_compiler():
    """The formal type rules agree with the production compiler on every
    (op, operand types) combination -- pick_intermediate_type itself is the
    oracle, so the spec cannot silently drift from the implementation."""
    from fpy import types as fpy_types
    from fpy.semantics import PickTypesAndResolveFields

    compiler = PickTypesAndResolveFields()

    def to_fpy(t):
        if t is None:
            return None
        if isinstance(t, BoolTy):
            return fpy_types.BOOL
        return getattr(fpy_types, t.name)

    bad = []
    n = 0
    for op in BinaryStackOp:
        for t1 in ALL_SPEC_TYS:
            for t2 in ALL_SPEC_TYS:
                got = compiler.pick_intermediate_type([to_fpy(t1), to_fpy(t2)], op)
                want = to_fpy(intermediate_type(op, [t1, t2]))
                n += 1
                if got != want:
                    bad.append(
                        f"{t1.name} {op.value} {t2.name}: compiler={got} spec={want}"
                    )
    for op in UnaryStackOp:
        for t1 in ALL_SPEC_TYS:
            got = compiler.pick_intermediate_type([to_fpy(t1)], op)
            want = to_fpy(intermediate_type(op, [t1]))
            n += 1
            if got != want:
                bad.append(f"{op.value} {t1.name}: compiler={got} spec={want}")
    if bad:
        record(FAIL, "type rules == pick_intermediate_type", "; ".join(bad[:10]))
    else:
        record(
            PASS, "type rules == pick_intermediate_type", f"all {n} combinations agree"
        )


def check_well_formed():
    """Every legal (op, types) combination builds a closed Result: the value
    has the result type's sort, halted is Boolean, and no free variable
    other than the two operands appears."""
    from z3.z3util import get_vars

    bad = []
    n = 0
    for op in BinaryStackOp:
        for t1 in ALL_SPEC_TYS:
            for t2 in ALL_SPEC_TYS:
                ty = intermediate_type(op, [t1, t2])
                if ty is None or not (coercible(t1, ty) and coercible(t2, ty)):
                    continue
                n += 1
                x = Const(f"wf_a_{t1.name}", t1.sort)
                y = Const(f"wf_b_{t2.name}", t2.sort)
                res = apply_binary(op, Value(t1, x), Value(t2, y))
                want_sort = (
                    BoolSort()
                    if op in COMPARISON_OPS or op in BOOLEAN_OPERATORS
                    else ty.sort
                )
                if res.value.expr.sort() != want_sort:
                    bad.append(
                        f"{t1.name} {op.value} {t2.name}: sort {res.value.expr.sort()}"
                    )
                    continue
                if res.halted.sort() != BoolSort():
                    bad.append(f"{t1.name} {op.value} {t2.name}: halted sort")
                    continue
                fv = get_vars(res.value.expr) + get_vars(res.halted)
                if not all(
                    v.eq(x) or v.eq(y) or v.decl().name() == "fpy_pow" for v in fv
                ):
                    bad.append(f"{t1.name} {op.value} {t2.name}: free vars {fv}")
    for op in UnaryStackOp:
        for t1 in ALL_SPEC_TYS:
            ty = intermediate_type(op, [t1])
            if ty is None or not coercible(t1, ty):
                continue
            n += 1
            x = Const(f"wf_u_{t1.name}", t1.sort)
            res = apply_unary(op, Value(t1, x))
            want_sort = BoolSort() if op == UnaryStackOp.NOT else ty.sort
            if res.value.expr.sort() != want_sort:
                bad.append(f"{op.value} {t1.name}: sort {res.value.expr.sort()}")
                continue
            fv = get_vars(res.value.expr) + get_vars(res.halted)
            if not all(v.eq(x) for v in fv):
                bad.append(f"{op.value} {t1.name}: free vars {fv}")
    if bad:
        record(FAIL, "well-formed results", "; ".join(bad[:10]))
    else:
        record(
            PASS, "well-formed results", f"all {n} legal combinations closed and sorted"
        )


def check_int_overflow_encoding():
    """The widening halt condition of int_exact equals Z3's independent
    built-in no-overflow/no-underflow predicates -- two unrelated encodings
    of 'the mathematical result is unrepresentable'.

    The encoding is uniform in the bit width (the same SignExt/Extract
    structure at every width), so the multiply equivalence -- nonlinear and
    beyond Z3 for signed widths above 16 -- is proved at the widths Z3 can
    finish, and 64-bit multiply is additionally cross-checked concretely
    against Python in check_int_ops_match_python."""
    for tyn in ("I64", "U64", "I8", "U8"):
        ty = TYPES[tyn]
        a = Const(f"ovf_a_{tyn}", ty.sort)
        b = Const(f"ovf_b_{tyn}", ty.sort)
        s = bool(ty.signed)

        _, h_add = int_exact(ty, 1, operator.add, a, b)
        ok_add = (
            And(BVAddNoOverflow(a, b, True), BVAddNoUnderflow(a, b))
            if s
            else BVAddNoOverflow(a, b, False)
        )
        prove(f"halt(+) == Z3 overflow predicates {tyn}", h_add == Not(ok_add))

        _, h_sub = int_exact(ty, 1, operator.sub, a, b)
        ok_sub = (
            And(BVSubNoOverflow(a, b), BVSubNoUnderflow(a, b, True))
            if s
            else BVSubNoUnderflow(a, b, False)
        )
        prove(f"halt(-) == Z3 overflow predicates {tyn}", h_sub == Not(ok_sub))

        if s:
            # (negation of unsigned operands is a compile error, so the
            # encoding is only an operator semantics for signed types)
            _, h_neg = int_exact(ty, 1, operator.neg, a)
            prove(
                f"halt(unary -) == Z3 overflow predicate {tyn}",
                h_neg == Not(BVSNegNoOverflow(a)),
            )

        # and when it does not halt, the value is the plain wrapped op
        prove(
            f"value(+) == bvadd {tyn}", int_exact(ty, 1, operator.add, a, b)[0] == a + b
        )
        prove(
            f"value(*) == bvmul {tyn}",
            int_exact(ty, ty.bits, operator.mul, a, b)[0] == a * b,
        )

    for tyn in ("I8", "U8", "I16", "U16", "U32", "U64"):
        ty = TYPES[tyn]
        a = Const(f"ovfm_a_{tyn}", ty.sort)
        b = Const(f"ovfm_b_{tyn}", ty.sort)
        _, h_mul = int_exact(ty, ty.bits, operator.mul, a, b)
        ok_mul = (
            And(BVMulNoOverflow(a, b, True), BVMulNoUnderflow(a, b))
            if ty.signed
            else BVMulNoOverflow(a, b, False)
        )
        prove(f"halt(*) == Z3 overflow predicates {tyn}", h_mul == Not(ok_mul))


def check_int_divmod_props():
    """Floored // and % are each other's complements and % is SMT-LIB's
    floored bvsmod: // and % halt on exactly the same inputs, and wherever
    they are defined,
        a == (a // b) * b + (a % b),
    the remainder has the divisor's sign (or is 0), and |a % b| < |b|.

    Nonlinear division proofs are expensive, and the encoding is uniform in
    the width, so the div/mod identity is proved at 8 bits and the cheaper
    remainder properties through 16 bits (bvsmod also at 64); 64-bit // and
    % are cross-checked concretely against Python in
    check_int_ops_match_python."""
    a64 = Const("dm_a_I64", I64TY.sort)
    b64 = Const("dm_b_I64", I64TY.sort)
    r64, _ = int_mod(a64, b64, I64TY)
    prove("a%b == bvsmod I64", Implies(b64 != BitVecVal(0, 64), r64 == a64 % b64))
    for tyn in ("I8", "U8", "I16", "U16"):
        ty = TYPES[tyn]
        a = Const(f"dm_a_{tyn}", ty.sort)
        b = Const(f"dm_b_{tyn}", ty.sort)
        zero = BitVecVal(0, ty.bits)
        q, hq = int_floor_div(a, b, ty)
        r, hr = int_mod(a, b, ty)
        if ty.bits == 8:
            prove(f"a == (a//b)*b + a%b {tyn}", Implies(Not(hq), a == q * b + r))
        if ty.signed:
            # z3py maps % on signed bitvectors to SMT-LIB's floored bvsmod
            prove(f"a%b == bvsmod {tyn}", Implies(b != zero, r == a % b))
            prove(
                f"a%b sign and range {tyn}",
                Implies(
                    b != zero,
                    And(
                        Implies(b > zero, And(r >= zero, r < b)),
                        Implies(b < zero, And(r <= zero, r > b)),
                    ),
                ),
            )
        else:
            prove(f"a%b range {tyn}", Implies(b != zero, ULT(r, b)))
        prove(f"// and % halt together {tyn}", hq == hr)


def check_narrow_types_never_halt():
    """Widening to the 64-bit intermediate type protects narrow operands:
    +, -, * on (up to) 32-bit operands can never overflow the intermediate
    type, hence never halt. (Signed - unsigned mixes are compile errors, and
    U64/I64 operands genuinely can halt: see the spot checks.)"""
    for t1n, t2n in [("U8", "U32"), ("U32", "U32"), ("I16", "I32"), ("I32", "I32")]:
        t1, t2 = TYPES[t1n], TYPES[t2n]
        x = Const(f"nh_a_{t1n}_{t2n}", t1.sort)
        y = Const(f"nh_b_{t1n}_{t2n}", t2.sort)
        a, b = Value(t1, x), Value(t2, y)
        for name, fn in [("+", add), ("-", sub), ("*", mul)]:
            if name == "-" and not t1.signed:
                # unsigned subtraction CAN halt even for narrow types
                continue
            prove(f"{t1n} {name} {t2n} never halts", Not(fn(a, b).halted))
    # ... and unsigned narrow subtraction halts exactly when a < b
    t = TYPES["U8"]
    x = Const("nh_sub_a", t.sort)
    y = Const("nh_sub_b", t.sort)
    res = sub(Value(t, x), Value(t, y))
    prove("U8 - U8 halts iff a < b", res.halted == ULT(x, y))


def check_float_mod_props():
    """Sanity of the fmod construction: for finite a and nonzero finite b,
    fmod is zero or has the sign of a, and its magnitude is < |b|.

    The construction is sort-generic and fpRem proofs blow up already at
    F32, so this is proved on the 16-bit IEEE format; F64 behavior is
    cross-checked concretely against math.fmod in
    check_float_ops_match_python."""
    f16s = FPSort(5, 11)
    a = Const("fm_a", f16s)
    b = Const("fm_b", f16s)
    m = float_fmod_trunc(a, b)
    inputs_ok = And(
        Not(fpIsNaN(a)),
        Not(fpIsNaN(b)),
        fpLT(fpAbs(a), FPVal(float("inf"), f16s)),
        fpLT(fpAbs(b), FPVal(float("inf"), f16s)),
        Not(fpIsZero(b)),
    )
    prove(
        "fmod sign == sign(a) or zero (F16)",
        Implies(inputs_ok, Or(fpIsZero(m), Not(Xor(fpIsNegative(m), fpIsNegative(a))))),
    )
    prove("fmod magnitude < |b| (F16)", Implies(inputs_ok, fpLT(fpAbs(m), fpAbs(b))))


# --- concrete cross-checks vs Python ----------------------------------------------


def iv(x: int, tyn: str) -> Value:
    ty = TYPES[tyn]
    return Value(ty, BitVecVal(x, ty.bits))


def fv(x: float, tyn: str = "F64") -> Value:
    ty = TYPES[tyn]
    return Value(ty, FPVal(x, ty.sort))


def concrete_int(expr, ty: Ty) -> int:
    v = simplify(expr)
    return v.as_signed_long() if ty.signed else v.as_long()


def concrete_bool(expr) -> bool:
    v = simplify(expr)
    assert is_true(v) or is_false(v), v
    return is_true(v)


def fp_matches(term, expected: float) -> bool:
    """Concrete FP term equals the Python float (fpEQ, so +-0 agree; NaN
    matches NaN)."""
    if math.isnan(expected):
        return concrete_bool(fpIsNaN(term))
    return concrete_bool(fpEQ(term, FPVal(expected, term.sort())))


def check_int_ops_match_python():
    """The integer operators agree with Python's unbounded-int arithmetic:
    same value when the mathematical result is representable, halt exactly
    when it is not (//, % additionally halt per their divisor conditions)."""
    random.seed(0)
    py_ops = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "//": operator.floordiv,
        "%": operator.mod,
    }
    spec_ops = {"+": add, "-": sub, "*": mul, "//": floor_div, "%": mod}
    for tyn in ("I64", "U64"):
        ty = TYPES[tyn]
        corners = {ty.min, ty.min + 1, 0, 1, 2, ty.max - 1, ty.max}
        if ty.signed:
            corners |= {-1, -2}
        samples = sorted(corners) + [
            random.randrange(ty.min, ty.max + 1) for _ in range(28)
        ]
        pairs = [(x, y) for x in samples for y in samples[:12]]
        for op_name, py_fn in py_ops.items():
            bad = None
            for x, y in pairs:
                res = spec_ops[op_name](iv(x, tyn), iv(y, tyn))
                halted = concrete_bool(res.halted)
                if op_name in ("//", "%"):
                    # // and % halt together: zero divisor, or MIN op -1
                    want_halt = y == 0 or (ty.signed and x == ty.min and y == -1)
                else:
                    exact = py_fn(x, y)
                    want_halt = not (ty.min <= exact <= ty.max)
                if halted != want_halt:
                    bad = f"{x} {op_name} {y}: halted={halted}, want {want_halt}"
                    break
                if not want_halt:
                    got = concrete_int(res.value.expr, ty)
                    want = py_fn(x, y)
                    if got != want:
                        bad = f"{x} {op_name} {y}: {got}, python says {want}"
                        break
            if bad:
                record(FAIL, f"{tyn} {op_name} == python", bad)
            else:
                record(
                    PASS, f"{tyn} {op_name} == python", f"{len(pairs)} sampled pairs"
                )


def check_float_ops_match_python():
    """The float operators agree with CPython's float arithmetic (which is
    the host's IEEE-754 double) on sampled values, including // and the
    fmod-based %."""
    random.seed(1)
    interesting = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        0.5,
        -0.5,
        7.5,
        -7.5,
        1e300,
        -1e300,
        1e-300,
        -1e-300,
        2.0**53,
        -(2.0**53),
        float("inf"),
        -float("inf"),
        float("nan"),
        3.141592653589793,
        -2.718281828459045,
    ]
    samples = interesting + [random.uniform(-1e6, 1e6) for _ in range(12)]

    def py_floor_div(x, y):
        # the spec's formula computed in host IEEE arithmetic: ONE division,
        # then floor. (CPython's own float // is fmod-based and can differ
        # in the last ulp, so it is not the reference here.)
        try:
            q = x / y
        except ZeroDivisionError:
            if x == 0.0 or math.isnan(x):
                return float("nan")
            return math.copysign(float("inf"), x) * math.copysign(1.0, y)
        if math.isnan(q) or math.isinf(q):
            return q
        return float(math.floor(q))

    def py_mod(x, y):
        # CPython float_rem: fmod, then add the divisor back on sign
        # mismatch (we keep fmod's zero sign where CPython copysigns it to
        # y, but fpEQ treats +-0 as equal so the comparison still holds)
        if y == 0.0 or math.isnan(x) or math.isnan(y) or math.isinf(x):
            return float("nan")
        m = math.fmod(x, y)  # fmod(finite, +-inf) == x, matching the spec
        if m != 0.0 and (m < 0.0) != (y < 0.0):
            m = m + y
        return m

    checks = [
        ("+", add, operator.add),
        ("-", sub, operator.sub),
        ("*", mul, operator.mul),
        ("/", div, None),
        ("//", floor_div, py_floor_div),
        ("%", mod, py_mod),
    ]
    for op_name, spec_fn, py_fn in checks:
        bad = None
        for x in samples:
            for y in samples[:14]:
                if py_fn is None:  # division: python raises on /0
                    if y == 0.0:
                        want = (
                            float("nan")
                            if (x == 0.0 or math.isnan(x))
                            else math.copysign(float("inf"), x) * math.copysign(1.0, y)
                        )
                    else:
                        want = x / y
                else:
                    try:
                        want = py_fn(x, y)
                    except (ZeroDivisionError, ValueError):
                        want = float("nan")
                res = spec_fn(fv(x), fv(y))
                assert concrete_bool(Not(res.halted))
                if not fp_matches(res.value.expr, want):
                    bad = f"{x!r} {op_name} {y!r}: want {want!r}, got {simplify(res.value.expr)}"
                    break
            if bad:
                break
        if bad:
            record(FAIL, f"F64 {op_name} == python", bad)
        else:
            record(
                PASS, f"F64 {op_name} == python", f"{len(samples) * 14} sampled pairs"
            )

    # fmod itself against math.fmod
    bad = None
    for x in samples:
        for y in samples[:14]:
            try:
                want = math.fmod(x, y)
            except ValueError:
                want = float("nan")
            if not fp_matches(float_fmod_trunc(fv(x).expr, fv(y).expr), want):
                bad = f"fmod({x!r}, {y!r}) != {want!r}"
                break
        if bad:
            break
    record(
        FAIL if bad else PASS, "float_fmod_trunc == math.fmod", bad or "sampled pairs"
    )


def check_spot_semantics():
    """Hand-picked cases that pin down the deliberate design decisions."""
    spots_int = [
        # floored division and modulo (Python, not C, semantics)
        (floor_div(iv(-7, "I64"), iv(2, "I64")), -4, False, "-7 // 2 == -4"),
        (floor_div(iv(7, "I64"), iv(-2, "I64")), -4, False, "7 // -2 == -4"),
        (mod(iv(-7, "I64"), iv(2, "I64")), 1, False, "-7 % 2 == 1"),
        (mod(iv(7, "I64"), iv(-2, "I64")), -1, False, "7 % -2 == -1"),
        # OQ-1: overflow is a halt, not a wrap
        (
            add(iv(TYPES["I64"].max, "I64"), iv(1, "I64")),
            None,
            True,
            "I64_MAX + 1 halts",
        ),
        (sub(iv(0, "U64"), iv(1, "U64")), None, True, "U64 0 - 1 halts"),
        (
            mul(iv(1 << 32, "U64"), iv(1 << 32, "U64")),
            None,
            True,
            "U64 2^32 * 2^32 halts",
        ),
        (
            floor_div(iv(TYPES["I64"].min, "I64"), iv(-1, "I64")),
            None,
            True,
            "I64_MIN // -1 halts",
        ),
        (
            mod(iv(TYPES["I64"].min, "I64"), iv(-1, "I64")),
            None,
            True,
            "I64_MIN % -1 halts (like //, like Rust)",
        ),
        (floor_div(iv(1, "I64"), iv(0, "I64")), None, True, "1 // 0 halts"),
        (mod(iv(1, "U64"), iv(0, "U64")), None, True, "1 % 0 halts"),
        (neg(iv(TYPES["I64"].min, "I64")), None, True, "-I64_MIN halts"),
        # widening: narrow ops that would wrap in their own width don't halt
        (add(iv(200, "U8"), iv(100, "U8")), 300, False, "U8 200 + 100 == 300 (in U64)"),
        (
            mul(iv(-(2**31), "I32"), iv(2, "I32")),
            -(2**32),
            False,
            "I32 min * 2 ok in I64",
        ),
    ]
    for res, want_val, want_halt, label in spots_int:
        halted = concrete_bool(res.halted)
        ok = halted == want_halt and (
            want_halt or concrete_int(res.value.expr, res.value.ty) == want_val
        )
        record(
            PASS if ok else FAIL,
            f"spot {label}",
            "ok" if ok else f"halted={halted}, value={simplify(res.value.expr)}",
        )

    # negation of an unsigned operand is rejected at the type level (as in
    # Rust), not turned into a runtime halt
    try:
        neg(iv(5, "U32"))
        record(FAIL, "spot -U32(5) is a compile error", "spec accepted it")
    except AssertionError:
        record(PASS, "spot -U32(5) is a compile error", "rejected at type level")

    spots_float = [
        # / never halts; IEEE specials
        (
            div(iv(1, "I64"), iv(0, "I64")),
            float("inf"),
            "1 / 0 == +inf (computed in F64)",
        ),
        (div(fv(0.0), fv(0.0)), float("nan"), "0.0 / 0.0 == NaN"),
        (div(fv(-1.0), fv(0.0)), -float("inf"), "-1.0 / 0.0 == -inf"),
        # float floor div floors toward -inf
        (floor_div(fv(-7.0), fv(2.0)), -4.0, "-7.0 // 2.0 == -4.0"),
        (floor_div(fv(7.5), fv(2.0)), 3.0, "7.5 // 2.0 == 3.0"),
        # floored float modulo
        (mod(fv(7.5), fv(2.0)), 1.5, "7.5 % 2.0 == 1.5"),
        (mod(fv(-7.5), fv(2.0)), 0.5, "-7.5 % 2.0 == 0.5"),
        (mod(fv(1.0), fv(0.0)), float("nan"), "1.0 % 0.0 == NaN (no halt)"),
        # int operands of / are rounded to F64 first (coercion rounds)
        (
            div(iv((1 << 62) + 1, "I64"), iv(1, "I64")),
            float((1 << 62) + 1),
            "big I64 / 1 rounds",
        ),
    ]
    for res, want, label in spots_float:
        ok = concrete_bool(Not(res.halted)) and fp_matches(res.value.expr, want)
        record(
            PASS if ok else FAIL,
            f"spot {label}",
            "ok" if ok else f"got {simplify(res.value.expr)}",
        )

    nan = Value(F64TY, fpNaN(F64S))
    spots_bool = [
        (apply_binary(BinaryStackOp.EQUAL, nan, nan), False, "NaN == NaN is False"),
        (
            apply_binary(BinaryStackOp.NOT_EQUAL, nan, nan),
            True,
            "NaN != NaN is True (OQ-3)",
        ),
        (
            apply_binary(BinaryStackOp.LESS_THAN, nan, fv(1.0)),
            False,
            "NaN < 1.0 is False",
        ),
        (
            apply_binary(BinaryStackOp.GREATER_THAN_OR_EQUAL, nan, fv(1.0)),
            False,
            "NaN >= 1.0 is False",
        ),
        # mixed-width comparison happens at the common widened type
        (
            apply_binary(BinaryStackOp.LESS_THAN, iv(-1, "I8"), iv(1, "I64")),
            True,
            "I8 -1 < I64 1",
        ),
        (
            apply_binary(BinaryStackOp.EQUAL, iv(255, "U8"), iv(255, "U64")),
            True,
            "U8 255 == U64 255",
        ),
    ]
    for res, want, label in spots_bool:
        ok = concrete_bool(res.value.expr) == want
        record(
            PASS if ok else FAIL, f"spot {label}", "ok" if ok else "wrong truth value"
        )


def main():
    print("fpy arithmetic operator specification checks (see MATH.md)\n")
    check_intermediate_type_matches_compiler()
    check_well_formed()
    check_int_overflow_encoding()
    check_int_divmod_props()
    check_narrow_types_never_halt()
    check_float_mod_props()
    check_int_ops_match_python()
    check_float_ops_match_python()
    check_spot_semantics()

    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s not in (PASS, FAIL))
    print(f"\n{n_pass} passed, {n_fail} failed, {n_warn} unknown/timeout")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
