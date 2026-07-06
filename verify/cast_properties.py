#!/usr/bin/env python3
"""Mechanized consistency checks for the fpy cast specification.

Spec (MATH_CASTS_DRAFT.md): a cast is a total function
cast_{S->T} : [[S]] -> [[T]] defined by four equations:

    int   -> int    wrap_T(x)
    int   -> float  round_T(x)              (single direct RNE rounding)
    float -> float  round_T([[x]])          (sign of zero preserved)
    float -> int    clamp_T(trunc([[x]]))   (saturating; NaN -> 0)

Each equation is encoded as a quantifier-free Z3 term over the SMT bitvector
and IEEE-754 floating-point theories. Totality and determinism (theorem T1)
hold *by construction*: every Z3 expression tree denotes exactly one value of
the result sort for every input, and the SMT FP sorts have a single NaN,
matching the spec's canonical-NaN policy. The checks below verify the
remaining theorems (T2-T7 in the draft).

GOTCHA encoded here deliberately: SMT-LIB's fp.to_sbv/fp.to_ubv are
*underspecified* for NaN and out-of-range inputs, so the spec functions guard
them and only apply them on their defined domain. The "definedness" checks
prove those guards are sufficient.

Run:
    uv run --with z3-solver python verify/cast_properties.py
"""

import sys
import time
from dataclasses import dataclass

from z3 import (
    RNE,
    RTZ,
    And,
    BitVecSort,
    BitVecVal,
    Const,
    Extract,
    FPSort,
    FPVal,
    If,
    Implies,
    Not,
    SignExt,
    Solver,
    ULE,
    ZeroExt,
    substitute,
    fpGEQ,
    fpIsNaN,
    fpLEQ,
    fpLT,
    fpMinusZero,
    fpNaN,
    fpPlusInfinity,
    fpMinusInfinity,
    fpPlusZero,
    fpRoundToIntegral,
    fpSignedToFP,
    fpToFP,
    fpToSBV,
    fpToUBV,
    fpUnsignedToFP,
    sat,
    simplify,
    unsat,
)

F32S = FPSort(8, 24)
F64S = FPSort(11, 53)


@dataclass(frozen=True)
class Ty:
    name: str
    kind: str  # 'int' | 'float'
    bits: int
    signed: bool | None  # None for floats

    @property
    def sort(self):
        if self.kind == "int":
            return BitVecSort(self.bits)
        return F32S if self.bits == 32 else F64S

    @property
    def min(self) -> int:
        assert self.kind == "int"
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def max(self) -> int:
        assert self.kind == "int"
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1


TYPES = {
    t.name: t
    for t in [
        Ty("U8", "int", 8, False),
        Ty("U16", "int", 16, False),
        Ty("U32", "int", 32, False),
        Ty("U64", "int", 64, False),
        Ty("I8", "int", 8, True),
        Ty("I16", "int", 16, True),
        Ty("I32", "int", 32, True),
        Ty("I64", "int", 64, True),
        Ty("F32", "float", 32, None),
        Ty("F64", "float", 64, None),
    ]
}
INT_TYPES = [t for t in TYPES.values() if t.kind == "int"]
FLOAT_TYPES = [t for t in TYPES.values() if t.kind == "float"]


# --- the four spec equations -------------------------------------------------


def wrap_int_to_int(x, frm: Ty, to: Ty):
    """cast(x, int S, int T) = wrap_T(x).

    Structural encoding: extract low bits (narrowing) / extend by the
    *source* signedness (widening). check_wrap_matches_mod proves this equals
    the mathematical definition r = x mod 2^n.
    """
    if to.bits == frm.bits:
        return x
    if to.bits < frm.bits:
        return Extract(to.bits - 1, 0, x)
    ext = SignExt if frm.signed else ZeroExt
    return ext(to.bits - frm.bits, x)


def int_to_float(x, frm: Ty, to: Ty):
    """cast(x, int S, float T) = round_T(x): one direct RNE rounding."""
    conv = fpSignedToFP if frm.signed else fpUnsignedToFP
    return conv(RNE(), x, to.sort)


def float_to_float(x, frm: Ty, to: Ty):
    """cast(x, float S, float T) = round_T([[x]]), sign of zero preserved.

    SMT fp.to_fp on FP input is IEEE convertFormat: it preserves the sign of
    zero and of underflowed results, and maps NaN to the sort's single NaN --
    exactly the spec including the sign-of-zero clause (checked below).
    """
    if frm.name == to.name:
        return x
    return fpToFP(RNE(), x, to.sort)


def float_to_int(x, frm: Ty, to: Ty):
    """cast(x, float S, int T) = clamp_T(trunc([[x]])).

    fp.to_sbv/fp.to_ubv are only applied when the RTZ-rounded value provably
    fits in T (see check_tosbv_guards_sufficient); all other inputs are
    resolved by the clamp arms. The float bounds used in the guards
    (+-2^(n-1), 2^n) are powers of two, exactly representable in both F32 and
    F64 for n <= 64.
    """
    n = to.bits
    zero = BitVecVal(0, n)
    if to.signed:
        lo_f = FPVal(-(2.0 ** (n - 1)), frm.sort)
        hi_f = FPVal(2.0 ** (n - 1), frm.sort)
        return If(
            fpIsNaN(x),
            zero,
            If(
                fpLT(x, lo_f),
                BitVecVal(to.min, n),
                If(
                    fpGEQ(x, hi_f),
                    BitVecVal(to.max, n),
                    fpToSBV(RTZ(), x, BitVecSort(n)),
                ),
            ),
        )
    hi_f = FPVal(2.0**n, frm.sort)
    return If(
        fpIsNaN(x),
        zero,
        If(
            fpLT(x, FPVal(0.0, frm.sort)),
            zero,  # trunc of anything in (-inf, -0] clamps/truncates to 0
            If(fpGEQ(x, hi_f), BitVecVal(to.max, n), fpToUBV(RTZ(), x, BitVecSort(n))),
        ),
    )


def cast(x, frm: Ty, to: Ty):
    if frm.kind == "int" and to.kind == "int":
        return wrap_int_to_int(x, frm, to)
    if frm.kind == "int" and to.kind == "float":
        return int_to_float(x, frm, to)
    if frm.kind == "float" and to.kind == "float":
        return float_to_float(x, frm, to)
    return float_to_int(x, frm, to)


# --- check runner ------------------------------------------------------------

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def record(status, name, detail):
    results.append((status, name, detail))
    print(f"[{status}] {name}: {detail}", flush=True)


def prove(name, claim, timeout_ms=120_000):
    """Expect `claim` valid (negation unsat)."""
    s = Solver()
    s.set("timeout", timeout_ms)
    s.add(Not(claim))
    t0 = time.time()
    r = s.check()
    dt = time.time() - t0
    if r == unsat:
        record(PASS, name, f"proved ({dt:.1f}s)")
    elif r == sat:
        record(FAIL, name, f"COUNTEREXAMPLE: {s.model()}")
    else:
        record(WARN, name, f"unknown/timeout after {dt:.1f}s")


def find(name, condition, witness_vars, timeout_ms=120_000):
    """Expect `condition` satisfiable (a witness exists)."""
    s = Solver()
    s.set("timeout", timeout_ms)
    s.add(condition)
    t0 = time.time()
    r = s.check()
    dt = time.time() - t0
    if r == sat:
        m = s.model()
        vals = ", ".join(
            f"{v} = {m.eval(v, model_completion=True)}" for v in witness_vars
        )
        record(PASS, name, f"witness found ({dt:.1f}s): {vals}")
    elif r == unsat:
        record(FAIL, name, "expected a witness but none exists")
    else:
        record(WARN, name, f"unknown/timeout after {dt:.1f}s")


# --- T1: totality / determinism ----------------------------------------------


def check_totality_note():
    record(
        PASS,
        "T1 totality+determinism (all 100 pairs)",
        "per-model: by construction (quantifier-free terms of total ops; SMT FP "
        "sorts have one NaN = canonical-NaN policy). Across models: mechanized "
        "below (well-formedness, guard sufficiency, model-independence)",
    )


def check_well_formed():
    """T1 mechanized, part 1: every pair constructs a closed function of x.

    For all 100 (S, T) pairs, cast(x, S, T) must (a) build without error --
    the case table is exhaustive, (b) have sort [[T]], and (c) mention no
    free variable other than x. (c) rules out the builder accidentally
    capturing a constant from another scope, which would make the "function"
    depend on a hidden second input.
    """
    from z3.z3util import get_vars

    bad = []
    for frm in TYPES.values():
        for to in TYPES.values():
            x = Const(f"wf_{frm.name}_{to.name}", frm.sort)
            term = cast(x, frm, to)
            if term.sort() != to.sort:
                bad.append(f"{frm.name}->{to.name}: sort {term.sort()}")
                continue
            fv = get_vars(term)
            if not (len(fv) == 1 and fv[0].eq(x)):
                bad.append(f"{frm.name}->{to.name}: free vars {fv}")
    if bad:
        record(FAIL, "T1 well-formed terms (all 100 pairs)", "; ".join(bad))
    else:
        record(
            PASS,
            "T1 well-formed terms (all 100 pairs)",
            "each term built, sorted [[T]], free vars exactly {x}",
        )


def check_model_independence():
    """T1 mechanized, part 2: the spec is ONE function, not one per Z3 model.

    fp.to_sbv/fp.to_ubv are total in every model but SMT-LIB only pins their
    value where the RTZ-rounded input is representable; elsewhere each model
    may answer differently. Determinism of the spec therefore means: the value
    of the cast term never depends on that unpinned region.

    Proof, direct form: take the actual float->int term and substitute the
    fp.to_sbv(x) occurrence with two distinct fresh constants g1, g2 standing
    for the answers of two arbitrary models, constrained to agree with
    fp.to_sbv only on the pinned (representable) region. If the two resulting
    terms are provably equal, every model computes the same function.
    """
    for f in FLOAT_TYPES:
        for t in INT_TYPES:
            x = Const(f"mi_{f.name}_{t.name}", f.sort)
            n = t.bits
            term = cast(x, f, t)
            conv = fpToSBV if t.signed else fpToUBV
            app = conv(RTZ(), x, BitVecSort(n))
            g1 = Const(f"mi_g1_{f.name}_{t.name}", BitVecSort(n))
            g2 = Const(f"mi_g2_{f.name}_{t.name}", BitVecSort(n))
            t1 = substitute(term, (app, g1))
            t2 = substitute(term, (app, g2))
            assert not t1.eq(t2), "substitution did not fire; wrong app term?"

            # SMT-LIB pins fp.to_sbv/to_ubv exactly when the value obtained by
            # rounding toward zero is representable in the target type.
            rounded = fpRoundToIntegral(RTZ(), x)
            if t.signed:
                lo_f = FPVal(-(2.0 ** (n - 1)), f.sort)
                hi_f = FPVal(2.0 ** (n - 1), f.sort)
                pinned_region = And(
                    Not(fpIsNaN(x)), fpGEQ(rounded, lo_f), fpLT(rounded, hi_f)
                )
            else:
                pinned_region = And(
                    Not(fpIsNaN(x)),
                    fpGEQ(rounded, FPVal(0.0, f.sort)),
                    fpLT(rounded, FPVal(2.0**n, f.sort)),
                )
            pinned = Implies(pinned_region, And(g1 == app, g2 == app))
            prove(
                f"T1 model-independence {f.name}->{t.name}", Implies(pinned, t1 == t2)
            )


def check_tosbv_guards_sufficient():
    """The in-range branch never feeds fp.to_sbv/to_ubv an undefined input:
    if the guards pass, the RTZ-rounded value provably lies in T's range."""
    for f in FLOAT_TYPES:
        for t in INT_TYPES:
            x = Const(f"x_{f.name}_{t.name}", f.sort)
            n = t.bits
            rounded = fpRoundToIntegral(RTZ(), x)
            if t.signed:
                lo_f = FPVal(-(2.0 ** (n - 1)), f.sort)
                hi_f = FPVal(2.0 ** (n - 1), f.sort)
                guards = And(Not(fpIsNaN(x)), Not(fpLT(x, lo_f)), Not(fpGEQ(x, hi_f)))
                in_range = And(fpGEQ(rounded, lo_f), fpLT(rounded, hi_f))
            else:
                hi_f = FPVal(2.0**n, f.sort)
                guards = And(
                    Not(fpIsNaN(x)),
                    Not(fpLT(x, FPVal(0.0, f.sort))),
                    Not(fpGEQ(x, hi_f)),
                )
                in_range = And(fpGEQ(rounded, FPVal(0.0, f.sort)), fpLT(rounded, hi_f))
            prove(
                f"T1 to_sbv/to_ubv guard sufficient {f.name}->{t.name}",
                Implies(guards, in_range),
            )


# --- T2: identity is by construction (cast returns x for S == T) --------------


def check_identity_note():
    record(
        PASS,
        "T2 identity casts (all 10 types)",
        "by construction: cast(x, T, T) is literally x",
    )


# --- T4: saturation boundaries -----------------------------------------------


def check_saturation_boundaries():
    for f in FLOAT_TYPES:
        for t in INT_TYPES:
            claims = [
                cast(fpNaN(f.sort), f, t) == BitVecVal(0, t.bits),
                cast(fpPlusInfinity(f.sort), f, t) == BitVecVal(t.max, t.bits),
                cast(fpMinusInfinity(f.sort), f, t) == BitVecVal(t.min, t.bits),
                cast(fpPlusZero(f.sort), f, t) == BitVecVal(0, t.bits),
                cast(fpMinusZero(f.sort), f, t) == BitVecVal(0, t.bits),
            ]
            prove(f"T4 NaN/inf/zero boundaries {f.name}->{t.name}", And(*claims))

    f64, f32 = TYPES["F64"], TYPES["F32"]
    spot = [
        # the LLVM-vs-bytecode divergence example from MATH_TODO.txt:
        (
            cast(FPVal(300.0, F64S), f64, TYPES["I8"]) == BitVecVal(127, 8),
            "I8(300.0) == 127",
        ),
        (
            cast(FPVal(-300.0, F64S), f64, TYPES["I8"]) == BitVecVal(-128, 8),
            "I8(-300.0) == -128",
        ),
        (
            cast(FPVal(300.0, F64S), f64, TYPES["U8"]) == BitVecVal(255, 8),
            "U8(300.0) == 255",
        ),
        (cast(FPVal(-3.0, F64S), f64, TYPES["U8"]) == BitVecVal(0, 8), "U8(-3.0) == 0"),
        (cast(FPVal(-0.5, F64S), f64, TYPES["U8"]) == BitVecVal(0, 8), "U8(-0.5) == 0"),
        (cast(FPVal(-1.0, F64S), f64, TYPES["U8"]) == BitVecVal(0, 8), "U8(-1.0) == 0"),
        (
            cast(FPVal(126.7, F64S), f64, TYPES["I8"]) == BitVecVal(126, 8),
            "I8(126.7) == 126 (trunc)",
        ),
        (
            cast(FPVal(-126.7, F64S), f64, TYPES["I8"]) == BitVecVal(-126, 8),
            "I8(-126.7) == -126 (trunc toward 0)",
        ),
        # 2^63 is exactly representable in F64 but out of I64 range -> saturate:
        (
            cast(FPVal(2.0**63, F64S), f64, TYPES["I64"])
            == BitVecVal((1 << 63) - 1, 64),
            "I64(2^63) == I64_MAX",
        ),
        (
            cast(FPVal(2.0**63, F32S), f32, TYPES["I64"])
            == BitVecVal((1 << 63) - 1, 64),
            "I64(F32 2^63) == I64_MAX",
        ),
        (
            cast(FPVal(1e300, F64S), f64, TYPES["U64"]) == BitVecVal((1 << 64) - 1, 64),
            "U64(1e300) == U64_MAX",
        ),
    ]
    for claim, label in spot:
        prove(f"T4 spot {label}", claim)


# --- T5: monotonicity ----------------------------------------------------------


def check_monotonicity():
    mono_pairs = [
        ("F32", "I8"),
        ("F32", "U8"),
        ("F32", "I32"),
        ("F32", "U32"),
        ("F64", "I16"),
        ("F64", "I64"),
        ("F64", "U64"),
    ]
    for fn, tn in mono_pairs:
        f, t = TYPES[fn], TYPES[tn]
        x = Const(f"mx_{fn}_{tn}", f.sort)
        y = Const(f"my_{fn}_{tn}", f.sort)
        le = (lambda a, b: a <= b) if t.signed else ULE
        claim = Implies(
            And(Not(fpIsNaN(x)), Not(fpIsNaN(y)), fpLEQ(x, y)),
            le(cast(x, f, t), cast(y, f, t)),
        )
        prove(f"T5 monotone float->int {fn}->{tn}", claim)

    for fn, tn in [("I64", "F32"), ("U64", "F32"), ("I32", "F64")]:
        f, t = TYPES[fn], TYPES[tn]
        x = Const(f"nx_{fn}_{tn}", f.sort)
        y = Const(f"ny_{fn}_{tn}", f.sort)
        hyp = (x <= y) if f.signed else ULE(x, y)
        claim = Implies(hyp, fpLEQ(cast(x, f, t), cast(y, f, t)))
        prove(f"T5 monotone int->float {fn}->{tn}", claim)


# --- T3/T6: exactness and round trips ------------------------------------------


def check_round_trips():
    # S -> F -> S must be the identity when S's values all fit in F's mantissa
    for sn, fn in [
        ("I8", "F32"),
        ("I16", "F32"),
        ("U16", "F32"),
        ("I32", "F64"),
        ("U32", "F64"),
    ]:
        s, f = TYPES[sn], TYPES[fn]
        v = Const(f"rt_{sn}_{fn}", s.sort)
        prove(f"T6 round trip {sn}->{fn}->{sn} == id", cast(cast(v, s, f), f, s) == v)

    # ... and must NOT be the identity when it doesn't fit
    for sn, fn in [("I32", "F32"), ("I64", "F64"), ("U64", "F64")]:
        s, f = TYPES[sn], TYPES[fn]
        v = Const(f"nrt_{sn}_{fn}", s.sort)
        find(
            f"T6 round trip {sn}->{fn}->{sn} != id (expected counterexample)",
            cast(cast(v, s, f), f, s) != v,
            [v],
        )

    # float widening is exact: F32 -> F64 -> F32 == id for ALL values,
    # including NaN and both zeros (SMT `=` distinguishes +0 from -0,
    # so this also proves the sign-of-zero clause for widening)
    f32, f64 = TYPES["F32"], TYPES["F64"]
    x = Const("wide_x", F32S)
    prove(
        "T3 F32->F64->F32 == id (incl NaN, +-0)", cast(cast(x, f32, f64), f64, f32) == x
    )


def check_sign_of_zero_narrowing():
    f64, f32 = TYPES["F64"], TYPES["F32"]
    claims = [
        (
            cast(fpMinusZero(F64S), f64, f32) == fpMinusZero(F32S),
            "F32(-0.0 F64) == -0.0",
        ),
        (cast(fpPlusZero(F64S), f64, f32) == fpPlusZero(F32S), "F32(+0.0 F64) == +0.0"),
        # narrowing underflow keeps the sign: -1e-300 is far below F32's
        # smallest subnormal, RNE rounds it to zero -- must be MINUS zero
        (
            cast(FPVal(-1e-300, F64S), f64, f32) == fpMinusZero(F32S),
            "F32(-1e-300) == -0.0",
        ),
        # narrowing overflow goes to infinity, NOT max-finite (resolves the
        # question mark in the old MATH.md draft)
        (
            cast(FPVal(1e300, F64S), f64, f32) == fpPlusInfinity(F32S),
            "F32(1e300) == +inf",
        ),
        (
            cast(FPVal(-1e300, F64S), f64, f32) == fpMinusInfinity(F32S),
            "F32(-1e300) == -inf",
        ),
    ]
    for claim, label in claims:
        prove(f"T4/T3 narrowing {label}", claim)


# --- wrap == mod definition -----------------------------------------------------


WRAP_PAIRS = [
    ("I8", "U8"),
    ("U8", "I8"),
    ("I64", "U8"),
    ("I8", "I64"),
    ("U8", "I64"),
    ("I32", "I64"),
    ("U32", "U16"),
    ("I64", "U64"),
    ("U64", "I8"),
    ("I16", "U32"),
]


def check_wrap_matches_mod():
    """The structural ext/extract encoding of int->int equals the spec's
    mathematical definition wrap_T(z) = z mod 2^n.

    NOTE: the direct encoding Int2BV(BV2Int(x, signed), n) makes Z3 diverge
    for every signed-source widening pair (even I8->I64), so the mod-2^n
    semantics is checked without the integer theory, two ways:
      (a) universally, against the value embedded in a 128-bit ring, where
          taking mod 2^n is Extract of the low n bits;
      (b) concretely, evaluating the actual Z3 term against plain Python
          integer arithmetic -- exhaustive for 8-bit sources, corners plus
          random samples for wider sources.
    """
    for fn, tn in WRAP_PAIRS:
        f, t = TYPES[fn], TYPES[tn]
        x = Const(f"w_{fn}_{tn}", f.sort)
        ext = SignExt if f.signed else ZeroExt
        embedded = ext(128 - f.bits, x)  # two's complement value of x, mod 2^128
        mathematical = Extract(t.bits - 1, 0, embedded)  # ... mod 2^n
        prove(
            f"wrap == mod 2^n in BV ring {fn}->{tn}",
            wrap_int_to_int(x, f, t) == mathematical,
        )


def py_wrap(z: int, to: Ty) -> int:
    """The spec definition in plain Python: unique r in [[T]] with r == z (mod 2^n)."""
    r = z % (1 << to.bits)
    if to.signed and r >= 1 << (to.bits - 1):
        r -= 1 << to.bits
    return r


def check_wrap_matches_mod_concrete():
    import random

    random.seed(0)
    for fn, tn in WRAP_PAIRS:
        f, t = TYPES[fn], TYPES[tn]
        if f.bits == 8:
            samples = list(range(f.min, f.max + 1))
            how = "exhaustive"
        else:
            samples = {f.min, f.min + 1, -1 if f.signed else 2, 0, 1, f.max - 1, f.max}
            samples |= {random.randrange(f.min, f.max + 1) for _ in range(4096)}
            samples = sorted(samples)
            how = f"{len(samples)} samples"
        bad = None
        for z in samples:
            term = simplify(wrap_int_to_int(BitVecVal(z, f.bits), f, t))
            got = term.as_signed_long() if t.signed else term.as_long()
            want = py_wrap(z, t)
            if got != want:
                bad = (z, got, want)
                break
        if bad is None:
            record(PASS, f"wrap == mod 2^n concrete {fn}->{tn}", how)
        else:
            record(
                FAIL,
                f"wrap == mod 2^n concrete {fn}->{tn}",
                f"z={bad[0]}: term gives {bad[1]}, spec says {bad[2]}",
            )


# --- T7: double rounding ---------------------------------------------------------


def check_double_rounding():
    f32, f64 = TYPES["F32"], TYPES["F64"]
    # A witness must exist that direct U64->F32 differs from U64->F64->F32
    # (this is the sanity check that the harness can see double rounding at all)
    u64 = TYPES["U64"]
    v = Const("dr_v", u64.sort)
    direct = int_to_float(v, u64, f32)
    via_f64 = float_to_float(int_to_float(v, u64, f64), f64, f32)
    find(
        "T7 direct U64->F32 != U64->F64->F32 (expected witness)", direct != via_f64, [v]
    )

    # ... but for I32 sources the two coincide (I32 -> F64 is exact,
    # so only one true rounding happens)
    i32 = TYPES["I32"]
    w = Const("dr_w", i32.sort)
    direct32 = int_to_float(w, i32, f32)
    via32 = float_to_float(int_to_float(w, i32, f64), f64, f32)
    prove("T7 direct I32->F32 == I32->F64->F32", direct32 == via32)


def main():
    print("fpy cast specification checks (see MATH_CASTS_DRAFT.md)\n")
    check_totality_note()
    check_identity_note()
    check_well_formed()
    check_tosbv_guards_sufficient()
    check_model_independence()
    check_saturation_boundaries()
    check_sign_of_zero_narrowing()
    check_round_trips()
    check_wrap_matches_mod()
    check_wrap_matches_mod_concrete()
    check_double_rounding()
    check_monotonicity()

    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    print(f"\n{n_pass} passed, {n_fail} failed, {n_warn} unknown/timeout")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
