# Casts — draft normative section for MATH.md

> Draft to be merged into MATH.md. Mechanized checks: `verify/cast_properties.py`.

## Value sets

For each numeric type `T`, the **value set** `⟦T⟧` is the set of values an
expression of type `T` may evaluate to.

For `n ∈ {8, 16, 32, 64}`:

* `⟦Un⟧ = { z ∈ ℤ | 0 ≤ z ≤ 2ⁿ − 1 }`
* `⟦In⟧ = { z ∈ ℤ | −2ⁿ⁻¹ ≤ z ≤ 2ⁿ⁻¹ − 1 }`

For `n ∈ {32, 64}`, `⟦Fn⟧` is the set of IEEE 754 binary-`n` data:

* all finite values, including the two zeros `+0` and `−0`,
* the two infinities `+∞` and `−∞`,
* a single value `NaN`.

Fpy does not distinguish NaN payloads and does not distinguish quiet from
signaling NaNs. Every operation that produces a NaN produces the **canonical
NaN** (sign bit 0, quiet bit set, payload 0); a NaN *input* may have any
encoding, and all encodings denote the single value `NaN`.

> Note: canonical-NaN output matters because struct equality compares
> serialized bytes, so NaN payloads would otherwise be observable through
> `==` on aggregates even though float `==` is numeric.

## Denotation

The **denotation** `⟦x⟧` of a value `x` is the mathematical object it
represents. Let `ℝ* = ℝ ∪ {+∞, −∞, NaN}`.

* If `x` is a value of an integer type: `⟦x⟧ = x ∈ ℤ`.
* If `x` is a finite float value: `⟦x⟧ ∈ ℝ` is the real number assigned by
  IEEE 754. In particular `⟦+0⟧ = ⟦−0⟧ = 0` (the denotation is not injective
  at zero; see the sign-of-zero clause below).
* `⟦+∞⟧ = +∞`, `⟦−∞⟧ = −∞`, `⟦NaN⟧ = NaN`, as elements of `ℝ*`.

## Primitive conversion functions

All casts are composed from four primitives. Each is **total** on its stated
domain; totality of every cast follows.

**`round_T : ℝ* → ⟦T⟧`** for float `T` (IEEE 754 `convertFormat` /
`convertFromInt` with rounding attribute `roundTiesToEven`):

1. `round_T(NaN) = NaN`
2. `round_T(+∞) = +∞`, `round_T(−∞) = −∞`
3. `round_T(0) = +0`
4. For nonzero real `r`: the value of `⟦T⟧` nearest to `r`, ties to the one
   with even least significant mantissa digit. If `|r|` exceeds the IEEE
   overflow threshold for `T`, the result is `+∞` or `−∞` with the sign of
   `r` (**not** the largest finite value). If `r` rounds to zero, the result
   is `+0` or `−0` with the sign of `r`.

**`trunc : ℝ* → ℤ ∪ {+∞, −∞, NaN}`** (round toward zero):

1. `trunc(NaN) = NaN`, `trunc(±∞) = ±∞`
2. For real `r`: the integer with largest magnitude such that
   `|trunc(r)| ≤ |r|` and `sign(trunc(r)) ∈ {0, sign(r)}`.

**`clamp_T : ℤ ∪ {+∞, −∞, NaN} → ⟦T⟧`** for integer `T` with bounds
`T_min`, `T_max`:

1. `clamp_T(NaN) = 0`
2. `clamp_T(z) = T_min` if `z = −∞` or `z < T_min`
3. `clamp_T(z) = T_max` if `z = +∞` or `z > T_max`
4. `clamp_T(z) = z` otherwise

**`wrap_T : ℤ → ⟦T⟧`** for integer `T` of bitwidth `n`: the unique
`r ∈ ⟦T⟧` with `r ≡ z (mod 2ⁿ)`.

## Definition of cast

A cast converts a value `x ∈ ⟦S⟧` to a value of type `T`. It is the total
function `cast_{S→T} : ⟦S⟧ → ⟦T⟧` defined by exactly one of the following
equations, selected by the kinds of `S` and `T`:

| `S` | `T` | `cast_{S→T}(x)` |
|---|---|---|
| integer | integer | `wrap_T(x)` |
| integer | float | `round_T(x)` |
| float | float | `±0_T` if `x = ±0_S` (same sign); else `round_T(⟦x⟧)` |
| float | integer | `clamp_T(trunc(⟦x⟧))` |

The sign-of-zero clause is required because `⟦·⟧` identifies `+0` and `−0`:
float→float casts preserve the sign of zero; float→integer casts map both
zeros to `0`.

Integer→float casts are a **single direct rounding** into `T`. In
particular, `int → F32` is *not* defined as `int → F64 → F32`; the two
differ for some `|x| > 2⁵³` where the intermediate `F64` result lands
exactly on an `F32` tie (double rounding).

> Note (rationale): casts never end the program. Arithmetic overflow ends
> the program, but a cast is an explicit request for conversion, so
> truncation/saturation is presumed intended; a program that wants different
> overflow handling can test the value before casting.
>
> Note (correspondence): these semantics coincide with Rust `as`, WASM
> `wrap`/`extend`/`convert`/`promote`/`demote`/`trunc_sat`, LLVM
> `trunc`/`sext`/`zext`/`sitofp`/`uitofp`/`fpext`/`fptrunc`/
> `llvm.fptosi.sat`/`llvm.fptoui.sat`, and Java primitive conversions.
> They do **not** coincide with LLVM's plain `fptosi`/`fptoui` (poison on
> out-of-range) or WASM's plain `trunc` (traps).

## Theorems

The following hold for all numeric `S`, `T` and all `x ∈ ⟦S⟧`
(mechanically checked in `verify/cast_properties.py`):

* **T1 (totality, determinism).** `cast_{S→T}` assigns exactly one value of
  `⟦T⟧` to every `x ∈ ⟦S⟧`. *(By construction: each equation is a
  composition of total functions with exhaustive, mutually exclusive
  cases.)*
* **T2 (identity).** `cast_{T→T}(x) = x`.
* **T3 (widening exactness).** If `⟦x⟧` is exactly representable in `T`,
  then `⟦cast_{S→T}(x)⟧ = ⟦x⟧`. Consequences: `F32→F64` is exact;
  integer→integer with same signedness and greater width is exact;
  integer→float is exact when the integer fits in the mantissa
  (`|x| ≤ 2²⁴` for `F32`, `|x| ≤ 2⁵³` for `F64`).
* **T4 (saturation boundaries).** For float→integer:
  `cast(NaN) = 0`, `cast(+∞) = T_max`, `cast(−∞) = T_min`, `cast(±0) = 0`,
  and for finite `x`, `cast(x) ∈ [T_min, T_max]` with equality at the
  bounds exactly when `trunc(⟦x⟧)` is out of range on that side.
* **T5 (monotonicity).** Float→integer and integer→float casts are monotone
  on non-NaN inputs: `x ≤ y ⇒ cast(x) ≤ cast(y)`.
* **T6 (round trips).** `S→T→S` is the identity when every value of `S` is
  exactly representable in `T` (e.g. `I32→F64→I32`). It is **not** the
  identity otherwise (e.g. `I64→F64→I64`, `I32→F32→I32`); the checker
  exhibits counterexamples.
* **T7 (no double rounding).** Direct `U64→F32` differs from
  `U64→F64→F32` for some inputs (the checker exhibits one); for `I32` and
  narrower sources the two coincide. The spec mandates the direct form.
