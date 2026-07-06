# Fpy arithmetic semantics vs Rust and C#

How fpy's arithmetic rules (MATH.md, mechanized in `verify/arith_properties.py`)
compare to Rust and C#. **Every Rust and C# claim in this document was verified
empirically** by compiling and running probe programs; the raw outputs are in
the appendix.

Verified with:

* Rust: `rustc 1.96.0`, run twice: default profile (debug assertions ON) and
  `-O` (debug assertions OFF), on x86-64 Linux.
* C#: .NET SDK `8.0.422` (RyuJIT, x86-64 Linux), default `unchecked` context
  plus explicit `checked` expressions.

## Integer arithmetic

| Behavior | fpy | Rust | C# |
|---|---|---|---|
| `+ - *` overflow | halts, always | debug: panics; release: wraps (still defined as a bug; `checked_*`/`wrapping_*` express intent) | unchecked (default): wraps; `checked`: throws `OverflowException` |
| divide by zero | halts (`DOMAIN_ERROR`) | panics, **both profiles** | throws `DivideByZeroException`, always |
| remainder by zero | halts (`DOMAIN_ERROR`) | panics, both profiles | throws `DivideByZeroException`, always |
| `MIN / -1` (fpy `//`) | halts (`ARITHMETIC_OVERFLOW`) | panics, both profiles | throws `OverflowException`, **even unchecked** |
| `MIN % -1` | halts (`ARITHMETIC_OVERFLOW`) | panics, both profiles (`checked_rem` returns `None`) | throws `OverflowException`, even unchecked |
| division style | `//` floors toward -inf, `%` takes divisor's sign (Python) | `/` truncates toward zero, `%` takes dividend's sign (C); `div_euclid`/`rem_euclid` as methods | `/` truncates, `%` takes dividend's sign (C) |
| `/` on integers | always computes in F64 (true division) | stays integer | stays integer |

Notes:

* Both Rust and C# treat `+ - *` overflow checking as a *mode* (debug profile,
  `checked` context) but check division unconditionally: div-by-zero and
  `MIN/-1` are hard errors in every configuration, because the underlying
  hardware/LLVM operation is undefined there. fpy's always-halt rule for
  `+ - *` is Rust's debug behavior made permanent, which fpy's VM already
  implements (`ARITHMETIC_OVERFLOW`/`UNDERFLOW`); the LLVM backend still wraps
  (open question OQ-1 in `verify/arith_properties.py`).
* `MIN % -1` halting (rather than evaluating to 0) was decided 2026-07-06 to
  match Rust and C#: **both** error here in every mode, even though the
  mathematical remainder is 0. It also keeps `//` and `%` halting on exactly
  the same inputs, so `a == (a // b) * b + (a % b)` holds wherever the pair is
  defined (proved in `verify/arith_properties.py`).
* fpy is alone in floored division. Rust and C# are C-style truncating. fpy
  follows Python because sequences are written by Python users; the VM,
  the LLVM backend, and the spec all agree on floored.

## Type discipline

| Behavior | fpy | Rust | C# |
|---|---|---|---|
| unary `-` on unsigned | **compile error** (as of 2026-07-06) | compile error `E0600` (`wrapping_neg()` expresses intent) | `uint`: legal, **promotes to `long`** (result is correct, e.g. `-5`); `ulong`: compile error `CS0023` |
| mixed-width same-signedness (`u8 + u64`) | implicit widening to the 64-bit intermediate | compile error `E0277` (no implicit conversions at all) | implicit promotion (both sides widen) |
| mixed signedness (`i32 + u32`) | compile error | compile error `E0277` | promotes both to `long`; `ulong + int` is compile error `CS0034` (no type holds both) |
| int -> float implicit | allowed (RNE rounding, can be lossy for wide ints) | compile error `E0308` (`as` required) | allowed, even lossy `long -> float` |
| narrowing implicit (`i64 -> i32`) | compile error | compile error | compile error `CS0266` |

Notes:

* fpy sits between the two: stricter than C# (no signed/unsigned mixing) but
  looser than Rust (implicit widening and int->float are allowed because the
  64-bit intermediate makes them value-preserving or explicitly rounding).
* C#'s `-uint -> long` promotion is the mathematically honest alternative to
  rejecting unary minus on unsigned. fpy chose Rust's rule (reject) instead:
  fpy's unsigned intermediate would have been U64, where C#'s trick has no
  wider signed type to escape to -- exactly why C# rejects `-ulong`.

## Floating point

| Behavior | fpy | Rust | C# |
|---|---|---|---|
| `NaN != NaN` | True | true | True |
| `NaN == NaN`, `NaN < x` | False | false | False |
| `1.0 / 0.0` | +inf, no halt | inf, no panic | Infinity, no throw |
| `0.0 / 0.0` | NaN | NaN | NaN |
| float `%` style | floored, sign of divisor (`-7.5 % 2.0 == 0.5`, Python) | truncated fmod, sign of dividend (`-7.5 % 2.0 == -1.5`, C) | truncated fmod, sign of dividend (`-1.5`, C) |
| float `% 0.0` | NaN (never halts) | NaN, no panic | NaN, no throw |
| overflow to +-inf, subnormals | IEEE-754 | IEEE-754 | IEEE-754 |

Note on `% 0.0` (OQ-5, resolved 2026-07-06): Rust and C# both give NaN --
float ops never trap in either language -- and fpy follows them: the spec,
the LLVM backend (frem), and the VM model all produce NaN. CPython raises
`ZeroDivisionError` instead; this is a deliberate divergence from Python.
The C++ `op_fmod` still returns DOMAIN_ERROR and needs the upstream fix
below.

## Numeric casts

| Behavior | fpy cast | Rust `as` | C# cast |
|---|---|---|---|
| int -> smaller int | wrap (truncate bits) | wrap (`-1i8 as u8 == 255`, `300i64 as u8 == 44`) | unchecked: wrap (`(byte)300 == 44`); checked: throws |
| float -> int, in range | truncate toward zero | truncate | truncate |
| float -> int, out of range | saturate to MIN/MAX | saturate (`300.0 as u8 == 255`, `-300.0 as i8 == -128`) | unchecked: **unspecified value** (measured on x64 .NET 8: sentinel `int.MinValue` for `(int)3e10`); checked: throws |
| NaN -> int | 0 | 0 | unchecked: measured `int.MinValue` (unspecified per spec); checked: throws |
| int -> float | round to nearest (RNE) | round to nearest | round to nearest |

# TODO i think we might want to follow C# for the Nan-> int and float->int out of range

fpy's cast spec (MATH_CASTS_DRAFT.md) is exactly Rust's `as` semantics -- Rust
moved float->int to saturating in 1.45 to remove the same UB fpy avoids. C# is
the outlier: its unchecked out-of-range float->int is explicitly "an
unspecified value of the destination type" (memory-safe UB), and on x64 .NET 8
it produces a sentinel, not saturation.

## User exit vs runtime fault

Question: should a user-invoked `exit(code)` be distinguishable from a runtime
arithmetic fault, so a sequence cannot fake (or accidentally collide with) a
`DOMAIN_ERROR`? Precedent says yes, with one caveat about *where* the
distinction lives:

* **Rust** separates *in process*: a panic runs the panic hook, prints
  diagnostics, and can be observed by `catch_unwind`; `std::process::exit(n)`
  does none of that. But at the OS level the channels collapse to one integer:
  a panicking process exits with code 101, and `process::exit(101)` is
  indistinguishable to the parent (verified). The separation is real only
  because supervisors look at the panic diagnostics/hook, not the code.
* **C#** likewise: an unhandled exception has its own termination path,
  diagnostics, and type; `Environment.Exit(n)` is just a code. In process the
  channels are distinct; the integer alone is spoofable.
* **F Prime C++ FpySequencer already implements the separation correctly**:
  `exit_directiveHandler` raises a dedicated event
  (`SequenceExitedWithError(path, userCode)`) carrying the user's code, and
  reports the directive error as `EXIT_WITH_ERROR` -- always. A user exit can
  never surface as `DirectiveError::DOMAIN_ERROR`; the fault enum values are
  reserved for actual faults.
* The **Python VM model** also keeps them apart: `handle_exit` sets
  `error_code` (user-owned I32) and returns no directive error; faults return
  a `DirectiveErrorCode` from the handler.
* The **LLVM/wasm backend** used to be the one place they were conflated (a
  single `fpy_exit(i32)` host import shared by `exit()`, `assert`, and the
  arithmetic guards). As of 2026-07-06 the wasm ABI has two noreturn host
  imports: `fpy_exit(user_code)` for user-requested termination (exit(),
  assert) and `fpy_fault(directive_error)` for runtime faults (division by
  zero, arithmetic overflow). The test runner reports them as distinct
  outcomes (`exit <code>` vs `fault <code>`), and the test helpers refuse
  cross-channel matches: `exit(10)` does not satisfy an expected
  `DOMAIN_ERROR` even though `DOMAIN_ERROR`'s value is 10.

The general lesson (which the Rust exit-code collision demonstrates): the
separation must live in the *channel*, not in an integer convention.

## Known divergences to fix upstream (C++ FpySequencer)

Found while verifying, in `Svc/FpySequencer/FpySequencerDirectives.cpp`:

1. `op_sdiv` computes `lhs / rhs` with only a zero-divisor guard:
   `INT64_MIN / -1` is C++ UB (SIGFPE on x86). Needs the overflow guard the
   fpy model/backends now have.
2. `op_smod` computes `lhs % rhs` with only a zero-divisor guard:
   `INT64_MIN % -1` is likewise UB. Same guard needed
   (Rust/C# both error here; the fpy spec now halts with
   `ARITHMETIC_OVERFLOW`).
3. `op_fmod` returns `DOMAIN_ERROR` on a zero divisor, but the fpy semantics
   (spec, LLVM backend, VM model, Rust, C#, IEEE) is NaN with no halt. It
   also computes `lhs - rhs * floor(lhs / rhs)`, which rounds at every step;
   the spec's formula is exact truncated fmod plus at most one rounded
   addition of the divisor (what `frem` + `fadd` and the VM model compute),
   and the two differ in the last ulp for extreme operand ratios.
4. `exit_directiveHandler` pops the exit code as `U8`, while the fpy compiler
   and Python model treat it as `I32` -- worth checking version alignment.

## Appendix: probe outputs

### Rust runtime (`rustc 1.96.0`)

Left: default build (debug assertions on). Right: `-O` (off). Lines identical
between profiles are shown once.

```
                                   debug    release
i64_add_overflow_panics:           true     false
u64_sub_underflow_panics:          true     false
i64_mul_overflow_panics:           true     false
i64_div_by_zero_panics:            true     true
i64_rem_by_zero_panics:            true     true
i64_min_div_neg1_panics:           true     true
i64_min_rem_neg1_panics:           true     true
checked_rem_min_neg1:              None
trunc_div_7_by_neg2:               -3
trunc_div_neg7_by_2:               -3
rem_neg7_by_2:                     -1
rem_7_by_neg2:                     1
div_euclid_neg7_by_2:              -4
rem_euclid_neg7_by_2:              1
wrapping_neg_5u32:                 4294967291
nan_ne_nan:                        true
nan_eq_nan:                        false
nan_lt_1:                          false
f64_1_div_0:                       inf
f64_0_div_0:                       NaN
f64_rem_neg7p5_by_2:               -1.5
f64_rem_7p5_by_neg2:               1.5
f64_rem_1_by_0:                    NaN
as_sat_300f64_to_u8:               255
as_sat_neg300f64_to_i8:            -128
as_nan_to_i32:                     0
as_wrap_neg1i8_to_u8:              255
as_trunc_300i64_to_u8:             44
as_2pow63_f64_to_i64_saturates:    true
panic exit code:                   101
process::exit(101) exit code:      101   (indistinguishable to the parent)
```

Compile errors (all rejected):

```
-x where x: u32                 error[E0600]: cannot apply unary operator `-` to type `u32`
i32 + u32                       error[E0277]: cannot add `u32` to `i32`
u8 + u64                        error[E0277]: cannot add `u64` to `u8`
let _: f64 = 1i64               error[E0308]: mismatched types
```

### C# runtime (.NET SDK 8.0.422, x64, default unchecked)

```
unchecked_add_wraps_to_min:     True
checked_add:                    OverflowException
div_by_zero:                    DivideByZeroException
rem_by_zero:                    DivideByZeroException
min_div_neg1_unchecked:         OverflowException
min_rem_neg1_unchecked:         OverflowException
trunc_div_7_by_neg2:            -3
trunc_div_neg7_by_2:            -3
rem_neg7_by_2:                  -1
rem_7_by_neg2:                  1
neg_uint_type:                  Int64 value -5
uint_plus_int_type:             Int64 value -1
implicit_long_to_float:         9.223372E+18
implicit_long_to_double:        9.223372036854776E+18
nan_ne_nan:                     True
nan_eq_nan:                     False
nan_lt_1:                       False
f64_1_div_0:                    Infinity
f64_0_div_0:                    NaN
f64_rem_neg7p5_by_2:            -1.5
f64_rem_7p5_by_neg2:            1.5
f64_rem_1_by_0:                 NaN
unchecked_double300_to_byte:    44
unchecked_doubleNeg300_to_sbyte: -44
unchecked_double3e10_to_int:    -2147483648
unchecked_nan_to_int:           -2147483648
checked_double3e10_to_int:      OverflowException
unchecked_int300_to_byte:       44
checked_int300_to_byte:         OverflowException
long_to_double_rounds:          True
```

Compile errors (all rejected):

```
-x where x: ulong               error CS0023: Operator '-' cannot be applied to operand of type 'ulong'
ulong + int                     error CS0034: Operator '+' is ambiguous on operands of type 'ulong' and 'int'
int x = (long)y implicit        error CS0266: Cannot implicitly convert type 'long' to 'int'
```
