# Compiler / bytecode / VM interaction scan, 2026-09-07

Independent scan of the compiler front end, both backends (fpybc and LLVM/wasm),
and the flight `Svc::FpySequencer` VM, looking for compiler crashes,
miscompilations and backend divergences. Arithmetic overflow/rounding/truncation
discrepancies were deliberately skipped. Every open and closed GitHub issue was
checked before listing a finding here; anything already covered by an issue
(e.g. #184, #185, #191-#209, #219) is omitted.

Nothing here has been filed. Each finding has a minimal repro that was run
against the tree at `a69b17d` (devel) using the Ref dictionary.

## Findings

### 1. Const folding `**` crashes the compiler when the fold yields a complex number (filed as #228)

`float.__pow__` returns a complex number for a negative base and a fractional
exponent. The folder only expects int/float/Decimal/bool and hits
`assert False, folded_value` in `CalculateConstExprValues.visit_AstBinaryOp`.

Reachable whenever at least one operand is a *concrete* float constant (so the
intermediate type is F64 and the operands become Python floats rather than
Decimals; with two literals the Decimal path raises `InvalidOperation`, which is
caught).

```
x: F64 = F64(-8.0) ** 0.5
```
```
x: F64 = (-2.0) ** F64(0.5)
```

Both backends: `AssertionError: (1.7319121124709868e-16+2.8284271247461903j)`.

Fix sketch: treat a complex fold result as a domain error (`state.err(...)`),
or fold float `**` through `math.pow` inside the existing `ValueError` handler.

### 2. Casting an infinite constant float to an integer type crashes the compiler (filed as #229)

`const_convert_type` guards NaN (`int()` raises `ValueError`, which is caught)
but not infinity: `int(float('inf'))` raises `OverflowError`, which is not in the
`except (ValueError, struct.error)` clause. Infinite constants are easy to make
(`F64(1e999)` folds to `inf`, cf. #110), and the crash fires in every position
that coerces the cast result:

```
x: I64 = I64(F64(1e999))
x: U8 = U8(F64(1e999))
x: I64 = I64(-F64(1e999))
x: I64 = I64(F64(1e999) // 2)
CdhCore.cmdDisp.CMD_TEST_CMD_1(I32(F64(1e999)), 1.0, 1)
a: Ref.FpyExampleArray = [1, 2, 3]
y: U32 = a[I64(F64(1e999))]
for i in 0..I64(F64(1e999)):
    pass
exit(I32(F64(1e999)))
```

Both backends: `OverflowError: cannot convert float infinity to integer`
(uncaught, traceback). The NaN variant of the same casts correctly reports
`For type F64: cannot convert float NaN to integer`.

Fix sketch: add `OverflowError` to the `except` in `const_convert_type`, or
reject non-finite values before `int()`.

### 3. Dictionary-configured type singletons leak across compiles in one process

`state._build_global_scopes` is `lru_cache`d on `(dictionary path, time base)`,
but the dictionary-dependent *mutation* of module-level singletons happens
inside the cached body: `update_configurable_types_from_dict` (FwSizeStoreType,
FwChanIdType, FwPrmIdType, FwOpcodeType, FwIndexType, FwPacketDescriptorType,
SerialPortIndex), `_update_time_base_from_dict` (TIME_BASE enum), 
`_update_time_context_type_from_dict` (TIME's context member), and
`_update_seq_args_from_dict` (SEQ_ARGS buffer length). On a cache hit none of
that is redone, so after compiling against dictionary B, compiling against
dictionary A again uses B's values. Only the time() default time base is
re-applied per call (`get_base_compile_state` special-cases it).

Repro (`FwSizeStoreType` = U32 and `Svc.SeqArgs` buffer = 100 in the modified
copy; the program is `CdhCore.cmdDisp.CMD_NO_OP_STRING("hi")`):

```
A:       SeqArgs buffer len=255 FwSizeStoreType=U16 cmd directive bytes=390000...
B:       SeqArgs buffer len=100 FwSizeStoreType=U32 cmd directive bytes=3900000000...
A again: SeqArgs buffer len=100 FwSizeStoreType=U32 cmd directive bytes=3900000000...
```

"A again" emits every directive header's length field as 4 bytes and would pad
`Svc.SeqArgs` to 100 bytes, i.e. a binary the A deployment's sequencer cannot
deserialize (or, for a subtler difference such as the TimeBase enum, one it
runs with wrong constants). The `fprime-fpyc` CLI compiles one file per process
and is unaffected; any long-lived process that compiles against more than one
dictionary (a test session, a GDS-side compile service) is affected. The cache is
also keyed by path, so a dictionary rewritten on disk under the same path is
never reloaded.

Script: `scan-2026-09-07-compiler-vm-scripts/probe_cache.py` (creates the modified dictionary
from `test/fpy/RefTopologyDictionary.json`).

Fix sketch: move the singleton updates out of the cached function and re-apply
them on every `get_base_compile_state`, or drop the cache / key it by content.

### 4. Inconsistent tabs and spaces are accepted silently, with tab = 8 (filed as #227)

`PythonIndenter` counts a tab as 8 columns and never checks consistency, so a
file mixing tabs and spaces compiles with whatever nesting the 8-column rule
produces. Python 3 rejects the same file with `TabError`. A sequence authored in
an editor showing tabs as 4 columns silently changes meaning:

```
a: bool = True
b: bool = False
n: I64 = 0
if a:
<TAB>if b:
<TAB><TAB>n = n + 1
        n = n + 10        # 8 spaces
assert n == 0, 1
```

The author (tab = 4) reads line 7 as inside `if b`; the compiler (tab = 8) puts
it inside `if a` only, so `n == 10` and the assert fails at runtime on both
backends. No warning is produced. Given that sequences are hand-edited flight
products, rejecting mixed indentation (or at least mixed tab/space prefixes
within one block) seems worth it.

### 5. Observation, lower priority

- **F32 literal comparison** (possibly out of scope as a rounding matter, listed
  because it bites `check` conditions): `x: F32 = 0.1` then `assert x == 0.1`
  fails on both backends. The common type of `F32` and `Float` is F32, but the
  comparison is performed at the widened F64 intermediate type and the literal
  is rounded to F64, not to F32 first, so `F64(F32(0.1)) != 0.1`. Same for
  `s.time == 0.1` with an F32 member. `x == y` with two F32 variables is fine.
  Rounding the literal to the common (F32) type before widening would make the
  comparison behave as written.

## Checked and found consistent (no finding)

All of the following were run on both the fpybc harness (real `Svc::FpySequencer`)
and the wasm harness (real `Svc::WasmSequencer`) and agreed with each other and
with the expected result:

- Control flow: nested for/while with break/continue (including continue in a
  desugared for loop, break inside check bodies and timeout bodies), loop
  variable and bound mutation inside the body, empty/negative ranges, range
  bounds with side effects evaluated once, `exit(0)` from nested calls.
- Functions: recursion, mutual recursion, value semantics for struct and array
  parameters, struct returns with direct member access (`f().seconds`), array
  returns with runtime index (`f()[i]`, `f()[i].m`), returns from nested loops
  with locals live, globals read/written from functions with and without
  sequence arguments, default arguments (const, anonymous struct/array, used
  before definition), named-argument reordering.
- Aggregates: nested struct/array member stores with runtime indices at top
  level, in functions on locals, globals, parameters and sequence arguments;
  bool/enum arrays; copy semantics on assignment; runtime out-of-bounds
  (negative and >= length) faulting with ARRAY_OUT_OF_BOUNDS on both.
- Commands: identical dispatched byte buffers for constant, runtime and mixed
  arguments (ints, floats, bools, enums, strings, structs, arrays), response
  capture, `flags.assert_cmd_success` toggling from functions, nested sequence
  calls with runtime/const/named/aggregate arguments and failure propagation.
- Telemetry/parameter reads including struct members and runtime-indexed
  elements; `write_to_port` byte output; `log` events; time arithmetic,
  `sleep`, `sleep_until`, and `check` timeout/persist/period timing; time
  operators nested in member accesses, arguments and check clauses.
- Imports: nested modules, cycles, re-export, star imports, aliasing that
  shadows dictionary modules, recursion across files.
- Stack discipline on fpybc (the harness checks the final stack size exactly)
  for bare expression statements of every kind inside loops and functions.
- A random differential fuzzer (Python reference vs fpybc vs wasm) over the
  subset of the language whose semantics coincide with Python's: ints, bools,
  U8, enums, `Ref.FpyExampleArray`, `Ref.ScalarStruct`, `Ref.ChoiceSlurry`
  (nested arrays of enums), if/elif/else, bounded while, for, break/continue,
  functions with params and early returns, aggregate copies, asserts and
  out-of-bounds faults. 108 programs ran to a verdict (36 + 72), 0 mismatches,
  0 compiler crashes; 3 more were skipped because the stock wasm harness cannot
  load modules of that size (ERR_OUT_OF_MEMORY), a known harness limit.
  `scan-2026-09-07-compiler-vm-scripts/fuzz.py <seed> <count> --keep DIR`.
