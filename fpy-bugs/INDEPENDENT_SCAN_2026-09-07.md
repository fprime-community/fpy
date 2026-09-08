# Independent compiler / bytecode / VM scan, 2026-09-07

Scope: interactions between the compiler, the emitted fpybc bytecode and the
LLVM/wasm module, and the two flight VMs; miscompilations; compiler crashes.
Arithmetic overflow/rounding/truncation semantics were deliberately skipped.
Existing GitHub issues (#37-#210) were checked first; nothing below duplicates
one.

Method:

* Read the whole front end, both code generators, the assembler, and the
  FpySequencer / WasmSequencer sources.
* Ran ~120 hand-written sequences targeting frames, calling convention,
  aggregates, temporaries, check/for desugaring, commands, tlm/prm, seq args,
  time ops, on both harnesses (fpybc on Svc::FpySequencer, wasm on
  Svc::WasmSequencer), comparing serial-port traces.
* Ran a differential fuzzer (random I64/bool/control-flow programs executed
  in Python as the reference, then on both backends): 600 seeds, 0
  disagreements.
* Ran a second differential fuzzer (structs, arrays, enums, bools, U32/F64,
  commands, member/element stores, aggregate params/returns) comparing fpybc
  against wasm: ~600 seeds, 0 disagreements between the backends.
* Ran a mutation-based crash fuzzer on the compiler (analysis + both
  codegens): ~2900 mutated programs, 0 crashes.
* Compiled representative programs against dictionaries that change the
  deployment-configurable F Prime types (FwSizeType, FwSizeStoreType,
  FwTimeContextStoreType, TimeBase representation, ...).

Overall the two backends agree closely; every finding below comes from the
compiler's handling of constants and dictionary configuration rather than
from codegen or the VMs.

---

## 1. `time()` hardcodes `U8` for `timeContext`: wrong-size constant (fpybc stack corruption) and wasm backend crash when `FwTimeContextStoreType` is not `U8`

**Severity: high** (silent miscompile on fpybc; only affects deployments that
configure `FwTimeContextStoreType` to something other than `U8`)

`Fw.TimeValue`'s `timeContext` member takes its type from the dictionary
(`_update_time_context_type_from_dict` sets `TIME.members[1].type`), and
every other producer of an `Fw.Time` (the `Fw.Time` constructor, `now()`,
`PUSH_TIME`, `WAIT_ABS`) follows it. The `time()` builtin does not:

* `TIME_MACRO.args[2]` is `("timeContext", U8, FpyValue(U8, 0))`
  (`src/fpy/macros.py`), so the argument is coerced and range-checked as `U8`.
* `_parse_time_string` builds the value with `FpyValue(U8, time_context)`
  (`src/fpy/semantics.py`), so the member value's own type is `U8` regardless
  of the struct member's type.

`FpyValue.serialize` serializes a struct member by the *member value's* type,
not the struct's declared member type, so the constant serializes one byte
short.

Repro (dictionary = RefTopologyDictionary with `FwTimeContextStoreType`
changed to `U16`; the scan script wrote it as `dict_ctx16.json`):

```
x: Fw.Time = time("2025-01-01T00:00:00Z")
y: U32 = x.seconds
```

fpybc output:

```
2 PUSHVAL(val=b'\x00\x02\x00gt\x85\x80\x00\x00\x00\x00')   <- 11 bytes
3 STORERELCONSTOFFSET(lvar_offset=1, size=12)             <- pops 12
```

`Fw.Time` is 12 bytes on that deployment (TIME.max_size == 12), the constant
is 11. The store pops 12 bytes: the 11-byte constant plus one byte from the
frame below it, so every field of `x` is shifted by one byte and the
expression stack is left one byte short for the rest of the sequence. No
error is reported at compile time or at run time.

wasm output: `RuntimeError: LLVM IR parsing error ... element 1 of struct
initializer doesn't match struct element type` (`{i16 2, i8 0, ...}` stored
into `{i16, i16, i32, i32}`), an uncaught Python exception rather than a
`BackendError`.

Also: `time("...", TimeBase.TB_WORKSTATION_TIME, 300)` is rejected with
"300 is out of range for type U8" although the deployment's context type is
`U16`.

Fix sketch: build the `timeContext` argument spec and the parsed value from
`TIME.members[1].type` (after the dictionary load), the same way
`get_base_compile_state` re-applies the `timeBase` default.

---

## 2. `time()` parses the microsecond field through a float and is off by one for about half of all inputs

**Severity: medium** (wrong constant, both backends, default configuration)

`_parse_time_string` computes

```python
timestamp = dt.timestamp()
seconds = int(timestamp)
useconds = int((timestamp - seconds) * 1_000_000)
```

A 2025 timestamp is ~1.7e9 s, which leaves a double only ~2e-7 s of
resolution, so `(timestamp - seconds) * 1e6` is frequently `N - 1e-6` and
`int()` truncates it. Over all microsecond values 0..999999 (step 7), 71422 of
142858 parse one microsecond low.

End-to-end on both harnesses:

```
t: Fw.Time = time("2025-12-19T14:30:00.000007Z")
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, I64(t.useconds))
```

emits `6`.

Fix: use `dt.microsecond` directly and `calendar.timegm`/integer arithmetic
for the seconds.

---

## 3. Deployments with `FwSizeType = U32` cannot compile anything: `AssertionError` in `_update_seq_args_from_dict`

**Severity: medium** (compiler crash; blocks every compile on such a
deployment, which includes 32-bit flight targets)

`Svc.SeqArgs` is `struct SeqArgs { $size: FwSizeType, buffer: [N] U8 }`
(`Svc/Seq/Seq.fpp`). The compiler's canonical `SEQ_ARGS` hardcodes the size
member as `U64` and `_update_seq_args_from_dict` (`src/fpy/state.py:409`)
asserts the dictionary agrees:

```
AssertionError: Dictionary Svc.SeqArgs.size has type FpyType(U32), expected FpyType(U64)
```

This fires from `get_base_compile_state`, i.e. before any source is looked
at, for any program, with a raw traceback. Unlike the other canonical types
this is an `assert`, not a `DictionaryError`, and unlike the buffer length
(which is adopted from the dictionary) the size type is not adopted. Since
`_emit_seq_run_cmd` and `_create_command_buffer` serialize the size field
with `SEQ_ARGS.members[0].type`, adopting the dictionary's type there is all
that is needed for the emitted `RUN_ARGS` payload to match the flight
`Svc::SeqArgs` layout.

---

## 4. Compiler crashes with `struct.error` instead of a diagnostic when a serialized length does not fit `FwSizeStoreType`

**Severity: low** (crash, only with a non-default `FwSizeStoreType`)

With a dictionary that sets `FwSizeStoreType = U8`, a string literal longer
than 255 bytes (`log("x"*300)`, `write_to_port(port, "y"*300)`) or any
directive whose argument bytes exceed 255 (`x: Svc.SeqArgs = Svc.SeqArgs()`
emits a 263-byte `PUSH_VAL`) crashes in `FpyValue.serialize`
(`src/fpy/types.py:549`) with `struct.error: 'B' format requires 0 <=
number <= 255`. The same limit exists on the flight side, so these should be
compile errors; today the `FinalChecks` size check never runs because
`dir.serialize()` raises first.

---

## 5. `rand()`, `randf()`, `set_seed()` crash the wasm backend with an uncaught `NotImplementedError`

**Severity: low** (crash with traceback instead of a diagnostic)

```
$ fprime-fpyc -d RefTopologyDictionary.json --emit wasm randtest.fpy
...
  File ".../src/fpy/symbols.py", line 30, in _generate_llvm_unsupported
    raise NotImplementedError("this builtin has no LLVM/wasm lowering yet")
```

`compile_main` only catches `BackendError` around codegen, so the user gets a
Python traceback. Raising `BackendError` from `_generate_llvm_unsupported`
(with the builtin's name and source position) would turn this into a normal
compile error.

---

## Configurable-type sweep against rebuilt harnesses (2026-09-07, later)

Both harnesses were rebuilt with a project config module overriding
`FpConfig.fpp` (`register_fprime_config(... CONFIGURATION_OVERRIDES ...)`,
added after `FPrime-Code.cmake`; `-DFPRIME_CONFIG_DIR` and settings.ini do not
reach this F Prime version's config system). Each variant was compiled with a
dictionary patched the same way and driven through nine sequences covering
commands (const and runtime args, failure + `flags`), tlm/prm reads, `now()`,
`Fw.Time` ctor, `sleep`/`sleep_until`, check statements, `time()` literals,
strings/`log`/serial, sequence arguments, aggregate params/returns, and a
child sequence run via `RUN_ARGS`.

* **wide** (`FwSizeStoreType=U32`, `FwTimeContextStoreType=U16`,
  `FwOpcodeType`/`FwChanIdType`/`FwPrmIdType=U64`, `TimeBase: U32`): every
  sequence agrees on both backends except the `time()` literal (finding 1:
  fpybc `STACK_ACCESS_OUT_OF_BOUNDS`, wasm LLVM IR error).
* **narrow** (`FwSizeStoreType=U8`, ids `U16`, `TimeBase: U8`, ids remapped
  to fit): everything agrees except the two sequences carrying a >255-byte
  directive (finding 4: `struct.error` on fpybc, wasm fine), and ground
  `RUN_ARGS` is rejected with `FORMAT_ERROR` by the sequencer itself because a
  255-byte `Svc.SeqArgs` cannot travel through a port whose buffer length is
  a `U8` (a flight config consistency matter, not a compiler bug).
* **size32** (`FwSizeType=U32`): the compiler crashes (finding 3). With the
  size type adopted in a scratch monkeypatch both backends emit a consistent
  271-byte `RUN_ARGS` payload (4-byte size), but the flight side could not be
  built on this host: the Posix OSAL static-asserts `FwSizeType` holds
  `size_t`, and no 32-bit multilib is installed.

Variant deployments and build caches are under `build-variants/` (untracked).

Filed on 2026-09-07: the time() context type, time() microseconds,
FwSizeType, FwSizeStoreType, Fw.LogSeverity and serialOutMax issues.

## Notes (not filed as bugs)

* `Fw.LogSeverity` is the one canonical enum layered over the dictionary
  without `_validate_and_replace_type`; a dictionary that defined it with a
  different representation type would be silently ignored (the VM pops
  `Fw::LogSeverity::SerialType`). The framework defines it as `U8`, so this
  is only a robustness gap.
* Anonymous struct/array literals cannot be compared: `a == [1, 2]` and
  `t == {seconds: 1}` (for a non-time struct) fail with "Op == undefined for
  ..., array literal" even though the literal coerces to the variable's type
  everywhere else. A limitation rather than a miscompile.
* `write_to_port` of a value larger than the WasmSequencer's configured
  `serialOutMax` (256 in the harness; e.g. a `Svc.SeqArgs`) traps on wasm and
  succeeds on fpybc. The limit is a WasmSequencer configuration value that is
  not in the dictionary, so the compiler cannot check it.
* The differential fuzzers confirmed the following behave identically on
  both backends: frames with early `return`/`break`/`continue` at any depth,
  recursion with aggregate locals, global aggregates mutated from functions,
  aggregate params/returns, temp-slot addressing of call results on wasm,
  named/default/anonymous arguments, check statements nested in loops and
  functions, command responses in expressions with temporaries below them,
  time operators with anonymous-struct operands, sequence arguments of
  aggregate type, and the `flags.assert_cmd_success` path from inside
  functions.
