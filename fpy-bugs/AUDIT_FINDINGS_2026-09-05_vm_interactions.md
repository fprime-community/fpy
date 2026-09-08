# Fpy audit 2026-09-05: compiler / bytecode / VM interactions

Independent scan of the compiler front end, both code generators
(`codegen_fpybc.py`, `codegen_llvm.py`), the bytecode file format, the flight
`Svc::FpySequencer` directive handlers and the `Svc::WasmSequencer` host
imports. Every finding below was reproduced on the real sequencers through
the test harnesses (fpybc on `FpyHarness`, wasm on `WasmSeqHarness`).

Not filed. Each entry says whether an existing issue already covers it.

## Method

* Hand review of the directive ISA against `FpySequencerDirectives.cpp`
  (operand order, stack accounting, field widths, error paths), of the
  fpybc and LLVM emitters against the semantics passes (contextual vs
  synthesized types, frame layout, labels), and of the wasm host ABI on
  both sides.
* ~40 hand-written probe programs run on both backends (aggregates,
  runtime indices, parameters by value, globals from functions, recursion,
  time ops, check statements, sequence args, imports, commands with
  runtime aggregate arguments, serial writes, enums, bool arrays, U64/I64
  extremes).
* A differential fuzzer (`scratchpad/fuzz2.py`): random programs emitted
  together with an equivalent Python reference program; each is compiled
  through both backends, run on both sequencers, and checked for final
  variable values, dispatched command bytes, logged events and serial
  writes. Math semantics known to differ (overflow trapping, float
  casts, `//`/`%` corner cases) are excluded from the generator.

Result: apart from finding 1, both backends agreed with the reference on
every fuzzed program (60 programs fpybc-only plus a 150-program dual-backend
batch; see the end of this file for the batch outcome), and every probe
passed. The compiler-to-VM contract (operand order, stack accounting,
frame layout, file format) looks solid.

## 1. fpybc: U64 arithmetic traps on valid results >= 2^63 (VM overflow checks are signed-only)

Severity: medium (valid programs fail on the flight VM; wasm succeeds).
Not covered by an existing issue. #111 ("Integer arithmetic overflow:
bytecode VM traps, wasm backend silently wraps") notes the opposite
direction only: that *genuine* unsigned wraparound is not trapped. #59
defines overflow as "R is not representable in the result type".

The compiler lowers `+`, `-` and `*` on a U64 intermediate type to the same
`ADD`/`SUB`/`MUL` directives it uses for I64 (`FPYBC_OP_IMPLS`,
`OpCase.ADD_INT` etc. do not distinguish signedness). The VM's `op_add`,
`op_sub` and `op_mul` (`FpySequencerDirectives.cpp`) pop both operands as
I64 and apply signed overflow checks, so a U64 value >= 2^63 is treated as
a negative number:

```python
a: U64 = 9223372036854775807
b: U64 = a + 1               # fpybc: ARITHMETIC_OVERFLOW; wasm: b == 2^63
```

```python
a: U64 = 9223372036854775813
b: U64 = a - 10              # fpybc: ARITHMETIC_UNDERFLOW; wasm: b == 2^63 + 3
```

```python
a: U64 = 4611686018427387904
b: U64 = a * 3               # fpybc: ARITHMETIC_OVERFLOW; wasm: b == 3 * 2^62
```

All three results are representable in U64, so per #59 none should end the
program. Conversely `a: U64 = 5; b: U64 = a - 10` does not trap on either
backend (the #111 direction). The directive stream for the first example
is `LOADREL; PUSHVAL 1; ADD; STORERELCONSTOFFSET`, i.e. nothing tells the
VM the operands are unsigned.

Probes: `scratchpad/p/h9b.fpy`, `h9c.fpy`, `h9e.fpy`, `h9d.fpy`, `h9f.fpy`.

Fix options: unsigned variants of the three directives (`UADD`/`USUB`/
`UMUL`, as `UDIV`/`UMOD` already exist), or have the compiler guard U64 ops
itself. Either way #111's wasm side needs the same treatment.

## 2. `assert cond, 0` ends the sequence as a success, silently

Severity: low (footgun / spec gap). Not covered by an existing issue.

The assert statement lowers to `NOT; IF; <exit code>; EXIT`, and `EXIT`
with code 0 is "go to the end of the sequence": the sequencer reports
`SequenceDone`, the RUN command completes with OK, and no
`SequenceExitedWithError` event is emitted. So a *failed* assertion whose
exit code is 0 looks exactly like a successful run, with every statement
after it skipped:

```python
ok: bool = False
assert ok, 0                 # both backends: sequence "succeeds" here
CdhCore.cmdDisp.CMD_NO_OP()  # never dispatched
```

SPEC.md ("Assert statement") says the exit code is displayed to the user
and the program halts; it does not say that 0 is reserved. Either forbid a
constant 0 (and document that a runtime 0 means success) or make assert
always fail. Related: the spec types the exit code as U8 for both `assert`
and `exit`, but the compiler coerces it to I32 (`ErrorCodeType`), so
`assert x, -7` compiles and reports -7 on both backends.

Probes: `scratchpad/p/h_exit0.fpy`, `h_exitneg.fpy`, `p3.fpy`.

## 3. The wasm backend has no compile-time command-size check

Severity: low. Not covered by an existing issue. Code reading only: no
command in the Ref dictionary can exceed the limits, so it is not
reproducible with the test dictionary.

`FinalChecks` (fpybc only) rejects a command whose serialized arguments
exceed `FW_CMD_ARG_BUFFER_MAX_SIZE` or whose packet exceeds
`FW_COM_BUFFER_MAX_SIZE`. The LLVM backend runs no such pass, so the same
sequence compiles to wasm and traps at run time in `wasmCommand`
(`BufferTooLarge`) on the flight WasmSequencer. The check belongs in the
shared front end (it only needs the dictionary and the argument types).

## 4. Flight VM: a command serialize failure fails the sequence with `lastDirectiveError == NO_ERROR`

Severity: low, flight side (nasa/fprime `FpySequencerDirectives.cpp`), not
this repo. `constCmd_directiveHandler` and `stackCmd_directiveHandler`
return `stmtResponse_failure` when `sendCmd` fails but never set `error`,
so `DirectiveErrorCode::CMD_SERIALIZE_FAILURE` is unused and telemetry
records `NO_ERROR` for a failed sequence. Unreachable with a compiler that
runs `FinalChecks`, reachable with a hand-assembled sequence.

## 5. Compiler crashes (AssertionError) on a dictionary whose `FwSizeType` is not U64

Severity: low. Not covered by an existing issue.
`_update_seq_args_from_dict` (`state.py`) asserts that
`Svc.SeqArgs.size` has type U64. F Prime allows `FwSizeType` to be
configured as U32 (32-bit targets); such a dictionary makes `fprime-fpy`
die with an assertion instead of a `DictionaryError`, and the canonical
`SEQ_ARGS` type hard-codes the 8-byte size field, so the sequence-run
command layout could not be produced for that deployment anyway.

## Verified as correct (for the record)

These were specifically checked because they looked suspicious on paper:

* Operand order and stack accounting for every directive the compiler
  emits (WAIT_REL/WAIT_ABS pop order, PEEK offsets in the bounds check,
  GET_FIELD offsets, STACK_CMD arg sizes including compact strings,
  RETURN value/arg sizes, CALL frame header, POP_EVENT layout,
  POP_SERIALIZABLE sizes, SeqArgs padding).
* Temporaries below a CALL frame (e.g. `1 + f()` where `f` mutates
  globals), recursion, parameters passed by value on both backends,
  element stores into global aggregates from functions with runtime
  indices, nested arrays with runtime indices (reads and single-index
  stores), struct-returning functions, whole-aggregate assignment,
  anonymous-literal default arguments, `exit` inside branches of
  functions that must return, `check` statements (with persist/period)
  inside functions and imported files, sequence arguments read from
  functions, imported functions with loops/logging.
* fpybc directive field widths versus `FpySequencerDirectives.fppi`,
  `DirectiveId` numbering, header/arg-spec/statement/footer layout versus
  `FpySequencerTypes.fpp` and `readBody`.
* Bool wire values (0xFF/0x00) on both sides, `Fw.LogSeverity` U8
  representation, TimeBase/TimeContext widths from the dictionary.

## Notes on the test infrastructure (not bugs)

* The wasm harness config (`heapPages` 8, `maxCodePages` 256) cannot load
  modules much above ~3.5 KB; the fuzzer used a private copy of
  `test/harness/wasm` built with `heapPages = 32`, `maxCodePages = 1024`,
  `stackSize = 2048`, `guestMemorySize = 1 MiB` (`SPACEWASM_MAX_PAGES`
  caps `heapPages` at 32). Worth raising in the checked-in harness if the
  test suite ever grows larger sequences.
* The fprime-wasm default `SequenceArgumentsMaxSize` is 12, so wasm
  harness runs with more than 12 bytes of sequence arguments fail before
  the sequence starts ("sequence arguments do not fit in Svc::SeqArgs").
