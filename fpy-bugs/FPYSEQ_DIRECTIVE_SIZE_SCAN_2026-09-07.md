# Independent scan: FpySequencer.cpp / FpySequencerDirectives.cpp

Date: 2026-09-07
Scope requested: line-by-line read of `FpySequencer.cpp` and
`FpySequencerDirectives.cpp`, compiler/bytecode/VM interactions,
miscompilations, compiler crashes. Arithmetic overflow/rounding/truncation
divergences deliberately skipped.

Everything below was checked against the open/closed GitHub issue list and the
other reports in `fpy-bugs/` before being written down. Nothing here is filed.

> **Overlap with #224 — read this first.** Finding 1's `log()` trigger is
> largely covered by #224 ("FwSizeStoreType = U8: oversized strings and
> directives crash the compiler with struct.error"). Despite its title, #224's
> body is about the same code path (the `FinalChecks` size check running only
> after `dir.serialize()` succeeds), uses the same example
> (`log("...300 characters...")`), and already states the general principle:
> "The flight side cannot represent these either, so they should be compile
> errors."
>
> What #224 does **not** say, and what is new here:
> * #224 is scoped to `FwSizeStoreType = U8` dictionaries, where the limit is
>   255 and the symptom is a compiler crash. In the **default** config
>   (`FwSizeStoreType = U16`) there is no crash: the limit is 506 and the
>   sequence fails at flight validation instead.
> * #224 implicates `FwSizeStoreType`'s range as the limit. The actual limit is
>   `FW_STATEMENT_ARG_BUFFER_MAX_SIZE`. A fix that checks only what #224
>   describes would leave the default-config bug in place.
> * The `CONST_CMD` off-by-four in Reproduction B is not in #224 or any other
>   issue.
>
> Recommendation: comment on #224 with the corrected constant and Reproduction
> B rather than filing a new issue. Also worth a correction on #203, whose body
> asserts that a long `log()` message "compiles cleanly and then reports
> something shorter" — above 506 bytes the sequence does not run at all.

---

## Finding 1 (primary): the compiler's per-directive size limit is the wrong constant, so it emits sequences that cannot be loaded

**Severity:** a clean sequence that passes every compiler check is rejected by
the flight sequencer at validation time. No memory unsafety, but a sequence
that looks good on the ground never runs on the spacecraft, and the ground
gets a low-level deserialize error rather than a diagnostic.

### The mismatch

A statement's argument buffer in the sequence file is an
`Fw::StatementArgBuffer`, whose capacity is

```
FW_STATEMENT_ARG_BUFFER_MAX_SIZE = FW_CMD_ARG_BUFFER_MAX_SIZE
                                 = FW_COM_BUFFER_MAX_SIZE - sizeof(FwOpcodeType) - sizeof(FwPacketDescriptorType)
                                 = 512 - 4 - 2 = 506
```

The only per-directive size check the compiler makes is in
`FinalChecks.run` (`src/fpy/codegen_fpybc.py`), against a completely
unrelated constant:

```python
dir_size = len(dir.serialize())
if dir_size > state.max_directive_size:      # Svc.Fpy.MAX_DIRECTIVE_SIZE == 2048
```

`Svc.Fpy.MAX_DIRECTIVE_SIZE` (2048) sizes the *in-memory* `argBuf`/`val`
arrays of the deserialized directive structs. It has nothing to do with how
many bytes a statement can carry in the file. The effective limit is 506
bytes of directive arguments; the compiler permits 2045.

### Reproduction A: `log()` with a long message

`log()` lowers to a `PUSH_VAL` holding the raw UTF-8 message bytes, with no
length bound of its own (`src/fpy/macros.py`, the `log` entry in `MACROS` —
the message is an `InternalString`, so no `string size N` cap applies).

```python
import fpy.test_helpers as th
for n in (506, 507):
    state, dirs, _ = th.compile_seq('log("%s")\n' % ("A" * n))
    try:
        th.run_seq(None, dirs); print(n, "loaded")
    except th.ValidationError as e:
        print(n, "VALIDATION-FAIL:", e)
```

Observed:

| message bytes | statement argBuf | result |
|---|---|---|
| 506 | 506 | loads and runs |
| 507 | 507 | validation fails |

The 507 case compiles with no error or warning and then fails at load with

```
FileReadDeserializeError : Deserialize error encountered while reading BODY (1)
of file s0.bin: 5 (510 bytes left out of 521)
```

The compiler only objects at 2046+ bytes, with a message naming the wrong
limit:

```
Directive PUSH_VAL in sequence too large (expected at most 2048 bytes, was 2103)
```

### Reproduction B: an ordinary command with large arguments

This one is worse because the compiler has a check that *looks* like it covers
the case and is off by exactly the opcode width. `FinalChecks` bounds a
command's arguments by `FW_CMD_ARG_BUFFER_MAX_SIZE` (506) and the whole packet
by `FW_COM_BUFFER_MAX_SIZE` (512), both inclusive. But a `CONST_CMD`
statement's argument buffer holds the 4-byte `FwOpcodeType` *plus* the command
arguments, so the real ceiling on constant command arguments is 502, not 506.

Confirmed directly against the sequencer:

| command args | statement argBuf (4 + args) | result |
|---|---|---|
| 502 | 506 | loads and runs |
| 503 | 507 | validation fails |
| 506 | 510 | validation fails |

506 bytes of arguments is not hypothetical: `Ref.seqDisp.RUN_ARGS`
(`String_240`, `Svc.BlockState`, `Svc.SeqArgs`) has a maximum argument
payload of exactly 506 bytes, so a seq-run call with a long enough child path
lands in the failing 503..506 window while passing every compiler check.

### Suggested fix

Read `FW_STATEMENT_ARG_BUFFER_MAX_SIZE` from the dictionary and check each
directive's *argument* bytes against it (the compiler's `dir_size` includes
the 1-byte opcode and the `FwSizeStoreType` length prefix, which live outside
the statement's arg buffer). For `CONST_CMD` the check must account for the
`FwOpcodeType` that shares the buffer with the command arguments.

### Related observation

Because `MAX_DIRECTIVE_SIZE` is 2048 while the statement buffer is 506,
`ConstCmdDirective::argBuf` and `PushValDirective::val` are each declared
`[Fpy.MAX_DIRECTIVE_SIZE] U8` in `FpySequencerDirectives.fppi` and can never be
more than a quarter full. `DirectiveUnion` contains both, and it is stack
allocated on every statement dispatch (`FpySequencerRunState.cpp`
`dispatchStatement`) and again on every telemetry tick while paused
(`FpySequencer.cpp` `updateDebugTelemetryStruct`). That is roughly 2 KB of
task stack per dispatch for a buffer bounded at 506 bytes.

---

## Finding 2 (secondary, config-dependent): the `PUSH_TLM_VAL_AND_TIME` overflow guard can underflow and turn a clean error into an `FW_ASSERT` abort

`FpySequencerDirectives.cpp:418`:

```cpp
if (Fpy::MAX_STACK_SIZE - tlmValue.getSize() - timeEsb.getSize() < this->m_runtime.stack.size) {
    error = DirectiveError::STACK_OVERFLOW;
```

This is unsigned arithmetic with two subtractions. The header's static asserts
only guarantee

```cpp
static_assert(Svc::Fpy::MAX_STACK_SIZE >= static_cast<FwSizeType>(FW_TLM_BUFFER_MAX_SIZE), ...)
```

They do not guarantee `MAX_STACK_SIZE >= FW_TLM_BUFFER_MAX_SIZE + Fw::Time::SERIALIZED_SIZE`.
In a configuration where `MAX_STACK_SIZE` is within 11 bytes of
`FW_TLM_BUFFER_MAX_SIZE` (which the static asserts accept), the expression
wraps to a huge value, the guard passes, and `Stack::push` trips its own
`FW_ASSERT`, aborting the flight `FpySequencer` instead of failing the
directive with `STACK_OVERFLOW`.

Not reachable in the default config (65535 vs 506), so this is a latent
robustness hole in a safety check rather than a live bug. It matters for the
config-variant work (#166). Every other `MAX_STACK_SIZE - x` guard in the file
subtracts a single quantity that a static assert or a type bound already
covers; line 418 is the only one with two.

Fix: either add the missing static assert, or reorder as
`this->m_runtime.stack.size > Fpy::MAX_STACK_SIZE - tlmValue.getSize() - timeEsb.getSize()`
computed with an explicit guard, or simply check the two pushes separately.

---

## Finding 3 (minor): `StatementsFailed` telemetry is never incremented

`m_tlm.statementsFailed` is declared in `FpySequencer.hpp:752`, initialised to
0, and read once in `tlmWrite_handler` (`FpySequencer.cpp:416`). Nothing ever
increments it, so the `StatementsFailed` channel is a constant 0.

This is the same class of defect as #154 (`SequencesFailed` not incremented)
but a different counter, and #154's title names only `SequencesFailed`. Worth
folding into that issue rather than filing separately.

---

## Checked and found correct

Recorded so the same ground is not re-walked.

**Stack discipline and bounds.** `storeHelper`, `loadHelper`,
`getField`, `peek`, `memCmp`, `discard`, `allocate`, `pushVal`, `stackCmd`,
`call`, `return`, `storeRel{,ConstOffset}`, `loadRel`, `storeAbs{,ConstOffset}`,
`loadAbs`, `popEvent`, `popSerializable`. Every overflow-safe rewrite is
correct; the non-overlap preconditions of `Stack::copy` are genuinely
established by the callers; `peek`'s source and destination provably never
overlap; `return`'s `memmove` out of the truncated region is in bounds.

**`Stack::pop` byte swapping.** The signed left shifts
(`static_cast<I64>(valBytes[0]) << 56`, `static_cast<I32>(...) << 24`) look
like signed-overflow UB and would be under C++11, but the project builds with
`CMAKE_CXX_STANDARD 14` (`test/fprime/cmake/settings.cmake`), where
[expr.shift] permits the result if it fits the *corresponding unsigned type*.
It does in both cases. Not a defect.

**Command response correlation** (`cmdResponseIn_handler`). The `cmdUid`
sequence/statement index packing is consistent with `sendCmd`, including the
subtlety that `m_statementsDispatched` is incremented after the directive is
queued but before the queued handler calls `sendCmd`. Stale responses from a
timed-out or cancelled sequence are correctly rejected. `Fw::CmdResponse::SerialType`
is `U8`, matching the compiler's `CMD_RESPONSE` (`rep_type=U8`).

**State machine.** Unhandled signals are silently ignored (`break`) in the
generated `sendSignal_*` switches; the `default: FW_ASSERT(false, ...)` arm
fires only on a corrupt `m_state`. So the SLEEPING race where `checkShouldWake`
and `checkStatementTimeout` both queue a signal in one tick drops the timeout
harmlessly rather than killing the following statement.
`STATEMENT_TIMEOUT_SECS` defaults to 0 (no timeout), so long sleeps are not
killed by default.

**Configurable types.** `FwIndexType`, `FwChanIdType`, `FwOpcodeType`,
`FwSizeStoreType`, `FwPacketDescriptorType`, `FwSizeType`,
`FwTimeContextStoreType`, `Svc.Fpy.SerialPortIndex` are all present in the
dictionary and agree between compiler and flight. `SeqArgs::get_buffer()`
returns `U8 (&)[255]`, so the `sizeof` in `validate()`'s
`ArgSizeExceedsCapacity` check is the real 255, not a pointer size.

**Builtin time library** (`src/fpy/builtin/time.fpy`). `time_add`,
`time_sub`, `time_interval_add`, `time_interval_sub` all check overflow and
underflow before converting, and normalise `useconds` below 1e6 on the way
out. `time_cmp` ignores `timeContext`, matching `checkShouldWake`.

**Compiler crash battery.** 45 adversarial programs (bad indices, recursive
and mutually recursive functions, arity and default-argument errors, huge and
malformed literals, out-of-loop `break`/`continue`/`return`, constructor
misuse, modules/types/commands used as values, bad time strings, bad
`write_to_port` ports, deeply nested parentheses and 400-term expressions,
shadowing the builtin `flags`). Every failure was a clean `CompileError`; no
Python traceback escaped.

**Semantic/miscompilation battery.** 50 self-checking sequences run on both
backends through the real `Svc::FpySequencer` and `Svc::WasmSequencer`:
nested struct/array reads and writes with constant and runtime indices,
two runtime indices in one access chain, struct and array copy semantics,
struct equality, functions taking and returning structs and arrays, globals
mutated from functions, argument evaluation order, short-circuit `and`/`or`
with side effects, recursion, `break`/`continue`, `return` from inside a loop,
default arguments, and for-loop semantics (bound evaluated once, loop variable
mutation, nesting, reuse). All agreed with the expected semantics on both
backends, and the harness's exact end-of-run stack size check passed each time.

Two spec/implementation mismatches noted in passing, both documentation-level:
the for-loop spec's step 4 says execution returns to step 1, which would
re-evaluate the range each iteration (the implementation evaluates it once,
which is the sane reading); and the spec says the loop variable is added to
the *resolving* scope, but it is not visible after the loop ends.
