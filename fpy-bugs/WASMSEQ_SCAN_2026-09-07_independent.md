# Independent WasmSequencer / compiler-VM scan — 2026-09-07 (devel @ a69b17d)

Scope requested: line-by-line reading of the WasmSeq sources, compiler ↔
bytecode ↔ VM interactions, miscompilations, and compiler crashes. Arithmetic
overflow / rounding / truncation deliberately skipped.

**Honest summary:** most of what this session reproduced empirically turned out
to be already recorded in `fpy-bugs/` or already filed on GitHub (see
"Already known — re-derived and deduped" at the bottom). Two observations
below are new; one is a latent contract violation found by reading, the other
sharpens an already-triaged finding. The bulk of the session's value is the
"verified clean" list: a large amount of the WasmSequencer surface was checked
and found correct.

---

## 1. (New, latent) The WasmSequencer stores the spacewasm `caller` handle and uses it after the host call has returned

**Where:** `WasmSequencer.hpp:1053` (`PendingHostFunction::caller`), every
`wasm*` host function in `WasmSequencerHost.cpp` (e.g. `:188`, `:219`, `:277`,
`:325`, `:362`, `:399`, `:488`, `:530`), and every use in
`WasmSequencerInterpreter.cpp` via `readGuestMemory` / `writeGuestMemory`
(`:232`, `:245`).

The spacewasm C API documents the handle as call-scoped:

```
/// Opaque handle passed to C host callbacks, wrapping a borrowed core
/// [`Engine`]. Valid only for the duration of the call.
pub struct SpacewasmCaller;
```
(`spacewasm_c_api-0.6.4/src/host.rs:15-17`, mirrored in `spacewasm_include/spacewasm.h:370-375`)

Every guest host call that touches guest memory violates this. `wasmReadTelemetry`
stashes `caller` and returns `SPACEWASM_PAUSE`; the pointer is then dereferenced
on a **later component dispatch** — `spacewasm_run` has already returned to
`action_spin`, which queued `interpreterPause`; the state machine only reaches
`dispatchPendingHostFunction` when that queued signal is processed. For a
`COMMAND` the gap spans the entire wait for the command response.

**Why it does not bite today:** the trampoline hands out the address of the
long-lived engine itself —
`let caller = state as *const Engine as *mut SpacewasmCaller;`
(`host.rs:110`) — so the pointer stays valid until `spacewasm_destroy`. I
traced the cancel/reset paths (`resetStore` → `destroyStore` while a host
function is pending) and the component's FIFO queue orders
`clearPendingHostFunction` before any stale dereference, so I could not reach a
live use-after-free.

**Why it still matters for flight:** correctness rests entirely on an
undocumented implementation detail of a pinned third-party crate. If spacewasm
ever makes the caller a per-call object (a stack temporary, or a struct
bundling engine + frame), every guest `tlm`/`prm`/`cmd`/`args`/`event`/
`serial_send` becomes a use-after-free — and `spacewasm_mem_write` writes
guest-controlled bytes at a guest-controlled offset. Nothing in the build would
flag the change; the version pin (`=0.6.4`) is the only thing holding it.

**Suggested fix:** don't retain `caller`. Either copy the guest bytes in/out
inside the host function (before returning `SPACEWASM_PAUSE`), or ask spacewasm
for an explicitly engine-scoped memory accessor (`spacewasm_mem_read/write`
taking `spacewasm_t*`). At minimum, document the dependency and add a
compile-time pin/assert tying it to the spacewasm version.

---

## 2. FILED AS #231 — The compiler reports the shared buffer's capacity as the tlm/prm value size, so the host's only size check is against the wrong number

`COMPILER_VM_SCAN_2026-09-05.md` §3 already records that neither VM validates a
telemetry/parameter value's serialized size. New detail: on the wasm side the
size the host checks against is not the channel's type size at all.

`_emit_tlm_prm_read` (`codegen_llvm.py:236-242`) takes
`buf_size = buf.type.pointee.count` from the single shared `tlm_prm_buffer`,
which `CreateTlmPrmBuffers` sizes to the largest read anywhere in the module,
and passes it as `value_size` for **every** read. Emitted IR for the same U32
channel (id 16777216, `max_size` 4):

```llvm
; alone in the module
%".4"  = call i32 @"tlm"(i64 16777216, i8* %".3", i32 11, i8* %".2", i32 4)
; identical read, but a 42-byte struct channel is read elsewhere
%".60" = call i32 @"tlm"(i64 16777216, i8* %".59", i32 11, i8* %".58", i32 42)
```

The buffer is shared across telemetry *and* parameters, so an unrelated
telemetry read also changes what a parameter read reports:

```llvm
%".59" = call i32 @"prm"(i64 268574720, i8* %".58", i32 42)   ; a U32 parameter
```

Observable effect, with `CdhCore.cmdDisp.CommandsDispatched` (U32) returning
12 bytes:

| sequence | wasm result |
|---|---|
| reads only the U32 channel | `BufferTooSmall`, sequence fails (correct) |
| same read, plus a read of `Ref.typeDemo.ScalarStructCh` (42 B) elsewhere | **no error, sequence succeeds** |

(fpybc silently returns the wrong bytes — the *tail* of the value, not the head
— and reports success, in both variants.)

**The structural gap.** The compiler is arguably ABI-compliant: `fprime.h`
documents `value_size` as "Size allocated for value_ptr", i.e. the allocation
size. But that is the host's *only* validation, so `dispatchTelemetry` /
`dispatchParameter` (`WasmSequencerInterpreter.cpp:312,335`) can never compare
the incoming value against the expected type size, which they are never told.
And `tlm`/`prm` return only a validity enum, so the guest is never told how
many bytes arrived and cannot check either. Neither side holds both numbers.

Contrast the two host functions that do get this right: `time` requires
`time_size == Fw::Time::SERIALIZED_SIZE` and rejects both directions
(`WasmSequencerHost.cpp:209-217`); `args` returns the byte count and the
compiler's prologue faults `INVALID_ARG` unless it equals the declared total
(`codegen_llvm.py:1403-1408`). Telemetry and parameters are the only reads with
no size contract at all.

**Status:** filed as #231 (covers the whole missing-size-validation problem on
both backends). The compiler half is applied in the working tree, uncommitted:
`_emit_tlm_prm_read` now passes `state.synthesized_types[node].max_size`.
Verified: the same oversized read now fails identically with and without an
unrelated bigger read; 241 tests pass (telemetry, parameters, wasm, commands,
types, golden). The undersized/leak direction is unchanged and still needs the
host-side exact-equality check.

**Suggested fix:** pass this read's own `value_type.max_size` as `value_size`
instead of the buffer capacity — a one-line compiler change; the shared buffer
is sized to the maximum so it still always fits. That alone makes the existing
`>` comparison a real check and removes the action-at-a-distance. Pair it with
an exact-equality check in the host to also close the undersized direction,
which is the half that currently reads leftover bytes from the previous read.

## 3. (Note, low severity) A NaN timeout parameter passes both range guards and reaches an undefined float→integer conversion

`WasmSequencer::checkTimeout` (`WasmSequencerInterpreter.cpp:619`) and
`FpySequencer::checkStatementTimeout` (`FpySequencerRunState.cpp:712`) both
guard with `timeout <= 0 || timeout >= max`. Both comparisons are false for
NaN, so a NaN `HOST_FUNCTION_TIMEOUT_SECS` / `STATEMENT_TIMEOUT_SECS` reaches
`static_cast<U32>(NaN)` / `static_cast<U64>(NaN * 1e6f)`, which is undefined
behaviour. Ground can set any 4 bytes for an F32 parameter.

I checked the consequences and they are mild on common targets: the wasm side
normalizes `useconds` before `Fw::Time::add`, so the `FW_ASSERT(newUSeconds <
1999999)` is not reachable; the fpybc side just makes a wrong timeout decision.
Worth an `std::isnan` guard for UB hygiene, but not a crash path I could reach.

---

## Verified clean this session (no finding)

WasmSequencer, read line by line and, where testable, exercised on the real
component through the wasm harness:

- **Guest ABI matches the compiler exactly.** All 11 imports the compiler
  declares match the sequencer's registered signatures (`fprime_v1`, plus the
  patched `env.pow/fmod/log`). Compiled 20+ programs across every operator and
  builtin and diffed the emitted wasm import section against what the
  sequencer provides: the complete libcall set is `pow`, `fmod`, `log`, all
  double. F32 operands always widen to F64, so no `powf`/`fmodf`/`logf` is ever
  required — there is no "compiles but cannot load" gap beyond the known `env`
  patch.
- **Malformed module robustness.** 365 corrupted modules (60 truncations, 260
  single-bit flips, 40 multi-byte corruptions of a module with functions,
  commands, telemetry, logging and serial output): zero aborts, zero hangs,
  zero Rust panics, zero silent deaths. Every one either loaded and ran or
  failed cleanly.
- **Guest bump allocator** (`guestAlloc`/`guestRealloc`, `WasmSequencerHelpers.cpp:66-125`):
  bounds arithmetic is overflow-safe and the `size > guestMemorySize` pre-check
  makes the pre-subtracted comparison sound; grow-in-place is correctly
  restricted to the last allocation.
- **fpybc stack primitives** (`FpySequencerStack.cpp`): `storeHelper`,
  `loadHelper`, `getField`, `peek`, `memCmp` and `return` all satisfy
  `Stack::copy`'s non-overlap assertion and `Stack::move`'s bounds assertion for
  every input that passes their own guards — verified by hand for each.
- **Sequence-argument path**, ground → binary arg specs → VM → stack/guest
  memory: identical results on both backends for U32/I8/U64+U8/bool/F32/F64/
  enum/struct/array/mixed-order arguments, arguments mutated then read, and
  arguments read from inside a function. Wrong total sizes rejected by both.
- **Command dispatch byte layout**: byte-identical dispatched buffers on both
  backends for all-const, all-runtime and mixed const/runtime arguments,
  including string literals followed by runtime arguments (the compact-vs-
  max_size accounting), empty strings, arrays, enums, response capture,
  commands in functions and loops, and `flags.assert_cmd_success`.
- **Types that are not constant-sized cannot reach the stack**: variables,
  telemetry reads and parameter reads of string-containing types
  (`Svc.CustomVersionDb`, `String_40`) are all rejected with a clean
  diagnostic on both backends, so the compact-serialization/`max_size`
  mismatch I went looking for is unreachable.
- **Sequence-run filename** must be a string literal; a dictionary string
  constant is rejected cleanly rather than crashing `_emit_seq_run_cmd`.
- **AST node equality** is id-based in practice (dataclass `__eq__` includes
  the `id` field, `__hash__` inherited from `Ast`), so the `value == node`
  rebinding of `break`/`continue` in `DesugarForLoops` cannot capture a
  structurally identical sibling loop.
- **`Fw::ExternalString` over the guest event buffer** is safe: the two-arg
  constructor writes only `buffer[0]`, the message is read in afterwards, and
  `length()` is computed live (`strnlen`), not cached.
- Random differential fuzzing (120 generated programs: nested if/for/while,
  break/continue, checks, functions, commands, serial writes) — 0 divergences,
  0 crashes. Front-end robustness probe (36 pathological inputs: BOM, CR-only,
  NUL, form feed, 5000-deep parens, 100k identifiers, 3000 variables, 500
  functions, 300 parameters) — no new crashes.
- **WasmSequencer controller command surface** (LOAD / INVOKE / WAIT / PAUSE /
  CONTINUE / CANCEL / GLOBAL_*) is covered by the component's own 4878-line
  unit test, including path traversal, base-dir joining, malformed magic,
  truncation, bad imports and start-function traps. Not re-tested here.

## Already known — re-derived and deduped, not re-reported

Reached independently this session, then found to be already recorded/filed:
telemetry & parameter size mismatch (`COMPILER_VM_SCAN_2026-09-05.md` §3 and
`AUDIT_FINDINGS_2026-09-04.md`); oversized directive length field crashing the
compiler with `struct.error` (#224 — note it also triggers at the **default**
`FwSizeStoreType = U16`, via a `log()` literal > 65535 bytes, not only the U8
config); `RecursionError` escaping codegen (#195 — also affects the wasm
backend, not just fpybc); sequence arguments silently dropped when the sequence
declares none (#205); `env.pow/fmod/log` absent from unpatched flight
(`test/harness/patches/README.md`); `MAX_SERIAL_PORTS` and max-stack analysis
gaps (#108); wasm cancel semantics (#220).
