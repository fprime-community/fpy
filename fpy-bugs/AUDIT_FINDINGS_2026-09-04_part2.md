# Fpy audit findings, part 2 — 2026-09-04 (branch devel @ a69b17d)

Scope: interactions between the compiler, the fpybc bytecode + Svc::FpySequencer,
and the LLVM/wasm backend + Svc::WasmSequencer; miscompilations and backend
divergences. Math/IEEE discrepancies deliberately excluded (#109-#131, #59).
Nothing here overlaps #183-#186 or #191-#195, nor the widening-directive
FW_ASSERT finding in AUDIT_FINDINGS_2026-09-04.md (part 1).

Method: every candidate was compiled through both backends and run on both real
sequencers via test/harness (probe runner and probe scripts in the session
scratchpad, `probes/`). A differential fuzzer (`probes/fuzz3.py`) also ran
360 random programs over control flow, functions, arrays, structs, enums,
commands with failing responses and `flags` toggles, comparing every global's
serialized bytes, the dispatched command bytes and the exit outcome across the
two backends. Only finding 1 below showed up in the fuzz runs (5 of 60 early
programs; 0 of 300 once the generator avoided that pattern).

Filed 2026-09-04: finding 1 = #198, finding 2 = #199, finding 3 = #196,
finding 4 = #197. Each finding keeps its "novelty" note against the tracker.

---

## Finding 1 — backends disagree on `a[f()]` when `f` mutates `a`

**Severity:** medium (silent value divergence between backends; reachable from
ordinary code)
**Kind:** miscompilation / unspecified evaluation order

```python
a: Ref.FpyExampleArray = Ref.FpyExampleArray(10, 20, 30)
def f() -> I64:
    a[0] = 99
    return 0
x: U32 = a[f()]     # fpybc: 10     wasm: 99
```

- fpybc `emit_AstIndexExpr` (src/fpy/codegen_fpybc.py:894) pushes a *copy of
  the whole parent* onto the operand stack first, then evaluates the index
  (:897-898) and GET_FIELDs out of the stale copy. The element the sequence
  reads is the value from *before* the index expression ran.
- wasm `_emit_ptr` (src/fpy/codegen_llvm.py:285-307) forms a pointer to the
  variable's live storage, evaluates the index, then GEPs and loads, so it
  reads the value *after* the index expression ran (Python semantics).
- Same asymmetry for any variable-rooted parent (local, global, parameter,
  `s.arr[f()]`, `a[i][f()]`). Telemetry/parameter-rooted parents agree
  (both read the channel before the index: wasm copies a non-addressable
  parent to a temp slot first).
- Stores agree (`a[f()] = v`: both evaluate rhs, then index, then store in
  place), as do `a[i] + f()` / `f() + a[i]` and call-argument order
  (verified).

SPEC.md specifies the order for element *assignment* (rhs, then item) but the
element *read* order is a `[element](todo)` link, so neither backend is
"wrong"; the divergence is the bug. Implementing the fpybc TODO at
codegen_fpybc.py:891 ("read the element in place at its frame offset instead
of copying the whole array") would make fpybc match wasm.

Repro: `probes/batch2.py` A1/A1b; fuzz seeds 11, 17, 59 in
`probes/fuzz_findings/`.

**Novelty:** no open issue covers evaluation order of index expressions
(#116 is const-fold short-circuit order; #103 is untyped expressions).

---

## Finding 2 — relative `sleep()` whose wake-up overflows U32 seconds wakes immediately on fpybc, sleeps forever on wasm

**Severity:** medium-high on fpybc (a "sleep forever / park" wait returns at
once, so whatever follows runs early); divergence between backends
**Kind:** VM (flight) + backend divergence

```python
t0: Fw.Time = now()
sleep(4294967290, 0)        # at t = 100 s
assert now().seconds > t0.seconds + 1000, 3   # fpybc: fails at once (exit 3); wasm: passes
```

- fpybc: `waitRel_directiveHandler` (FpySequencerDirectives.cpp:300-314) calls
  `Fw::Time::add(seconds, useconds)` (Fw/Time/Time.cpp:240-249), which computes
  `newSeconds = m_seconds + seconds` in U32 and wraps. The wake-up time lands
  in the past, `checkShouldWake` wakes on the next timer tick, and the sequence
  continues with no fault or event. Also reproduced with `sleep(4000000000)`
  at t = 1e9 s.
- wasm: `dispatchRelativeSleep` (WasmSequencerInterpreter.cpp:398-415) computes
  the deadline in U64 and clamps to `(U32 max, 999999)`, i.e. sleeps ~136
  years. No fault either.
- `check ... period/timeout` is safe: the builtin `time_add` in
  src/fpy/builtin/time.fpy checks U32 overflow and asserts. Only the raw
  `sleep()` builtin is affected.

Suggested fix: reject the overflow in the WAIT_REL handler (fail the sequence
with an error code) and/or have the compiler warn on constant seconds that can
overflow any plausible clock; in any case the two backends should agree.

Repro: `probes/batch2.py` J3/J3b.

**Novelty:** #193 is the neighbouring bug (useconds >= 1e6 trips FW_ASSERT in
Fw::Time; the U32-wrapping *useconds* sleeps too short). This is the *seconds*
field wrapping silently inside `Fw::Time::add`, plus the wasm clamp
divergence. Could be filed as its own issue or added to #193 -- your call.

---

## Finding 3 — `==`/`!=` between two void calls is accepted; fpybc emits MEMCMP(0), wasm codegen crashes

**Severity:** low
**Kind:** semantics gap + compiler crash + one more unguarded stack grower

```python
def f():
    pass
b: bool = f() == f()   # accepted by semantics
```

- `pick_intermediate_type` (src/fpy/semantics.py:1745-1749) allows `==`/`!=`
  on any two expressions of the same non-numeric type, including `Nothing`.
  Every other use of a void call (`y = f()`, `f() + 1`, `if f():`, `not f()`,
  `g(f())`, `write_to_port(p, f())`) is rejected with a proper error.
- fpybc: `_emit_bytes_equal` (src/fpy/codegen_fpybc.py:1313-1324) emits
  `MEMCMP(size=0)`; the VM handler (FpySequencerDirectives.cpp:1335-1366)
  pops 0 bytes and pushes the 1-byte result with **no STACK_OVERFLOW guard**
  -- the same class of unguarded growth as the seven widening directives in
  part 1 (net +1 byte; reachable only via this construct with the operand
  stack at exactly MAX_STACK_SIZE, so contrived).
- wasm: `_emit_value_equal` (src/fpy/codegen_llvm.py:783-806) receives the
  void call result and dies at :799 with `AssertionError: void` -- an internal
  crash instead of a compile error.

Fix: reject `Nothing` operands in `pick_intermediate_type` (require
`is_concrete`).

Repro: `probes/batch6.py` V7/V8, `probes/batch3.py` S7.

**Novelty:** #103 ("Compiler crashes or gives misleading errors for expressions
with no concrete runtime type") is the closest; `Nothing` is such a type, so
this may belong as a comment on #103 rather than a new issue.

---

## Finding 4 — semantic errors and warnings inside imported files are reported against the importing file

**Severity:** low-medium (misleading diagnostics; could send someone debugging
a flight sequence to the wrong file)
**Kind:** compiler diagnostics

`bad.fpy` line 4 contains `return ok() + 10` (U64 into a U8 return). Compiling
`main.fpy` (`import bad` + 3 more lines) with `fprime-fpyc ... -i <dir>` prints:

```
main.fpy:4 Expected U8, found U64
     3 | y: U8 = 2
     4 | z: U8 = bad.ok()
                ^^^^^^^^^
```

i.e. the importing file's name, its line 4 and a caret under an unrelated
expression. A warning inside an imported file is misattributed the same way
(`main4.fpy:3 warning [shadow-callable] ...` shows main's line 2). Parse errors
are attributed correctly, because `_lex_and_parse` (src/fpy/imports.py:259)
scopes `diagnostic_context` around parsing only; nodes carry only line/column
`meta`, and `format_diagnostic` (src/fpy/error.py:104-123) reads the
module-global `file_name`/`input_lines` of whatever file is current when the
semantic passes run (always the main file).

Fix: record the source file on each block/node (or per imported block) and
have `format_diagnostic` pick the file's name and lines from the node.

Repro: `probes/impdiag/` (main.fpy + bad.fpy; main4.fpy + warn.fpy).

**Novelty:** not tracked. #134 ("Import + definitions refactor") and #72
("Debug line numbers") are adjacent but neither describes diagnostics
attribution; check #134's body before filing.

---

## Observations (design questions / not bugs; no action unless you want one)

- **`assert cond, 0` and `exit(0)` end the sequence with SUCCESS** on both
  backends (the RUN command answers OK; a parent calling it via RUN_ARGS sees
  OK). Consistent with the spec's `exit` semantics, but an assert with exit
  code 0 is a footgun (SPEC.md "Assert statement" says "halt the program",
  which reads as failure). `probes/batch1` P4.
- **A sequence can emit a FATAL event** via `log(msg, Fw.LogSeverity.FATAL)`
  on both backends (`LogFatal`). In a deployment with the standard
  FatalHandler that resets/aborts the FSW. Worth deciding whether sequences
  should be allowed to do that.
- **STATEMENT_TIMEOUT_SECS applies to sleeps on fpybc but not on wasm.** The
  fpybc SLEEPING state runs `checkStatementTimeout` on every timer tick
  (FpySequencerStateMachine.fppi, `state SLEEPING`), so a `sleep()` or a
  `check ... period` longer than the timeout fails the sequence with
  DirectiveTimedOut; the wasm sequencer explicitly exempts sleeps from
  HOST_FUNCTION_TIMEOUT_SECS (WasmSequencerInterpreter.cpp:605-611). Code
  reading only: the harness jumps its clock straight to the wake-up time, so
  it cannot reproduce the timeout ordering.
- **No compile-time size feedback for wasm.** A 40-line fuzz program (830
  fpybc directives, 4.1 KB wasm) fails to load on the WasmSequencer with the
  default config (8 heap pages x 8 KiB): `ModuleLoadFailed:
  ERR_OUT_OF_MEMORY`. fpybc gets a hard compile-time cap (2048 directives);
  the wasm backend has none, so a sequence can compile cleanly and be
  unloadable on the target. Fuzz seeds 203, 228.
- **`sendCmd` failure leaves LastDirectiveError = NO_ERROR.** Both cmd
  handlers return `stmtResponse_failure` without setting `error`
  (FpySequencerDirectives.cpp:461-463, :1397-1399); `CMD_SERIALIZE_FAILURE`
  exists but is never assigned. Unreachable from the compiler while the
  dictionary carries FW_COM_BUFFER_MAX_SIZE (FinalChecks rejects such
  commands), so telemetry-only.
- **Test infrastructure:** the fprime-wasm submodule's
  `SequenceArgumentsMaxSize` is 12 (default/config/AcConstants.fpp:71) vs 255
  in the Ref dictionary, so the wasm harness rejects any sequence with more
  than 12 bytes of arguments ("sequence arguments do not fit in
  Svc::SeqArgs"); wasm coverage of sequence arguments is limited to that.
- **Reminder:** the compile-time FW_CMD_STRING_MAX_SIZE check for command
  string arguments (commit 1999e8f) lives only on `build-real-sequencers`, not
  on devel.

---

## Verified clean this session (both backends agree, no leak, no fault)

- Command wire bytes for runtime args of I32/F32/U8/bool/enum/string mixes,
  seq-run commands with runtime args (parent + child), `flags` toggles in
  callers/callees, failing responses (CMD_FAIL 17 on both).
- `write_to_port` bytes for bool, struct, enum, array, string, "", F64
  expression, array element, I8 member, enum constant.
- Telemetry struct member / array element reads with constant and runtime
  indices, inside functions and main.
- Dynamic global element stores (`a[i] = v`, `s[i].value = ...`) from inside
  functions; by-value struct parameters; sequence-argument mutation and access
  from functions (fpybc).
- `check` in functions (with `return` in both bodies), nested checks,
  `break`/`continue` in a check body inside a `for`, timeout path with a
  period longer than the remaining timeout, anon-struct timeout/period.
- Call-argument evaluation order, `for` bound capture / loop-var assignment,
  assert exit-code expression not evaluated on success, imports with
  same-named functions in two modules plus a local one.
- wasm: globals/functions named `__heap_base`, `__data_end`, `__memory_base`,
  `__table_base`, `__tls_base`, `__wasm_call_ctors`, `memory`,
  `__indirect_function_table`, `fmod`, and every fprime_v1 host import name
  (only `pow`/`log`/`__stack_pointer` from #191 misbehave).
- 360 random differential programs (`probes/fuzz.py`, `fuzz2.py`,
  `fuzz3.py`): no divergence other than finding 1.
