# Compiler crash / backend interaction scan — 2026-09-05 (devel @ a69b17d)

Independent scan focused on compiler crashes, miscompilations and
compiler/bytecode/VM interactions. Arithmetic overflow, rounding and
truncation were deliberately skipped. Every finding below was checked
against the open/closed issue list (#37–#207) and the other reports in this
directory before being written down; nothing here duplicates them.

Method: ~330 compile-only probes (syntax/semantic edge cases, looking for
Python exceptions instead of clean diagnostics) plus ~120 runtime
differential probes run on the real FpySequencer and WasmSequencer through
the two harnesses, comparing exit codes, `write_to_port` bytes, dispatched
command bytes and fpybc stack balance. Scratch scripts live in the session
scratchpad, not in the repo.

## Findings

### 1. `time()` with a runtime `timeBase` or `timeContext` crashes both backends

`time()` is documented as "always const-evaluated" (macros.py: "The generate
function should never be called since this is always const-evaluated"), but
only argument 0 is checked to be a constant, and only incidentally: the
builtin declares no `const_arg_indices` at all. When `timeBase` or
`timeContext` is a runtime value, semantics accepts the call, const folding
records "unknown value" for it, and codegen takes the runtime-builtin path:

```python
x: U8 = 1
t: Fw.Time = time("2025-01-01T00:00:00Z", TimeBase.TB_NONE, x)
```

* fpybc: `AssertionError: FpyValue(InternalString, '2025-01-01T00:00:00Z')`
  in `try_emit_expr_as_const` (codegen_fpybc.py) — it tries to push the
  timestamp string literal as a stack value.
* wasm: `NotImplementedError: this builtin has no LLVM/wasm lowering yet`
  from `_generate_llvm_unsupported` (symbols.py).

Same with `timeContext=x` as a named argument, a `TimeBase`-typed variable as
the base, or inside a function taking the context as a parameter. Expected: a
clean "Argument 'timeBase' of 'time' must be a compile-time constant"
diagnostic, which is exactly what `CalculateConstExprValues.visit_AstFuncCall`
already produces for builtins that declare `const_arg_indices`. One-line fix:
`const_arg_indices=frozenset({0, 1, 2})` on `TIME_MACRO`.

### 2. wasm backend: constant aggregate accessed with a constant index, then a runtime index, crashes with `KeyError`

A folded constructor call that is first indexed/member-accessed with a
constant (so that sub-access folds too) and then indexed with a runtime value
crashes the LLVM backend. The fpybc backend compiles and runs it correctly.

```python
j: I64 = 1
c: Ref.Choice = Ref.TooManyChoices(Ref.ManyChoices(Ref.Choice.ONE, Ref.Choice.TWO),
                                   Ref.ManyChoices(Ref.Choice.RED, Ref.Choice.BLUE))[0][j]
```

```
File "src/fpy/codegen_llvm.py", line 285, in _emit_ptr
    parent_ptr = self._emit_ptr(sym.parent_expr, state)
File "src/fpy/codegen_llvm.py", line 280, in _emit_ptr
    return self._emit_to_temp_slot(expr, state)
File "src/fpy/codegen_llvm.py", line 312, in _emit_to_temp_slot
    slot = state.backend.temp_slots[expr]
KeyError: AstFuncCall(func=AstGetAttr(... 'TooManyChoices') ...)
```

Two more shapes that crash the same way:

```python
# array member of a constant struct, runtime index
j: I64 = 1
x: F32 = Ref.SignalInfo(Ref.SignalType.SINE, Ref.SignalSet(1.0, 2.0, 3.0, 4.0),
                        Ref.SignalPairSet()).history[j]

# nested: constant struct -> member -> constant index -> runtime index
c: Ref.Choice = Ref.ChoiceSlurry(<...>, Ref.Choice.ONE, <...>, Array_U8_2(1, 2)).tooManyChoices[0][j]
```

Cause: `AssignAddresses._temp_slot_parent` (codegen_llvm.py) creates a temp
slot for a non-addressable parent only when the *access itself* is not
const-folded. In `Ctor(...)[0][j]` the inner access `Ctor(...)[0]` is folded,
so it never asks for a slot for `Ctor(...)`; the outer access `[j]` sees an
addressable parent (`Ctor(...)[0]` is a `FieldAccess`) and asks for nothing
either. At emit time `_emit_ptr` walks the chain down to `Ctor(...)`, which
has no slot. `Ctor(...)[j][0]`, `Ctor(...)[j].m` and `f()[0][j]` (a non-folded
call as the root) all work, which is why the existing tests never hit it.
fpybc pushes the whole folded constant and uses GET_FIELD, so it is unaffected.

Suggested fix: in `_temp_slot_parent`, when the access is folded but is itself
the parent of a non-folded access, still walk down to the non-addressable
root and give it a slot — or, simpler, have `_emit_ptr` fold the constant
prefix of the chain (`Ctor(...)[0]` is a constant; store *it* to a temp slot
and GEP from there).

### 3. (low) fpybc evaluates a non-canonical bool byte inconsistently; wasm faults

The compiler assumes a bool is always `FW_SERIALIZE_TRUE_VALUE` (0xFF) or
`FW_SERIALIZE_FALSE_VALUE` (0x00) and emits `MEMCMP` for `==`/`!=` on bools
(and on structs containing bools), while the flight VM's `IF` and `NOT`
handlers treat any nonzero byte as true
(`FpySequencerDirectives.cpp`: `if_directiveHandler` pops a U8 and branches on
`!= 0`; `op_not` pushes TRUE iff the byte `== 0`). A bool that arrives as
0x01 — from a telemetry channel or from the sequence-argument buffer, neither
of which the fpybc VM validates — therefore behaves as true in `if`/`not` but
compares unequal to `True`:

```python
sequence(b: bool)          # args buffer = 0x01
x: bool = not b            # 0x00  (b treated as true)
if b: ...                  # taken
y: bool = b == True        # False (memcmp 0x01 vs 0xFF)
```

Observed on the harness with both a tlm channel (`Ref.cmdSeq0.BreakpointInUse`
= 0x01) and a bool sequence argument. The wasm backend instead faults the
sequence with `DESERIALIZE_ERROR_INVALID_BOOL` (20) on the read (#144's fix),
so the backends diverge. F Prime's own serializer always writes 0xFF/0x00, so
this needs a non-framework producer (a hand-built ground tool, a foreign
component filling `Svc::SeqArgs`) to trigger; I'm listing it because the two
backends disagree and because fpybc's own answer is self-inconsistent. Either
validate bool bytes in `PUSH_TLM_VAL`/argument loading on fpybc, or make `IF`/
`NOT`/`==` agree on what "true" means.

### 4. (design note, not filed) `now() - {seconds: 1}` compiles as Time − Time

`Fw.Time - Fw.TimeIntervalValue` is not in `TIME_OPS`, so an interval-shaped
struct literal on the right of `-` is adapted to the only overload that
accepts it, `(TIME, TIME, -)`: the literal becomes
`Fw.TimeValue(<default base>, 0, 1, 0)` and the result is an interval, not a
time. With the default `--time-base` this fails at run time with
"time_sub: operands have different time bases" (exit 1) on both backends;
with a matching base it silently computes "now minus one second after the
epoch". `now() - i` with `i: Fw.TimeIntervalValue` is at least rejected
("Op - undefined"). Worth either adding a `Time - TimeInterval` overload or
refusing to adapt an anonymous struct to `Fw.Time` when a `TimeInterval`
overload of the operator exists.

## Notes (covered elsewhere, not repeated as findings)

* A bare range statement (`1..2`) and a range in `==` (`(1..2) == (1..2)`)
  pass semantics and crash fpybc codegen with
  `KeyError: <class 'fpy.syntax.AstRange'>` (wasm reports a `BackendError`).
  #103 already asks for ranges-as-statements to be rejected cleanly, so this
  is that issue's range case; noting it here only because the crash is in
  codegen rather than semantics.
* `from helper import *` where `helper.fpy` defines `sleep` makes every
  `check` statement loop forever (imported-name flavour of #202).
* A user function named after a dictionary *module* (`import Ref` with a
  `Ref.fpy` alongside) shadows the dictionary module in the callable group;
  `Ref.cmdDisp.CMD_NO_OP()` then fails with "Unknown callable". Clean error,
  no silent change; types/values keep resolving to the dictionary because
  name groups are separate.

## Checked and found consistent (both backends agree; no crash)

* By-value semantics of arrays/structs across assignment, parameters and
  returns; swaps through a temporary; nested runtime-index reads (`m[i][j]`)
  and single-runtime-index writes at either level; struct-in-array member
  writes with runtime index; runtime index into a call result and into a
  telemetry array; member of a call result.
* Evaluation order: global/element/member modified by a call on the other
  side of `+` and of `==`; index modified by the rhs of an element store;
  `while` conditions and `for` bounds with side effects (bounds evaluated
  once); `and`/`or` short-circuit with side-effecting calls; call-argument
  nesting; assert exit-code expression not evaluated on success.
* Control flow: `return` from nested `for` loops and from a `check` body in
  a function called in a loop; `break`/`continue` across nested `for`s;
  loop variable at the I64 limit; `exit()` from a function in a loop
  (stack intentionally not unwound on both); recursion returning structs.
* Sequence arguments (U8/F32/array, bool/enum/I8) read and written from
  functions, runtime-index store into an argument array from a function.
* Telemetry/parameter reads: two reads in one expression, struct read then
  member modify, struct equality vs a constructed value, I16 sign extension,
  enum compare, missing channel/param (TLM_CHAN_NOT_FOUND / PRM_NOT_FOUND on
  both; wasm reports them as exit codes 3/5).
* Commands: string + runtime args, named-argument reordering, enum/struct
  args, response captured in a function, `flags.assert_cmd_success` cleared
  from a function or from within a command argument, 40-char string arg.
* Time: `now()` differences across `sleep`, `check` with persist/period,
  timeout hit, body-less check with a variable timeout, nested checks, check
  condition with side effects; `sleep_until` in the past; `time()` vs
  `now()` base mismatch (both exit 1 with the library's warning).
* Casts between I8/U8/I64/U64 round-trip identically; F32/F64 chains.
* Imports: cyclic, self-import, package paths, aliasing over dictionary
  module/type/builtin names, duplicate/star imports, imported file with
  variables, sequence args, or errors in unused functions (all clean errors).
* ~330 malformed-program probes (bad targets, void values everywhere, wrong
  types in every builtin slot, keyword-named identifiers, indentation/line
  continuation/unicode edge cases, 300-parameter functions, 6000-variable
  programs, 500-deep parentheses): all produced clean diagnostics except the
  two crashes above and the #103 range case.
* fpyasm text round trip (`fpybc_directives_to_fpyasm` → parse → assemble)
  reproduces the binary byte-for-byte for strings with quotes/backslashes/
  `#`/`;`, unicode logs, -0.0, sequence args, functions and checks.

---

# Part 2 — wasm / LLVM / WasmSequencer integration pass (same day)

Wider pass over the LLVM lowering, the linked modules, and the flight
`Svc::WasmSequencer` (test/fprime-wasm) as it would sit in a real deployment.
Findings 1–4 of part 1 were filed as #208–#210 (the bool note stayed
unfiled).

## Findings

### 5. Blocking commands longer than 60 s fail on wasm but not on fpybc, under default parameters

`Svc::WasmSequencer` applies `HOST_FUNCTION_TIMEOUT_SECS` (default **60**,
WasmSequencerParams.fppi) to every command it dispatches: `dispatchCommand`
starts the clock and `checkTimeout` fails the sequence with `REPLY_TIMEOUT`
when the response is late. `Svc::FpySequencer`'s equivalent,
`STATEMENT_TIMEOUT_SECS`, defaults to **0**, which its `checkStatementTimeout`
treats as "never time out". So the same `.fpy` that issues a blocking command
taking more than a minute — `Ref.cmdSeq1.RUN_ARGS("child.bin",
Svc.BlockState.BLOCK, ...)` where the child sleeps, a large
`FileHandling.fileDownlink.SendFile`, any command whose handler waits on
hardware — completes on the fpybc backend and fails on the wasm backend unless
the deployment changes a parameter. Nothing in the language, the compiler or
the docs mentions the difference. Code reading only: the harness answers a
blocking child immediately without advancing its clock, so it cannot reproduce
either timeout. Suggest aligning the two defaults (both 0, or both 60) and
documenting that a blocking command is subject to the sequencer's timeout on
the wasm backend.

### 6. (note) Guest `log()` at FATAL / COMMAND severity contradicts WASM-SEQ-014

Another report already observed that `log("x", Fw.LogSeverity.FATAL)` emits a
FATAL on both backends; the new piece is that the WasmSequencer SDD requirement
WASM-SEQ-014 says the sequencer "shall let a sequence emit events at the
non-reserved F´ severities … restricting to WARNING_LO/HI, ACTIVITY_LO/HI and
DIAGNOSTIC prevents untrusted code from triggering the FATAL handler", while
`dispatchEvent` (WasmSequencerInterpreter.cpp) has explicit `FATAL` and
`COMMAND` cases that call `log_FATAL_LogFatal` / `log_COMMAND_LogCommand`.
Verified on the harness: the wasm run reports the guest event at severity 1.
Whichever side is fixed, the compiler could reject those two severities in
`log()` (they are dictionary enum constants, so it is a one-line check in the
`log` builtin).

### 7. (extends #195) Deeply nested *statements* also crash both codegens with an uncaught `RecursionError`

200 nested `if` statements pass every front-end pass and then both codegens
die with a raw `RecursionError` traceback (fpybc in `emit_AstBlock`, wasm in
`EmitLlvmExpr.emit`). #195 describes the expression case on fpybc; the
statement case and the wasm backend are the same hole. `text_to_ast` and
`analyze_ast` are wrapped in `except RecursionError` in main.py but the codegen
stage is not.

### 8. (minor) A dictionary without `Svc.SeqArgs` crashes the compiler with an `AssertionError`

`_update_seq_args_from_dict` (state.py) asserts `"Svc.SeqArgs" in
dict_type_name_dict` instead of raising `DictionaryError` like the other
dictionary checks. Only a deployment with no sequencer at all produces such a
dictionary (WasmSequencer's `seqRunIn` port pulls the type in), so this is a
diagnostics nit. A dictionary with no `Svc.Fpy.*` types or constants at all
(a wasm-only deployment) compiles fine, which is the important case.

## Checked and found consistent (integration)

* **Imports.** The import sections of 111 linked modules (all golden tests
  plus this session's probes, including 255-byte array copies, struct
  parameters/returns, recursion with aggregates, pow/ln/%/`//`, float↔int
  casts, telemetry structs, 40 log calls) contain only the twelve
  `fprime_v1` functions and `env.pow`/`env.fmod`/`env.log` — exactly what
  `hostFprimeV1`/`hostEnv` register. No `memcpy`/`memset` or other libcall
  is ever emitted, so `--allow-undefined` cannot currently produce an
  unresolvable import.
* **Framework type aliases.** `FwOpcodeType`, `FwChanIdType`, `FwPrmIdType`,
  `FwIndexType`, `FwSizeStoreType`, `FwPacketDescriptorType`,
  `FwTimeBaseStoreType`, `FwTimeContextStoreType` are all read from the
  dictionary (`_update_configurable_type`), so the fpybc directive widths, the
  wasm command buffer's opcode, and the 11-byte `Fw.Time` buffer the host
  insists on (`Fw::Time::SERIALIZED_SIZE`, exact-match checked in
  `wasmTime`/`wasmReadTelemetry`) follow the deployment's config.
* **Host status checks.** fpybc's `pushTlmVal`/`pushPrm` and the wasm guest
  both require `Fw::TlmValid::VALID` / `Fw::ParamValid::VALID`, with the enum
  values taken from the dictionary; an unconnected `getTlmChan`/`getParam`
  port fails the sequence gracefully on both.
* **Time semantics.** `Fw::Time::compare` and `FpySequencer::checkShouldWake`
  both ignore the time context, as does the Fpy time library, so a context
  change cannot make one side incomparable while the other is not. The wasm
  sleep timer is derived from the host's own `getTime()`, so its
  `TIMER_INCOMPARABLE` path needs a mid-sleep base change.
* **Command dispatch.** The host copies the guest buffer after the packet
  descriptor and rejects `len > FW_COM_BUFFER_MAX_SIZE - sizeof(descriptor)`
  (#206 covers the missing compile-time check); stale responses are
  rejected by `cmdUid`; arguments are fully evaluated before the buffer is
  written, so a command site inside a recursive function cannot clobber its
  own buffer.
* **Guest memory.** With `--stack-first -zstack-size=4096 --page-size=1`, the
  largest probe declares a 4504-byte minimum memory; the default flight
  `guestMemorySize` is 8192 with `stackSize` 1024 and `heapPages` 8 — the
  known "no compile-time size feedback" situation, nothing new.
* **Reserved names.** `main` is uniqued away from the export; user functions
  named `memcpy`/`memset` compile and run correctly on both backends.
