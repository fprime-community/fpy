# Compiler / bytecode / VM interaction scan — 2026-09-05

Independent scan of the compiler front end, the fpybc code generator, the
LLVM/wasm code generator, and both flight VMs (Svc::FpySequencer in
`test/fprime`, Svc::WasmSequencer in `test/fprime-wasm`). Focus: contract
mismatches and miscompilations, not arithmetic semantics. Every finding was
checked against the open issue list (nothing below duplicates #1–#199) and,
unless marked "code reading only", was reproduced on the real sequencers
through `test/harness` (fpybc) and `test/harness/wasm` (wasm).

Method: read every source file on both sides of the bytecode contract, then
ran ~60 hand-written probe sequences and ~1,000 randomly generated
struct/array/function/loop programs through both backends, comparing the
bytes each backend wrote to `write_to_port` (details at the end).

Filed 2026-09-05 after review: finding 1 → #201, finding 2 → #202, finding 5 →
#205, finding 8 → #206, log truncation (finding 9) → #203, the unused
`CMD_SERIALIZE_FAILURE` note → #204. Findings 3, 4, 6 and 7 are not filed.
Items are ordered by how much I think they matter for flight.

---

## 1. String literal escapes are mangled: `strip()` eats quotes and `\"` / `\\` / `\n` are never unescaped

**Where:** `src/fpy/syntax.py:461` `handle_str` (`s.strip("'").strip('"')`),
used for every `STRING` token.

**What happens:** the grammar's `STRING` regex explicitly accepts escaped
quotes (`(?<!\\)(\\\\)*?"`), but the transformer never processes escapes; it
just strips *every* leading/trailing quote character of either kind. The
constant that ends up in a command argument or event is therefore not what
the source says:

| Source literal        | Emitted bytes / event text | Expected      |
|-----------------------|----------------------------|---------------|
| `'"hi"'`              | `hi`                       | `"hi"`        |
| `"say \"hi\""`        | `say \"hi\`                | `say "hi"`    |
| `"a\"b"`              | `a\"b` (4 bytes)           | `a"b`         |
| `"a\nb"`              | `a\nb` (backslash, n)      | newline (or a diagnostic) |
| `"path\\to"`          | `path\\to` (two backslashes) | `path\to`   |

Identical on both backends (front-end bug). Repro:

```
log('"hi"')                                   # event text is: hi
CdhCore.cmdDisp.CMD_NO_OP_STRING("a\"b")      # arg bytes 0004 61 5c 22 62
```

**Why it matters:** command string arguments (file names, ping entries) and
log messages are silently altered; a quote-bearing path never reaches the
spacecraft as written. **Suggested fix:** strip exactly one quote of the kind
the token starts with, and either implement the escape set the regex accepts
(`\"`, `\'`, `\\`, `\n`, `\t`) or reject backslashes with a compile error; add
the escape rules to SPEC.md "String literals".

## 2. A user function named `sleep`, `now`, `time_cmp`, `time_add`, `time_sub` or `time_interval_cmp` silently hijacks every `check` statement

**Where:** `src/fpy/desugaring.py` `DesugarCheckStatements` builds plain
`AstIdent("sleep")`, `AstIdent("now")`, `AstIdent("time_cmp")`, ... nodes that
are then resolved in the *user's* scope. `DesugarTimeOperators` was
deliberately hardened against exactly this (it resolves `time_add` etc. in
`state.base_scope`, see its comment and
`test_sequence_function_does_not_hijack_time_desugaring`), but the check
desugaring was not.

**Reproduced (both backends):**

```
def sleep(seconds: U32 = 0, useconds: U32 = 0):   # only a shadow-callable warning
    log("user sleep called")
x: U8 = 0
check x == 1 timeout {seconds: 1, useconds: 0} period {seconds: 0, useconds: 500000}:
    log("cond true")
timeout:
    log("timed out")
```
→ the check's per-iteration `sleep(period)` calls the user's function, the
loop never sleeps, and the sequence spins forever (harness dispatch cap hit
on fpybc and wasm). `def now() -> Fw.Time` likewise loops forever; a
`def time_cmp(...)` that returns `GT` makes every check time out instantly
(observed: "user time_cmp called" then "timed out" with a 100 s timeout).
The same happens through `from lib import sleep`. A top-level *variable*
named `Fw` also breaks every check with a confusing "U8 is not a struct"
error, because `Fw.TimeComparison.INCOMPARABLE` is resolved in the value
group.

**Why it matters:** `check` is the flight-rule primitive; one innocuous
helper name changes its timing and its timeout behaviour with only a generic
shadow warning. **Suggested fix:** resolve the desugaring's callees in the
base scope like `DesugarTimeOperators` does (e.g. pre-resolved
`resolved_symbols` entries, or `$`-escaped aliases such as `$check_sleep`
registered in the base callable scope that user code cannot spell), and add a
regression test per helper name.

## 3. Neither VM checks that a telemetry/parameter value has the size the compiled code assumed — a stale ground dictionary produces silently wrong values

**Where:** fpybc `pushTlmVal_directiveHandler` / `pushPrm_directiveHandler`
push `tlmValue.getSize()` bytes (`FpySequencerDirectives.cpp:389,450`) while
the generated code consumes `ch_type.max_size` bytes; the directives carry no
expected size. wasm `dispatchTelemetry`/`dispatchParameter` only reject a
value *larger* than the guest buffer; a smaller one leaves stale bytes that
the guest deserializes as the value (`WasmSequencerInterpreter.cpp:312,335`).

**Reproduced** with a channel whose live value is a different size than the
dictionary type (exactly what a ground dictionary from a previous FSW build
produces):

| Live value vs dictionary type | fpybc                                                        | wasm |
|-------------------------------|--------------------------------------------------------------|------|
| 1 byte, type `Ref.Choice` (4) | STACK_ACCESS_OUT_OF_BOUNDS in this program; in others the compare eats 3 bytes of the frame first | runs, value read as `02 00 00 00` (stale buffer) |
| 8 bytes, type `U32` (4)       | **succeeds**, variable gets `00000000` (low half), 4 stray bytes left on the operand stack per read | clean BufferTooSmall failure |

```
c: U32 = CdhCore.cmdDisp.CommandsDispatched     # tlm provided as 8 bytes 0000000500000000
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, c)   # fpybc writes 00000000, cmdResponse OK
```

**Why it matters:** the sequence keeps running with garbage in a flight rule
condition instead of failing. **Suggested fix:** add the expected size to
`PUSH_TLM_VAL`/`PUSH_PRM` (schema bump) and fail with a dedicated error when
`getSize()` differs; make the wasm `tlm`/`prm` hosts require
`tlmBuffer.getSize() == valueLen`. (A dictionary hash in the file header would
catch the general problem earlier.)

## 4. `write_to_port` of a value shorter than the typed port's arguments asserts the flight FpySequencer (code reading only)

**Where:** `FpySequencerDirectives.cpp:1795-1796`

```
Fw::SerializeStatus portStatus = this->serialOut_out(portIndex, buf);
FW_ASSERT(portStatus == Fw::SerializeStatus::FW_SERIALIZE_OK, ...);
```
When `serialOut[i]` is connected to a *typed* input port (the normal flight
wiring), the generated `invokeSerial` deserializes the port arguments and
returns `FW_DESERIALIZE_BUFFER_EMPTY` etc. on a short buffer
(`TimePortAc.cpp:122-125`); the sequencer treats that status as a coding error
and FW_ASSERTs, taking down the FSW. The compiler cannot catch it:
`write_to_port`'s value parameter is `SIZED`, so `write_to_port(PORT, U8(1))`
compiles against a port that expects a `U32`. Extra bytes (a value longer
than the port's arguments) are silently ignored instead. The wasm sequencer
handles the same status gracefully (`SerialPortSendFailed` +
`hostResponseFailure`).

Not reproduced empirically: both test harnesses wire `serialOut` to serial
input ports, which never deserialize. Same class as #193 (sequence content
reaching an FW_ASSERT). #179 is the long-term fix (port signatures known to
the compiler); until then the directive should return a
`SERIAL_PORT_...` error instead of asserting.

## 5. Sequence arguments given to a sequence that declares none: fpybc rejects, wasm silently ignores

`RUN_ARGS` with a 2-byte argument buffer on a sequence without `sequence()`:
fpybc fails validation (ArgSizeMismatch, cmdResponse EXECUTION_ERROR); the
wasm module never calls the `args` host import, so the run succeeds and the
arguments are dropped. Small divergence in the runner contract; the wasm
prologue (`_emit_seq_args_prologue`) is only emitted when there are
declared arguments, so the check `got != total` never runs for the
zero-argument case. Low.

## 6. Float → narrow integer casts differ between backends (and from const folding)

fpybc lowers `U8(f64)` as `FPTOUI` (saturate to U64) then `ITRUNC_64_8`
(wrap); the LLVM backend uses `llvm.fptoui.sat.i8` (saturate to U8). The
constant folder wraps.

| Expression (runtime `f = 300.0`, `g = -3.7`) | fpybc | wasm | folded |
|----------------------------------------------|-------|------|--------|
| `U8(f)`                                      | 44    | 255  | 44     |
| `I8(f)`                                      | 44    | 127  | 44     |
| `U8(g)`                                      | 0     | 0    | 253    |

Casting semantics rather than arithmetic, and #59 is the umbrella issue;
listed so the backend deltas are known. Low.

## 7. `for` over an existing variable creates a shadowing variable instead of reusing it (spec divergence)

SPEC.md "For loop statement": "If `loop_var` resolves to a previously-defined
variable ... that variable becomes the loop variable". The implementation
(`DefineVariables.visit_AstFor`) always defines a fresh variable in the body
scope and emits `shadow-value`:

```
x: I64 = 100
for x in 0..3:
    pass
write_to_port(P0, x)      # 100 on both backends; spec says 3
```
Both backends agree with each other, so this is a spec-vs-implementation
question, not a miscompile. (Noted before in an earlier assessment; not filed.)

## 8. The wasm backend has none of fpybc's `FinalChecks`

`FinalChecks` (fpybc) rejects at compile time commands whose arguments exceed
`FW_CMD_ARG_BUFFER_MAX_SIZE` or whose packet exceeds
`FW_COM_BUFFER_MAX_SIZE`, and sequences with too many directives. The LLVM
backend emits the same command buffers with no size check; an oversized
command traps at run time in `wasmCommand` (`BufferTooLarge`). Latent with the
Ref dictionary (no command can exceed 506 bytes), real for a dictionary with
larger commands. Low.

## 9. `log()` messages longer than `FW_LOG_STRING_MAX_SIZE` are silently truncated, with no compile-time warning

A 300-character `log("AAA...")` is cut to 128 characters by both VMs
(`popEvent` clamp; wasm `wasmEvent` clamp). The compiler knows the message at
compile time and could warn or error. Low.

## Notes (not filed as findings)

* `src/fpy/macros.py` `MACRO_SLEEP_FLOAT` / `generate_sleep_float` is dead
  code (not in `MACROS`) and is wrong as written: it pushes `1_000_000.0` but
  never multiplies (missing `FloatMultiplyDirective`), so `WAIT_REL` would pop
  1,000,000 µs and then 4 bytes of the fraction as "seconds". Delete it or fix
  it before it is ever wired in.
* `sendCmd` failure paths return `stmtResponse_failure` without setting an
  error, so `LastDirectiveError` stays `NO_ERROR` and the defined
  `CMD_SERIALIZE_FAILURE` code (6) is never used. Unreachable from the
  compiler thanks to `FinalChecks`, but the telemetry would be misleading if
  it ever happened.
* `STATEMENT_TIMEOUT_SECS` is also checked while the sequencer is `SLEEPING`
  (state machine `SLEEPING: on checkTimersIn do { checkShouldWake,
  checkStatementTimeout }`), so an operator who sets a 30 s command timeout
  also makes any `sleep(60)` fail the sequence. Probably intended; worth a
  sentence in the parameter's documentation.
* Already filed as #200 (not repeated here): the widening stack ops
  (`SIEXT_*`/`ZIEXT_*`/`SITOFP` …) push 8 bytes after popping fewer without
  a `STACK_OVERFLOW` guard, so a guest sequence can hit `Stack::push`'s
  FW_ASSERT. Finding 4 above is the same class of hazard on a different
  directive.

## What was checked and found consistent

* Frame layout and calling convention: mixed-size parameters, parameter
  mutation, nested calls inside argument lists, recursion returning structs,
  returns from inside `for` loops, functions reading/writing globals
  (LOAD_ABS/STORE_ABS) and sequence arguments, local arrays modified through
  runtime indices and returned, 2-D arrays and struct-member arrays with
  runtime indices, element/member stores into globals from functions.
* Control flow: `break`/`continue` in desugared `for` loops, `while True`
  with commands, `check` inside functions and loops (`persist`, `period`,
  `timeout`), short-circuit `and`/`or` with side-effecting calls, `exit()` in
  nested calls, exit codes outside 0..255, `assert ..., 0` (ends the sequence
  successfully on both backends, matching `exit(0)`).
* Command dispatch: constant and runtime struct/array/enum/bool arguments,
  nested calls as arguments, captured `Fw.CmdResponse`, `flags.assert_cmd_success`
  at top level and inside functions, seq-run commands with a `Fw.Time`
  subtraction as an argument (in-place AST rewrite keeps `resolved_args`
  in sync), child sequence argument buffers.
* Telemetry/parameters of enum, bool, nested-array and struct-member-array
  types; equality of structs/arrays/enums; time comparison operators and
  `sleep_until`/`sleep(0)` with clock jumps; empty `log("")`; anonymous
  struct default arguments; `sequence()` with no parameters.
* Differential fuzzing: 457 random programs (globals, functions with
  struct/array parameters, loops, `if/elif/else`, member/element stores with
  runtime indices, narrow integer widening/truncation) produced identical
  `write_to_port` bytes on both backends; the only compiler failures were the
  known #185 (`struct.error` on a runtime-index store into a parameter
  array). A second run of 800 programs adding I8/I16/I32/U8/U16/F32
  variables (sign/zero extension, truncation, F32<->F64 widening through the
  cast and coercion paths) produced 650 valid programs, all with identical
  bytes on both backends (the remainder were generator artifacts: empty
  ranges and loops that reassign their own counter, which spin on both).

Scratch scripts (driver, probe batches, fuzzer) live in the session
scratchpad, not in the repo.
