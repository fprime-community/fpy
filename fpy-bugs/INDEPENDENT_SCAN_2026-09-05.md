# Independent compiler / bytecode / VM scan — 2026-09-05

Scope: interactions between the compiler front end, the fpybc codegen, the
flight `Svc::FpySequencer` VM (test/fprime @ cd3845f), the LLVM/wasm codegen
and the `Svc::WasmSequencer` host (test/fprime-wasm @ 8d5ad09f + the local
harness patch). Mathematical discrepancies (fold-vs-runtime rounding,
overflow policy, div/mod corner cases) were deliberately left out. Existing
open issues #39–#200 were read first; nothing below duplicates one of them
unless noted. No other in-progress reports were consulted.

Method:

* Full read of `codegen_fpybc.py`, `codegen_llvm.py`, `directives.py`,
  `assembler.py`, `macros.py`, `desugaring.py`, `semantics.py`,
  `wasm_host.py`, and of `FpySequencer{Directives,RunState,Stack,
  StateMachine,ValidationState}.cpp` and `WasmSequencer{Host,Interpreter,
  Controller,Helpers}.cpp`, checking every directive's operand order, field
  types and bounds checks against what the compiler emits.
* ~60 hand-written programs run on both real sequencers through the
  harnesses (`scratchpad/batch*.py`), covering frames/globals/params at
  negative offsets, nested member/element chains with runtime indices,
  recursion, defaults, imports, seq-run commands, tlm/prm reads, commands
  with runtime args, `check`, time ops, serial writes, sequence args.
* Two differential fuzzers (`scratchpad/fuzz.py`, `fuzz2.py`): 600 random
  well-typed programs (aggregates, functions, loops, globals, tlm/prm reads,
  commands, sequence args, serial writes) run on both backends, comparing
  exit codes, serial bytes and command buffers. 463 ran on both backends and
  matched byte-for-byte; 55 hit the known fpybc codegen crashes #184/#185;
  the rest were generator type errors. No mismatch and no new crash.

## Findings

### 1. `assert` whose exit code evaluates to 0 makes the sequence *succeed* (both backends)

```python
x: I32 = 0
assert x == 1, x        # fires; exit code expression is 0
exit(5)                 # never reached; sequence reports OK
```

Also `assert False, 0`. Verified on both harnesses: the RUN command answers
OK and `sequencesSucceeded` is incremented.

Cause: `emit_AstAssert` (`codegen_fpybc.py:1234`, `codegen_llvm.py:1237`)
lowers a failed assertion to the same `EXIT` path as the `exit()` builtin,
and both VMs define exit code 0 as normal completion
(`FpySequencerDirectives.cpp` `exit_directiveHandler`, "exit(0), no error";
`WasmSequencerController.cpp:489` `guard_interpreterSucceeded`). SPEC
"Assert statement" says a failed assertion halts the program; nothing there
says a 0 code turns it into success. A runtime-computed code (e.g.
`assert v > 0, I32(v)`, a common pattern to surface the bad value) is 0
exactly in the case the author most wants to catch. Suggested fix: reject a
constant 0 at compile time and have the lowering substitute a nonzero code
(e.g. `EXIT_WITH_ERROR`) when the runtime value is 0, or give assert its own
failure directive. Severity: high (silent success on a failed assertion).

### 2. `write_to_port` payload the connected typed port cannot deserialize `FW_ASSERT`s the flight FpySequencer (wasm sequencer fails the sequence gracefully)

`popSerializable_directiveHandler` (`FpySequencerDirectives.cpp:1795-1796`):

```cpp
Fw::SerializeStatus portStatus = this->serialOut_out(portIndex, buf);
FW_ASSERT(portStatus == Fw::SerializeStatus::FW_SERIALIZE_OK, ...);
```

When `serialOut` is connected to a *typed* input port, the generated port
code deserializes the arguments and returns the status instead of asserting
(`*PortAc.cpp` `invokeSerial`: `_status = _serializer.deserializePortArgs(_buffer);
if (_status != FW_SERIALIZE_OK) return _status;`). So a sequence that writes
fewer bytes than the port's argument list needs, e.g.
`write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, U8(1))` into a port
that takes a `U32`, gets `FW_DESERIALIZE_BUFFER_EMPTY` back and the
sequencer asserts, taking down the flight software. The compiler cannot
type-check the port (that is #179), so this is reachable from any sequence
that compiles. The `WasmSequencer` handles the same status with
`SerialPortSendFailed` + `hostResponseFailure`
(`WasmSequencerInterpreter.cpp:476-497`). Same class as #200. Severity:
high (guest-reachable FW_ASSERT); fix is a `DirectiveErrorCode` failure in
the VM (there is no code for it yet; `CMD_SERIALIZE_FAILURE` is the nearest).
Could not be demonstrated on the harness because the harness's `serialIn`
is itself a serial port.

### 3. A user function named `now`, `time_cmp`, `time_add`, `time_sub`, `time_interval_cmp` or `sleep` silently hijacks the `check` statement's generated code

```python
def time_cmp(lhs: Fw.Time, rhs: Fw.Time) -> Fw.TimeComparison:
    return Fw.TimeComparison.GT
def now() -> Fw.Time:
    return Fw.Time(TimeBase.TB_WORKSTATION_TIME, 0, 5, 0)
n: U32 = 0
check n > 0 timeout Fw.TimeInterval(2, 0):
    exit(1)
timeout:
    exit(2)
exit(3)
```

Both backends exit 2 immediately: the check's deadline test called the
user's `time_cmp`. `DesugarCheckStatements` builds plain identifiers
(`desugaring.py:562, 588, 602, 614, 666-668, 705`) that resolve through the
main sequence's scope, while `DesugarTimeOperators._make_func_call`
deliberately resolves in `base_scope` for exactly this reason (the
"does not hijack" test). The only signal is the `shadow-callable` warning,
which is not an error by default. Fix: resolve the check statement's helper
calls in the base scope the same way the operator desugaring does (or emit
`$`-prefixed unnameable aliases). Severity: medium.

### 4. Float-to-integer cast on fpybc wraps instead of saturating (diverges from wasm and from #59)

```python
x: F64 = 300.0
y: U8 = U8(x)      # fpybc: 44, wasm: 255
```

`convert_numeric_type` (`codegen_fpybc.py:546-593`) lowers the cast as
`FPTOUI` (saturates to U64) followed by `ITRUNC_64_8` (wraps), so the
saturation the VM implements is thrown away for every narrower target; the
wasm backend uses `llvm.fptoui.sat.i8` and saturates, which is what #59
specifies ("Float to integer overflow saturates the integer"). Same for
signed targets and for F32 sources. Listed here because it is a
backend/spec divergence in the cast lowering rather than an arithmetic
corner case; it is casting semantics, so lower priority if #59 is still in
flux. Severity: medium.

### 5. Valid U64 arithmetic above 2^63 traps `ARITHMETIC_OVERFLOW` on fpybc

```python
x: U64 = 0x4000000000000000
y: U64 = x * 3       # 3*2^62 fits in U64; fpybc traps, wasm gives 0xC000000000000000
```

`ADD`/`SUB`/`MUL` are shared by signed and unsigned operands
(`pick_binary_op_case` maps both to `ADD_INT`), and `op_add/op_sub/op_mul`
(`FpySequencerDirectives.cpp:728-799`) pop `I64` and apply signed overflow
checks, so any unsigned operand or result with the top bit set is
misclassified. #111 covers unsigned *wrapping* silently; this is the
opposite failure (a false trap on a representable result). Math-adjacent,
so listed last among the semantic items. Severity: low-medium.

### 6. String literal escape sequences are kept verbatim

`"a\"b"` compiles to the 4 bytes `a\"b` (backslash included) in command
arguments and `log` messages; `"\t"` is a backslash and a `t`. The lexer
accepts backslash escapes (`grammar.lark:254`), but `handle_str`
(`syntax.py:461`) only strips the quotes. Whatever the intended rule is, a
literal that lexes as an escape but is not unescaped sends bytes the author
did not write. Severity: low.

### 7. `sendCmd` failure leaves `lastDirectiveError == NO_ERROR`

`constCmd_directiveHandler` / `stackCmd_directiveHandler` return
`stmtResponse_failure` without setting `error` when `sendCmd` fails
(`FpySequencerDirectives.cpp:461-463, 1397-1399`), so the sequence fails
while `LastDirectiveError` telemetry says no error; `CMD_SERIALIZE_FAILURE`
exists for this and is never used. Only reachable when a dictionary lacks
`FW_COM_BUFFER_MAX_SIZE` (the compiler's `FinalChecks` otherwise rejects
the command at compile time). Severity: low (diagnosability).

### 8. `rand()`, `randf()`, `set_seed()` crash the wasm backend with an uncaught `NotImplementedError`

`_generate_llvm_unsupported` (`symbols.py:27`) raises `NotImplementedError`,
which none of the CLI's handlers catch (they catch `BackendError`), so
`fprime-fpy --wasm` on a sequence using the RNG builtins prints a Python
traceback instead of a compile error. Severity: low.

## Test-infrastructure note (not a compiler bug)

The wasm harness deployment builds `Svc::SeqArgs` with
`SequenceArgumentsMaxSize = 12` (its `config/AcConstants.fpp`), while the
dictionary and the fpybc harness use 255. Any seq-calling test whose child
takes more than 12 bytes of arguments fails on the wasm backend with
"child sequence ...: sequence arguments do not fit in Svc::SeqArgs". The
existing suite only passes small argument sets, so this is invisible today,
but it caps what seq-calling can be tested on wasm. The compiled command
buffers themselves are identical on both backends.

## Checked and found consistent

Directive operand order and field widths for every directive the compiler
emits (`WAIT_REL/ABS`, `IF/GOTO`, `PEEK`, `GET_FIELD`, `STORE_*`/`LOAD_*`,
`CALL/RETURN` frame layout incl. parameters at negative offsets,
`STACK_CMD` size accounting with compact strings, `POP_EVENT`,
`POP_SERIALIZABLE`, `EXIT`); sequence-argument push and frame layout
(args, flags, locals) at stack offset 0; `LOAD_ABS/STORE_ABS` for globals
from functions; recursion; six-parameter mixed-type layouts; runtime-index
stores through struct/array chains (one runtime index); nested loops with
`break`/`continue` and desugared `for` increments; `return` inside loops;
`exit()` from nested calls; bare expression statements of every kind leave
the stack balanced (harness leak check); defaults incl. forward calls;
imports; `check` nested and inside functions/loops; time desugaring in
conditions; tlm/prm struct/array/enum/bool reads incl. runtime element
index; command buffers for const, runtime, struct, array and seq-run
commands (byte-identical across backends); `log`; `write_to_port` of
structs, strings and scalars; sequence args of every scalar kind read and
written from functions.
