# Fpy safety audit findings — 2026-09-04 (branch devel @ a69b17d)

Method: each candidate is written as an Fpy sequence and run on BOTH real
sequencers (Svc::FpySequencer via the fpy harness, Svc::WasmSequencer via the
wasm harness) through fpy.test_helpers, comparing exit codes, faults, traps and
"process died" outcomes. Math/IEEE discrepancies are out of scope (tracked as
#109-#131). Nothing here overlaps the already-filed #183-#186, #191-#195,
#111/#112.

---

## Finding 1 — widening directives skip the stack-overflow guard, so a guest sequence can FW_ASSERT-abort the flight FpySequencer
**Status: filed as #200 (fprime-community/fpy).**

**Severity:** high (guest-controlled FW_ASSERT in flight code -> process abort / DoS)
**Backend:** fpybc only (Svc::FpySequencer). Wasm traps or rejects instead.

**Related issues (checked against the live tracker):**
- #108 "Max stack size analysis" (open) is the compile-time preventive side of
  this: it proposes erroring/warning when a program could exceed
  MAX_STACK_SIZE, with recursion left as a TODO. It is not implemented, which
  is why this is reachable; and even once it is, the runtime VM should still
  fault gracefully rather than FW_ASSERT. This finding is the runtime-VM half
  (an inconsistency among the directive handlers), complementary to #108, not a
  duplicate.
- #193 "sleep/sleep_until useconds >= 1e6 trip FW_ASSERT in the flight
  FpySequencer" (open) is the same failure class (guest value -> flight
  FW_ASSERT/FATAL) at a different assert site (Fw::Time). Its recommended fix,
  "the flight handlers also need a range check that returns a DirectiveError",
  is exactly the remedy the widening handlers need.
- #196 "Comparing two void function calls ... crashes the wasm backend" (open)
  is the memCmp(size=0) sibling below. #196 documents the type-check bug and the
  wasm codegen crash and describes fpybc's memCmp(0) only as "always pushes
  True" -- it does NOT note that memCmp(0) is an unguarded stack grower that can
  FW_ASSERT-abort the flight sequencer at a full stack. #196's proposed fix
  (reject void operands) would incidentally close that fpybc path.
- Not the same: #184/#185 (fpybc crashes, but at compile time in codegen, not a
  runtime VM abort).

### What happens
The seven widening conversion directives pop a narrow operand and push a wider
one with no bounds check:

    op_siext_8_64  op_siext_16_64  op_siext_32_64      (I8/I16/I32 -> I64)
    op_ziext_8_64  op_ziext_16_64  op_ziext_32_64      (U8/U16/U32 -> U64)
    op_fpext                                           (F32 -> F64)

(test/fprime/Svc/FpySequencer/FpySequencerDirectives.cpp). Each handler checks
only for STACK_UNDERFLOW, then calls Stack::push<T>() with the wider T. Every
other growing directive (pushVal, allocate, pushTime, pushRand, peek, load,
tlm/prm/cmdResp reads, call) first checks `size > MAX_STACK_SIZE - stack.size`
and returns DirectiveError::STACK_OVERFLOW gracefully. The widening handlers do
not, so when the operand stack is within 1..7 bytes of Svc.Fpy.MAX_STACK_SIZE
(65535) the wider push trips the assertion in Stack::push:

    FpySequencerStack.cpp:60
    FW_ASSERT(this->size + sizeof(val) <= Fpy::MAX_STACK_SIZE, ...)

FW_ASSERT in flight code aborts the process. Under the harness the sequencer
subprocess dies with SIGABRT rather than failing the sequence.

### Why it is reachable (not defended elsewhere)
There is no compile-time or load-time peak-operand-stack analysis. The only
stack-related front-end check is that the sequence's *argument* bytes fit
MAX_STACK_SIZE (semantics.py CheckSequenceArgs / the load-time
ArgTotalSizeExceedsStackLimit). Runtime directive handlers are the sole defense
against operand-stack overflow, and they are graceful for every growing
directive except these seven. So a sequence that passes validation can drive
the operand stack to ~MAX at runtime and abort the sequencer through a widening.

The operand stack is pushed to the 7-byte window using call arguments (each a
guarded load that stays live until the CALL) plus cheap U8 filler locals to
raise the frame, then a widening final argument.

### Minimal repro (aborts the real FpySequencer)
248 arguments of a 263-byte type + 6 one-byte args + 28 U8 filler locals put the
operand stack at 65528; the final `U64(t)` argument's ziext_8_64 then pushes 8
bytes -> 65536 > 65535 -> abort.

```python
from fpy.test_helpers import analyze_seq, _fpybc_codegen, run_seq
from fpy.error import WarningType
n263, n1, nfill = 248, 6, 28
params = ", ".join([f"p{i}: Svc.SeqArgs" for i in range(n263)] +
                   [f"q{i}: U8" for i in range(n1)] + ["z: U64"])
argv = ", ".join(["a"]*n263 + ["u"]*n1 + ["U64(t)"])
fillers = "".join(f"g{i}: U8 = 0\n" for i in range(nfill))
seq = f"""
def f({params}) -> U32:
    return 0
a: Svc.SeqArgs = Svc.SeqArgs()
u: U8 = 1
t: U8 = 1
{fillers}r: U32 = f({argv})
exit(I32(r))
"""
st = analyze_seq(seq, ignored_warnings=set(WarningType))
dirs, argtypes = _fpybc_codegen(st)
run_seq(None, dirs, arg_name_types=argtypes)   # -> harness process dies (SIGABRT)
```

Observed:
- nfill=28 -> abort, "FpySequencerStack.cpp:60 ... 65528 8"
- nfill=32 -> abort, "... 65532 8"
- nfill>=36 -> graceful RUNTIME STACK_OVERFLOW (a guarded push crosses MAX first)

The same short expression `c: U64 = U64(b)` compiles to LoadRel + ziext_8_64
(no overflow check between them), so the bug is the widening directive itself,
not the argument-marshalling; the big call is only the vehicle for putting the
operand stack near the limit under the 2048-directive and 255-parameter caps.

### Contrast with wasm
On Svc::WasmSequencer the operand stack is the engine's own stack; an overflow
is a clean STACK_OVERFLOW trap (the sequence fails, the sequencer survives).
(The 255-parameter repro above is not a clean A/B on wasm because that many
263-byte params inflate the module past the code-page limit and it fails to
load; the divergence is nonetheless real — wasm never aborts on operand-stack
growth.)

### Suggested fix
Give the seven widening handlers the same guard the other growers use, e.g.
before the wider push:

    if (sizeof(wide) - sizeof(narrow) > Fpy::MAX_STACK_SIZE - this->m_runtime.stack.size)
        return DirectiveError::STACK_OVERFLOW;

or make Stack::push return a status the handler can turn into STACK_OVERFLOW,
so no guest-reachable path can hit the FW_ASSERT.

### The unguarded growers
Every `stack.push` / `pushZeroes` / `size +=` site in FpySequencerDirectives.cpp
was checked for a preceding MAX_STACK_SIZE/STACK_OVERFLOW guard. All growers are
guarded (pushVal, allocate, pushTime, pushRand, peek, load, tlm/prm/cmdResp,
call header, return) except:
- op_siext_8/16/32_64, op_ziext_8/16/32_64, op_fpext (the widenings above).
- memCmp_directiveHandler when `size == 0`: it pops `2*size == 0` bytes and
  pushes a 1-byte result through the same unguarded `Stack::push`, so it is a
  net +1 grower. Reachable from `void == void` (e.g. two void function calls
  compared), which semantics accepts. Same root cause and same fix. This case
  is written up separately as finding 3 in
  AUDIT_FINDINGS_2026-09-04_part2.md; recorded here because it belongs to this
  exact class. (For `size >= 1` memCmp is net-non-growing and safe.)

The comparison/arithmetic/trunc/floor/abs/itof/ftoi ops are net-non-growing, so
they need no guard.

---

## Secondary observation — telemetry/parameter size mismatch diverges between backends

Requires a misconfigured dictionary (flight value size != dictionary type size),
which the audit brief already flags as an unvalidated case. Recording the
*behavioral divergence* only:

- Flight value SHORTER than the dictionary type:
  - fpybc: faults STACK_ACCESS_OUT_OF_BOUNDS (read rejected).
  - wasm: silently proceeds; the value buffer is left partially filled, so the
    guest reads a partially-uninitialized value (e.g. a 1-byte U32 channel read
    back as 0x09000000). No fault.
- Flight value LONGER than the dictionary type:
  - fpybc: silently reads the first N bytes (truncated value), no fault.
  - wasm: traps (BufferTooSmall), sequence fails.

So neither backend is "both silent": they disagree on both directions (fault vs
garbage vs truncate vs trap). Low priority since it needs a bad dictionary, but
the wasm silent-partial-read on a short value is the concerning half.

---

## Checked and found clean (no divergence / no reachable abort) this session
- Large straight-line sequences: fpybc rejects at codegen (>2048 directives);
  wasm runs them (no directive cap). Expected, not a safety bug.
- Recursion without a widening: both fault gracefully (fpybc STACK_OVERFLOW,
  wasm STACK_OVERFLOW trap) at their respective depths.
- log() message of 300 chars: both truncate to FW_LOG_STRING_MAX_SIZE cleanly.
- write_to_port with a string / oversized array / MAX_SERIAL_PORTS index: both
  reject the bad port/size gracefully (fpybc SERIAL_PORT_INVALID_INDEX, wasm
  HostFunctionInvalidPort trap).
- flags.assert_cmd_success toggled inside functions / across the call that
  fails: both backends agree (same exit codes) in every combination tried.
- Imported sequences defining same-named functions/globals: sectioned correctly;
  both backends agree.
- Child-sequence RUN_ARGS (arg pass, child failure, flags off, arg-count/size
  mismatch): both agree; count/size mismatches are caught at compile or
  validation time.
- check-statement (timeout, persist, period 0, immediate pass/fail): both agree.
