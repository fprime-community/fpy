# Model vs. real FpySequencer — divergences found

Found while migrating the test suite from `fpy.model.FpySequencerModel` to a harness
running the real `Svc::FpySequencer` (`test/harness`, `pytest --harness`).

Status of the suite: **1612/1616 pass on the real sequencer** (40s, versus 1616 in 41s on
the model). The 4 failures are item 2 below.

All three are resolved; nothing was filed upstream.

---

## 1. A sequence path longer than 40 characters cannot be run

**What happens.** `Svc.Fpy` declares `RUN`/`RUN_ARGS`'s `fileName` as
`string size FileNameStringSize` (240, `test/fprime/default/config/AcConstants.fpp:52`).
The generated command handler deserializes it into `Fw::CmdStringArg`
(`test/fprime/Fw/Cmd/CmdString.hpp:14`), which is `FW_CMD_STRING_MAX_SIZE` = **40**
characters. A longer path is truncated on the wire, the handler returns `FORMAT_ERROR`, and
a sequence that called it stops with `CMD_FAIL` (17).

The Python model has no such limit — it opens whatever path it is given — so the compiler
and the test suite have never had reason to notice.

**Why it matters beyond tests.** `Ref.seqDisp.RUN_ARGS("<path>", ...)` is the supported way
for one sequence to call another. Any user whose sequence directory pushes a child path past
40 characters gets a runtime `CMD_FAIL` with no indication that the path was the problem,
and nothing at compile time warns them. 40 characters is not much for an absolute path.

**How the test suite hid it.** `test_seq_calling.py` builds children under
`tempfile.TemporaryDirectory()`, so path length followed `TMPDIR`. With the usual `/tmp` the
paths came to ~26 characters and passed; with a 61-character `TMPDIR`, 20 of that file's 36
tests fail. Now pinned: `test/conftest.py` sets `tempfile.tempdir` to `/tmp/fpy` so the
budget is deterministic instead of ambient.

**How general is it.** Not specific to sequence paths, and not specific to FpySequencer:
fpp generates `Fw::CmdStringArg` for *every* command string argument. Ports are unaffected —
the sequencer's `seqRunIn` port, declared at the same 240, generates a correctly sized
`Fw::ExternalString`. So the rule is: a command's string arguments are capped at 40, a
port's are not.

**Fixed in fpy (compile time).** A new `CheckCommandStringArgs` pass
(`src/fpy/semantics.py`) rejects any string literal passed as a command argument that
exceeds the limit, for every command rather than only sequence-run ones:

    String argument 'fileName' is too long for a command: 41 bytes exceeds
    FW_CMD_STRING_MAX_SIZE (40)

The limit is read from the dictionary as `FW_CMD_STRING_MAX_SIZE`, falling back to
`DEFAULT_MAX_CMD_STRING_SIZE = 40` — the same read-from-dictionary-or-default pattern the
other sequencer limits use (`src/fpy/state.py`). Note the constant is **not** currently in
the dictionary (it is a plain `constant` in `test/fprime/default/config/FpConstants.fpp:29`,
not a `dictionary constant`), so today the default is what applies. Exporting it would make
the check track a deployment that configures the limit differently.

**Not being raised upstream** (decided 2026-08-07). The declared and actual widths still
disagree — `RUN`, `RUN_ARGS`, `VALIDATE`, `VALIDATE_ARGS` and `DUMP_STACK_TO_FILE` advertise
`string size 240` for a field that can only carry 40 — but fpy's compile-time check makes it
a clear error at the point a user could hit it, which is where it matters.

---

## 2. `rand()`/`randf()` — Python Mersenne Twister vs C++ `std::mt19937`

**What happens.** The model seeds Python's `random.Random`; flight uses `std::mt19937`
(`FpySequencerDirectives.cpp:1454`). Same algorithm family, different seeding and output.

This is already known and was already tested both ways — `test_rng.py` carried paired
model/GDS expectations. **The three parked `_gds` twins were verified to pass unchanged on
the harness** (`rand() == 2288500408`, `randf() == 0.5328330229967833`, …), which is good
independent evidence that the harness matches a real deployment rather than merely being
self-consistent.

**Resolution:** no bug — delete the model expectations and unskip the C++ ones when the
harness becomes the default runner. One test, `test_rand_uses_time_as_initial_seed`, has no
`_gds` counterpart and needs its expected value regenerated against
`std::seed_seq{base, ctx, secs, usecs}`.

---

## 3. `validate_cmd` coverage is lost, not broken

The model deserialized every dispatched command's arguments against the dictionary and hard
-failed on a mismatch (`model.py`). Neither the real sequencer nor the harness does this —
on a deployment the *receiving* component would reject the command. No test asserts on it, so
nothing goes red, but the migration does drop an argument-serialization check.

**Resolution: accept the loss — equivalent coverage already exists.**
`test_wasm.py::TestWasmCommands` asserts the exact command wire bytes against an independent
`struct.pack` encoding (opcode, string length prefixes, utf-8 byte counts, runtime scalar
packing), which is a stronger check than "the arguments deserialize". `test_assembler.py`
covers `ConstCmdDirective`/`StackCmdDirective` serialization round-trips on the bytecode
side. Nothing new is needed; the harness now returns dispatched command buffers, so
byte-exact assertions can be added for the bytecode path later if a gap shows up.

---

## 4. WasmSeq: a compiled module loads but never finishes

The `WasmSeqHarness` (`test/harness/WasmSeqHarnessMain.cpp`) drives a real
`Svc::WasmSequencer` through the same protocol as the bytecode harness. A module
compiled by fpy loads and starts, then spins: the state machine sits in
`RUNNING_SPINNING` and never reaches a terminal state, so the run ends on the
harness's dispatch bound rather than on an outcome.

What is known:

- **The component builds against devel unchanged.** Its branch has diverged 48
  commits against devel's 59, but nothing it uses has moved. No merge is needed.
- **Guest memory has to be raised.** The component's own config allows a 2048
  byte guest, and fpy links every module with a 4 KiB stack
  (`-zstack-size=4096`, `codegen_llvm.py`), so a module fails to load with
  `ERR_GUEST_MEMORY_ALLOC_FAILED` until the config is widened. The harness
  overrides it (`test/harness/config/WasmSequencerConfig.hpp`); **a real
  deployment running fpy sequences would have to do the same**, which is worth
  knowing before anyone tries.
- **Execution is sliced.** `INSTRUCTION_FUEL` defaults to 1000 instructions,
  after which the guest yields and something must resume it. The harness ticks
  `checkTimers`, which moves the state machine but does not appear to advance
  the guest -- so either resumption comes from elsewhere, or the module cannot
  make progress. That is the next thing to pin down.

The two gaps recorded earlier still stand and are the reason this cannot be
green yet: flight `wasmExit`/`wasmPanic` discard the code and trap (so even
`exit(0)` cannot report success), and there is no `env` module for the float
libcalls LLVM materialises.

### WasmSeq: current test results

`pytest --wasm-seq test/fpy/test_wasm.py` runs the wasm suite on the real
component: **106 of 137 pass**, against 137 on the spacewasm runner.

**30 of the 31 failures are one bug**, not thirty: every one is
`assert 0 == <code>`, because flight's `wasmExit`/`wasmPanic` discard the code
and trap, so no error code ever reaches the caller. The expected codes span
EXIT_WITH_ERROR (7), DOMAIN_ERROR (10), ARRAY_OUT_OF_BOUNDS (11), CMD_FAIL (17)
and DESERIALIZE_ERROR_INVALID_BOOL (20) -- all of which the spacewasm runner
reports correctly. Fixing the host functions to carry their code should convert
essentially all of them in one go.

The 31st is a FATAL-severity `log()` not arriving as an event.

Two harness-side prerequisites were needed to get this far, both worth knowing
for anyone wiring the component into a deployment:

- **`loadParameters()` must be called at init.** Without it
  `INSTRUCTION_FUEL` reads as zero, `spacewasm_run` executes no instructions,
  and the module spins out of fuel forever with no diagnostic.
- **Guest memory must exceed the linked stack** (see above).

One genuine defect found in the component while debugging that spin:
`WasmSequencer`'s constructor initialises every flag except **`m_pendingRun`
and `m_pendingPause`**, so `PAUSE_CHECK` branches on indeterminate memory and a
run may park in PAUSED depending on stack garbage. Fixed locally in the
submodule working tree; it needs reporting upstream or the fix will be lost on
the next bump.
