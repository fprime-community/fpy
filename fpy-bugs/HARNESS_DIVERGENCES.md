# Model vs. real FpySequencer — divergences found

Found while migrating the test suite from `fpy.model.FpySequencerModel` to a harness
running the real `Svc::FpySequencer` (`test/harness`, `pytest --harness`).

Status of the suite: **1612/1616 pass on the real sequencer** (40s, versus 1616 in 41s on
the model). The 4 failures are item 2 below.

Nothing here has been filed. Each needs a decision.

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

**Still open upstream.** The declared and actual widths disagree: `RUN`, `RUN_ARGS`,
`VALIDATE`, `VALIDATE_ARGS` and `DUMP_STACK_TO_FILE` all advertise `string size 240` for a
field that can only ever carry 40. Either the declaration should say 40, or
`FW_CMD_STRING_MAX_SIZE` should be raised for deployments that need real paths — 40
characters is not much for an absolute path. Worth raising on nasa/fprime.

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
