# Local fixes to the WasmSequencer checkout

`wasm-sequencer-local-fixes.patch` holds changes to the `test/fprime-wasm`
submodule that have not been reported upstream. A submodule bump discards
them, so `conftest` refuses to run the WasmSeq harness without them and says
how to reapply:

```sh
git -C test/fprime-wasm apply ../harness/patches/wasm-sequencer-local-fixes.patch
```

What it changes, and why each is temporary:

- **`wasmExit`/`wasmPanic` report their code.** Both discarded the guest's exit
  code and left through a trap, so no sequence could report *why* it failed --
  30 of the 31 wasm test failures were this one bug. They now emit the
  `ProgramExited`/`PanicOccurred` events the component already declares but
  never raised. The real fix belongs in SpaceWasm, which has no "exit"
  hostcall result to return; until it does, leaving through a trap and
  carrying the code out in an event is the workaround.
- **`m_pendingRun` and `m_pendingPause` are initialised.** The constructor sets
  every other flag but missed these two, so the state machine's `PAUSE_CHECK`
  branched on indeterminate memory and a run could park in `PAUSED` depending
  on stack contents. This one is an outright defect and should go upstream
  as-is.
