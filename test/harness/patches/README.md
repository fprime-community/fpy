# Local patches to the fprime-wasm submodule

`wasm-sequencer-fixes.patch` carries two fixes to `Svc::WasmSequencer` the
wasm harness needs:

1. **Exit codes get reported.** The code a sequence passes to the `exit` and
   `panic` host functions was thrown away (a host call can only end the
   interpreter with a trap), so every explicit exit failed the sequence and
   the code never reached the ground. With the patch, a code of 0 finishes
   the sequence cleanly and a nonzero code is reported through the new
   `SequenceExitedWithError` event.
2. **The `env` float imports exist.** LLVM materializes library calls for
   guest float arithmetic (`pow`, `fmod`, `log`) under wasm's default import
   module `env`; without them, any sequence using float modulo, exponent or
   log fails to load.
3. **Members are initialized.** Four members (`m_invokeStatus`,
   `m_loadStatus`, `m_pendingRun`, `m_pendingPause`) were missing from the
   constructor initializer list; a garbage `m_pendingPause` intermittently
   paused runs nobody paused.

The patch is applied to `test/fprime-wasm` automatically before the wasm
harness is built (see `build_wasm_harness` in `src/fpy/harness.py`). It
should be dropped once the fixes land upstream.
