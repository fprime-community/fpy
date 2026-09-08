# Local patches to the fprime-wasm submodule

`wasm-sequencer-fixes.patch` carries two fixes to `Svc::WasmSequencer` the
wasm harness needs:

1. **The `env` float imports exist.** LLVM materializes library calls for
   guest float arithmetic (`pow`, `fmod`, `log`) under wasm's default import
   module `env`; without them, any sequence using float modulo, exponent or
   log fails to load.
2. **Guest FATAL/COMMAND events are emitted.** The guest-severity
   restriction (HostFunctionInvalidSeverity) is being removed upstream;
   until that lands, the patch restores the LogFatal/LogCommand events so a
   guest `log()` at any F Prime severity round-trips, matching the
   FpySequencer.

The patch is applied to `test/fprime-wasm` automatically before the wasm
harness is built (see `build_wasm_harness` in `src/fpy/harness.py`). It
should be dropped once the fixes land upstream.
