Reproduction scripts for WASMSEQ_SCAN_2026-09-07_independent.md.
Run from the repo root with `uv run python3 fpy-bugs/scan-2026-09-07-wasmseq-scripts/<script>`.

- shared_tlm_buffer.py   -- finding 2: the wasm oversize check depends on an
                            unrelated telemetry read elsewhere in the module.
- wasm_imports.py        -- parses the emitted wasm import section and compares
                            it against what the sequencer registers.
- malformed_modules.py   -- corruption/truncation sweep of a compiled module,
                            looking for sequencer aborts or hangs.
- diff_seq_args.py       -- ground-supplied sequence arguments, both backends.
- diff_cmd_layout.py     -- dispatched command byte layout, both backends.
- diff_runner.py         -- shared differential helper (control flow cases).
- frontend_robustness.py -- pathological source inputs.

diff_cmd_layout.py imports diff_runner.py by absolute scratch path; adjust the
path at the top if you move it.
