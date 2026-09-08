# Fpy bug-scan reports

Findings from a series of independent scans of the Fpy compiler, both code
generators, and the two flight sequencers (`Svc::FpySequencer` in
`test/fprime`, `Svc::WasmSequencer` in `test/fprime-wasm`). Every scan used the
same method: read both sides of the bytecode/wasm contract, then run
hand-written probes and differential fuzzers on the real sequencers through
`test/harness`, comparing exit codes, faults, dispatched command bytes and
`write_to_port` output between the backends.

Each report keeps its own "checked and found consistent" section; those negative
results are the bulk of the coverage and are worth reading before re-walking the
same ground.

## Reports

| Report | Scope | Filed from it |
|---|---|---|
| `AUDIT_FINDINGS_2026-09-04.md` | guest-reachable FW_ASSERT / DoS on flight | #200 |
| `AUDIT_FINDINGS_2026-09-04_part2.md` | miscompiles and backend divergence; 360-program fuzz | #196-#199 |
| `AUDIT_FINDINGS_2026-09-05_vm_interactions.md` | ISA and stack-accounting contract; 210-program fuzz | (see below) |
| `INDEPENDENT_SCAN_2026-09-05.md` | full front-end and VM read; 600-program fuzz | (see below) |
| `COMPILER_VM_SCAN_2026-09-05.md` | contract mismatches; ~1000-program fuzz | #201-#206 |
| `CRASH_SCAN_2026-09-05.md` | compiler crashes; wasm integration pass | #207-#210 |
| `INDEPENDENT_SCAN_2026-09-07.md` | dictionary-configurable types (`build-variants/`) | #221-#226 |
| `COMPILER_VM_INTERACTIONS_SCAN_2026-09-07.md` | const folding, indentation, dictionary caching | #227-#229 |
| `FPYSEQ_DIRECTIVE_SIZE_SCAN_2026-09-07.md` | line-by-line `FpySequencer*.cpp` | #233 |
| `WASMSEQ_SCAN_2026-09-07_independent.md` | line-by-line WasmSeq; 365 malformed modules | #231 |

Scan scripts are in `scan-2026-09-05-scripts/`,
`scan-2026-09-07-compiler-vm-scripts/` and `scan-2026-09-07-wasmseq-scripts/`.

Filed from these reports: #196-#210, #221-#229, #231-#234 (29 issues).

## Findings not yet filed

Ordered by how much they matter for flight. Each one names the report that
carries the full write-up and repro.

### Medium

- **fpybc float-to-narrow-integer casts wrap where wasm saturates.**
  `U8(300.0)` is 44 on fpybc, 255 on wasm, and the constant folder wraps
  differently again; #59 specifies saturation. `INDEPENDENT_SCAN_2026-09-05.md`
  §4, `COMPILER_VM_SCAN_2026-09-05.md` §6.
- **Dictionary-configured type singletons leak across compiles in one
  process.** `_build_global_scopes` is `lru_cache`d around the mutations, so a
  long-lived process that compiles against two dictionaries emits a binary for
  the wrong deployment. `COMPILER_VM_INTERACTIONS_SCAN_2026-09-07.md` §3.
- **`rand()`, `randf()`, `set_seed()` crash the wasm backend** with an uncaught
  `NotImplementedError` instead of a diagnostic.
  `INDEPENDENT_SCAN_2026-09-05.md` §8, `INDEPENDENT_SCAN_2026-09-07.md` §5.
- **The WasmSequencer retains the spacewasm `caller` handle past its documented
  call scope.** Latent: correctness rests on an undocumented detail of the
  pinned `=0.6.4` crate. `WASMSEQ_SCAN_2026-09-07_independent.md` §1.
- **Timeout defaults disagree between sequencers.** wasm
  `HOST_FUNCTION_TIMEOUT_SECS` is 60, fpybc `STATEMENT_TIMEOUT_SECS` is 0, so a
  sequence with a blocking command over a minute long passes on one backend and
  fails on the other. `CRASH_SCAN_2026-09-05.md` §5.

### Low

- `PUSH_TLM_VAL_AND_TIME`'s overflow guard subtracts two quantities and can
  underflow, turning a clean `STACK_OVERFLOW` into an FW_ASSERT in
  configurations the static asserts accept. Matters for #166.
  `FPYSEQ_DIRECTIVE_SIZE_SCAN_2026-09-07.md` §2.
- `StatementsFailed` telemetry is never incremented (same class as #154, which
  names only `SequencesFailed`). `FPYSEQ_DIRECTIVE_SIZE_SCAN_2026-09-07.md` §3.
- fpybc evaluates a non-canonical bool byte inconsistently: `IF`/`NOT` treat
  `0x01` as true while `==` memcmps it against `0xFF`; wasm faults instead.
  `CRASH_SCAN_2026-09-05.md` §3.
- `for` over an existing variable defines a shadowing variable; SPEC.md says the
  existing variable becomes the loop variable. `COMPILER_VM_SCAN_2026-09-05.md`
  §7.
- `x: F32 = 0.1` then `x == 0.1` is false: the comparison happens at the widened
  F64 intermediate type and the literal is rounded to F64, not to F32 first.
  `COMPILER_VM_INTERACTIONS_SCAN_2026-09-07.md` §5.
- A dictionary without `Svc.SeqArgs` crashes with an `AssertionError` instead of
  a `DictionaryError`. `CRASH_SCAN_2026-09-05.md` §8.
- Anonymous struct and array literals cannot be compared (`a == [1, 2]`) even
  though they coerce everywhere else. `INDEPENDENT_SCAN_2026-09-07.md`, notes.
- `macros.py` `MACRO_SLEEP_FLOAT` is dead code and wrong as written (it pushes
  `1_000_000.0` and never multiplies). `COMPILER_VM_SCAN_2026-09-05.md`, notes.
- A NaN `HOST_FUNCTION_TIMEOUT_SECS` / `STATEMENT_TIMEOUT_SECS` passes both
  range guards and reaches an undefined float-to-integer conversion.
  `WASMSEQ_SCAN_2026-09-07_independent.md` §3.
- No compile-time size feedback for the wasm backend: a sequence can compile
  cleanly and be unloadable on the target (`ERR_OUT_OF_MEMORY`). Distinct from
  #206, which is about command size. `AUDIT_FINDINGS_2026-09-04_part2.md`,
  observations.
- `DirectiveUnion` is stack-allocated on every statement dispatch at
  `MAX_DIRECTIVE_SIZE` (2048) for a buffer bounded at 506 bytes, i.e. ~2 KB of
  task stack per dispatch. `FPYSEQ_DIRECTIVE_SIZE_SCAN_2026-09-07.md`, related
  observation.

## Caveats

- **One finding is an artifact of the local harness patch.**
  `CRASH_SCAN_2026-09-05.md` §6 reports that the WasmSequencer's
  `dispatchEvent` has FATAL and COMMAND cases contradicting WASM-SEQ-014. That
  is `test/harness/patches/wasm-sequencer-fixes.patch`, which deliberately
  restores those events; upstream fprime-wasm restricts them. The fpybc half of
  the observation (a guest `log()` can emit a FATAL) is real.
- **The compiler half of #231 was lost.** The report describes it as applied and
  tested in the working tree but uncommitted; `_emit_tlm_prm_read` still passes
  the shared buffer's capacity as `value_size`. It needs redoing.
- **Deliberately out of scope in every scan:** arithmetic overflow, rounding and
  truncation semantics (#59, #109-#131). Findings that drifted math-adjacent
  were dropped rather than filed; #234 is one that was recovered later.
- **Could not be exercised on the harnesses:** statement-timeout ordering (the
  harness jumps its clock), a 32-bit `FwSizeType` flight build (the Posix OSAL
  static-asserts `FwSizeType` holds `size_t`, and no 32-bit multilib was
  installed), wasm sequence arguments over 12 bytes (fprime-wasm's
  `SequenceArgumentsMaxSize`), and wasm modules much above 3.5 KB with the
  checked-in harness config.
- The scans were run independently and did not read each other, so several
  findings were discovered two or three times (`assert ..., 0`, the
  telemetry/parameter size mismatch, the unused `CMD_SERIALIZE_FAILURE`, the
  U64 trap, the cast divergence, the RNG crash, the typed-port assert).

## Reports that no longer exist

An earlier compiler-robustness sweep (2026-07-21, `codegen` @907e815 versus
`../fpy-bugfix`) was written to `COMPILER_ROBUSTNESS_FINDINGS.md` in an
untracked directory and is gone. Its findings, none of them filed, were:

- Parse errors `exit(1)` instead of raising: `x = @` (`UnexpectedCharacters`)
  exits with no message at all, and an unclosed bracket at EOF (`x = (1`) raises
  a raw `TypeError` in `format_diagnostic` (line is `None`).
- `dictionary.py` leaks `FileNotFoundError`, `JSONDecodeError` and `KeyError`
  raw, and its resolve loop swallows `KeyError` as "unresolved".
- The coercion matrix has no lossless unsigned-to-signed widening, so no
  unsigned variable can index an array, while the lossy int-to-float coercion is
  allowed.
- `FW_SERIALIZE_TRUE_VALUE` / `FW_SERIALIZE_FALSE_VALUE` are process-wide
  globals mutated per compile (the same cross-dictionary hazard as the singleton
  leak above).
- `x = 1` (assignment to an undefined name) reports "Unknown NameGroup.VALUE
  'x'", leaking an enum into the diagnostic.
