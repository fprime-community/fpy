"""Compile an fpy program and run it on both backends, printing outcomes."""

import sys, traceback, json

sys.path.insert(0, "src")
import fpy.harness as H
from fpy.harness import SequencerHarness, FPY_HARNESS_BINARY, WASM_HARNESS_BINARY

# bypass the fprime-util build step: reuse the existing binaries
H._fpy_harness = SequencerHarness(FPY_HARNESS_BINARY)
H._wasm_harness = SequencerHarness(WASM_HARNESS_BINARY)
import fpy.test_helpers as T
from fpy.test_helpers import (
    analyze_seq,
    _fpybc_codegen,
    _wasm_codegen,
    run_seq,
    run_wasm,
    CompilationFailed,
    ValidationError,
)
from fpy.bytecode.assembler import fpybc_directives_to_fpyasm
from fpy.types import FpyValue


def run_both(
    name,
    src,
    tlm=None,
    prms=None,
    args=None,
    failing_opcodes=None,
    asm=False,
    seq_dir=None,
    initial_time_us=0,
):
    print(f"=== {name}")
    try:
        state = analyze_seq(src, seq_dir=seq_dir, ignored_warnings=set(T.ALL_WARNINGS))
    except CompilationFailed as e:
        print("  analysis FAILED:", "\n    ".join(str(e).splitlines()[1:3]))
        return
    except Exception as e:
        print("  analysis CRASH:", type(e).__name__, e)
        traceback.print_exc(limit=4)
        return
    args_bytes = b"".join(v.serialize() for v in args) if args else None
    # fpybc
    try:
        dirs, arg_types = _fpybc_codegen(state)
        if asm:
            print(fpybc_directives_to_fpyasm(dirs))
        try:
            run_seq(
                None,
                dirs,
                tlm=tlm,
                prms=prms,
                args=args_bytes,
                arg_name_types=arg_types,
                failing_opcodes=failing_opcodes,
                seq_dir=seq_dir,
                initial_time_us=initial_time_us,
            )
            print("  fpybc: SUCCESS")
        except ValidationError as e:
            print("  fpybc: VALIDATION ERROR", e)
        except RuntimeError as e:
            print("  fpybc: RUNTIME FAIL", e.args)
        except Exception as e:
            print("  fpybc: HARNESS/OTHER", type(e).__name__, str(e)[:300])
    except CompilationFailed as e:
        print("  fpybc codegen FAILED:", "\n    ".join(str(e).splitlines()[1:3]))
    except Exception as e:
        print("  fpybc codegen CRASH:", type(e).__name__, e)
        traceback.print_exc(limit=4)
    # wasm
    try:
        wasm = _wasm_codegen(state)
        try:
            code, events, cmds, serial = run_wasm(
                wasm,
                tlm=tlm,
                prms=prms,
                args=args_bytes,
                failing_opcodes=failing_opcodes,
                seq_dir=seq_dir,
                initial_time_us=initial_time_us,
            )
            print("  wasm: code", code, "events", [e[1] for e in events][:5])
        except Exception as e:
            print("  wasm: HARNESS/OTHER", type(e).__name__, str(e)[:300])
    except CompilationFailed as e:
        print("  wasm codegen FAILED:", "\n    ".join(str(e).splitlines()[1:3]))
    except Exception as e:
        print("  wasm codegen CRASH:", type(e).__name__, e)
        traceback.print_exc(limit=4)


if __name__ == "__main__":
    import importlib.util

    spec = importlib.util.spec_from_file_location("cases", sys.argv[1])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    only = sys.argv[2:]
    for name, kw in m.CASES.items():
        if only and name not in only:
            continue
        if isinstance(kw, str):
            kw = {"src": kw}
        run_both(name, **kw)
    H.close_all()
