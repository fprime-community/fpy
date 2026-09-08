"""Compile and run on both backends, printing dispatched cmd bytes, serial writes, events for comparison."""

import sys, traceback

sys.path.insert(0, "src")
import fpy.harness as H
from fpy.harness import SequencerHarness, FPY_HARNESS_BINARY, WASM_HARNESS_BINARY

H._fpy_harness = SequencerHarness(FPY_HARNESS_BINARY)
H._wasm_harness = SequencerHarness(WASM_HARNESS_BINARY)
import fpy.test_helpers as T
from fpy.test_helpers import (
    analyze_seq,
    _fpybc_codegen,
    _wasm_codegen,
    run_wasm,
    CompilationFailed,
    _write_seq_for_harness,
    load_dictionary,
    default_dictionary,
)
from fpy.bytecode.assembler import fpybc_directives_to_fpyasm


def run_fpybc_raw(
    dirs,
    arg_types,
    args_bytes=None,
    tlm=None,
    prms=None,
    failing_opcodes=None,
    initial_time_us=0,
):
    d = load_dictionary(default_dictionary)
    cwd, f = _write_seq_for_harness(dirs, arg_types)
    req = {
        "seqFile": f,
        "cwd": cwd,
        "time": {
            "base": 0,
            "context": 0,
            "seconds": initial_time_us // 1_000_000,
            "useconds": initial_time_us % 1_000_000,
        },
        "tlm": {
            str(d["ch_name_dict"][k].ch_id): bytes(v).hex()
            for k, v in (tlm or {}).items()
        },
        "prms": {
            str(d["prm_name_dict"][k].prm_id): bytes(v).hex()
            for k, v in (prms or {}).items()
        },
        "failOpcodes": sorted(
            {d["cmd_name_dict"]["Ref.cmdSeq0.RUN"].opcode} | set(failing_opcodes or ())
        ),
    }
    if args_bytes is not None:
        req["args"] = args_bytes.hex()
    return H.fpy_harness().run(req)


def run_both(
    name,
    src,
    tlm=None,
    prms=None,
    args=None,
    failing_opcodes=None,
    asm=False,
    initial_time_us=0,
):
    print(f"=== {name}")
    try:
        state = analyze_seq(src, ignored_warnings=set(T.ALL_WARNINGS))
    except CompilationFailed as e:
        print("  analysis FAILED:", "\n    ".join(str(e).splitlines()[1:3]))
        return
    except Exception as e:
        print("  analysis CRASH:", type(e).__name__, e)
        traceback.print_exc(limit=4)
        return
    args_bytes = b"".join(v.serialize() for v in args) if args else None
    fres = None
    try:
        dirs, arg_types = _fpybc_codegen(state)
        if asm:
            print(fpybc_directives_to_fpyasm(dirs))
        r = run_fpybc_raw(
            dirs, arg_types, args_bytes, tlm, prms, failing_opcodes, initial_time_us
        )
        if "error" in r:
            print("  fpybc HARNESS ERROR", r["error"])
        else:
            fres = r
            print(
                "  fpybc: resp",
                r["cmdResponse"],
                "exit",
                r.get("exitCode"),
                "dirErr",
                r.get("lastDirectiveError"),
                "cmds",
                r.get("cmds"),
                "serial",
                r.get("serial"),
                "events",
                [e["text"][:60] for e in r.get("events", []) if e.get("guest")],
            )
    except CompilationFailed as e:
        print("  fpybc codegen FAILED:", "\n    ".join(str(e).splitlines()[1:3]))
    except Exception as e:
        print("  fpybc CRASH/OTHER:", type(e).__name__, str(e)[:300])
        traceback.print_exc(limit=3)
    try:
        wasm = _wasm_codegen(state)
        code, events, cmds, serial = run_wasm(
            wasm,
            tlm=tlm,
            prms=prms,
            args=args_bytes,
            failing_opcodes=failing_opcodes,
            initial_time_us=initial_time_us,
        )
        print(
            "  wasm : code",
            code,
            "cmds",
            [c.hex() for c in cmds],
            "serial",
            [(p, b.hex()) for p, b in serial],
            "events",
            [e[1][:60] for e in events],
        )
        if fres is not None:
            fc = fres.get("cmds")
            if fc is not None and [c.hex() for c in cmds] != fc:
                print("  !!! CMD BYTES DIFFER")
            fs = fres.get("serial")
            if (
                fs is not None
                and [{"port": p, "data": b.hex()} for p, b in serial] != fs
            ):
                print("  !!! SERIAL DIFFER", fs)
    except CompilationFailed as e:
        print("  wasm codegen FAILED:", "\n    ".join(str(e).splitlines()[1:3]))
    except Exception as e:
        print("  wasm CRASH/OTHER:", type(e).__name__, str(e)[:300])
        traceback.print_exc(limit=3)


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
