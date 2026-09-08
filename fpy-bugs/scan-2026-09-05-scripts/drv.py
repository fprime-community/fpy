"""Scratch driver: compile a sequence and run it on both backends, printing
what happened (cmd response / exit code, dispatched cmds, events, serial)."""

import sys, traceback

sys.path.insert(0, "/home/threelambda/work/fpy/src")
import fpy.test_helpers as th
from fpy.test_helpers import (
    analyze_seq,
    _fpybc_codegen,
    _wasm_codegen,
    run_wasm,
    run_seq,
    CompilationFailed,
    load_dictionary,
    default_dictionary,
    _write_seq_for_harness,
    _seq_args_buffer_len,
)
from fpy.harness import fpy_harness, HarnessError
from fpy.test_helpers import _materialize_children, FPYBC, WASM
from fpy.bytecode.assembler import serialize_directives
from fpy.bytecode.directives import DirectiveErrorCode


def compile_both(seq, **kw):
    state = analyze_seq(seq, **kw)
    out = {}
    try:
        out["fpybc"] = _fpybc_codegen(state)
    except Exception as e:
        out["fpybc"] = e
    try:
        out["wasm"] = _wasm_codegen(state)
    except Exception as e:
        out["wasm"] = e
    return state, out


def run_fpybc_raw(
    directives,
    arg_types=None,
    tlm=None,
    prms=None,
    args=None,
    seq_dir=None,
    time_base=0,
    time_context=0,
    initial_time_us=0,
    failing_opcodes=None,
    seq_run_opcodes=None,
):
    d = load_dictionary(default_dictionary)
    always_failing = {d["cmd_name_dict"]["Ref.cmdSeq0.RUN"].opcode} | set(
        failing_opcodes or ()
    )
    cwd, f = _write_seq_for_harness(directives, arg_types, directory=seq_dir)
    req = {
        "seqFile": f,
        "cwd": cwd,
        "time": {
            "base": time_base,
            "context": time_context,
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
        "failOpcodes": sorted(always_failing),
    }
    if args is not None:
        req["args"] = args.hex()
    if seq_run_opcodes or seq_dir is not None:
        req["seqRunOpcodes"] = sorted(
            seq_run_opcodes or {d["cmd_name_dict"]["Ref.seqDisp.RUN_ARGS"].opcode}
        )
        req["seqArgsBufferSize"] = _seq_args_buffer_len(d)
    return fpy_harness().run(req)


def show(
    name,
    seq,
    tlm=None,
    prms=None,
    args=None,
    seq_dir=None,
    time_base=0,
    time_context=0,
    initial_time_us=0,
    failing_opcodes=None,
    backends=("fpybc", "wasm"),
    expected_warnings=None,
    ignored_warnings=None,
    dump_dirs=False,
    cmd_response=None,
    seq_run_opcodes=None,
):
    print(f"\n===== {name} =====")
    try:
        state, out = compile_both(
            seq,
            expected_warnings=expected_warnings,
            ignored_warnings=ignored_warnings,
            seq_dir=seq_dir,
        )
    except CompilationFailed as e:
        print("COMPILE FAILED:", str(e).strip()[:600])
        return None
    except Exception as e:
        print("COMPILE CRASHED:", type(e).__name__, str(e)[:400])
        traceback.print_exc(limit=3)
        return None
    results = {}
    if "fpybc" in backends:
        if isinstance(out["fpybc"], Exception):
            print(
                "[fpybc] codegen error:",
                type(out["fpybc"]).__name__,
                str(out["fpybc"])[:300],
            )
        else:
            dirs, arg_types = out["fpybc"]
            if dump_dirs:
                for i, dd in enumerate(dirs):
                    print(f"   {i:3d} {dd}")
            try:
                _materialize_children(seq_dir, FPYBC)
                r = run_fpybc_raw(
                    dirs,
                    arg_types,
                    tlm=tlm,
                    prms=prms,
                    args=args,
                    seq_dir=seq_dir,
                    time_base=time_base,
                    time_context=time_context,
                    initial_time_us=initial_time_us,
                    failing_opcodes=failing_opcodes,
                    seq_run_opcodes=seq_run_opcodes,
                )
                ev = [e["text"] for e in r.get("events", [])]
                err = r.get("lastDirectiveError")
                print(
                    f"[fpybc] cmdResponse={r.get('cmdResponse')} lastDirErr={DirectiveErrorCode(err).name if err is not None else None} "
                    f"exit={r.get('exitCode')} stackBytes={len(bytes.fromhex(r.get('stack','')))} error={r.get('error')}"
                )
                print("[fpybc] cmds:", r.get("cmds"))
                if r.get("serial"):
                    print("[fpybc] serial:", r.get("serial"))
                for e in ev:
                    print("   evt:", e[:160])
                results["fpybc"] = r
            except HarnessError as e:
                print("[fpybc] HARNESS ERROR:", str(e)[:800])
    if "wasm" in backends:
        if isinstance(out["wasm"], Exception):
            print(
                "[wasm] codegen error:",
                type(out["wasm"]).__name__,
                str(out["wasm"])[:300],
            )
        else:
            try:
                _materialize_children(seq_dir, WASM)
                code, events, cmds, serial = run_wasm(
                    out["wasm"],
                    failing_opcodes=failing_opcodes,
                    tlm=tlm,
                    prms=prms,
                    time_base=time_base,
                    time_context=time_context,
                    initial_time_us=initial_time_us,
                    args=args,
                    seq_dir=seq_dir,
                    cmd_response=cmd_response,
                )
                print(
                    f"[wasm] code={code} ({DirectiveErrorCode(code).name if code in [c.value for c in DirectiveErrorCode] else '?'})"
                )
                print("[wasm] cmds:", [c.hex() for c in cmds])
                if serial:
                    print("[wasm] serial:", serial)
                for e in events:
                    print("   evt:", e)
                results["wasm"] = (code, events, cmds, serial)
            except HarnessError as e:
                print("[wasm] HARNESS ERROR:", str(e)[:800])
    return results


if __name__ == "__main__":
    import runpy

    runpy.run_path(sys.argv[1], run_name="__main__")
