import sys, traceback

sys.path.insert(0, "src")
from fpy.test_helpers import (
    analyze_seq,
    _fpybc_codegen,
    _wasm_codegen,
    run_seq,
    run_wasm,
    CompilationFailed,
)
from fpy.harness import HarnessError


def run_both(seq, **kw):
    """Return (fpybc_result, wasm_result) as comparable strings."""
    out = {}
    for backend in ("fpybc", "wasm"):
        try:
            st = analyze_seq(seq, error_warnings=frozenset())
        except CompilationFailed as e:
            out[backend] = "COMPILE-REJECT: " + str(e).splitlines()[1][:70]
            continue
        except Exception as e:
            out[backend] = f"COMPILE-CRASH {type(e).__name__}: {e}"
            continue
        try:
            if backend == "fpybc":
                dirs, _ = _fpybc_codegen(st)
                # capture serial writes via the harness result
                import fpy.test_helpers as th
                from fpy.harness import fpy_harness
                from fpy.dictionary import load_dictionary

                d = load_dictionary(th.default_dictionary)
                cwd, f = th._write_seq_for_harness(dirs, None, None)
                req = {
                    "seqFile": f,
                    "cwd": cwd,
                    "time": {"base": 0, "context": 0, "seconds": 0, "useconds": 0},
                    "tlm": {},
                    "prms": {},
                    "failOpcodes": [],
                }
                r = fpy_harness().run(req)
                out[backend] = fmt(r)
            else:
                w = _wasm_codegen(st)
                import fpy.test_helpers as th
                from fpy.harness import wasm_harness

                cwd, f = th._write_wasm_for_harness(w, None)
                req = {
                    "seqFile": f,
                    "cwd": cwd,
                    "time": {"base": 0, "context": 0, "seconds": 0, "useconds": 0},
                    "tlm": {},
                    "prms": {},
                    "failOpcodes": [],
                }
                r = wasm_harness().run(req)
                out[backend] = fmt(r)
        except Exception as e:
            out[backend] = f"RUNERR {type(e).__name__}: {str(e)[:120]}"
    return out["fpybc"], out["wasm"]


def fmt(r):
    if "error" in r:
        return "HARNESS-ERROR: " + r["error"][:100]
    parts = [f"resp={r.get('cmdResponse')}"]
    if r.get("exited"):
        parts.append(f"exit={r.get('exitCode')}")
    ser = [(s["port"], s["data"]) for s in r.get("serial", [])]
    parts.append(f"serial={ser}")
    ev = [e["text"] for e in r.get("events", []) if e.get("guest")]
    parts.append(f"events={ev}")
    parts.append("cmds=" + str(r.get("cmds", [])))
    return " ".join(parts)


W = lambda v: f"write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, {v})\n"
T = "timeout {seconds: 1}"

CASES = {
    "for loop with continue": "x: I64 = 0\nfor i in 0 .. 4:\n    if i == 1:\n        continue\n    x = x + 1\n"
    + W("x"),
    "nested for break": "x: I64 = 0\nfor i in 0 .. 3:\n    for j in 0 .. 3:\n        if j == 1:\n            break\n        x = x + 1\n"
    + W("x"),
    "while with break/continue": "x: I64 = 0\ni: I64 = 0\nwhile i < 5:\n    i = i + 1\n    if i == 2:\n        continue\n    if i == 4:\n        break\n    x = x + 1\n"
    + W("x"),
    "check body runs": f"y: I64 = 0\ncheck True {T}:\n    y = 1\n" + W("y"),
    "check timeout body runs": f"y: I64 = 0\ncheck False timeout {{useconds: 1}}:\n    y = 1\ntimeout:\n    y = 2\n"
    + W("y"),
    "check inside for + continue": f"x: I64 = 0\nfor i in 0 .. 3:\n    check i > 0 {T}:\n        continue\n    x = x + 1\n"
    + W("x"),
    "check inside for + break": f"x: I64 = 0\nfor i in 0 .. 3:\n    check True {T}:\n        break\n    x = x + 1\n"
    + W("x"),
    "recursion": "def f(n: I64) -> I64:\n    if n <= 0:\n        return 0\n    return n + f(n - 1)\n"
    + W("f(5)"),
    "mutual recursion": "def a(n: I64) -> I64:\n    if n <= 0:\n        return 0\n    return b(n-1)\ndef b(n: I64) -> I64:\n    return a(n-1) + 1\n"
    + W("a(5)"),
    "array runtime index read": "a: Ref.ManyChoices = [Ref.Choice.ONE, Ref.Choice.TWO]\ni: I64 = 1\n"
    + W("a[i]"),
    "array runtime index write": "a: Ref.ManyChoices = [Ref.Choice.ONE, Ref.Choice.TWO]\ni: I64 = 0\na[i] = Ref.Choice.TWO\n"
    + W("a[0]"),
    "struct member rw": "t: Fw.TimeIntervalValue = {seconds: 1, useconds: 2}\nt.seconds = U32(7)\n"
    + W("t.seconds"),
    "global written in fn": "g: I64 = 0\ndef f():\n    g = 9\nf()\n" + W("g"),
    "fn param shadows global": "g: I64 = 1\ndef f(g2: I64) -> I64:\n    return g2 + g\n"
    + W("f(2)"),
    "short circuit and": "def s(x: I64) -> bool:\n    write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, x)\n    return False\nb: bool = s(1) and s(2)\n",
    "short circuit or": "def s(x: I64) -> bool:\n    write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, x)\n    return True\nb: bool = s(1) or s(2)\n",
    "nested fn call order": "def s(x: I64) -> I64:\n    write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, x)\n    return x\ndef f(a: I64, b: I64) -> I64:\n    return a\nc: I64 = f(s(1), s(2))\n",
    "bool from struct eq": "a: Fw.TimeIntervalValue = {seconds: 1, useconds: 2}\nb: Fw.TimeIntervalValue = {seconds: 1, useconds: 2}\n"
    + W("a == b"),
    "early exit 0": "exit(0)\n" + W("1"),
    "assert passes": "assert True, 5\n" + W("1"),
    "assert fails": "assert False, 5\n" + W("1"),
}
for name, seq in CASES.items():
    a, b = run_both(seq)
    flag = "" if a == b else "   <<<<<< DIVERGE"
    print(f"--- {name}\n    fpybc: {a}\n    wasm : {b}{flag}")
