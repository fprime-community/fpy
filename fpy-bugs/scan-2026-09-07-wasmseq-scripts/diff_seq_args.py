import sys, struct

sys.path.insert(0, "src")
import fpy.test_helpers as th
from fpy.test_helpers import (
    analyze_seq,
    _fpybc_codegen,
    _wasm_codegen,
    CompilationFailed,
)
from fpy.harness import fpy_harness, wasm_harness


def run(backend, seq, args):
    st = analyze_seq(seq, error_warnings=frozenset())
    if backend == "fpybc":
        dirs, argt = _fpybc_codegen(st)
        cwd, f = th._write_seq_for_harness(dirs, argt, None)
        h = fpy_harness()
    else:
        w = _wasm_codegen(st)
        cwd, f = th._write_wasm_for_harness(w, None)
        h = wasm_harness()
    req = {
        "seqFile": f,
        "cwd": cwd,
        "time": {"base": 0, "context": 0, "seconds": 0, "useconds": 0},
        "tlm": {},
        "prms": {},
        "failOpcodes": [],
    }
    if args is not None:
        req["args"] = args.hex()
    r = h.run(req)
    if "error" in r:
        return "HARNESS-ERROR " + r["error"][:90]
    return (
        r.get("cmdResponse"),
        r.get("exitCode") if r.get("exited") else None,
        tuple((s["port"], s["data"]) for s in r.get("serial", [])),
    )


P = "Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0"
CASES = [
    (
        "U32 arg",
        "sequence(a: U32)\n" + f"write_to_port({P}, a)\n",
        struct.pack(">I", 0x11223344),
    ),
    ("I8 arg", "sequence(a: I8)\n" + f"write_to_port({P}, a)\n", struct.pack(">b", -5)),
    (
        "U64+U8",
        "sequence(a: U64, b: U8)\n" + f"write_to_port({P}, a)\nwrite_to_port({P}, b)\n",
        struct.pack(">Q", 0x0102030405060708) + struct.pack(">B", 0xAB),
    ),
    ("bool arg", "sequence(a: bool)\n" + f"write_to_port({P}, a)\n", b"\xff"),
    ("bool false", "sequence(a: bool)\n" + f"write_to_port({P}, a)\n", b"\x00"),
    (
        "F64 arg",
        "sequence(a: F64)\n" + f"write_to_port({P}, a)\n",
        struct.pack(">d", 1.5),
    ),
    (
        "F32 arg",
        "sequence(a: F32)\n" + f"write_to_port({P}, a)\n",
        struct.pack(">f", 1.5),
    ),
    (
        "enum arg",
        "sequence(a: Ref.Choice)\n" + f"write_to_port({P}, a)\n",
        struct.pack(">I", 1),
    ),
    (
        "struct arg",
        "sequence(a: Fw.TimeIntervalValue)\n"
        + f"write_to_port({P}, a.seconds)\nwrite_to_port({P}, a.useconds)\n",
        struct.pack(">II", 7, 9),
    ),
    (
        "array arg",
        "sequence(a: Ref.ManyChoices)\n"
        + f"write_to_port({P}, a[0])\nwrite_to_port({P}, a[1])\n",
        struct.pack(">II", 1, 0),
    ),
    (
        "array runtime idx",
        "sequence(a: Ref.ManyChoices, i: I64)\n" + f"write_to_port({P}, a[i])\n",
        struct.pack(">II", 1, 0) + struct.pack(">q", 1),
    ),
    (
        "mixed order",
        "sequence(a: U8, b: U32, c: U16)\n"
        + f"write_to_port({P}, a)\nwrite_to_port({P}, b)\nwrite_to_port({P}, c)\n",
        struct.pack(">B", 1) + struct.pack(">I", 2) + struct.pack(">H", 3),
    ),
    (
        "arg mutated then read",
        "sequence(a: U32)\n" + "a = U32(99)\n" + f"write_to_port({P}, a)\n",
        struct.pack(">I", 5),
    ),
    (
        "arg read in function",
        "sequence(a: I64)\ndef f() -> I64:\n    return a\n"
        + f"write_to_port({P}, f())\n",
        struct.pack(">q", 42),
    ),
    (
        "WRONG SIZE (short)",
        "sequence(a: U32)\n" + f"write_to_port({P}, a)\n",
        b"\x01\x02",
    ),
    (
        "WRONG SIZE (long)",
        "sequence(a: U32)\n" + f"write_to_port({P}, a)\n",
        b"\x01\x02\x03\x04\x05",
    ),
    (
        "no args declared, args given",
        "" + f"write_to_port({P}, U8(1))\n",
        b"\x01\x02\x03\x04",
    ),
]
for name, seq, args in CASES:
    try:
        a = run("fpybc", seq, args)
    except Exception as e:
        a = f"ERR {type(e).__name__}: {str(e)[:80]}"
    try:
        b = run("wasm", seq, args)
    except Exception as e:
        b = f"ERR {type(e).__name__}: {str(e)[:80]}"
    flag = "" if a == b else "   <<<<<< DIVERGE"
    print(f"--- {name}\n    fpybc: {a}\n    wasm : {b}{flag}")
