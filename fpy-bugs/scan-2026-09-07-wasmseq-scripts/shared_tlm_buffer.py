import sys

sys.path.insert(0, "src")
import fpy.test_helpers as th
from fpy.test_helpers import analyze_seq, _wasm_codegen, _fpybc_codegen
from fpy.harness import wasm_harness, fpy_harness
from fpy.dictionary import load_dictionary

d = load_dictionary(th.default_dictionary)


def run(backend, seq, tlm):
    st = analyze_seq(seq, error_warnings=frozenset())
    if backend == "wasm":
        w = _wasm_codegen(st)
        cwd, f = th._write_wasm_for_harness(w, None)
        h = wasm_harness()
    else:
        dirs, _ = _fpybc_codegen(st)
        cwd, f = th._write_seq_for_harness(dirs, None, None)
        h = fpy_harness()
    r = h.run(
        {
            "seqFile": f,
            "cwd": cwd,
            "time": {"base": 0, "context": 0, "seconds": 0, "useconds": 0},
            "tlm": {str(d["ch_name_dict"][k].ch_id): v.hex() for k, v in tlm.items()},
            "prms": {},
            "failOpcodes": [],
        }
    )
    if "error" in r:
        return "HARNESS-ERROR " + r["error"][:80]
    ev = [
        e["text"]
        for e in r.get("events", [])
        if "BufferTooSmall" in e["text"] or "TooLarge" in e["text"]
    ]
    return (
        r.get("cmdResponse"),
        tuple((s["port"], s["data"]) for s in r.get("serial", [])),
        ev,
    )


P = "Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0"
SMALL = "CdhCore.cmdDisp.CommandsDispatched"  # U32, 4 bytes
BIG = "Ref.typeDemo.ScalarStructCh"  # 42 bytes

only_small = f"x: U32 = {SMALL}\nwrite_to_port({P}, x)\n"
also_big = (
    f"y: Ref.ScalarStruct = Ref.typeDemo.ScalarStructCh\n"
    f"x: U32 = {SMALL}\nwrite_to_port({P}, x)\n"
)
bigval = bytes(42)
oversized_small = bytes.fromhex("11223344") + b"\xaa" * 8  # 12 bytes for a U32 channel

print("A) module reads ONLY the small U32 channel; channel returns 12 bytes")
print("   wasm :", run("wasm", only_small, {SMALL: oversized_small}))
print("   fpybc:", run("fpybc", only_small, {SMALL: oversized_small}))
print()
print("B) SAME small read, but the module ALSO reads a 42-byte channel elsewhere")
print("   wasm :", run("wasm", also_big, {SMALL: oversized_small, BIG: bigval}))
print("   fpybc:", run("fpybc", also_big, {SMALL: oversized_small, BIG: bigval}))
