import sys, random, json, subprocess, tempfile, os, time

sys.path.insert(0, "src")
import fpy.test_helpers as th
from fpy.test_helpers import analyze_seq, _wasm_codegen
from fpy.harness import WASM_HARNESS_BINARY

# a richer baseline: commands, tlm, loops, functions -> more sections
SEQ = """
def f(n: I64) -> I64:
    if n <= 0:
        return 0
    return n + f(n - 1)
g: I64 = f(3)
x: U32 = CdhCore.cmdDisp.CommandsDispatched
CdhCore.cmdDisp.CMD_NO_OP()
log("hi")
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, g)
"""
base = _wasm_codegen(analyze_seq(SEQ, error_warnings=frozenset()))
print("baseline module size:", len(base), flush=True)
d = tempfile.mkdtemp()
REQ = (
    json.dumps(
        {
            "seqFile": "m0.wasm",
            "cwd": d,
            "time": {"base": 0, "context": 0, "seconds": 0, "useconds": 0},
            "tlm": {},
            "prms": {},
            "failOpcodes": [],
        }
    ).encode()
    + b"\n"
)


def run_one(blob):
    open(os.path.join(d, "m0.wasm"), "wb").write(blob)
    try:
        p = subprocess.run(
            [str(WASM_HARNESS_BINARY)],
            input=REQ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT", b"", b""


def bad(rc, out, err):
    if rc == "TIMEOUT":
        return "TIMEOUT (hang)"
    if rc != 0:
        return f"process exit rc={rc}"
    if not out.strip():
        return "no reply (died silently)"
    if b"Rust panic" in out or b"Rust panic" in err:
        return "spacewasm Rust panic"
    if b"Assert" in err or b"ASSERT" in err:
        return "FW_ASSERT"
    return None


t0 = time.time()
rc, out, err = run_one(base)
print(
    "control:", bad(rc, out, err) or "clean", f"({time.time()-t0:.2f}s/run)", flush=True
)

cases = []
for n in range(0, len(base), max(1, len(base) // 60)):
    cases.append((f"trunc@{n}", base[:n]))
rnd = random.Random(1234)
for k in range(260):
    i = rnd.randrange(len(base))
    b = bytearray(base)
    b[i] ^= 1 << rnd.randrange(8)
    cases.append((f"bitflip@{i}", bytes(b)))
for k in range(40):
    r = random.Random(500 + k)
    b = bytearray(base)
    for _ in range(r.randrange(2, 12)):
        b[r.randrange(len(b))] = r.randrange(256)
    cases.append((f"multi{k}", bytes(b)))

aborts = 0
for name, blob in cases:
    rc, out, err = run_one(blob)
    why = bad(rc, out, err)
    if why:
        aborts += 1
        print(f"### {name}: {why}")
        print("   stdout:", out[-300:])
        print("   stderr:", err[-300:])
        fn = os.path.join(d, f"repro_{name.replace('@','_')}.wasm")
        open(fn, "wb").write(blob)
        print("   saved:", fn, flush=True)
        if aborts >= 4:
            break
print(f"tested {len(cases)} malformed modules, {aborts} aborts/hangs")
