import sys, traceback

sys.path.insert(0, "src")
from fpy.test_helpers import analyze_seq, _wasm_codegen, CompilationFailed


def wasm_imports(b: bytes):
    """Parse the wasm import section, returning (module,name,kind) tuples."""
    assert b[:4] == b"\0asm", b[:4]
    i = 8
    out = []

    def uleb(i):
        r = 0
        s = 0
        while True:
            x = b[i]
            i += 1
            r |= (x & 0x7F) << s
            s += 7
            if not x & 0x80:
                return r, i

    while i < len(b):
        sid = b[i]
        i += 1
        size, i = uleb(i)
        end = i + size
        if sid == 2:
            n, i = uleb(i)
            for _ in range(n):
                ml, i = uleb(i)
                mod = b[i : i + ml].decode()
                i += ml
                nl, i = uleb(i)
                nm = b[i : i + nl].decode()
                i += nl
                kind = b[i]
                i += 1
                if kind == 0:
                    _, i = uleb(i)
                elif kind == 1:
                    i += 1
                    fl = b[i]
                    i += 1
                    _, i = uleb(i)
                    if fl:
                        _, i = uleb(i)
                elif kind == 2:
                    fl = b[i]
                    i += 1
                    _, i = uleb(i)
                    if fl:
                        _, i = uleb(i)
                elif kind == 3:
                    i += 1
                    i += 1
                out.append((mod, nm, kind))
        i = end
    return out


# What the (patched) WasmSequencer actually provides
PROVIDED = {
    ("fprime_v1", n)
    for n in [
        "exit",
        "panic",
        "args",
        "time",
        "tlm",
        "prm",
        "cmd",
        "event",
        "rsleep",
        "asleep",
        "serial_send",
        "serial_recv",
    ]
} | {("env", "pow"), ("env", "fmod"), ("env", "log")}
FLIGHT_ONLY = {
    ("fprime_v1", n)
    for n in [
        "exit",
        "panic",
        "args",
        "time",
        "tlm",
        "prm",
        "cmd",
        "event",
        "rsleep",
        "asleep",
        "serial_send",
        "serial_recv",
    ]
}

CASES = {
    "float pow": "x: F64 = 2.0\ny: F64 = x ** 3.0\n",
    "float mod": "x: F64 = 5.5\ny: F64 = x % 2.0\n",
    "ln": "x: F64 = 5.5\ny: F64 = ln(x)\n",
    "float floordiv": "x: F64 = 5.5\ny: F64 = x // 2.0\n",
    "fabs": "x: F64 = -5.5\ny: F64 = fabs(x)\n",
    "iabs": "x: I64 = -5\ny: I64 = iabs(x)\n",
    "f64->i64 cast": "x: F64 = 5.5\ny: I64 = I64(x)\n",
    "f64->u64 cast": "x: F64 = 5.5\ny: U64 = U64(x)\n",
    "f64->u8 cast": "x: F64 = 5.5\ny: U8 = U8(x)\n",
    "i64->f64 cast": "x: I64 = 5\ny: F64 = F64(x)\n",
    "u64->f64 cast": "x: U64 = 5\ny: F64 = F64(x)\n",
    "f32 arith": "x: F32 = 1.5\ny: F32 = x * x + x / x - x\n",
    "f32 mod": "x: F32 = 5.5\ny: F32 = x % 2.0\n",
    "f32 pow": "x: F32 = 2.0\ny: F32 = x ** 3.0\n",
    "i64 mul/div/mod": "x: I64 = 7\ny: I64 = x*x // 3 % 5\n",
    "u64 div/mod": "x: U64 = 7\ny: U64 = x // 3 % 5\n",
    "i8 arith": "x: I8 = 7\ny: I8 = x*x // 3 % 5\n",
    "struct eq": "a: Fw.TimeIntervalValue = {seconds:1, useconds:2}\nb: bool = a == a\n",
    "array index": "a: Ref.ManyChoices = [Ref.Choice.ONE, Ref.Choice.TWO]\ni: I64 = 1\nb: Ref.Choice = a[i]\n",
    "cmd + tlm + prm": "Ref.recvBuffComp.parameter1\nx: U32 = Ref.recvBuffComp.parameter1\ny: U32 = CdhCore.cmdDisp.CommandsDispatched\nCdhCore.cmdDisp.CMD_NO_OP()\n",
    "time ops": "a: Fw.Time = now()\nb: Fw.Time = now()\nc: Fw.TimeIntervalValue = a - b\nd: bool = a < b\n",
    "sleep/log/serial": 'sleep(1)\nlog("hi")\nwrite_to_port(Svc.Fpy.SerialPortIndex.PORT_0, 5)\n',
    "deep recursion fn": "def f(n: I64) -> I64:\n    if n <= 0:\n        return 0\n    return f(n-1)\nx: I64 = f(3)\n",
}
allimp = set()
for name, seq in CASES.items():
    try:
        st = analyze_seq(seq)
        w = _wasm_codegen(st)
    except CompilationFailed as e:
        print(
            f"{name:22s} REJECTED: {[l for l in str(e).splitlines() if l.strip()][1][:80]}"
        )
        continue
    except Exception as e:
        print(f"{name:22s} *** CRASH {type(e).__name__}: {e}")
        continue
    imps = {(m, n) for m, n, k in wasm_imports(w) if k == 0}
    allimp |= imps
    missing = imps - PROVIDED
    flightmissing = imps - FLIGHT_ONLY
    tag = ""
    if missing:
        tag = f"  <<< NOT PROVIDED EVEN WITH PATCH: {sorted(missing)}"
    elif flightmissing:
        tag = f"  (needs env patch: {sorted(flightmissing)})"
    print(f"{name:22s} imports={sorted(n for m,n in imps)}{tag}")
print()
print("ALL func imports seen:", sorted(allimp))
print("NOT provided by patched sequencer:", sorted(allimp - PROVIDED))
