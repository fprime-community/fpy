import sys, traceback

sys.path.insert(0, "src")
import fpy.error
from fpy.test_helpers import (
    analyze_seq,
    _fpybc_codegen,
    _wasm_codegen,
    CompilationFailed,
)

CASES = {
    "empty file": "",
    "only newline": "\n",
    "only comment no newline": "# hi",
    "only whitespace": "   \n  \n",
    "BOM prefix": "﻿x: I64 = 1\n",
    "CRLF": "x: I64 = 1\r\ny: I64 = 2\r\n",
    "CR only": "x: I64 = 1\ry: I64 = 2\r",
    "form feed": "x: I64 = 1\n\x0cy: I64 = 2\n",
    "vertical tab in line": "x: I64 = 1\x0b\n",
    "NUL byte in source": "x: I64 = 1\n\x00\n",
    "deep parens 200": "x: I64 = " + "(" * 200 + "1" + ")" * 200 + "\n",
    "deep parens 5000": "x: I64 = " + "(" * 5000 + "1" + ")" * 5000 + "\n",
    "deep unary 2000": "x: I64 = " + "-" * 2000 + "1\n",
    "long add chain 3000": "x: I64 = " + "+".join(["1"] * 3000) + "\n",
    "deep member chain": "x: I64 = " + "a." * 500 + "b\n",
    "deep index chain": "x: I64 = a" + "[0]" * 500 + "\n",
    "huge int literal": "x: I64 = " + "9" * 5000 + "\n",
    "huge float literal": "x: F64 = 1e" + "9" * 100 + "\n",
    "huge negative exponent": "x: F64 = 1e-" + "9" * 100 + "\n",
    "very long identifier": "a" * 100000 + ": I64 = 1\n",
    "very long string": 'x: I64 = 1\nlog("' + "a" * 100000 + '")\n',
    "unicode identifier": "é: I64 = 1\n",
    "emoji in string": 'log("\U0001f600")\n',
    "deeply nested if 200": "".join("    " * i + f"if True:\n" for i in range(200))
    + "    " * 200
    + "pass\n",
    "deeply nested blocks 60": "".join("    " * i + f"if True:\n" for i in range(60))
    + "    " * 60
    + "pass\n",
    "many variables 3000": "".join(f"v{i}: I64 = {i}\n" for i in range(3000)),
    "many functions 500": "".join(
        f"def f{i}() -> I64:\n    return {i}\n" for i in range(500)
    )
    + "x: I64 = f0()\n",
    "deep anon nesting": "x: Fw.TimeIntervalValue = "
    + "{seconds: " * 1
    + "1"
    + "}" * 0
    + "\n",
    "trailing backslash": "x: I64 = 1 \\\n",
    "lone else": "else:\n    pass\n",
    "tab indent consistent": "if True:\n\tpass\n",
    "sequence 300 params": "sequence("
    + ", ".join(f"a{i}: U8" for i in range(300))
    + ")\n",
    "sequence 255 params": "sequence("
    + ", ".join(f"a{i}: U8" for i in range(255))
    + ")\n",
    "func 300 params": "def f("
    + ", ".join(f"a{i}: U8" for i in range(300))
    + "):\n    pass\nf("
    + ",".join("U8(0)" for i in range(300))
    + ")\n",
    "range huge": "for i in 0 .. 99999999999999999999:\n    pass\n",
    "range negative": "for i in -5 .. -1:\n    pass\n",
    "range reversed": "for i in 5 .. 1:\n    pass\n",
}
for name, seq in CASES.items():
    try:
        st = analyze_seq(seq, error_warnings=frozenset())
    except CompilationFailed as e:
        ls = [l for l in str(e).splitlines() if l.strip()]
        print(f"{name:28s} rejected: {ls[1][:60] if len(ls)>1 else ''}")
        continue
    except RecursionError as e:
        print(f"{name:28s} *** UNCAUGHT RecursionError")
        continue
    except Exception as e:
        print(f"{name:28s} *** CRASH {type(e).__name__}: {str(e)[:110]}")
        traceback.print_exc(limit=4)
        continue
    res = []
    for backend, fn in (("fpybc", _fpybc_codegen), ("wasm", _wasm_codegen)):
        try:
            fn(st)
            res.append(backend + "=OK")
        except CompilationFailed as e:
            res.append(backend + "=rej")
        except Exception as e:
            res.append(f"{backend}=*CRASH* {type(e).__name__}: {str(e)[:80]}")
    print(f"{name:28s} analyzed OK; {' | '.join(res)}")
