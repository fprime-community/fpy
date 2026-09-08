import sys, traceback

sys.path.insert(0, "src")
from fpy.test_helpers import compile_seq, compile_seq_wasm, CompilationFailed
from fpy.bytecode.assembler import fpybc_directives_to_fpyasm


def try_compile(name, src, wasm=True):
    print(f"=== {name}")
    try:
        state, dirs, _ = compile_seq(src)
        print("fpybc OK:", len(dirs), "dirs")
        if "--asm" in sys.argv:
            print(fpybc_directives_to_fpyasm(dirs))
    except CompilationFailed as e:
        print(
            "fpybc CompilationFailed:",
            str(e).splitlines()[1] if len(str(e).splitlines()) > 1 else e,
        )
    except Exception as e:
        print("fpybc CRASH:", type(e).__name__, e)
        traceback.print_exc(limit=3)
    if wasm:
        try:
            compile_seq_wasm(src)
            print("wasm OK")
        except CompilationFailed as e:
            print(
                "wasm CompilationFailed:",
                str(e).splitlines()[1] if len(str(e).splitlines()) > 1 else e,
            )
        except Exception as e:
            print("wasm CRASH:", type(e).__name__, e)
            traceback.print_exc(limit=3)


cases = {
    "neg_float_pow": "x: F64 = F64(-8.0) ** 0.5\n",
    "int_of_inf": "x: I64 = I64(F64(1e999))\n",
    "uint_of_inf": "x: U8 = U8(F64(1e999))\n",
    "int_of_nan": "x: I64 = I64(F64(1e999) - F64(1e999))\n",
    "huge_pow": "x: F64 = 2 ** 100000\n",
    "neg_exp_int": "x: I64 = 2 ** -1\n",
    "float_mod_zero_const": "x: F64 = F64(1.5) % F64(0.0)\n",
    "float_floordiv_zero": "x: F64 = F64(1.5) // F64(0.0)\n",
    "inf_floordiv": "x: F64 = F64(1e999) // 2.0\n",
    "nan_cmp": "x: bool = F64(1e999) - F64(1e999) == 1.0\n",
}
for k, v in cases.items():
    try_compile(k, v)
