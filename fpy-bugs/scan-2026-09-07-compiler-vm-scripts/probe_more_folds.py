import sys, traceback

sys.path.insert(0, "src")
from fpy.test_helpers import compile_seq, CompilationFailed, ALL_WARNINGS

cases = {
    "neginf_pow_half": "x: F64 = F64(-1e999) ** 0.5\n",
    "neg_f64_pow_f64_frac": "x: F64 = (-2.0) ** F64(0.5)\n",
    "neg_lit_pow_f64_int": "x: F64 = (-2.0) ** F64(2)\n",
    "f32_cast_inf_to_u8": "x: U8 = U8(F32(1e39))\n",
    "inf_cast_in_cmd_arg": "CdhCore.cmdDisp.CMD_TEST_CMD_1(I32(F64(1e999)), 1.0, 1)\n",
    "inf_as_index": "a: Ref.FpyExampleArray = [1,2,3]\nx: U32 = a[I64(F64(1e999))]\n",
    "inf_seconds": "t: Fw.TimeIntervalValue = {seconds: U32(F64(1e999))}\n",
    "inf_range": "for i in 0..I64(F64(1e999)):\n    pass\n",
    "inf_exit_code": "exit(I32(F64(1e999)))\n",
    "inf_assert_code": "assert False, I32(F64(1e999))\n",
    "inf_neg_cast": "x: I64 = I64(-F64(1e999))\n",
    "nan_cast": "x: I64 = I64(F64(1e999) - F64(1e999))\n",
    "huge_decimal_cast": "x: I64 = I64(1e400)\n",
    "huge_decimal_f32": "x: F32 = F32(1e400)\n",
    "inf_compare_fold": "assert F64(1e999) > 1\nassert not (F64(1e999) < 1)\n",
    "inf_floor_div_int": "x: I64 = I64(F64(1e999) // 2)\n",
    "zero_pow_neginf": "x: F64 = F64(0.0) ** F64(-1e999)\n",
    "neg_zero_pow_half": "x: F64 = F64(-0.0) ** 0.5\n",
    "int_pow_huge_float": "x: F64 = F64(2.0) ** 100000\n",
    "mod_inf": "x: F64 = 7 % F64(1e999)\n",
    "cast_bool_of_float": "x: bool = bool(1.5)\n",
}
for k, v in cases.items():
    try:
        compile_seq(v, ignored_warnings=set(ALL_WARNINGS))
        print(f"{k}: OK")
    except CompilationFailed as e:
        print(
            f"{k}: CompilationFailed:",
            str(e).splitlines()[1] if len(str(e).splitlines()) > 1 else e,
        )
    except Exception as e:
        print(f"{k}: CRASH {type(e).__name__}: {str(e)[:120]}")
