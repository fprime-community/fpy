from fpy.types import U32

from fpy.test_helpers import assert_compile_failure, assert_run_success


class TestBasicComparisons:

    def test_geq(self, fprime_test_api):
        seq = """
if 2 >= 1:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_float_cmp(self, fprime_test_api):
        seq = """
if 4.0 > 5.0:
    exit(1)
exit(0)
"""

        assert_run_success(fprime_test_api, seq)

    def test_literal_comparison(self, fprime_test_api):
        seq = """
if 255 > 254:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_literal_comparison_false(self, fprime_test_api):
        seq = """
if 255 < 254:
    exit(1)
exit(0)
"""
        assert_run_success(fprime_test_api, seq)

class TestCrossTypeComparisons:

    def test_f32_f64_cmp(self, fprime_test_api):
        seq = """
val: F32 = 0.0
val2: F64 = 1.0
if val > val2:
    exit(1)
exit(0)
"""

        assert_run_success(fprime_test_api, seq)

    def test_i32_f64_cmp(self, fprime_test_api):
        seq = """
val: I32 = 2
val2: F64 = 1.0
if val > val2:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_i32_u32_cmp(self, fprime_test_api):
        seq = """
val: I32 = -2
val2: U32 = 2
# fails to compile, can't compare types of diff signedness
if val < val2:
    exit(1)
exit(0)
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_float_int_literal_cmp(self, fprime_test_api):
        seq = """
if 1 < 2.0:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_mixed_numeric_comparisons(self, fprime_test_api):
        seq = """
val_u8: U8 = 255
val_i8: I8 = -10
val_u32: U32 = 4294967295
val_i32: I32 = -2147483648
val_f32: F32 = 3.14159
val_f64: F64 = -3.14159265359

# i32 > u32 because the cmp happens as unsigned, and so the
# two's complement negative is really large
if val_u8 < val_i8 and val_i32 > val_u32:
    if val_f64 <= val_f32 and val_f32 >= val_f64:
        if val_u8 != val_i8 and not (val_u32 == val_i32):
            exit(0)
exit(1)
"""
        assert_compile_failure(fprime_test_api, seq)

class TestAllOperatorsByType:

    def test_all_comparison_operators_u8(self, fprime_test_api):
        seq = """
val1: U8 = 200
val2: U8 = 100

if val1 > val2 and val2 < val1:
    if val1 >= val2 and val2 <= val1:
        if val1 != val2 and not (val1 == val2):
            exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_all_comparison_operators_i8(self, fprime_test_api):
        seq = """
val1: I8 = 100
val2: I8 = -100

if val1 > val2 and val2 < val1:
    if val1 >= val2 and val2 <= val1:
        if val1 != val2 and not (val1 == val2):
            exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_all_comparison_operators_u32(self, fprime_test_api):
        seq = """
val1: U32 = 4294967295
val2: U32 = 0

if val1 > val2 and val2 < val1:
    if val1 >= val2 and val2 <= val1:
        if val1 != val2 and not (val1 == val2):
            exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_all_comparison_operators_i32(self, fprime_test_api):
        seq = """
val1: I32 = 2147483647
val2: I32 = -2147483648

if val1 > val2 and val2 < val1:
    if val1 >= val2 and val2 <= val1:
        if val1 != val2 and not (val1 == val2):
            exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_all_comparison_operators_f32(self, fprime_test_api):
        seq = """
val1: F32 = 3.14159
val2: F32 = -3.14159

if val1 > val2 and val2 < val1:
    if val1 >= val2 and val2 <= val1:
        if val1 != val2 and not (val1 == val2):
            exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_all_comparison_operators_f64(self, fprime_test_api):
        seq = """
val1: F64 = 3.14159265359
val2: F64 = -3.14159265359

if val1 > val2 and val2 < val1:
    if val1 >= val2 and val2 <= val1:
        if val1 != val2 and not (val1 == val2):
            exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

class TestEdgeCases:

    def test_equality_edge_cases(self, fprime_test_api):
        seq = """
val1: U8 = 0
val2: U8 = 0
val3: F32 = 0.0
val4: F64 = 0.0
val5: I32 = 0

if val1 == val2 and val3 == val4 and val4 == val5:
    if not (val1 != val2) and not (val3 != val4) and not (val4 != val5):
        exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_maximum_integer_comparisons(self, fprime_test_api):
        seq = """
val_max: I64 = 9223372036854775807  # Max I64
val_mid: I64 = 1
val_min: I64 = -9223372036854775808  # Min I64

if val_max > val_mid and val_mid > val_min:
    if val_min < val_max:
        exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_complex_type_assignments(self, fprime_test_api):
        seq = """
val1: I8 = 127
val2: U8 = 255
val3: F32 = 127.0

if val1 == val3:  # Integer to float comparison
    if val2 > val3:  # Unsigned vs float comparison
        exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)
