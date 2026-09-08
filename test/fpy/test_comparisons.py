import pytest

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


class TestAllOperatorsByType:

    @pytest.mark.parametrize(
        "type_name,big,small",
        [
            ("U8", "200", "100"),
            ("I8", "100", "-100"),
            ("U16", "60000", "100"),
            ("I16", "30000", "-30000"),
            ("U32", "4294967295", "0"),
            ("I32", "2147483647", "-2147483648"),
            ("U64", "4294967295", "0"),
            ("I64", "2147483647", "-2147483648"),
            ("F32", "3.14159", "-3.14159"),
            ("F64", "3.14159265359", "-3.14159265359"),
        ],
    )
    def test_all_comparison_operators(self, fprime_test_api, type_name, big, small):
        seq = f"""
val1: {type_name} = {big}
val2: {type_name} = {small}

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

    def test_bool_inequality_fails(self, fprime_test_api):
        """Booleans do not support <, >, <=, >= operators."""
        seq = """
a: bool = True
b: bool = False
if a > b:
    exit(0)
exit(1)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_nan_comparisons(self, fprime_test_api):
        """IEEE 754 NaN semantics: every ordered comparison and == are false
        when an operand is NaN, and != (the negation of ==) is true."""
        seq = """
zero: F64 = 0.0
nan: F64 = zero / zero
assert nan != nan
assert nan != 0.0
assert not (nan == nan)
assert not (nan < 0.0)
assert not (nan <= 0.0)
assert not (nan > 0.0)
assert not (nan >= 0.0)
"""
        assert_run_success(fprime_test_api, seq)


class TestStringComparisons:
    """Strings exist at runtime only as compile-time constants (a variable,
    telemetry channel or parameter of string type is rejected), so a string
    comparison always folds at compile time."""

    def test_const_string_eq(self, fprime_test_api):
        seq = """
assert "a" == "a"
assert not ("a" == "b")
"""
        assert_run_success(fprime_test_api, seq)

    def test_const_string_neq(self, fprime_test_api):
        seq = """
assert "a" != "b"
assert not ("a" != "a")
"""
        assert_run_success(fprime_test_api, seq)

    def test_const_string_eq_different_lengths(self, fprime_test_api):
        # A prefix is not equal to the longer string.
        seq = """
assert not ("a" == "ab")
assert "a" != "ab"
"""
        assert_run_success(fprime_test_api, seq)

    def test_const_string_eq_into_variable(self, fprime_test_api):
        seq = """
x: bool = "a" == "a"
y: bool = "a" == "b"
assert x
assert not y
"""
        assert_run_success(fprime_test_api, seq)

    def test_string_compared_to_number_rejected(self, fprime_test_api):
        seq = """
assert "a" == 1
"""
        assert_compile_failure(fprime_test_api, seq)
