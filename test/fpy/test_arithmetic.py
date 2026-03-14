import pytest

from fpy.types import U32

from fpy.test_helpers import assert_compile_failure, assert_run_success

def test_overflow_compile_error(fprime_test_api):
    seq = """
val1: U8 = 256  # Should fail: value too large for U8
"""
    assert_compile_failure(fprime_test_api, seq)


def test_add_unsigned(fprime_test_api):
    seq = """
var1: U32 = 500
var2: U32 = 1000
if var1 + var2 == 1500 and (var1 + 1) > var1:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_add_signed(fprime_test_api):
    seq = """
var1: I32 = -255
var2: I32 = 255
if var1 + var2 == 0 and (var1 + 1) > (var1 + -1):
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_add_float(fprime_test_api):
    seq = """
var1: F32 = -255.0
var2: F32 = 255.0
if var1 + var2 == 0.0 and (var1 + 1.0) > (var1 + -1.0):
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_const_fold_specific_float_binop(fprime_test_api):
    """Const folding binary ops on specific float types (F64, F32).

    When both operands are cast to a specific float type (e.g. F64(1.5)),
    their .val is a Python float, not Decimal. The const folder must handle
    the bare `float` result from `float + float` etc.
    """
    seq = """
if F64(1.5) + F64(2.5) == 4.0:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_float_truncate_stack_size(fprime_test_api):
    seq = """
var2: F64 = 123.0
var1: F32 = F32(-var2)
if var1 == -123.0:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_sub_unsigned(fprime_test_api):
    seq = """
var1: U32 = 1000
var2: U32 = 500
if var1 - var2 == 500 and (var1 - 1) < var1:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_sub_signed(fprime_test_api):
    seq = """
var1: I32 = 255
var2: I32 = 255
if var1 - var2 == 0 and (var1 - 1) < (var1 - -1):
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_sub_float(fprime_test_api):
    seq = """
var1: F32 = 255.0
var2: F32 = 255.0
if var1 - var2 == 0.0 and (var1 - 1.0) < (var1 - -1.0):
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_mul_unsigned(fprime_test_api):
    seq = """
var1: U32 = 5
var2: U32 = 20
if var1 * var2 == 100 and (var1 * 2) > var1:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_mul_signed(fprime_test_api):
    seq = """
var1: I32 = -5
var2: I32 = 20
if var1 * var2 == -100 and (var1 * 2) < var1:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_mul_float(fprime_test_api):
    seq = """
var1: F32 = 5.0
var2: F32 = 20.0
if var1 * var2 == 100.0 and (var1 * 2.0) > var1:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_div_unsigned(fprime_test_api):
    seq = """
var1: U32 = 20
var2: U32 = 5
if var1 / var2 == 4.0 and (var1 / 2) < var1:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_div_signed(fprime_test_api):
    seq = """
var1: I32 = -20
var2: I32 = 5
if var1 / var2 == -4.0: # and (var1 / -2) > var1:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_div_float(fprime_test_api):
    seq = """
var1: F32 = -20.0
var2: F32 = 5.0
if var1 / var2 == -4.0 and (var1 / -2.0) > var1:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_order_of_operations(fprime_test_api):
    seq = """
if 1 - 2 + 3 * 4 == 11 and 10.0 / 5.0 * 2.0 == 4.0:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_arithmetic_arg_to_builtin_bad_type(fprime_test_api):
    seq = """
sleep(1 + 2 * 0, (0 + 1 / 2))
"""
    assert_compile_failure(fprime_test_api, seq)


def test_arithmetic_arg_to_builtin(fprime_test_api):
    seq = """
sleep(1 + 2 * 0, (0 + 1 // 2))
"""
    assert_run_success(fprime_test_api, seq)


def test_chain_mul(fprime_test_api):
    seq = """
var1: I32 = 1
var2: I32 = 2
var3: I32 = 3
if var1 * var2 * var3 == 6:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_chain_add(fprime_test_api):
    seq = """
var1: I32 = 1
var2: I32 = 2
var3: I32 = 3
if var1 + var2 + var3 == 6:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_chain_sub(fprime_test_api):
    seq = """
var1: I32 = 1
var2: I32 = 2
var3: I32 = 3
if var1 - var2 - var3 == -4:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_chain_div(fprime_test_api):
    seq = """
var1: I32 = 3
var2: I32 = 2
var3: I32 = 1
if var1 / var3 / var2 == 3/2:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_pow_unsigned(fprime_test_api):
    seq = """
var1: U32 = 20
var2: U32 = 2
if var1 ** var2 == 400:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_pow_signed(fprime_test_api):
    seq = """
var1: I32 = -20
var2: I32 = 2
if var1 ** var2 == 400:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_pow_float(fprime_test_api):
    seq = """
var1: F32 = 4.0
var2: F32 = 0.5
if var1 ** var2 == 2:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_log(fprime_test_api):
    seq = """
if log(4.0) > 1.385 and log(4.0) < 1.387:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_mod_float(fprime_test_api):
    seq = """
var1: F32 = 25.25
var2: F32 = 5
if var1 % var2 == 0.25 and (var1 + 1) % var2 == 1.25:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_mod_unsigned(fprime_test_api):
    seq = """
var1: U32 = 5
var2: U32 = 20
if var2 % var1 == 0 and (var2 + 1) % var1 == 1:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_mod_signed(fprime_test_api):
    seq = """
var1: I32 = -5
var2: I32 = 20
if var2 % var1 == 0 and (var2 + 1) % var1 == -4:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


@pytest.mark.parametrize(
    "lhs_type,rhs_type,lhs_value,rhs_value,result_type,expected_value",
    [
        ("U64", "U64", "9", "2", "U64", "4"),
        ("F64", "F64", "5.5", "2.0", "F64", "2.0"),
        ("F64", "F64", "-5.5", "2.0", "F64", "-2.0"),
        ("F64", "I64", "5.5", "2", "F64", "2.0"),
        ("I64", "F64", "5", "2.5", "F64", "2.0"),
        ("U64", "F64", "9", "2.0", "F64", "4.0"),
        ("F64", "U64", "9.0", "2", "F64", "4.0"),
    ],
)
def test_floor_divide_64_bit_numeric_types(
    fprime_test_api,
    lhs_type,
    rhs_type,
    lhs_value,
    rhs_value,
    result_type,
    expected_value,
):
    seq = f"""
lhs: {lhs_type} = {lhs_value}
rhs: {rhs_type} = {rhs_value}
result: {result_type} = lhs // rhs
assert result == {expected_value}
"""

    assert_run_success(fprime_test_api, seq)


@pytest.mark.parametrize(
    "lhs_type,rhs_type,lhs_value,rhs_value,result_type,expected_value",
    [
        ("I64", "U64", "9", "2", "U64", "4"),
        ("U64", "I64", "9", "2", "U64", "4"),
    ],
)
def test_floor_divide_64_bit_bad_type_pairs(
    fprime_test_api,
    lhs_type,
    rhs_type,
    lhs_value,
    rhs_value,
    result_type,
    expected_value,
):
    seq = f"""
lhs: {lhs_type} = {lhs_value}
rhs: {rhs_type} = {rhs_value}
result: {result_type} = lhs // rhs
assert result == {expected_value}
"""

    assert_compile_failure(fprime_test_api, seq)


def test_unary_plus_unsigned(fprime_test_api):
    seq = """
var: U32 = 1
if +var == var:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_unary_plus_signed(fprime_test_api):
    seq = """
var: I32 = 1
if +var == var:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_unary_plus_float(fprime_test_api):
    seq = """
var: F32 = 1.0
if +var == var:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_unary_minus_signed(fprime_test_api):
    seq = """
var: I32 = 1
if -var == -1:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_unary_minus_float(fprime_test_api):
    seq = """
var: F32 = 1.0
if -var == -1.0:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_negative_int_literal_unsigned_op(fprime_test_api):
    seq = """
var: U32 = 1
if -var == -1:
    exit(0)
exit(1)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_abs_float(fprime_test_api):
    seq = """
assert fabs(1.0) == 1.0
assert fabs(-1.0) == 1.0
assert fabs(0.0) == 0.0
"""

    assert_run_success(fprime_test_api, seq)


def test_abs_i64(fprime_test_api):
    seq = """
assert iabs(I64(-1)) == 1
assert iabs(I64(1)) == 1
assert iabs(I64(0)) == 0
# need to use a large subtract here cuz otherwise float precision kills us... this is kinda sus
assert iabs(I64(2**63 - 6556)) == 2**63 - 6556
"""

    assert_run_success(fprime_test_api, seq)


def test_abs_u64(fprime_test_api):
    seq = """
# fails, iabs takes signed
assert iabs(U64(1)) == 1
assert iabs(U64(0)) == 0
"""

    assert_compile_failure(fprime_test_api, seq)


def test_abs_literal_int(fprime_test_api):
    seq = """
assert iabs(1) == 1
assert iabs(-1) == 1
"""

    assert_run_success(fprime_test_api, seq)


def test_abs_literal_float(fprime_test_api):
    seq = """
assert fabs(1.0) == 1.0
assert fabs(-1.0) == 1.0
"""

    assert_run_success(fprime_test_api, seq)


def test_const_divide_by_zero(fprime_test_api):
    seq = """
1 / 0
"""

    assert_compile_failure(fprime_test_api, seq)


def test_const_complex_pow(fprime_test_api):
    seq = """
(-1) ** 0.5
"""

    assert_compile_failure(fprime_test_api, seq)


def test_very_large_const_pow(fprime_test_api):
    seq = """
10.0 ** 1000
"""

    assert_run_success(fprime_test_api, seq)
