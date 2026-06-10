import pytest

from fpy.types import U32

from fpy.test_helpers import assert_compile_failure, assert_run_success


class TestConstantFolding:

    def test_overflow_compile_error(self, fprime_test_api):
        seq = """
val1: U8 = 256  # Should fail: value too large for U8
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_const_fold_specific_float_binop(self, fprime_test_api):
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

    def test_float_truncate_stack_size(self, fprime_test_api):
        seq = """
var2: F64 = 123.0
var1: F32 = F32(-var2)
if var1 == -123.0:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_const_divide_by_zero(self, fprime_test_api):
        seq = """
1 / 0
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_const_complex_pow(self, fprime_test_api):
        seq = """
(-1) ** 0.5
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_very_large_const_pow(self, fprime_test_api):
        seq = """
10.0 ** 1000
"""

        assert_run_success(fprime_test_api, seq)

    def test_const_pow_decimal_overflow(self, fprime_test_api):
        """A huge constant exponent must produce a clean compile error.

        Folding ``10.0 ** 100000000000`` evaluates ``Decimal ** Decimal``,
        which raises ``decimal.Overflow`` (a sibling of the builtin
        ``OverflowError``, not a subclass). The fold's except clauses only
        caught ``OverflowError``/``decimal.InvalidOperation``, so this escaped
        as an uncaught Python traceback — a compiler crash rather than a
        diagnostic.
        """
        seq = """
x: F64 = 10.0 ** 100000000000
"""

        assert_compile_failure(fprime_test_api, seq, match="[Oo]verflow")

class TestBasicArithmetic:

    def test_add_unsigned(self, fprime_test_api):
        seq = """
var1: U32 = 500
var2: U32 = 1000
if var1 + var2 == 1500 and (var1 + 1) > var1:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_add_signed(self, fprime_test_api):
        seq = """
var1: I32 = -255
var2: I32 = 255
if var1 + var2 == 0 and (var1 + 1) > (var1 + -1):
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_add_float(self, fprime_test_api):
        seq = """
var1: F32 = -255.0
var2: F32 = 255.0
if var1 + var2 == 0.0 and (var1 + 1.0) > (var1 + -1.0):
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_sub_unsigned(self, fprime_test_api):
        seq = """
var1: U32 = 1000
var2: U32 = 500
if var1 - var2 == 500 and (var1 - 1) < var1:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_sub_signed(self, fprime_test_api):
        seq = """
var1: I32 = 255
var2: I32 = 255
if var1 - var2 == 0 and (var1 - 1) < (var1 - -1):
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_sub_float(self, fprime_test_api):
        seq = """
var1: F32 = 255.0
var2: F32 = 255.0
if var1 - var2 == 0.0 and (var1 - 1.0) < (var1 - -1.0):
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_mul_unsigned(self, fprime_test_api):
        seq = """
var1: U32 = 5
var2: U32 = 20
if var1 * var2 == 100 and (var1 * 2) > var1:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_mul_signed(self, fprime_test_api):
        seq = """
var1: I32 = -5
var2: I32 = 20
if var1 * var2 == -100 and (var1 * 2) < var1:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_mul_float(self, fprime_test_api):
        seq = """
var1: F32 = 5.0
var2: F32 = 20.0
if var1 * var2 == 100.0 and (var1 * 2.0) > var1:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_div_unsigned(self, fprime_test_api):
        seq = """
var1: U32 = 20
var2: U32 = 5
if var1 / var2 == 4.0 and (var1 / 2) < var1:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_div_signed(self, fprime_test_api):
        seq = """
var1: I32 = -20
var2: I32 = 5
if var1 / var2 == -4.0: # and (var1 / -2) > var1:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_div_float(self, fprime_test_api):
        seq = """
var1: F32 = -20.0
var2: F32 = 5.0
if var1 / var2 == -4.0 and (var1 / -2.0) > var1:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_order_of_operations(self, fprime_test_api):
        seq = """
if 1 - 2 + 3 * 4 == 11 and 10.0 / 5.0 * 2.0 == 4.0:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

class TestArithmeticWithBuiltins:

    def test_arithmetic_arg_to_builtin_bad_type(self, fprime_test_api):
        seq = """
sleep(1 + 2 * 0, (0 + 1 / 2))
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_arithmetic_arg_to_builtin(self, fprime_test_api):
        seq = """
sleep(1 + 2 * 0, (0 + 1 // 2))
"""
        assert_run_success(fprime_test_api, seq)

class TestChainedOperations:

    def test_chain_mul(self, fprime_test_api):
        seq = """
var1: I32 = 1
var2: I32 = 2
var3: I32 = 3
if var1 * var2 * var3 == 6:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_chain_add(self, fprime_test_api):
        seq = """
var1: I32 = 1
var2: I32 = 2
var3: I32 = 3
if var1 + var2 + var3 == 6:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_chain_sub(self, fprime_test_api):
        seq = """
var1: I32 = 1
var2: I32 = 2
var3: I32 = 3
if var1 - var2 - var3 == -4:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_chain_div(self, fprime_test_api):
        seq = """
var1: I32 = 3
var2: I32 = 2
var3: I32 = 1
if var1 / var3 / var2 == 3/2:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

class TestPowerModLog:

    def test_pow_unsigned(self, fprime_test_api):
        seq = """
var1: U32 = 20
var2: U32 = 2
if var1 ** var2 == 400:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_pow_signed(self, fprime_test_api):
        seq = """
var1: I32 = -20
var2: I32 = 2
if var1 ** var2 == 400:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_pow_float(self, fprime_test_api):
        seq = """
var1: F32 = 4.0
var2: F32 = 0.5
if var1 ** var2 == 2:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_pow_zero_to_negative_is_inf(self, fprime_test_api):
        """0 ** <negative> is a pole: C/IEEE pow() yields +inf, not a crash.

        With runtime (non-constant) operands the VM evaluates ``base ** exp``
        directly. Python's ``0.0 ** -1.0`` raises ``ZeroDivisionError``, which
        the handler did not catch, crashing the sequencer instead of producing
        the infinite result the source expression denotes.
        """
        seq = """
base: F64 = 0.0
result: F64 = base ** -1.0
assert result > 100000000000000000000.0
exit(0)
"""
        assert_run_success(fprime_test_api, seq)

    def test_log(self, fprime_test_api):
        seq = """
if ln(4.0) > 1.385 and ln(4.0) < 1.387:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_mod_float(self, fprime_test_api):
        seq = """
var1: F32 = 25.25
var2: F32 = 5
if var1 % var2 == 0.25 and (var1 + 1) % var2 == 1.25:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_mod_unsigned(self, fprime_test_api):
        seq = """
var1: U32 = 5
var2: U32 = 20
if var2 % var1 == 0 and (var2 + 1) % var1 == 1:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_mod_signed(self, fprime_test_api):
        seq = """
var1: I32 = -5
var2: I32 = 20
if var2 % var1 == 0 and (var2 + 1) % var1 == -4:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_pow_result_is_f64(self, fprime_test_api):
        """pow() of integers should produce F64 so fractional exponents work."""
        seq = """
result: F64 = 2 ** 0.5
assert result > 1.41 and result < 1.42
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
    def test_floor_divide_64_bit_numeric_types(self, 
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
    def test_floor_divide_64_bit_bad_type_pairs(self, 
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

    def test_const_floor_divide_large_int_precision(self, fprime_test_api):
        """Constant folding of integer floor division must be exact.

        The compiler folds ``a // b`` for integer operands via Python's
        ``int(a / b)``, which routes through 64-bit float division and loses
        precision for operands beyond 2**53.  The mathematically-correct value
        of ``7000000000000000001 // 3`` is 2333333333333333333, but the
        const-folded value baked into the bytecode is 2333333333333333504.

        The assert below is true by construction, so a correct compiler runs
        this sequence to a clean exit.  With the precision bug, the baked
        constant differs from the literal and the assert fails at runtime.
        """
        seq = """
result: I64 = 7000000000000000001 // 3
assert result == 2333333333333333333
"""
        assert_run_success(fprime_test_api, seq)

class TestUnaryOperators:

    @pytest.mark.parametrize("type_name,value", [
        ("U8", "1"),
        ("I8", "1"),
        ("U16", "1"),
        ("I16", "1"),
        ("U32", "1"),
        ("I32", "1"),
        ("U64", "1"),
        ("I64", "1"),
        ("F32", "1.0"),
        ("F64", "1.0"),
    ])
    def test_unary_plus(self, fprime_test_api, type_name, value):
        seq = f"""
var: {type_name} = {value}
if +var == var:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_unary_minus_signed(self, fprime_test_api):
        seq = """
var: I32 = 1
if -var == -1:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_unary_minus_float(self, fprime_test_api):
        seq = """
var: F32 = 1.0
if -var == -1.0:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_negative_int_literal_unsigned_op(self, fprime_test_api):
        seq = """
var: U32 = 1
if -var == -1:
    exit(0)
exit(1)
"""
        assert_compile_failure(fprime_test_api, seq)

class TestAbs:

    def test_abs_float(self, fprime_test_api):
        seq = """
assert fabs(1.0) == 1.0
assert fabs(-1.0) == 1.0
assert fabs(0.0) == 0.0
"""

        assert_run_success(fprime_test_api, seq)

    def test_abs_i64(self, fprime_test_api):
        seq = """
assert iabs(I64(-1)) == 1
assert iabs(I64(1)) == 1
assert iabs(I64(0)) == 0
# need to use a large subtract here cuz otherwise float precision kills us... this is kinda sus
assert iabs(I64(2**63 - 6556)) == 2**63 - 6556
"""

        assert_run_success(fprime_test_api, seq)

    def test_abs_u64(self, fprime_test_api):
        seq = """
# fails, iabs takes signed
assert iabs(U64(1)) == 1
assert iabs(U64(0)) == 0
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_abs_literal_int(self, fprime_test_api):
        seq = """
assert iabs(1) == 1
assert iabs(-1) == 1
"""

        assert_run_success(fprime_test_api, seq)


class TestFloorDivision:
    """Floor division uses C++ truncation semantics (toward zero).
    Both const-folded and runtime paths should agree."""

    def test_int_floor_div_negative_const_vs_runtime(self, fprime_test_api):
        """Runtime -7 // 2 should give -3 (truncation toward zero)."""
        seq = """
a: I64 = -7
b: I64 = 2
result: I64 = a // b
assert result == -3
"""
        assert_run_success(fprime_test_api, seq)

    def test_int_floor_div_negative_const_folded(self, fprime_test_api):
        """Const-folded (-7) // 2 should also give -3 (truncation toward zero)."""
        seq = """
result: I64 = (-7) // 2
assert result == -3
"""
        assert_run_success(fprime_test_api, seq)

    def test_float_floor_div_negative_const_vs_runtime(self, fprime_test_api):
        """Runtime float floor division: -5.5 // 2.0 = -2.0 (truncation toward zero)."""
        seq = """
a: F64 = -5.5
b: F64 = 2.0
result: F64 = a // b
assert result == -2.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_float_floor_div_negative_const_folded(self, fprime_test_api):
        """Const-folded (-5.5) // 2.0 should also give -2.0 (truncation toward zero)."""
        seq = """
result: F64 = (-5.5) // 2.0
assert result == -2.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_int_floor_div_positive(self, fprime_test_api):
        """Positive floor division: 7 // 2 = 3."""
        seq = """
result: I64 = 7 // 2
assert result == 3
"""
        assert_run_success(fprime_test_api, seq)

    def test_int_floor_div_negative_divisor(self, fprime_test_api):
        """7 // (-2) = -3 (truncation toward zero)."""
        seq = """
result: I64 = 7 // (-2)
assert result == -3
"""
        assert_run_success(fprime_test_api, seq)


