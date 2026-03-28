from fpy.types import U32

from fpy.model import DirectiveErrorCode
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_failure,
    assert_run_success,
)


class TestEnums:

    def test_bad_enum_ctor(self, fprime_test_api):
        seq = """
Ref.SG5.Settings(123, 0.5, 0.5, Ref.SignalType(1))
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_var_with_enum_type(self, fprime_test_api):
        seq = """
var: Ref.Choice = Ref.Choice.ONE
"""

        assert_run_success(fprime_test_api, seq)

class TestStructs:

    def test_get_const_struct_member(self, fprime_test_api):
        seq = """
var: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
if var.priority == 3:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_get_member_of_anon_expr(self, fprime_test_api):
        seq = """
var: U32 = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED).priority
if var == 3:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_struct_ctor_var_arg(self, fprime_test_api):
        seq = """
id: U32 = 111
priority: U32 = 3
state: Fw.DpState = Fw.DpState.UNTRANSMITTED
var: Svc.DpRecord = Svc.DpRecord(id, 1, 2, priority, 4, 5, state)
if var.priority == priority and var.id == id and state == var.state:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_struct_eq(self, fprime_test_api):
        seq = """
var: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
var2: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
var3: Svc.DpRecord = Svc.DpRecord(123, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
if var == var2 and var != var3:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_calling_struct_field_should_fail_gracefully(self, fprime_test_api):
        seq = """
var: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
var.priority()
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_missing_struct_member(self, fprime_test_api):
        seq = """
record: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
value: U32 = record.missing_field
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_assign_field_with_type_ann_bad(self, fprime_test_api):
        seq = """
var: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
var.priority = 123
if var.priority == 123:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_assign_field_with_type_ann_bad_2(self, fprime_test_api):
        seq = """
var: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
var.priority: U8 = 123
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_assign_field_before_declare(self, fprime_test_api):
        seq = """
var.priority = 123
var: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_set_member_of_anon_expr(self, fprime_test_api):
        seq = """
Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED).priority = 5
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_write_struct_array_member_const_idx(self, fprime_test_api):
        """Write to a struct's array member at a constant index.

        Ref.SignalInfo has:
          type: Ref.SignalType (4 bytes)
          history: F32[4] (16 bytes)  -- offset 4 within struct
          pairHistory: Ref.SignalPairSet (32 bytes)
        """
        seq = """
info: Ref.SignalInfo = Ref.SignalInfo( \\
    Ref.SignalType.TRIANGLE, \\
    Ref.SignalSet(0.0, 0.0, 0.0, 0.0), \\
    Ref.SignalPairSet( \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0)))
info.history[1] = 42.0
assert info.history[1] == 42.0
assert info.history[0] == 0.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_write_struct_array_member_var_idx(self, fprime_test_api):
        """Write to a struct's array member with a variable index."""
        seq = """
info: Ref.SignalInfo = Ref.SignalInfo( \\
    Ref.SignalType.TRIANGLE, \\
    Ref.SignalSet(0.0, 0.0, 0.0, 0.0), \\
    Ref.SignalPairSet( \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0)))
idx: I64 = 1
info.history[idx] = 42.0
assert info.history[1] == 42.0
assert info.history[0] == 0.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_write_struct_array_member_nonzero_history(self, fprime_test_api):
        """Write to history[0] — the first element of an array member at a
        non-zero offset within the struct."""
        seq = """
info: Ref.SignalInfo = Ref.SignalInfo( \\
    Ref.SignalType.TRIANGLE, \\
    Ref.SignalSet(0.0, 0.0, 0.0, 0.0), \\
    Ref.SignalPairSet( \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0)))
info.history[0] = 77.0
assert info.history[0] == 77.0
"""
        assert_run_success(fprime_test_api, seq)

class TestArrays:

    def test_array_ctor_var_arg(self, fprime_test_api):
        seq = """
arr_0: U32 = 123
arr_1: U32 = 456
val: Svc.ComQueueDepth = Svc.ComQueueDepth(arr_0, arr_1)
if val[0] == arr_0 and val[1] == arr_1:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_construct_array(self, fprime_test_api):
        seq = """
val: Svc.ComQueueDepth = Svc.ComQueueDepth(0, 0)
"""

        assert_run_success(fprime_test_api, seq)

    def test_get_item_of_array(self, fprime_test_api):
        seq = """
val: Svc.ComQueueDepth = Svc.ComQueueDepth(222, 111)
if val[0] == 222:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_get_item_of_anon_expr(self, fprime_test_api):
        seq = """
if Svc.ComQueueDepth(123, 456)[1] == 456:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_assign_array_element(self, fprime_test_api):
        seq = """
val: Svc.ComQueueDepth = Svc.ComQueueDepth(0, 0)
val[0] = 55
if val[0] == 55:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_assign_array_element_with_type_ann_bad(self, fprime_test_api):
        seq = """
val: Svc.ComQueueDepth = Svc.ComQueueDepth(0, 0)
val[0]: U8 = 55
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_set_item_of_anon_expr(self, fprime_test_api):
        seq = """
Svc.ComQueueDepth(123, 456)[1] = 456
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_array_oob_1(self, fprime_test_api):
        seq = """
val: Svc.ComQueueDepth = Svc.ComQueueDepth(0, 0)
val[2] = 3
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_array_oob_2(self, fprime_test_api):
        seq = """
val: Svc.ComQueueDepth = Svc.ComQueueDepth(123, 456)
if val[-1] == 456:
    exit(0)
exit(1)
"""
        # TODO in the future this should work, should be the last element
        assert_compile_failure(fprime_test_api, seq)

    def test_const_array_oob(self, fprime_test_api):
        """Out-of-bounds on a const array expression (not a variable).
    The parent is a type constructor call, so CalculateConstExprValues
    has a non-None parent_value. Without the bounds guard there,
    this would crash with a Python IndexError instead of a compile error.
    """
        seq = """
val: U32 = Svc.ComQueueDepth(10, 20)[2]
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_get_variable_array_idx(self, fprime_test_api):
        seq = """
val: Svc.ComQueueDepth = Svc.ComQueueDepth(456, 123)
idx: I8 = 1
if val[idx] == 123:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_get_variable_array_idx_oob(self, fprime_test_api):
        seq = """
val: Svc.ComQueueDepth = Svc.ComQueueDepth(456, 123)
idx: I8 = 2
if val[idx] == 123:
    exit(0)
exit(1)
"""

        assert_run_failure(fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR)

    def test_get_variable_array_idx_oob_2(self, fprime_test_api):
        seq = """
val: Svc.ComQueueDepth = Svc.ComQueueDepth(456, 123)
idx: I8 = -1
if val[idx] == 123:
    exit(0)
exit(1)
"""

        # TODO this really should also assert the failure code
        assert_run_failure(fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR)

    def test_set_variable_array_idx_oob(self, fprime_test_api):
        seq = """
val: Svc.ComQueueDepth = Svc.ComQueueDepth(456, 123)
idx: I8 = 2
val[idx] = 111
"""

        assert_run_failure(fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR)

    def test_set_variable_array_idx(self, fprime_test_api):
        seq = """
val: Svc.ComQueueDepth = Svc.ComQueueDepth(456, 123)
idx: I8 = 1
val[idx] = 111
if val[1] == 111:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_get_item_of_struct(self, fprime_test_api):
        seq = """
record: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
value: U32 = record[0]
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_write_array_elem_struct_member(self, fprime_test_api):
        """Ref.SignalPairSet is Ref.SignalPair[4].
        Ref.SignalPair has {time: F32, value: F32}.
        Writing to pairs[0].value should work."""
        seq = """
pairs: Ref.SignalPairSet = Ref.SignalPairSet( \\
    Ref.SignalPair(1.0, 2.0), \\
    Ref.SignalPair(3.0, 4.0), \\
    Ref.SignalPair(5.0, 6.0), \\
    Ref.SignalPair(7.0, 8.0))
pairs[0].value = 99.0
assert pairs[0].value == 99.0
assert pairs[0].time == 1.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_write_array_elem_struct_member_var_idx(self, fprime_test_api):
        """Write to array element's struct member with variable index."""
        seq = """
pairs: Ref.SignalPairSet = Ref.SignalPairSet( \\
    Ref.SignalPair(1.0, 2.0), \\
    Ref.SignalPair(3.0, 4.0), \\
    Ref.SignalPair(5.0, 6.0), \\
    Ref.SignalPair(7.0, 8.0))
idx: I64 = 1
pairs[idx].value = 99.0
assert pairs[1].value == 99.0
assert pairs[1].time == 3.0
"""
        assert_run_success(fprime_test_api, seq)

class TestConstFoldEquality:

    def test_const_fold_struct_eq(self, fprime_test_api):
        """Test that struct equality can be constant folded"""
        seq = """
# Both structs are constant expressions, should be folded at compile time
if Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED) == Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED):
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_const_fold_struct_neq(self, fprime_test_api):
        """Test that struct inequality can be constant folded"""
        seq = """
# Both structs are constant expressions with different values
if Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED) != Svc.DpRecord(123, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED):
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_const_fold_enum_eq(self, fprime_test_api):
        """Test that enum constant equality can be constant folded"""
        seq = """
# Both are enum constants, should be folded at compile time
if Fw.DpState.UNTRANSMITTED == Fw.DpState.UNTRANSMITTED:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_const_fold_enum_neq(self, fprime_test_api):
        """Test that enum constant inequality can be constant folded"""
        seq = """
# Both are enum constants with different values
if Fw.DpState.UNTRANSMITTED != Fw.DpState.TRANSMITTED:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_const_fold_array_eq(self, fprime_test_api):
        """Test that array equality can be constant folded"""
        seq = """
# Both arrays are constant expressions, should be folded at compile time
if Svc.ComQueueDepth(100, 200) == Svc.ComQueueDepth(100, 200):
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_const_fold_array_neq(self, fprime_test_api):
        """Test that array inequality can be constant folded"""
        seq = """
# Both arrays are constant expressions with different values
if Svc.ComQueueDepth(100, 200) != Svc.ComQueueDepth(100, 300):
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_runtime_array_equality(self, fprime_test_api):
        """Array equality with runtime (non-const) operands."""
        seq = """
arr1: Svc.ComQueueDepth = Svc.ComQueueDepth(100, 200)
arr2: Svc.ComQueueDepth = Svc.ComQueueDepth(100, 200)
if arr1 == arr2:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_runtime_enum_equality(self, fprime_test_api):
        """Enum equality with runtime (non-const) operands."""
        seq = """
e1: Fw.DpState = Fw.DpState.UNTRANSMITTED
e2: Fw.DpState = Fw.DpState.UNTRANSMITTED
if e1 == e2:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

class TestTypeErrors:

    def test_u8_too_large(self, fprime_test_api):
        seq = """
var: U8 = 123
var = 256
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_wrong_bool_type(self, fprime_test_api):
        seq = """
val: bool = 123
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_int_literal_as_float(self, fprime_test_api):
        seq = """
var: F32 = 1
if var == 1.0:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_complex_eq_fail(self, fprime_test_api):
        seq = """
var: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
var2: Fw.CmdResponse = Fw.CmdResponse.OK
exit(var == var2)
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_literal_float_coercion_overflow(self, fprime_test_api):
        seq = """
var: F32 = 999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_const_float_downcast_rounding(self, fprime_test_api):
        rounding_seq = """
narrow: F32 = 16777217.0
assert F64(narrow) == 16777216.0
"""

        assert_run_success(fprime_test_api, rounding_seq)

    def test_const_float_overflow(self, fprime_test_api):
        overflow_seq = """
narrow: F32 = 3.5e38
"""

        assert_compile_failure(fprime_test_api, overflow_seq)

    def test_calling_variable_should_fail_gracefully(self, fprime_test_api):
        seq = """
x: U32 = 1
x()
"""

        assert_compile_failure(fprime_test_api, seq)

class TestStringTypes:

    def test_string_eq(self, fprime_test_api):
        seq = """
exit("asdf" == "asdf")
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_string_type(self, fprime_test_api):
        seq = """
var: string = "test"
"""
        assert_compile_failure(fprime_test_api, seq)

class TestConstCasts:

    def test_signed_int_const_casts(self, fprime_test_api):
        seq = """
assert I8(-256) == 0
assert I8(-129) == 127
assert I16(-65536) == 0
assert I16(-32769) == 32767
assert I32(-4294967296) == 0
assert I32(-2147483649) == 2147483647
assert I64(-18446744073709551616) == 0
assert I64(-9223372036854775809) == 9223372036854775807
"""

        assert_run_success(fprime_test_api, seq)

    def test_unsigned_int_const_casts(self, fprime_test_api):
        seq = """
assert U8(-1) == 255
assert U8(256) == 0
assert U16(-1) == 65535
assert U16(65536) == 0
assert U32(-1) == 4294967295
assert U32(4294967296) == 0
assert U64(-1) == 18446744073709551615
assert U64(18446744073709551616) == 0
"""

        assert_run_success(fprime_test_api, seq)

    def test_float_const_casts(self, fprime_test_api):
        seq = """
assert F32(0.5) == 0.5
assert F32(-0.75) == -0.75
assert F32(1024.0) == 1024.0
assert F32(-2048.0) == -2048.0
assert F64(0.5) == 0.5
assert F64(-0.75) == -0.75
assert F64(123456789.5) == 123456789.5
assert F64(-987654321.25) == -987654321.25
"""

        assert_run_success(fprime_test_api, seq)

    def test_float_to_signed_int_const_casts(self, fprime_test_api):
        seq = """
assert I8(-128.0) == -128
assert I8(127.0) == 127
assert I16(-32768.0) == -32768
assert I16(32767.0) == 32767
assert I32(-2147483648.0) == -2147483648
assert I32(2147483647.0) == 2147483647
"""

        assert_run_success(fprime_test_api, seq)

    def test_float_to_unsigned_int_const_casts(self, fprime_test_api):
        seq = """
assert U8(0.0) == 0
assert U8(255.0) == 255
assert U16(65535.0) == 65535
assert U32(4294967295.0) == 4294967295
"""

        assert_run_success(fprime_test_api, seq)

    def test_signed_int_to_float_const_casts(self, fprime_test_api):
        seq = """
assert F32(-128) == -128.0
assert F64(-128) == -128.0
assert F32(32767) == 32767.0
assert F64(32767) == 32767.0
assert F64(-2147483648) == -2147483648.0
assert F64(2147483647) == 2147483647.0
"""

        assert_run_success(fprime_test_api, seq)

    def test_unsigned_int_to_float_const_casts(self, fprime_test_api):
        seq = """
assert F32(U32(0)) == 0.0
assert F64(U32(0)) == 0.0
assert F32(U32(65535)) == 65535.0
assert F64(U32(65535)) == 65535.0
assert F64(U64(4294967295)) == 4294967295.0
"""

        assert_run_success(fprime_test_api, seq)

class TestRuntimeCasts:

    def test_signed_int_runtime_casts(self, fprime_test_api):
        seq = """
src: I64 = -256
assert I8(src) == 0

src = -129
assert I8(src) == 127

src = -65536
assert I16(src) == 0

src = -32769
assert I16(src) == 32767

src = -4294967296
assert I32(src) == 0

src = -2147483649
assert I32(src) == 2147483647
"""

        assert_run_success(fprime_test_api, seq)

    def test_unsigned_int_runtime_casts(self, fprime_test_api):
        seq = """
signed_src: I64 = -1
assert U8(signed_src) == 255
assert U16(signed_src) == 65535
assert U32(signed_src) == 4294967295
assert U64(signed_src) == 18446744073709551615

unsigned_src: U64 = 256
assert U8(unsigned_src) == 0

unsigned_src = 65536
assert U16(unsigned_src) == 0

unsigned_src = 4294967296
assert U32(unsigned_src) == 0
"""

        assert_run_success(fprime_test_api, seq)

    def test_float_runtime_casts(self, fprime_test_api):
        seq = """
wide_src: F64 = 0.5
assert F32(wide_src) == 0.5

wide_src = -0.75
assert F32(wide_src) == -0.75

wide_src = 1024.0
assert F32(wide_src) == 1024.0

wide_src = -2048.0
assert F32(wide_src) == -2048.0

narrow_src: F32 = 0.5
assert F64(narrow_src) == 0.5

narrow_src = -0.75
assert F64(narrow_src) == -0.75

narrow_src = 123.5
assert F64(narrow_src) == 123.5

narrow_src = -987.25
assert F64(narrow_src) == -987.25
"""

        assert_run_success(fprime_test_api, seq)

    def test_float_to_signed_int_runtime_casts(self, fprime_test_api):
        seq = """
f_src: F64 = -128.0
assert I8(f_src) == -128

f_src = 127.0
assert I8(f_src) == 127

f_src = -32768.0
assert I16(f_src) == -32768

f_src = 32767.0
assert I16(f_src) == 32767

f_src = -2147483648.0
assert I32(f_src) == -2147483648

f_src = 2147483647.0
assert I32(f_src) == 2147483647
"""

        assert_run_success(fprime_test_api, seq)

    def test_float_to_unsigned_int_runtime_casts(self, fprime_test_api):
        seq = """
f_src: F64 = 0.0
assert U8(f_src) == 0

f_src = 255.0
assert U8(f_src) == 255

f_src = 65535.0
assert U16(f_src) == 65535

f_src = 4294967295.0
assert U32(f_src) == 4294967295
"""

        assert_run_success(fprime_test_api, seq)

    def test_signed_int_to_float_runtime_casts(self, fprime_test_api):
        seq = """
i_src: I32 = -128
assert F32(i_src) == -128.0
assert F64(i_src) == -128.0

i_src = 32767
assert F32(i_src) == 32767.0
assert F64(i_src) == 32767.0

wide_src: I64 = -2147483648
assert F64(wide_src) == -2147483648.0

wide_src = 2147483647
assert F64(wide_src) == 2147483647.0
"""

        assert_run_success(fprime_test_api, seq)

    def test_unsigned_int_to_float_runtime_casts(self, fprime_test_api):
        seq = """
u_src: U32 = 0
assert F32(u_src) == 0.0
assert F64(u_src) == 0.0

u_src = 65535
assert F32(u_src) == 65535.0
assert F64(u_src) == 65535.0

wide_src: U64 = 4294967295
assert F64(wide_src) == 4294967295.0
"""

        assert_run_success(fprime_test_api, seq)

    def test_downcast(self, fprime_test_api):
        seq = """
i: U32 = 123123
u: U8 = U8(i)
assert u == (i % 256)
"""

        assert_run_success(fprime_test_api, seq)

    def test_downcast_fail(self, fprime_test_api):
        seq = """
i: U32 = 123123
u: U8 = i
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_upcast(self, fprime_test_api):
        seq = """
i: U8 = 255
u: U32 = U32(i)
assert u == i
"""

        assert_run_success(fprime_test_api, seq)

    def test_downcast_large_literal(self, fprime_test_api):
        seq = """
val: U8 = U8(1231231231243) # this is allowed but suspicious
"""

        assert_run_success(fprime_test_api, seq)

class TestNamedArgsInCtors:

    def test_named_arg_type_ctor(self, fprime_test_api):
        """Named arguments work with type constructors."""
        seq = """
time: Fw.Time = Fw.Time(seconds=123, useconds=456, timeBase=TimeBase.TB_NONE, timeContext=0)
assert time.seconds == 123
assert time.useconds == 456
"""

        assert_run_success(fprime_test_api, seq)

    def test_named_arg_struct_ctor(self, fprime_test_api):
        """Named arguments work with struct constructors."""
        seq = """
pair: Ref.ChoicePair = Ref.ChoicePair(secondChoice=Ref.Choice.TWO, firstChoice=Ref.Choice.ONE)
assert pair.firstChoice == Ref.Choice.ONE
assert pair.secondChoice == Ref.Choice.TWO
"""

        assert_run_success(fprime_test_api, seq)

    def test_named_arg_cast(self, fprime_test_api):
        """Named arguments work with type casts."""
        seq = """
val: F64 = 3.14
result: U32 = U32(value=val)
assert result == 3
"""

        assert_run_success(fprime_test_api, seq)

    def test_named_arg_coercion_int_to_float(self, fprime_test_api):
        """Named arguments correctly coerce int to float."""
        seq = """
def test(x: F64) -> F64:
    return x + 0.5

# U32 should be coerced to F64
val: U32 = 10
assert test(x=val) == 10.5
"""

        assert_run_success(fprime_test_api, seq)

    def test_named_arg_coercion_narrow_to_wide(self, fprime_test_api):
        """Named arguments correctly coerce narrow int to wider int."""
        seq = """
def test(x: U64) -> U64:
    return x + 1

small: U8 = 100
# U8 should be coerced to U64
assert test(x=small) == 101
"""

        assert_run_success(fprime_test_api, seq)

    def test_named_arg_coercion_reordered(self, fprime_test_api):
        """Named arguments maintain correct coercion when reordered."""
        seq = """
def test(a: U64, b: F64, c: I32) -> F64:
    return b + F64(a) + F64(c)

# Pass in different order with finite bitwidth types
x: I8 = -5
y: U32 = 10
z: F32 = 0.5
assert test(c=x, a=y, b=z) == 5.5
"""

        assert_run_success(fprime_test_api, seq)

class TestNonConstSized:

    def test_non_const_sized_var_decl(self, fprime_test_api):
        """Variable declarations with non-constant-sized types should fail."""
        seq = """
val: Ref.DpDemo.StringArray = Ref.DpDemo.StringArray("a", "b")
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_non_const_sized_func_param(self, fprime_test_api):
        """Function parameters with non-constant-sized types should fail."""
        seq = """
def foo(x: Ref.DpDemo.StringArray):
    pass
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_non_const_sized_func_return(self, fprime_test_api):
        """Function return types with non-constant-sized types should fail."""
        seq = """
def foo() -> Ref.DpDemo.StringArray:
    pass
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_non_const_sized_ctor_call(self, fprime_test_api):
        """Calling constructors for non-constant-sized types should fail."""
        seq = """
Ref.DpDemo.StringArray("a", "b")
"""
        assert_compile_failure(fprime_test_api, seq)

class TestConstructorDefaults:

    def test_struct_ctor_all_defaults(self, fprime_test_api):
        """Struct constructor with no args should use all defaults from dictionary."""
        seq = """
pair: Ref.SignalPair = Ref.SignalPair()
assert pair.time == 0.0
assert pair.value == 0.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_struct_ctor_partial_defaults(self, fprime_test_api):
        """Struct constructor with some args should use defaults for the rest."""
        seq = """
pair: Ref.SignalPair = Ref.SignalPair(time=1.0)
assert pair.time == 1.0
assert pair.value == 0.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_struct_ctor_override_all_defaults(self, fprime_test_api):
        """Struct constructor with all args should ignore defaults."""
        seq = """
pair: Ref.SignalPair = Ref.SignalPair(3.0, 4.0)
assert pair.time == 3.0
assert pair.value == 4.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_array_ctor_all_defaults(self, fprime_test_api):
        """Array constructor with no args should use all defaults from dictionary."""
        seq = """
depths: Svc.ComQueueDepth = Svc.ComQueueDepth()
assert depths[0] == 0
assert depths[1] == 0
"""
        assert_run_success(fprime_test_api, seq)

    def test_array_ctor_partial_defaults(self, fprime_test_api):
        """Array constructor with some args should use defaults for the rest."""
        seq = """
depths: Svc.ComQueueDepth = Svc.ComQueueDepth(e0=42)
assert depths[0] == 42
assert depths[1] == 0
"""
        assert_run_success(fprime_test_api, seq)

    def test_struct_ctor_enum_member_default(self, fprime_test_api):
        """Struct with an enum member should be constructable with defaults."""
        seq = """
stat: Ref.PacketStat = Ref.PacketStat()
assert stat.BuffRecv == 0
assert stat.BuffErr == 0
"""
        assert_run_success(fprime_test_api, seq)

    def test_array_elem_non_first_struct_member(self, fprime_test_api):
        """Accessing a non-first struct member on an array element must not crash.
    """
        seq = """
val: Ref.SignalPairSet = Ref.SignalPairSet( \
    Ref.SignalPair(1.0, 2.0), \
    Ref.SignalPair(3.0, 4.0), \
    Ref.SignalPair(5.0, 6.0), \
    Ref.SignalPair(7.0, 8.0))
assert val[0].value == 2.0
assert val[1].value == 4.0
"""
        assert_run_success(fprime_test_api, seq)
