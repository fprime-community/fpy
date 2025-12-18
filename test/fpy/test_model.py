"""
Tests for fpy.model - FpySequencerModel directive handlers and error conditions.
"""

import pytest
from fpy.model import (
    DirectiveErrorCode,
    FpySequencerModel,
    overflow_check,
    MIN_INT64,
)
from fpy.bytecode.directives import (
    AllocateDirective,
    AndDirective,
    CallDirective,
    DiscardDirective,
    ExitDirective,
    FloatAddDirective,
    FloatDivideDirective,
    FloatEqualDirective,
    FloatExponentDirective,
    FloatExtendDirective,
    FloatGreaterThanDirective,
    FloatGreaterThanOrEqualDirective,
    FloatLessThanDirective,
    FloatLessThanOrEqualDirective,
    FloatLogDirective,
    FloatModuloDirective,
    FloatMultiplyDirective,
    FloatNotEqualDirective,
    FloatSubtractDirective,
    FloatToSignedIntDirective,
    FloatToUnsignedIntDirective,
    FloatTruncateDirective,
    GetFieldDirective,
    GotoDirective,
    IfDirective,
    IntAddDirective,
    IntEqualDirective,
    IntMultiplyDirective,
    IntNotEqualDirective,
    IntSubtractDirective,
    IntegerSignedExtend16To64Directive,
    IntegerSignedExtend32To64Directive,
    IntegerSignedExtend8To64Directive,
    IntegerTruncate64To16Directive,
    IntegerTruncate64To32Directive,
    IntegerTruncate64To8Directive,
    IntegerZeroExtend16To64Directive,
    IntegerZeroExtend32To64Directive,
    IntegerZeroExtend8To64Directive,
    LoadAbsDirective,
    LoadRelDirective,
    MemCompareDirective,
    NoOpDirective,
    NotDirective,
    OrDirective,
    PeekDirective,
    PushPrmDirective,
    PushTimeDirective,
    PushTlmValDirective,
    PushValDirective,
    ReturnDirective,
    SignedGreaterThanDirective,
    SignedGreaterThanOrEqualDirective,
    SignedIntDivideDirective,
    SignedIntToFloatDirective,
    SignedLessThanDirective,
    SignedLessThanOrEqualDirective,
    SignedModuloDirective,
    StackCmdDirective,
    StoreAbsConstOffsetDirective,
    StoreAbsDirective,
    StoreRelConstOffsetDirective,
    StoreRelDirective,
    UnsignedGreaterThanDirective,
    UnsignedGreaterThanOrEqualDirective,
    UnsignedIntDivideDirective,
    UnsignedIntToFloatDirective,
    UnsignedLessThanDirective,
    UnsignedLessThanOrEqualDirective,
    UnsignedModuloDirective,
    WaitAbsDirective,
    WaitRelDirective,
)


class TestOverflowCheck:
    """Tests for the overflow_check function."""

    def test_positive(self):
        assert overflow_check(100) == 100

    def test_negative(self):
        assert overflow_check(-1) == -1

    def test_large_positive_wraps(self):
        """Test wrapping behavior for large positive numbers."""
        result = overflow_check(2**63)
        assert result == -(2**63)


class TestStackAllocation:
    """Tests for stack allocation directives."""

    def test_allocate_stack_overflow(self):
        """Test that allocating more than max stack size returns STACK_OVERFLOW."""
        model = FpySequencerModel(stack_size=100)
        result = model.run([AllocateDirective(200)])
        assert result == DirectiveErrorCode.STACK_OVERFLOW

    def test_discard_out_of_bounds(self):
        """Test discarding more bytes than on stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(DiscardDirective(8))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS


class TestLoadDirectives:
    """Tests for load directive error conditions."""

    def test_load_rel_stack_overflow(self):
        """Test load_rel when result would overflow stack."""
        model = FpySequencerModel(stack_size=16)
        model.stack = bytearray(16)
        result = model.dispatch(LoadRelDirective(0, 8))
        assert result == DirectiveErrorCode.STACK_OVERFLOW

    def test_load_rel_negative_offset(self):
        """Test load_rel with offset that goes negative."""
        model = FpySequencerModel()
        model.stack = bytearray(16)
        model.stack_frame_start = 4
        result = model.dispatch(LoadRelDirective(-10, 8))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_load_rel_beyond_stack(self):
        """Test load_rel trying to read beyond stack."""
        model = FpySequencerModel()
        model.stack = bytearray(16)
        model.stack_frame_start = 0
        result = model.dispatch(LoadRelDirective(12, 8))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_load_abs_stack_overflow(self):
        """Test load_abs when result would overflow stack."""
        model = FpySequencerModel(stack_size=16)
        model.stack = bytearray(16)
        result = model.dispatch(LoadAbsDirective(0, 8))
        assert result == DirectiveErrorCode.STACK_OVERFLOW

    def test_load_abs_negative_offset(self):
        """Test load_abs with negative offset."""
        model = FpySequencerModel()
        model.stack = bytearray(16)
        result = model.dispatch(LoadAbsDirective(-1, 8))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_load_abs_beyond_stack(self):
        """Test load_abs trying to read beyond stack."""
        model = FpySequencerModel()
        model.stack = bytearray(16)
        result = model.dispatch(LoadAbsDirective(12, 8))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS


class TestStoreDirectives:
    """Tests for store directive error conditions."""

    def test_store_rel_stack_underflow(self):
        """Test store_rel with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(StoreRelDirective(8))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_store_abs_stack_underflow(self):
        """Test store_abs with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(StoreAbsDirective(8))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_store_rel_const_offset_underflow(self):
        """Test store_rel_const_offset with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(StoreRelConstOffsetDirective(0, 8))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_store_abs_const_offset_underflow(self):
        """Test store_abs_const_offset with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(StoreAbsConstOffsetDirective(0, 8))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS


class TestPushDirectives:
    """Tests for push directive error conditions."""

    def test_push_val_stack_overflow(self):
        """Test push_val when stack would overflow."""
        model = FpySequencerModel(stack_size=4)
        model.stack = bytearray(4)
        result = model.dispatch(PushValDirective(b"\x00" * 8))
        assert result == DirectiveErrorCode.STACK_OVERFLOW

    def test_push_tlm_not_found(self):
        """Test push_tlm when channel not in database."""
        model = FpySequencerModel()
        model.tlm_db = {}
        result = model.dispatch(PushTlmValDirective(999))
        assert result == DirectiveErrorCode.TLM_CHAN_NOT_FOUND

    def test_push_tlm_stack_overflow(self):
        """Test push_tlm when would overflow stack."""
        model = FpySequencerModel(stack_size=4)
        model.stack = bytearray(4)
        model.tlm_db = {1: bytearray(8)}
        result = model.dispatch(PushTlmValDirective(1))
        assert result == DirectiveErrorCode.STACK_OVERFLOW

    def test_push_prm_not_found(self):
        """Test push_prm when parameter not in database."""
        model = FpySequencerModel()
        model.prm_db = {}
        result = model.dispatch(PushPrmDirective(999))
        assert result == DirectiveErrorCode.PRM_NOT_FOUND

    def test_push_prm_stack_overflow(self):
        """Test push_prm when would overflow stack."""
        model = FpySequencerModel(stack_size=4)
        model.stack = bytearray(4)
        model.prm_db = {1: bytearray(8)}
        result = model.dispatch(PushPrmDirective(1))
        assert result == DirectiveErrorCode.STACK_OVERFLOW

    def test_push_time_stack_overflow(self):
        """Test push_time when would overflow stack."""
        model = FpySequencerModel(stack_size=4)
        model.stack = bytearray(4)
        result = model.dispatch(PushTimeDirective())
        assert result == DirectiveErrorCode.STACK_OVERFLOW


class TestWaitDirectives:
    """Tests for wait directive error conditions."""

    def test_wait_rel_stack_underflow(self):
        """Test wait_rel with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(WaitRelDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_wait_abs_stack_underflow(self):
        """Test wait_abs with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(WaitAbsDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS


class TestControlFlowDirectives:
    """Tests for control flow directive error conditions."""

    def test_stack_cmd_stack_underflow(self):
        """Test stack_cmd with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(StackCmdDirective(8))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_goto_out_of_bounds(self):
        """Test goto to invalid directive index."""
        model = FpySequencerModel()
        model.dirs = [NoOpDirective()]
        result = model.dispatch(GotoDirective(100))
        assert result == DirectiveErrorCode.STMT_OUT_OF_BOUNDS

    def test_if_out_of_bounds(self):
        """Test if with invalid false branch index."""
        model = FpySequencerModel()
        model.dirs = [NoOpDirective()]
        model.stack = bytearray(1)
        result = model.dispatch(IfDirective(100))
        assert result == DirectiveErrorCode.STMT_OUT_OF_BOUNDS

    def test_if_stack_underflow(self):
        """Test if with empty stack."""
        model = FpySequencerModel()
        model.dirs = [NoOpDirective()]
        model.stack = bytearray(0)
        result = model.dispatch(IfDirective(0))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_call_stack_underflow(self):
        """Test call with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(2)
        result = model.dispatch(CallDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_return_stack_underflow(self):
        """Test return with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(ReturnDirective(8, 0))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_exit_stack_underflow(self):
        """Test exit with empty stack."""
        model = FpySequencerModel()
        model.stack = bytearray(0)
        result = model.dispatch(ExitDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS


class TestLogicalDirectives:
    """Tests for logical operation directive error conditions."""

    def test_or_stack_underflow(self):
        """Test or with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(1)
        result = model.dispatch(OrDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_and_stack_underflow(self):
        """Test and with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(1)
        result = model.dispatch(AndDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_not_stack_underflow(self):
        """Test not with empty stack."""
        model = FpySequencerModel()
        model.stack = bytearray(0)
        result = model.dispatch(NotDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS


class TestComparisonDirectives:
    """Tests for comparison directive error conditions."""

    def test_int_equal_stack_underflow(self):
        """Test int_equal with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(IntEqualDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_comparison_handlers_stack_underflow(self):
        """Test various comparison handlers with insufficient stack."""
        comparisons = [
            IntNotEqualDirective(),
            UnsignedLessThanDirective(),
            UnsignedLessThanOrEqualDirective(),
            UnsignedGreaterThanDirective(),
            UnsignedGreaterThanOrEqualDirective(),
            SignedLessThanDirective(),
            SignedLessThanOrEqualDirective(),
            SignedGreaterThanDirective(),
            SignedGreaterThanOrEqualDirective(),
            FloatEqualDirective(),
            FloatNotEqualDirective(),
            FloatLessThanDirective(),
            FloatLessThanOrEqualDirective(),
            FloatGreaterThanDirective(),
            FloatGreaterThanOrEqualDirective(),
        ]

        for directive in comparisons:
            model = FpySequencerModel()
            model.stack = bytearray(8)  # Need 16
            result = model.dispatch(directive)
            assert (
                result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS
            ), f"{type(directive).__name__} should fail"


class TestTypeConversionDirectives:
    """Tests for type conversion directive error conditions."""

    def test_float_extend_stack_underflow(self):
        """Test fpext with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(2)
        result = model.dispatch(FloatExtendDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_signed_extend_8_to_64_stack_underflow(self):
        """Test siext 8->64 with empty stack."""
        model = FpySequencerModel()
        model.stack = bytearray(0)
        result = model.dispatch(IntegerSignedExtend8To64Directive())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_signed_extend_8_to_64_stack_overflow(self):
        """Test siext 8->64 when would overflow."""
        model = FpySequencerModel(stack_size=4)
        model.stack = bytearray(4)
        model.stack[3] = 1
        result = model.dispatch(IntegerSignedExtend8To64Directive())
        assert result == DirectiveErrorCode.STACK_OVERFLOW

    def test_signed_extend_16_to_64_stack_underflow(self):
        """Test siext 16->64 with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(1)
        result = model.dispatch(IntegerSignedExtend16To64Directive())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_signed_extend_16_to_64_stack_overflow(self):
        """Test siext 16->64 when would overflow."""
        model = FpySequencerModel(stack_size=4)
        model.stack = bytearray(4)
        result = model.dispatch(IntegerSignedExtend16To64Directive())
        assert result == DirectiveErrorCode.STACK_OVERFLOW

    def test_signed_extend_32_to_64_stack_underflow(self):
        """Test siext 32->64 with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(2)
        result = model.dispatch(IntegerSignedExtend32To64Directive())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_signed_extend_32_to_64_stack_overflow(self):
        """Test siext 32->64 when would overflow."""
        model = FpySequencerModel(stack_size=4)
        model.stack = bytearray(4)
        result = model.dispatch(IntegerSignedExtend32To64Directive())
        assert result == DirectiveErrorCode.STACK_OVERFLOW

    def test_zero_extend_8_to_64_stack_underflow(self):
        """Test ziext 8->64 with empty stack."""
        model = FpySequencerModel()
        model.stack = bytearray(0)
        result = model.dispatch(IntegerZeroExtend8To64Directive())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_zero_extend_8_to_64_stack_overflow(self):
        """Test ziext 8->64 when would overflow."""
        model = FpySequencerModel(stack_size=4)
        model.stack = bytearray(4)
        result = model.dispatch(IntegerZeroExtend8To64Directive())
        assert result == DirectiveErrorCode.STACK_OVERFLOW

    def test_zero_extend_16_to_64_stack_underflow(self):
        """Test ziext 16->64 with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(1)
        result = model.dispatch(IntegerZeroExtend16To64Directive())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_zero_extend_16_to_64_stack_overflow(self):
        """Test ziext 16->64 when would overflow."""
        model = FpySequencerModel(stack_size=4)
        model.stack = bytearray(4)
        result = model.dispatch(IntegerZeroExtend16To64Directive())
        assert result == DirectiveErrorCode.STACK_OVERFLOW

    def test_zero_extend_32_to_64_stack_underflow(self):
        """Test ziext 32->64 with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(2)
        result = model.dispatch(IntegerZeroExtend32To64Directive())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_zero_extend_32_to_64_stack_overflow(self):
        """Test ziext 32->64 when would overflow."""
        model = FpySequencerModel(stack_size=4)
        model.stack = bytearray(4)
        result = model.dispatch(IntegerZeroExtend32To64Directive())
        assert result == DirectiveErrorCode.STACK_OVERFLOW

    def test_float_truncate_stack_underflow(self):
        """Test fptrunc with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(FloatTruncateDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_int_truncate_64_to_8_stack_underflow(self):
        """Test itrunc 64->8 with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(IntegerTruncate64To8Directive())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_int_truncate_64_to_16_stack_underflow(self):
        """Test itrunc 64->16 with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(IntegerTruncate64To16Directive())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_int_truncate_64_to_32_stack_underflow(self):
        """Test itrunc 64->32 with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(IntegerTruncate64To32Directive())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_float_to_signed_int_stack_underflow(self):
        """Test fptosi with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(FloatToSignedIntDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_float_to_unsigned_int_stack_underflow(self):
        """Test fptoui with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(FloatToUnsignedIntDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_signed_int_to_float_stack_underflow(self):
        """Test sitofp with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(SignedIntToFloatDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_unsigned_int_to_float_stack_underflow(self):
        """Test uitofp with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(UnsignedIntToFloatDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS


class TestArithmeticDirectives:
    """Tests for arithmetic directive error conditions."""

    def test_int_add_stack_underflow(self):
        """Test iadd with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(IntAddDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_int_sub_stack_underflow(self):
        """Test isub with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(IntSubtractDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_int_mul_stack_underflow(self):
        """Test imul with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(IntMultiplyDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_unsigned_div_stack_underflow(self):
        """Test udiv with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(UnsignedIntDivideDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_unsigned_div_by_zero(self):
        """Test udiv with zero divisor."""
        model = FpySequencerModel()
        model.push(10, signed=False)
        model.push(0, signed=False)
        result = model.dispatch(UnsignedIntDivideDirective())
        assert result == DirectiveErrorCode.DOMAIN_ERROR

    def test_signed_div_stack_underflow(self):
        """Test sdiv with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(SignedIntDivideDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_signed_div_by_zero(self):
        """Test sdiv with zero divisor."""
        model = FpySequencerModel()
        model.push(10)
        model.push(0)
        result = model.dispatch(SignedIntDivideDirective())
        assert result == DirectiveErrorCode.DOMAIN_ERROR

    def test_signed_div_min_by_neg_one(self):
        """Test sdiv MIN_INT64 / -1 special case (overflow to MIN_INT64)."""
        model = FpySequencerModel()
        model.push(MIN_INT64)
        model.push(-1)
        result = model.dispatch(SignedIntDivideDirective())
        assert result == DirectiveErrorCode.NO_ERROR or result is None
        ret = model.pop() 
        assert ret == MIN_INT64

    def test_float_add_stack_underflow(self):
        """Test fadd with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(FloatAddDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_float_sub_stack_underflow(self):
        """Test fsub with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(FloatSubtractDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_float_mul_stack_underflow(self):
        """Test fmul with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(FloatMultiplyDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_float_div_stack_underflow(self):
        """Test fdiv with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(FloatDivideDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_signed_mod_stack_underflow(self):
        """Test smod with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(SignedModuloDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_unsigned_mod_stack_underflow(self):
        """Test umod with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(UnsignedModuloDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_float_mod_stack_underflow(self):
        """Test fmod with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(FloatModuloDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_float_pow_stack_underflow(self):
        """Test fpow with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(FloatExponentDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_float_log_stack_underflow(self):
        """Test log with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(FloatLogDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS


class TestMemoryDirectives:
    """Tests for memory operation directive error conditions."""

    def test_memcmp_stack_underflow(self):
        """Test memcmp with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(MemCompareDirective(16))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_get_field_stack_underflow(self):
        """Test get_field with insufficient stack."""
        model = FpySequencerModel()
        model.stack = bytearray(8)
        result = model.dispatch(GetFieldDirective(16, 4))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_get_field_member_larger_than_parent(self):
        """Test get_field when member size > parent size."""
        model = FpySequencerModel()
        model.push(0, size=4)
        model.push(b"\x00" * 8)
        result = model.dispatch(GetFieldDirective(8, 16))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_get_field_offset_out_of_bounds(self):
        """Test get_field when offset + member_size > parent_size."""
        model = FpySequencerModel()
        model.push(b"\x00" * 8)
        model.push(6, size=4)
        result = model.dispatch(GetFieldDirective(8, 4))
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_peek_stack_underflow_offset(self):
        """Test peek with insufficient stack for offset."""
        model = FpySequencerModel()
        model.stack = bytearray(4)
        result = model.dispatch(PeekDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_peek_offset_beyond_stack(self):
        """Test peek when offset is beyond stack."""
        model = FpySequencerModel()
        model.push(4, size=4)
        model.push(1000, size=4)
        result = model.dispatch(PeekDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS

    def test_peek_stack_overflow(self):
        """Test peek when result would overflow stack."""
        model = FpySequencerModel(stack_size=20)
        model.stack = bytearray(16)
        model.push(100, size=4)
        model.push(0, size=4)
        result = model.dispatch(PeekDirective())
        assert result == DirectiveErrorCode.STACK_OVERFLOW

    def test_peek_byte_count_plus_offset_beyond_stack(self):
        """Test peek when byte_count + offset exceeds available stack."""
        model = FpySequencerModel()
        model.push(b"\x00" * 8)
        model.push(8, size=4)
        model.push(4, size=4)
        result = model.dispatch(PeekDirective())
        assert result == DirectiveErrorCode.STACK_ACCESS_OUT_OF_BOUNDS
