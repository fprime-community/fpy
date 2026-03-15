"""
Test suite for the builtin time functions in time.fpy

Tests for:
- time_cmp: Compare two Fw.Time values
- time_interval_cmp: Compare two Fw.TimeIntervalValue values
- time_sub: Subtract two Fw.Time values to get a TimeIntervalValue
- time_add: Add a TimeIntervalValue to a Fw.Time
"""

import pytest

from fpy.model import DirectiveErrorCode
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_failure,
    assert_run_success,
)


# ==================== time_cmp Tests ====================


class TestTimeCmp:
    """Tests for time_cmp function."""

    def test_time_cmp_equal(self, fprime_test_api):
        """Test time_cmp returns EQ for equal times."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 500000)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 500000)
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.EQ
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_less_than_seconds(self, fprime_test_api):
        """Test time_cmp returns LT when lhs < rhs (different seconds)."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 200, 0)
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.LT
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_greater_than_seconds(self, fprime_test_api):
        """Test time_cmp returns GT when lhs > rhs (different seconds)."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 200, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.GT
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_less_than_useconds(self, fprime_test_api):
        """Test time_cmp returns LT when lhs < rhs (same seconds, different useconds)."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 100000)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 200000)
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.LT
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_greater_than_useconds(self, fprime_test_api):
        """Test time_cmp returns GT when lhs > rhs (same seconds, different useconds)."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 500000)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 100000)
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.GT
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_incomparable_different_time_base(self, fprime_test_api):
        """Test time_cmp returns INCOMPARABLE for different time bases."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)  # timeBase = 0
t2: Fw.Time = Fw.Time(TimeBase.TB_PROC_TIME, 0, 100, 0)  # timeBase = 1
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.INCOMPARABLE
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_with_now(self, fprime_test_api):
        """Test time_cmp works with now() values."""
        seq = """
t1: Fw.Time = now()
t2: Fw.Time = now()
result: Fw.TimeComparison = time_cmp(t1, t2)
# Should be comparable (not INCOMPARABLE)
assert result != Fw.TimeComparison.INCOMPARABLE
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_zero_times(self, fprime_test_api):
        """Test time_cmp with zero times."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 0, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 0, 0)
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.EQ
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_large_values(self, fprime_test_api):
        """Test time_cmp with large second values."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 4294967295, 999999)  # max U32 seconds
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 4294967294, 999999)
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.GT
"""
        assert_run_success(fprime_test_api, seq)


# ==================== time_interval_cmp Tests ====================


class TestTimeIntervalCmp:
    """Tests for time_interval_cmp function."""

    def test_time_interval_cmp_equal(self, fprime_test_api):
        """Test time_interval_cmp returns EQ for equal intervals."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 500000)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 500000)
result: Fw.TimeComparison = time_interval_cmp(i1, i2)
assert result == Fw.TimeComparison.EQ
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_interval_cmp_less_than_seconds(self, fprime_test_api):
        """Test time_interval_cmp returns LT when lhs < rhs (different seconds)."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 0)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(200, 0)
result: Fw.TimeComparison = time_interval_cmp(i1, i2)
assert result == Fw.TimeComparison.LT
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_interval_cmp_greater_than_seconds(self, fprime_test_api):
        """Test time_interval_cmp returns GT when lhs > rhs (different seconds)."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(200, 0)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 0)
result: Fw.TimeComparison = time_interval_cmp(i1, i2)
assert result == Fw.TimeComparison.GT
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_interval_cmp_less_than_useconds(self, fprime_test_api):
        """Test time_interval_cmp returns LT when lhs < rhs (same seconds, different useconds)."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 100000)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 200000)
result: Fw.TimeComparison = time_interval_cmp(i1, i2)
assert result == Fw.TimeComparison.LT
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_interval_cmp_greater_than_useconds(self, fprime_test_api):
        """Test time_interval_cmp returns GT when lhs > rhs (same seconds, different useconds)."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 500000)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 100000)
result: Fw.TimeComparison = time_interval_cmp(i1, i2)
assert result == Fw.TimeComparison.GT
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_interval_cmp_zero_intervals(self, fprime_test_api):
        """Test time_interval_cmp with zero intervals."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(0, 0)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(0, 0)
result: Fw.TimeComparison = time_interval_cmp(i1, i2)
assert result == Fw.TimeComparison.EQ
"""
        assert_run_success(fprime_test_api, seq)


# ==================== time_sub Tests ====================


class TestTimeSub:
    """Tests for time_sub function."""

    def test_time_sub_basic(self, fprime_test_api):
        """Test basic time subtraction."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 500000)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 50, 100000)
result: Fw.TimeIntervalValue = time_sub(t1, t2)
# Expected: 50 seconds and 400000 useconds
assert result.seconds == 50
assert result.useconds == 400000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_sub_equal_times(self, fprime_test_api):
        """Test subtraction of equal times produces zero interval."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 500000)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 500000)
result: Fw.TimeIntervalValue = time_sub(t1, t2)
assert result.seconds == 0
assert result.useconds == 0
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_sub_useconds_only(self, fprime_test_api):
        """Test subtraction with only microseconds difference."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 500000)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 100000)
result: Fw.TimeIntervalValue = time_sub(t1, t2)
assert result.seconds == 0
assert result.useconds == 400000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_sub_with_borrow(self, fprime_test_api):
        """Test subtraction that requires borrowing from seconds."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 100000)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 99, 500000)
result: Fw.TimeIntervalValue = time_sub(t1, t2)
# 100.100000 - 99.500000 = 0.600000
assert result.seconds == 0
assert result.useconds == 600000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_sub_underflow_asserts(self, fprime_test_api):
        """Test that subtracting a larger time from smaller asserts."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 50, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
result: Fw.TimeIntervalValue = time_sub(t1, t2)  # Should assert
"""
        assert_run_failure(fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR)

    def test_time_sub_different_time_base_asserts(self, fprime_test_api):
        """Test that subtracting times with different time bases asserts."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_PROC_TIME, 0, 50, 0)  # Different timeBase
result: Fw.TimeIntervalValue = time_sub(t1, t2)  # Should assert
"""
        assert_run_failure(fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR)

    def test_time_sub_large_difference(self, fprime_test_api):
        """Test subtraction with large second values."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 1000000, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 1, 0)
result: Fw.TimeIntervalValue = time_sub(t1, t2)
assert result.seconds == 999999
assert result.useconds == 0
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_sub_u32_overflow_case(self, fprime_test_api):
        """Test time_sub correctly handles values that would overflow U32 when converted to microseconds.
        
        If the seconds * 1_000_000 calculation happened in U32, values above 4294 seconds
        would overflow. This test verifies the calculation happens in U64.
        """
        seq = """
# Both values > 4294 seconds, so microsecond calculation would overflow U32
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 10000, 500000)  # 10,000,500,000 microseconds
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 5000, 200000)   # 5,000,200,000 microseconds
result: Fw.TimeIntervalValue = time_sub(t1, t2)
# Expected: 5000 seconds and 300000 useconds
assert result.seconds == 5000
assert result.useconds == 300000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_sub_max_u32_seconds(self, fprime_test_api):
        """Test time_sub with max U32 seconds.
        
        The result of time_sub is always <= the larger input, so it can't overflow U32.
        This test verifies the max case works correctly.
        """
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 4294967295, 999999)  # max U32 seconds
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 0, 0)
result: Fw.TimeIntervalValue = time_sub(t1, t2)
assert result.seconds == 4294967295
assert result.useconds == 999999
"""
        assert_run_success(fprime_test_api, seq)


# ==================== time_add Tests ====================


class TestTimeAdd:
    """Tests for time_add function."""

    def test_time_add_basic(self, fprime_test_api):
        """Test basic time addition."""
        seq = """
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 5, 100, 200000)
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(50, 300000)
result: Fw.Time = time_add(t, interval)
# Expected: 150 seconds and 500000 useconds, same timeBase and context
assert result.timeBase == TimeBase.TB_NONE
assert result.timeContext == 5
assert result.seconds == 150
assert result.useconds == 500000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_add_zero_interval(self, fprime_test_api):
        """Test adding zero interval produces same time."""
        seq = """
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 500000)
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(0, 0)
result: Fw.Time = time_add(t, interval)
assert result.timeBase == TimeBase.TB_NONE
assert result.timeContext == 0
assert result.seconds == 100
assert result.useconds == 500000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_add_useconds_overflow(self, fprime_test_api):
        """Test addition that causes useconds to overflow into seconds."""
        seq = """
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 700000)
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(0, 500000)
result: Fw.Time = time_add(t, interval)
# 700000 + 500000 = 1200000 useconds = 1 second + 200000 useconds
assert result.seconds == 101
assert result.useconds == 200000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_add_preserves_time_base(self, fprime_test_api):
        """Test that time_add preserves the time base."""
        seq = """
t: Fw.Time = Fw.Time(TimeBase.TB_PROC_TIME, 0, 100, 0)  # timeBase = 1
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(10, 0)
result: Fw.Time = time_add(t, interval)
assert result.timeBase == TimeBase.TB_PROC_TIME
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_add_preserves_time_context(self, fprime_test_api):
        """Test that time_add preserves the time context."""
        seq = """
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 42, 100, 0)  # timeContext = 42
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(10, 0)
result: Fw.Time = time_add(t, interval)
assert result.timeContext == 42
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_add_large_interval(self, fprime_test_api):
        """Test adding a large interval."""
        seq = """
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 0, 0)
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(1000000, 999999)
result: Fw.Time = time_add(t, interval)
assert result.seconds == 1000000
assert result.useconds == 999999
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_add_u32_overflow_case(self, fprime_test_api):
        """Test time_add correctly handles values that would overflow U32 when converted to microseconds.
        
        If the seconds * 1_000_000 calculation happened in U32, values above 4294 seconds
        would overflow. This test verifies the calculation happens in U64.
        """
        seq = """
# Both values > 4294 seconds, so microsecond calculation would overflow U32
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 5000, 100000)  # 5,000,100,000 microseconds
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(6000, 200000)  # 6,000,200,000 microseconds
result: Fw.Time = time_add(t, interval)
# Expected: 11000 seconds and 300000 useconds
assert result.seconds == 11000
assert result.useconds == 300000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_add_result_overflow_u32_asserts(self, fprime_test_api):
        """Test that time_add asserts when the result seconds would overflow U32.
        
        Adding max U32 seconds to a large interval could produce a result
        that doesn't fit in U32 seconds. This should assert.
        """
        seq = """
# Start with max U32 seconds
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 4294967295, 0)
# Add 1 second - result would be 4294967296 which overflows U32
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(1, 0)
result: Fw.Time = time_add(t, interval)  # Should assert
"""
        assert_run_failure(fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR)

    def test_time_add_with_now(self, fprime_test_api):
        """Test time_add works with now()."""
        seq = """
t: Fw.Time = now()
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(1, 0)
result: Fw.Time = time_add(t, interval)
# Result should be greater than original time
cmp: Fw.TimeComparison = time_cmp(result, t)
assert cmp == Fw.TimeComparison.GT
"""
        assert_run_success(fprime_test_api, seq)


# ==================== Integration Tests ====================


class TestTimeIntegration:
    """Integration tests using multiple time functions together."""

    def test_time_add_then_sub(self, fprime_test_api):
        """Test adding then subtracting gives back the original interval."""
        seq = """
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(50, 500000)
t_plus: Fw.Time = time_add(t, interval)
result: Fw.TimeIntervalValue = time_sub(t_plus, t)
assert result.seconds == 50
assert result.useconds == 500000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_with_add(self, fprime_test_api):
        """Test that adding makes time greater."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
t2: Fw.Time = time_add(t1, Fw.TimeIntervalValue(0, 1))
cmp: Fw.TimeComparison = time_cmp(t1, t2)
assert cmp == Fw.TimeComparison.LT  # t1 < t2
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_interval_cmp_with_sub(self, fprime_test_api):
        """Test comparing intervals from subtraction."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 200, 0)
t3: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 250, 0)

interval1: Fw.TimeIntervalValue = time_sub(t2, t1)  # 100 seconds
interval2: Fw.TimeIntervalValue = time_sub(t3, t2)  # 50 seconds

cmp: Fw.TimeComparison = time_interval_cmp(interval1, interval2)
assert cmp == Fw.TimeComparison.GT  # interval1 > interval2
"""
        assert_run_success(fprime_test_api, seq)

    def test_chained_time_adds(self, fprime_test_api):
        """Test multiple chained time additions."""
        seq = """
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 0, 0)
t = time_add(t, Fw.TimeIntervalValue(1, 0))
t = time_add(t, Fw.TimeIntervalValue(2, 0))
t = time_add(t, Fw.TimeIntervalValue(3, 0))
assert t.seconds == 6
assert t.useconds == 0
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_funcs_in_function(self, fprime_test_api):
        """Test time functions work inside user-defined functions."""
        seq = """
def add_interval(t: Fw.Time, secs: U32) -> Fw.Time:
    interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(secs, 0)
    return time_add(t, interval)

t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
result: Fw.Time = add_interval(t, 50)
assert result.seconds == 150
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_funcs_in_loop(self, fprime_test_api):
        """Test time functions work inside loops."""
        seq = """
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 0, 0)
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(1, 0)

for i in 0..10:
    t = time_add(t, interval)

assert t.seconds == 10
"""
        assert_run_success(fprime_test_api, seq)


# ==================== Type Error Tests ====================


class TestTimeTypeErrors:
    """Tests for type errors with time functions."""

    def test_time_cmp_wrong_type_first_arg(self, fprime_test_api):
        """Test time_cmp fails with wrong type for first argument."""
        seq = """
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
result: Fw.TimeComparison = time_cmp(123, t)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_time_cmp_wrong_type_second_arg(self, fprime_test_api):
        """Test time_cmp fails with wrong type for second argument."""
        seq = """
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
result: Fw.TimeComparison = time_cmp(t, 123)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_time_cmp_interval_instead_of_time(self, fprime_test_api):
        """Test time_cmp fails when given TimeIntervalValue instead of Time."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 0)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 0)
result: Fw.TimeComparison = time_cmp(i1, i2)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_time_interval_cmp_wrong_type(self, fprime_test_api):
        """Test time_interval_cmp fails with wrong type."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
result: Fw.TimeComparison = time_interval_cmp(t1, t2)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_time_sub_wrong_type_first_arg(self, fprime_test_api):
        """Test time_sub fails with wrong type for first argument."""
        seq = """
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
result: Fw.TimeIntervalValue = time_sub(123, t)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_time_sub_interval_args(self, fprime_test_api):
        """Test time_sub fails when given intervals instead of times."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 0)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(50, 0)
result: Fw.TimeIntervalValue = time_sub(i1, i2)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_time_add_wrong_second_arg(self, fprime_test_api):
        """Test time_add fails when second arg is Time instead of TimeIntervalValue."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 50, 0)
result: Fw.Time = time_add(t1, t2)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_time_add_wrong_first_arg(self, fprime_test_api):
        """Test time_add fails when first arg is TimeIntervalValue."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 0)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(50, 0)
result: Fw.Time = time_add(i1, i2)
"""
        assert_compile_failure(fprime_test_api, seq)


# ==================== Operator Overloading Tests ====================


class TestTimeOperatorOverloading:
    """Tests for operator overloading on Fw.Time and Fw.TimeIntervalValue types.
    
    These tests verify that binary operators are properly desugared to function calls:
    - Time - Time -> time_sub
    - Time + TimeInterval -> time_add
    - Time comparison operators -> time_cmp
    - TimeInterval arithmetic -> interval_add/interval_sub
    - TimeInterval comparison operators -> time_interval_cmp
    """

    def test_time_subtraction_operator(self, fprime_test_api):
        """Test Time - Time using operator syntax."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 200, 500000)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 200000)
result: Fw.TimeIntervalValue = t1 - t2
# Should be 100.3 seconds
assert result.seconds == 100
assert result.useconds == 300000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_addition_operator(self, fprime_test_api):
        """Test Time + TimeInterval using operator syntax."""
        seq = """
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(50, 500000)
result: Fw.Time = t + interval
assert result.seconds == 150
assert result.useconds == 500000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_less_than_operator(self, fprime_test_api):
        """Test Time < Time using operator syntax."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 200, 0)
assert t1 < t2
assert not (t2 < t1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_greater_than_operator(self, fprime_test_api):
        """Test Time > Time using operator syntax."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 200, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
assert t1 > t2
assert not (t2 > t1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_less_than_or_equal_operator(self, fprime_test_api):
        """Test Time <= Time using operator syntax."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 200, 0)
t3: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
assert t1 <= t2
assert t1 <= t3
assert not (t2 <= t1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_greater_than_or_equal_operator(self, fprime_test_api):
        """Test Time >= Time using operator syntax."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 200, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
t3: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 200, 0)
assert t1 >= t2
assert t1 >= t3
assert not (t2 >= t1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_equal_operator(self, fprime_test_api):
        """Test Time == Time using operator syntax."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 500000)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 500000)
t3: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
assert t1 == t2
assert not (t1 == t3)
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_not_equal_operator(self, fprime_test_api):
        """Test Time != Time using operator syntax."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 200, 0)
t3: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
assert t1 != t2
assert not (t1 != t3)
"""
        assert_run_success(fprime_test_api, seq)

    def test_interval_addition_operator(self, fprime_test_api):
        """Test TimeInterval + TimeInterval using operator syntax."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 600000)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(50, 500000)
result: Fw.TimeIntervalValue = i1 + i2
# 100.6 + 50.5 = 151.1 seconds
assert result.seconds == 151
assert result.useconds == 100000
"""
        assert_run_success(fprime_test_api, seq)

    def test_interval_subtraction_operator(self, fprime_test_api):
        """Test TimeInterval - TimeInterval using operator syntax."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 500000)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(50, 200000)
result: Fw.TimeIntervalValue = i1 - i2
# 100.5 - 50.2 = 50.3 seconds
assert result.seconds == 50
assert result.useconds == 300000
"""
        assert_run_success(fprime_test_api, seq)

    def test_interval_less_than_operator(self, fprime_test_api):
        """Test TimeInterval < TimeInterval using operator syntax."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(50, 0)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 0)
assert i1 < i2
assert not (i2 < i1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_interval_greater_than_operator(self, fprime_test_api):
        """Test TimeInterval > TimeInterval using operator syntax."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 0)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(50, 0)
assert i1 > i2
assert not (i2 > i1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_interval_equal_operator(self, fprime_test_api):
        """Test TimeInterval == TimeInterval using operator syntax."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 500000)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 500000)
i3: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 0)
assert i1 == i2
assert not (i1 == i3)
"""
        assert_run_success(fprime_test_api, seq)

    def test_interval_not_equal_operator(self, fprime_test_api):
        """Test TimeInterval != TimeInterval using operator syntax."""
        seq = """
i1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 0)
i2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(50, 0)
i3: Fw.TimeIntervalValue = Fw.TimeIntervalValue(100, 0)
assert i1 != i2
assert not (i1 != i3)
"""
        assert_run_success(fprime_test_api, seq)

    def test_chained_time_operations(self, fprime_test_api):
        """Test chaining multiple time operations."""
        seq = """
start: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
delta1: Fw.TimeIntervalValue = Fw.TimeIntervalValue(10, 0)
delta2: Fw.TimeIntervalValue = Fw.TimeIntervalValue(20, 0)
# Can't chain + directly, but can do in sequence
t1: Fw.Time = start + delta1
t2: Fw.Time = t1 + delta2
assert t2.seconds == 130
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_operators_with_now(self, fprime_test_api):
        """Test operators work with now()."""
        seq = """
current: Fw.Time = now()
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(60, 0)
future: Fw.Time = current + interval
# The future time should be greater than current
assert future > current
assert current < future
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_comparison_in_if_statement(self, fprime_test_api):
        """Test time comparison operators work in control flow."""
        seq = """
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 100, 0)
t2: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 200, 0)
result: U8 = 0
if t1 < t2:
    result = 1
assert result == 1
"""
        assert_run_success(fprime_test_api, seq)

    def test_interval_comparison_in_while_loop(self, fprime_test_api):
        """Test interval comparison operators work in control flow."""
        seq = """
target: Fw.TimeIntervalValue = Fw.TimeIntervalValue(5, 0)
current: Fw.TimeIntervalValue = Fw.TimeIntervalValue(0, 0)
increment: Fw.TimeIntervalValue = Fw.TimeIntervalValue(1, 0)
count: U64 = 0
while current < target:
    current = current + increment
    count = count + 1
assert count == 5
"""
        assert_run_success(fprime_test_api, seq)

# ==================== Time Builtins Tests ====================
# Tests for sleep, sleep_until, now(), time constructors, and simulated time.
# Migrated from test_seqs.py.


class TestTimeConstruction:

    def test_get_time_member(self, fprime_test_api):
        seq = """
if Fw.Time(TimeBase.TB_NONE, 1, 2, 3).useconds == 3:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_time_type_ctor(self, fprime_test_api):
        seq = """
var: Fw.Time = Fw.Time(TimeBase.TB_NONE, 1, 2, 3)
if var.timeBase == TimeBase.TB_NONE and var.timeContext == 1:# and var.seconds == 2 and var.useconds == 3:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_aliases(self, fprime_test_api):
        """Fw.Time is an alias for Fw.TimeValue, Fw.TimeInterval for Fw.TimeIntervalValue."""
        seq = """
# Fw.Time and Fw.TimeValue are interchangeable
t1: Fw.Time = Fw.TimeValue(TimeBase.TB_NONE, 0, 100, 500000)
t2: Fw.TimeValue = Fw.Time(TimeBase.TB_NONE, 0, 50, 0)
assert t1.seconds == 100
assert t2.seconds == 50

# Fw.TimeInterval and Fw.TimeIntervalValue are interchangeable
i1: Fw.TimeInterval = Fw.TimeIntervalValue(10, 500000)
i2: Fw.TimeIntervalValue = Fw.TimeInterval(5, 0)
assert i1.seconds == 10
assert i2.seconds == 5

# Cross-alias operations work
result: Fw.Time = t2 + i1
assert result.seconds == 60
diff: Fw.TimeInterval = t1 - t2
assert diff.seconds == 50
"""
        assert_run_success(fprime_test_api, seq)

    def test_get_time(self, fprime_test_api):
        seq = """
time: Fw.Time = now()
"""

        assert_run_success(fprime_test_api, seq)

    def test_const_folding_time_eq(self, fprime_test_api):
        seq = """
assert Fw.Time(TimeBase.TB_NONE, 0, 0, 0) == Fw.Time(TimeBase.TB_NONE, 0, 0, 0)
assert Fw.Time(TimeBase.TB_NONE, 0, 1, 0) != Fw.Time(TimeBase.TB_NONE, 0, 0, 0)
"""

        assert_run_success(fprime_test_api, seq)

class TestWait:

    def test_wait_rel(self, fprime_test_api):
        seq = """
sleep(1, 1000)
"""
        assert_run_success(fprime_test_api, seq)

    def test_wait_rel_default_usec(self, fprime_test_api):
        seq = """
sleep(seconds=1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_wait_rel_default_sec(self, fprime_test_api):
        seq = """
sleep(useconds=500)
"""
        assert_run_success(fprime_test_api, seq)

    def test_wait_rel_no_args(self, fprime_test_api):
        seq = """
sleep()
"""
        assert_run_success(fprime_test_api, seq)

    def test_wait_abs(self, fprime_test_api):
        seq = """
sleep_until(Fw.Time(TimeBase.TB_WORKSTATION_TIME, 0, 123, 123))
"""
        assert_run_success(fprime_test_api, seq)

    def test_wait_abs_var_arg(self, fprime_test_api):
        seq = """
x: U32 = 123
sleep_until(Fw.Time(TimeBase.TB_WORKSTATION_TIME, 0, x, 123))
"""
        assert_run_success(fprime_test_api, seq)

    def test_wait_abs_var_arg_2(self, fprime_test_api):
        seq = """
x: Fw.Time = Fw.Time(TimeBase.TB_WORKSTATION_TIME, 1, 2, 3)
sleep_until(x)
"""
        assert_run_success(fprime_test_api, seq)

    def test_wait_abs_bad_arg(self, fprime_test_api):
        seq = """
sleep_until(2, 1, 2, 3)
"""
        assert_compile_failure(fprime_test_api, seq)


@pytest.mark.skipif("config.getoption('--use-gds')", reason="simulated time is only available in the Python model")
class TestSimulatedTime:
    """Tests for simulated time functionality.

    These tests verify that the sequencer model properly:
    - Tracks simulated time
    - Advances time when sleep() is called
    - Returns the configured timeBase from now()
    - Correctly handles timeBase incompatibility

    These tests are skipped when --use-gds is passed because they rely on the
    Python model's simulated time, which cannot be configured on a live GDS.
    """

    def test_now_returns_initial_time(self, fprime_test_api):
        """Test that now() returns the configured initial time."""
        seq = """
t: Fw.Time = now()
# Initial time of 5 seconds = 5,000,000 microseconds
# timeBase=TimeBase.TB_NONE, timeContext=0
assert t.timeBase == TimeBase.TB_NONE
assert t.timeContext == 0
assert t.seconds == 5
assert t.useconds == 0
"""
        assert_run_success(fprime_test_api, seq, initial_time_us=5_000_000)

    def test_now_returns_configured_time_base(self, fprime_test_api):
        """Test that now() returns the configured timeBase."""
        seq = """
t: Fw.Time = now()
# Configured timeBase=TimeBase.TB_WORKSTATION_TIME
assert t.timeBase == TimeBase.TB_WORKSTATION_TIME
assert t.timeContext == 0
"""
        assert_run_success(fprime_test_api, seq, time_base=2)

    def test_now_returns_configured_time_context(self, fprime_test_api):
        """Test that now() returns the configured timeContext."""
        seq = """
t: Fw.Time = now()
# Configured timeContext=4
assert t.timeContext == 42
"""
        assert_run_success(fprime_test_api, seq, time_context=42)

    def test_sleep_advances_simulated_time(self, fprime_test_api):
        """Test that sleep() advances simulated time correctly."""
        seq = """
# Get time before sleep
t_before: Fw.Time = now()

# Sleep for 2 seconds and 500000 microseconds (2.5 seconds total)
sleep(2, 500000)

# Get time after sleep
t_after: Fw.Time = now()

# Calculate the elapsed time
elapsed: Fw.TimeIntervalValue = time_sub(t_after, t_before)

# Should have slept for exactly 2.5 seconds
assert elapsed.seconds == 2
assert elapsed.useconds == 500000
"""
        assert_run_success(fprime_test_api, seq)

    def test_sleep_multiple_times_accumulates(self, fprime_test_api):
        """Test that multiple sleep() calls accumulate time correctly."""
        seq = """
t_start: Fw.Time = now()

# Sleep 1 second
sleep(1, 0)
# Sleep 0.5 seconds
sleep(0, 500000)
# Sleep 0.25 seconds
sleep(0, 250000)

t_end: Fw.Time = now()
elapsed: Fw.TimeIntervalValue = time_sub(t_end, t_start)

# Total: 1.75 seconds
assert elapsed.seconds == 1
assert elapsed.useconds == 750000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_same_time_base_works(self, fprime_test_api):
        """Test that time_cmp works when both times have the same timeBase."""
        seq = """
t1: Fw.Time = now()
sleep(1, 0)
t2: Fw.Time = now()

# t2 should be greater than t1
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.LT  # t1 < t2
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_with_different_time_base_crashes(self, fprime_test_api):
        """Test that check crashes when now() and timeout have different time_bases.

        This tests the full check statement integration: the check desugars to use
        time_cmp(now(), timeout), and if the time_bases differ, the assert should crash.
        """
        seq = """
# Construct a timeout with a different timeBase than what now() returns
# now() returns timeBase=TimeBase.TB_NONE by default
# Set timeout with timeBase=TimeBase.TB_PROC_TIME
bad_timeout: Fw.Time = Fw.Time(TimeBase.TB_PROC_TIME, 0, 100, 0)

check True timeout bad_timeout persist Fw.TimeIntervalValue(0, 0) freq Fw.TimeIntervalValue(0, 100000):
    pass
timeout:
    pass
"""
        # Now run with default timeBase=0, but the timeout uses timeBase=1
        assert_run_failure(fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR)

    def test_check_with_simulated_time_timeout(self, fprime_test_api):
        """Test that check properly times out based on simulated time advancement.

        This test verifies the full check loop:
        1. now() returns simulated time
        2. sleep() advances simulated time
        3. Check properly detects timeout when simulated time exceeds deadline
        """
        seq = """
timed_out: bool = False

# Set timeout to be 100ms from now
# With freq of 10ms, we'll check ~10 times before timeout
check False timeout time_add(now(), Fw.TimeIntervalValue(0, 100000)) persist Fw.TimeIntervalValue(0, 0) freq Fw.TimeIntervalValue(0, 10000):
    # This shouldn't run because condition is always false
    assert False, 1
timeout:
    timed_out = True

assert timed_out
"""
        # Start at time 0, each sleep(0, 10000) advances 10ms
        # After ~10 iterations, we hit 100ms and timeout
        assert_run_success(fprime_test_api, seq)

    def test_check_condition_persists_over_simulated_time(self, fprime_test_api):
        """Test that persist duration is measured using simulated time.

        The check condition must remain true for the full persist duration
        (measured in simulated time).
        """
        seq = """
# Track how many times the condition is checked
check_count: I64 = 0

def condition() -> bool:
    check_count = check_count + 1
    return True  # Always true

# Require condition to persist for 50ms with 10ms frequency
# Should need ~5 checks to persist
check condition() timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) persist Fw.TimeIntervalValue(0, 50000) freq Fw.TimeIntervalValue(0, 10000):
    pass
timeout:
    assert False, 1

# With simulated time, we should have checked at least 5 times
# (initial + enough to accumulate 50ms of persistence)
assert check_count >= 5
"""
        assert_run_success(fprime_test_api, seq)

    def test_sleep_float_advances_time(self, fprime_test_api):
        """Test that sleep with float argument advances simulated time."""
        seq = """
t_before: Fw.Time = now()

# Sleep for 1.5 seconds (1 second + 500000 microseconds)
sleep(1, 500000)

t_after: Fw.Time = now()
elapsed: Fw.TimeIntervalValue = time_sub(t_after, t_before)

# Should have slept for 1.5 seconds
assert elapsed.seconds == 1
assert elapsed.useconds == 500000
"""
        assert_run_success(fprime_test_api, seq)

    def test_now_time_base_preserved_through_check(self, fprime_test_api):
        """Test that now() consistently returns the configured timeBase throughout check."""
        seq = """
# Verify timeBase is consistent inside check
check_count: I64 = 0
time_base_ok: bool = True

def check_time_base() -> bool:
    check_count = check_count + 1
    t: Fw.Time = now()
    # Should always have timeBase=TimeBase.TB_SC_TIME
    if t.timeBase != TimeBase.TB_SC_TIME:
        time_base_ok = False
    return check_count >= 3

check check_time_base() timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) persist Fw.TimeIntervalValue(0, 0) freq Fw.TimeIntervalValue(0, 10000):
    pass
timeout:
    assert False, 1

assert time_base_ok
assert check_count >= 3
"""
        assert_run_success(fprime_test_api, seq, time_base=3)


class TestTimeFunction:

    def test_time_function_basic(self, fprime_test_api):
        """time() parses ISO 8601 strings to Fw.Time."""
        seq = """
t: Fw.Time = time("2000-01-01T00:00:00Z")
# Unix timestamp for 2000-01-01T00:00:00Z is 946684800
assert t.seconds == 946684800
assert t.useconds == 0
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_function_with_microseconds(self, fprime_test_api):
        """time() parses ISO 8601 strings with microseconds."""
        seq = """
t: Fw.Time = time("2000-01-01T00:00:00.123456Z")
assert t.seconds == 946684800
assert t.useconds == 123456
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_function_sleep_until(self, fprime_test_api):
        """time() can be passed directly to sleep_until()."""
        seq = """
sleep_until(time("2000-01-01T00:00:00Z", timeBase=TimeBase.TB_WORKSTATION_TIME))
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_function_invalid_format(self, fprime_test_api):
        """Invalid time string format should fail at compile time."""
        seq = """
t: Fw.Time = time("not a valid time")
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_time_function_invalid_format_2(self, fprime_test_api):
        """Time string without Z suffix should fail."""
        seq = """
t: Fw.Time = time("2000-01-01T00:00:00")
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_time_function_default_time_base(self, fprime_test_api):
        """time() defaults to timeBase=0 and timeContext=0."""
        seq = """
t: Fw.Time = time("2000-01-01T00:00:00Z")
assert t.timeBase == TimeBase.TB_NONE
assert t.timeContext == 0
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_function_custom_time_base(self, fprime_test_api):
        """time() accepts custom timeBase parameter."""
        seq = """
t: Fw.Time = time("2000-01-01T00:00:00Z", timeBase=TimeBase.TB_WORKSTATION_TIME)
assert t.timeBase == TimeBase.TB_WORKSTATION_TIME
assert t.timeContext == 0
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_function_custom_time_context(self, fprime_test_api):
        """time() accepts custom timeContext parameter."""
        seq = """
t: Fw.Time = time("2000-01-01T00:00:00Z", timeContext=5)
assert t.timeBase == TimeBase.TB_NONE
assert t.timeContext == 5
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_function_all_params(self, fprime_test_api):
        """time() accepts all parameters."""
        seq = """
t: Fw.Time = time("2000-01-01T00:00:00Z", timeBase=TimeBase.TB_SC_TIME, timeContext=7)
assert t.timeBase == TimeBase.TB_SC_TIME
assert t.timeContext == 7
assert t.seconds == 946684800
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_function_named_args(self, fprime_test_api):
        """time() works with named arguments."""
        seq = """
t: Fw.Time = time(timestamp="2000-01-01T00:00:00Z", timeBase=TimeBase.TB_PROC_TIME)
assert t.timeBase == TimeBase.TB_PROC_TIME
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_function_negative_seconds(self, fprime_test_api):
        """Time before Unix epoch (1970) should fail with negative seconds error."""
        seq = """
t: Fw.Time = time("1969-01-01T00:00:00Z")
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_time_function_u32_overflow(self, fprime_test_api):
        """Time after year 2106 overflows U32 seconds."""
        # U32 max is 4,294,967,295 seconds after 1970 = year ~2106
        seq = """
t: Fw.Time = time("2200-01-01T00:00:00Z")
"""
        assert_compile_failure(fprime_test_api, seq)
