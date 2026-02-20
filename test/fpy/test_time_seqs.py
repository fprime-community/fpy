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
    assert_run_success,
    assert_compile_failure,
    assert_run_failure,
)


@pytest.fixture(name="fprime_test_api", scope="module")
def fprime_test_api_override():
    """A file-specific override that simply returns None."""
    return None


# ==================== time_cmp Tests ====================


class TestTimeCmp:
    """Tests for time_cmp function."""

    def test_time_cmp_equal(self, fprime_test_api):
        """Test time_cmp returns EQ for equal times."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 100, 500000)
t2: Fw.Time = Fw.Time(0, 0, 100, 500000)
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.EQ
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_less_than_seconds(self, fprime_test_api):
        """Test time_cmp returns LT when lhs < rhs (different seconds)."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 100, 0)
t2: Fw.Time = Fw.Time(0, 0, 200, 0)
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.LT
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_greater_than_seconds(self, fprime_test_api):
        """Test time_cmp returns GT when lhs > rhs (different seconds)."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 200, 0)
t2: Fw.Time = Fw.Time(0, 0, 100, 0)
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.GT
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_less_than_useconds(self, fprime_test_api):
        """Test time_cmp returns LT when lhs < rhs (same seconds, different useconds)."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 100, 100000)
t2: Fw.Time = Fw.Time(0, 0, 100, 200000)
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.LT
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_greater_than_useconds(self, fprime_test_api):
        """Test time_cmp returns GT when lhs > rhs (same seconds, different useconds)."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 100, 500000)
t2: Fw.Time = Fw.Time(0, 0, 100, 100000)
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.GT
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_incomparable_different_time_base(self, fprime_test_api):
        """Test time_cmp returns INCOMPARABLE for different time bases."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 100, 0)  # time_base = 0
t2: Fw.Time = Fw.Time(1, 0, 100, 0)  # time_base = 1
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
t1: Fw.Time = Fw.Time(0, 0, 0, 0)
t2: Fw.Time = Fw.Time(0, 0, 0, 0)
result: Fw.TimeComparison = time_cmp(t1, t2)
assert result == Fw.TimeComparison.EQ
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_cmp_large_values(self, fprime_test_api):
        """Test time_cmp with large second values."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 4294967295, 999999)  # max U32 seconds
t2: Fw.Time = Fw.Time(0, 0, 4294967294, 999999)
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
t1: Fw.Time = Fw.Time(0, 0, 100, 500000)
t2: Fw.Time = Fw.Time(0, 0, 50, 100000)
result: Fw.TimeIntervalValue = time_sub(t1, t2)
# Expected: 50 seconds and 400000 useconds
assert result.seconds == 50
assert result.useconds == 400000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_sub_equal_times(self, fprime_test_api):
        """Test subtraction of equal times produces zero interval."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 100, 500000)
t2: Fw.Time = Fw.Time(0, 0, 100, 500000)
result: Fw.TimeIntervalValue = time_sub(t1, t2)
assert result.seconds == 0
assert result.useconds == 0
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_sub_useconds_only(self, fprime_test_api):
        """Test subtraction with only microseconds difference."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 100, 500000)
t2: Fw.Time = Fw.Time(0, 0, 100, 100000)
result: Fw.TimeIntervalValue = time_sub(t1, t2)
assert result.seconds == 0
assert result.useconds == 400000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_sub_with_borrow(self, fprime_test_api):
        """Test subtraction that requires borrowing from seconds."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 100, 100000)
t2: Fw.Time = Fw.Time(0, 0, 99, 500000)
result: Fw.TimeIntervalValue = time_sub(t1, t2)
# 100.100000 - 99.500000 = 0.600000
assert result.seconds == 0
assert result.useconds == 600000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_sub_underflow_asserts(self, fprime_test_api):
        """Test that subtracting a larger time from smaller asserts."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 50, 0)
t2: Fw.Time = Fw.Time(0, 0, 100, 0)
result: Fw.TimeIntervalValue = time_sub(t1, t2)  # Should assert
"""
        assert_run_failure(fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR)

    def test_time_sub_different_time_base_asserts(self, fprime_test_api):
        """Test that subtracting times with different time bases asserts."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 100, 0)
t2: Fw.Time = Fw.Time(1, 0, 50, 0)  # Different time_base
result: Fw.TimeIntervalValue = time_sub(t1, t2)  # Should assert
"""
        assert_run_failure(fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR)

    def test_time_sub_large_difference(self, fprime_test_api):
        """Test subtraction with large second values."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 1000000, 0)
t2: Fw.Time = Fw.Time(0, 0, 1, 0)
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
t1: Fw.Time = Fw.Time(0, 0, 10000, 500000)  # 10,000,500,000 microseconds
t2: Fw.Time = Fw.Time(0, 0, 5000, 200000)   # 5,000,200,000 microseconds
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
t1: Fw.Time = Fw.Time(0, 0, 4294967295, 999999)  # max U32 seconds
t2: Fw.Time = Fw.Time(0, 0, 0, 0)
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
t: Fw.Time = Fw.Time(0, 5, 100, 200000)
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(50, 300000)
result: Fw.Time = time_add(t, interval)
# Expected: 150 seconds and 500000 useconds, same time_base and context
assert result.time_base == 0
assert result.time_context == 5
assert result.seconds == 150
assert result.useconds == 500000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_add_zero_interval(self, fprime_test_api):
        """Test adding zero interval produces same time."""
        seq = """
t: Fw.Time = Fw.Time(0, 0, 100, 500000)
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(0, 0)
result: Fw.Time = time_add(t, interval)
assert result.time_base == 0
assert result.time_context == 0
assert result.seconds == 100
assert result.useconds == 500000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_add_useconds_overflow(self, fprime_test_api):
        """Test addition that causes useconds to overflow into seconds."""
        seq = """
t: Fw.Time = Fw.Time(0, 0, 100, 700000)
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
t: Fw.Time = Fw.Time(1, 0, 100, 0)  # time_base = 1
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(10, 0)
result: Fw.Time = time_add(t, interval)
assert result.time_base == 1
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_add_preserves_time_context(self, fprime_test_api):
        """Test that time_add preserves the time context."""
        seq = """
t: Fw.Time = Fw.Time(0, 42, 100, 0)  # time_context = 42
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(10, 0)
result: Fw.Time = time_add(t, interval)
assert result.time_context == 42
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_add_large_interval(self, fprime_test_api):
        """Test adding a large interval."""
        seq = """
t: Fw.Time = Fw.Time(0, 0, 0, 0)
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
t: Fw.Time = Fw.Time(0, 0, 5000, 100000)  # 5,000,100,000 microseconds
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
t: Fw.Time = Fw.Time(0, 0, 4294967295, 0)
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
t: Fw.Time = Fw.Time(0, 0, 100, 0)
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
t1: Fw.Time = Fw.Time(0, 0, 100, 0)
t2: Fw.Time = time_add(t1, Fw.TimeIntervalValue(0, 1))
cmp: Fw.TimeComparison = time_cmp(t1, t2)
assert cmp == Fw.TimeComparison.LT  # t1 < t2
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_interval_cmp_with_sub(self, fprime_test_api):
        """Test comparing intervals from subtraction."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 100, 0)
t2: Fw.Time = Fw.Time(0, 0, 200, 0)
t3: Fw.Time = Fw.Time(0, 0, 250, 0)

interval1: Fw.TimeIntervalValue = time_sub(t2, t1)  # 100 seconds
interval2: Fw.TimeIntervalValue = time_sub(t3, t2)  # 50 seconds

cmp: Fw.TimeComparison = time_interval_cmp(interval1, interval2)
assert cmp == Fw.TimeComparison.GT  # interval1 > interval2
"""
        assert_run_success(fprime_test_api, seq)

    def test_chained_time_adds(self, fprime_test_api):
        """Test multiple chained time additions."""
        seq = """
t: Fw.Time = Fw.Time(0, 0, 0, 0)
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

t: Fw.Time = Fw.Time(0, 0, 100, 0)
result: Fw.Time = add_interval(t, 50)
assert result.seconds == 150
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_funcs_in_loop(self, fprime_test_api):
        """Test time functions work inside loops."""
        seq = """
t: Fw.Time = Fw.Time(0, 0, 0, 0)
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
t: Fw.Time = Fw.Time(0, 0, 100, 0)
result: Fw.TimeComparison = time_cmp(123, t)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_time_cmp_wrong_type_second_arg(self, fprime_test_api):
        """Test time_cmp fails with wrong type for second argument."""
        seq = """
t: Fw.Time = Fw.Time(0, 0, 100, 0)
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
t1: Fw.Time = Fw.Time(0, 0, 100, 0)
t2: Fw.Time = Fw.Time(0, 0, 100, 0)
result: Fw.TimeComparison = time_interval_cmp(t1, t2)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_time_sub_wrong_type_first_arg(self, fprime_test_api):
        """Test time_sub fails with wrong type for first argument."""
        seq = """
t: Fw.Time = Fw.Time(0, 0, 100, 0)
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
t1: Fw.Time = Fw.Time(0, 0, 100, 0)
t2: Fw.Time = Fw.Time(0, 0, 50, 0)
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
t1: Fw.Time = Fw.Time(0, 0, 200, 500000)
t2: Fw.Time = Fw.Time(0, 0, 100, 200000)
result: Fw.TimeIntervalValue = t1 - t2
# Should be 100.3 seconds
assert result.seconds == 100
assert result.useconds == 300000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_addition_operator(self, fprime_test_api):
        """Test Time + TimeInterval using operator syntax."""
        seq = """
t: Fw.Time = Fw.Time(0, 0, 100, 0)
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(50, 500000)
result: Fw.Time = t + interval
assert result.seconds == 150
assert result.useconds == 500000
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_less_than_operator(self, fprime_test_api):
        """Test Time < Time using operator syntax."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 100, 0)
t2: Fw.Time = Fw.Time(0, 0, 200, 0)
assert t1 < t2
assert not (t2 < t1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_greater_than_operator(self, fprime_test_api):
        """Test Time > Time using operator syntax."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 200, 0)
t2: Fw.Time = Fw.Time(0, 0, 100, 0)
assert t1 > t2
assert not (t2 > t1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_less_than_or_equal_operator(self, fprime_test_api):
        """Test Time <= Time using operator syntax."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 100, 0)
t2: Fw.Time = Fw.Time(0, 0, 200, 0)
t3: Fw.Time = Fw.Time(0, 0, 100, 0)
assert t1 <= t2
assert t1 <= t3
assert not (t2 <= t1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_greater_than_or_equal_operator(self, fprime_test_api):
        """Test Time >= Time using operator syntax."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 200, 0)
t2: Fw.Time = Fw.Time(0, 0, 100, 0)
t3: Fw.Time = Fw.Time(0, 0, 200, 0)
assert t1 >= t2
assert t1 >= t3
assert not (t2 >= t1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_equal_operator(self, fprime_test_api):
        """Test Time == Time using operator syntax."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 100, 500000)
t2: Fw.Time = Fw.Time(0, 0, 100, 500000)
t3: Fw.Time = Fw.Time(0, 0, 100, 0)
assert t1 == t2
assert not (t1 == t3)
"""
        assert_run_success(fprime_test_api, seq)

    def test_time_not_equal_operator(self, fprime_test_api):
        """Test Time != Time using operator syntax."""
        seq = """
t1: Fw.Time = Fw.Time(0, 0, 100, 0)
t2: Fw.Time = Fw.Time(0, 0, 200, 0)
t3: Fw.Time = Fw.Time(0, 0, 100, 0)
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
start: Fw.Time = Fw.Time(0, 0, 100, 0)
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
t1: Fw.Time = Fw.Time(0, 0, 100, 0)
t2: Fw.Time = Fw.Time(0, 0, 200, 0)
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
