from fpy.test_helpers import (
    assert_run_success,
    assert_compile_failure,
)

# ── Anonymous struct tests ──────────────────────────────────────────────


class TestAnonStructBasic:
    def test_anon_struct_all_members(self):
        """Anonymous struct with all members specified."""
        seq = """
val: Fw.TimeIntervalValue = {seconds: 10, useconds: 500}
assert val.seconds == 10
assert val.useconds == 500
"""
        assert_run_success(seq)

    def test_anon_struct_empty_with_defaults(self):
        """Empty anon struct {} should use all defaults."""
        seq = """
val: Ref.SignalPair = {}
assert val.time == 0.0
assert val.value == 0.0
"""
        assert_run_success(seq)

    def test_anon_struct_partial_with_defaults(self):
        """Anon struct with some members, rest from defaults."""
        seq = """
val: Ref.SignalPair = {time: 42.0}
assert val.time == 42.0
assert val.value == 0.0
"""
        assert_run_success(seq)

    def test_anon_struct_member_order_doesnt_matter(self):
        """Members can be specified in any order."""
        seq = """
val: Fw.TimeIntervalValue = {useconds: 999, seconds: 1}
assert val.seconds == 1
assert val.useconds == 999
"""
        assert_run_success(seq)

    def test_anon_struct_with_variable_values(self):
        """Anon struct with runtime variable values."""
        seq = """
s: U32 = 100
u: U32 = 200
val: Fw.TimeIntervalValue = {seconds: s, useconds: u}
assert val.seconds == 100
assert val.useconds == 200
"""
        assert_run_success(seq)


class TestAnonStructErrors:
    def test_anon_struct_unknown_member(self):
        """Anon struct with a member that doesn't exist in target type."""
        seq = """
val: Fw.TimeIntervalValue = {seconds: 1, nonexistent: 2}
"""
        assert_compile_failure(seq)

    def test_anon_struct_duplicate_member(self):
        """Duplicate member names should fail."""
        seq = """
val: Fw.TimeIntervalValue = {seconds: 1, seconds: 2}
"""
        assert_compile_failure(seq)

    def test_anon_struct_wrong_member_type(self):
        """Member value type incompatible with target member type."""
        seq = """
val: Fw.TimeIntervalValue = {seconds: True}
"""
        assert_compile_failure(seq)

    def test_anon_struct_assigned_to_non_struct(self):
        """Anonymous struct cannot be coerced to a non-struct type."""
        seq = """
val: U32 = {seconds: 1}
"""
        assert_compile_failure(seq)


class TestAnonStructAdvanced:
    def test_anon_struct_as_func_arg(self):
        """Anonymous struct passed as a function argument."""
        seq = """
def check_time(t: Fw.TimeIntervalValue) -> bool:
    return t.seconds == 5

assert check_time({seconds: 5, useconds: 0})
"""
        assert_run_success(seq)

    def test_anon_struct_as_return_value(self):
        """Anonymous struct returned from a function."""
        seq = """
def make_interval() -> Fw.TimeIntervalValue:
    return {seconds: 42, useconds: 0}

val: Fw.TimeIntervalValue = make_interval()
assert val.seconds == 42
"""
        assert_run_success(seq)


# ── Anonymous array tests ──────────────────────────────────────────────


class TestAnonArrayBasic:
    def test_anon_array_simple(self):
        """Anonymous array with matching element count."""
        seq = """
val: Svc.ComQueueDepth = [111, 222]
assert val[0] == 111
assert val[1] == 222
"""
        assert_run_success(seq)

    def test_anon_array_with_variables(self):
        """Anonymous array with runtime variable values."""
        seq = """
a: U32 = 100
b: U32 = 200
val: Svc.ComQueueDepth = [a, b]
assert val[0] == 100
assert val[1] == 200
"""
        assert_run_success(seq)

    def test_anon_array_partial_with_defaults(self):
        """Anonymous array with fewer elements than target, rest from defaults."""
        seq = """
val: Svc.ComQueueDepth = [42]
assert val[0] == 42
assert val[1] == 0
"""
        assert_run_success(seq)

    def test_anon_array_empty_with_defaults(self):
        """Empty array [] should use all defaults."""
        seq = """
val: Svc.ComQueueDepth = []
assert val[0] == 0
assert val[1] == 0
"""
        assert_run_success(seq)


class TestAnonArrayErrors:
    def test_anon_array_too_many_elements(self):
        """Too many elements should fail."""
        seq = """
val: Svc.ComQueueDepth = [1, 2, 3]
"""
        assert_compile_failure(seq)

    def test_anon_array_wrong_element_type(self):
        """Element type incompatible with target should fail."""
        seq = """
val: Svc.ComQueueDepth = [True, False]
"""
        assert_compile_failure(seq)

    def test_anon_array_assigned_to_non_array(self):
        """Anonymous array cannot be coerced to a non-array type."""
        seq = """
val: U32 = [1, 2, 3]
"""
        assert_compile_failure(seq)

    def test_anon_array_incompatible_element_types(self):
        """Anonymous array with incompatible element types should fail."""
        seq = """
[1, "hello"]
"""
        assert_compile_failure(seq, match="common type")

    def test_index_non_array_reports_not_an_array(self):
        """Indexing a non-array must report 'not an array', not 'contains strings'."""
        seq = """
val: U32 = 5
x: U32 = val[0]
"""
        assert_compile_failure(seq, match="not an array")


class TestAnonArrayAdvanced:
    def test_anon_array_as_func_arg(self):
        """Anonymous array passed as a function argument."""
        seq = """
def sum_arr(arr: Svc.ComQueueDepth) -> U64:
    return arr[0] + arr[1]

result: U64 = sum_arr([10, 20])
assert result == 30
"""
        assert_run_success(seq)

    def test_anon_array_as_return_value(self):
        """Anonymous array returned from a function."""
        seq = """
def make_arr() -> Svc.ComQueueDepth:
    return [99, 88]

val: Svc.ComQueueDepth = make_arr()
assert val[0] == 99
assert val[1] == 88
"""
        assert_run_success(seq)


# ── Nested anonymous expressions ───────────────────────────────────────


class TestAnonNested:
    def test_anon_array_of_anon_structs(self):
        """Array of anonymous structs: [{...}, {...}, ...]"""
        seq = """
val: Ref.SignalPairSet = [{time: 1.0, value: 2.0}, {time: 3.0, value: 4.0}, {time: 5.0, value: 6.0}, {time: 7.0, value: 8.0}]
assert val[0].time == 1.0
assert val[0].value == 2.0
assert val[1].time == 3.0
assert val[3].value == 8.0
"""
        assert_run_success(seq)

    def test_anon_struct_containing_anon_array(self):
        """Struct with an array member set via anon array: {arr: [...]}"""
        seq = """
val: Ref.SignalInfo = {type: Ref.SignalType.TRIANGLE, history: [1.0, 2.0, 3.0, 4.0]}
assert val.history[0] == 1.0
assert val.history[3] == 4.0
"""
        assert_run_success(seq)


# ── Direct member/index access on anonymous literals ────────────────────


class TestAnonDirectAccess:
    def test_anon_struct_member_access(self):
        """Access a specific member from an anonymous struct literal."""
        seq = """
a: U32 = {x: 10, y: 20, z: 30}.y
assert a == 20
"""
        assert_run_success(seq)

    def test_anon_struct_member_access_nonexistent(self):
        """Accessing a non-existent member should fail."""
        seq = """
x: U32 = {xyz: 123}.abc
"""
        assert_compile_failure(seq)

    def test_anon_array_index_access(self):
        """Index into an anonymous array literal with a constant index."""
        seq = """
x: U32 = [1, 2, 3][1]
assert x == 2
"""
        assert_run_success(seq)

    def test_anon_array_index_out_of_bounds(self):
        """Out-of-bounds index on anonymous array should fail."""
        seq = """
x: U32 = [1, 2, 3][3]
"""
        assert_compile_failure(seq)

    def test_anon_struct_member_access_with_variable(self):
        """Access member of anon struct where member value is a runtime variable."""
        seq = """
y: U32 = 42
x: U32 = {a: y}.a
assert x == 42
"""
        assert_run_success(seq)

    def test_anon_array_dynamic_index_fails(self):
        """Dynamic (non-constant) indexing on anonymous array should fail."""
        seq = """
i: I64 = 1
x: U32 = [10, 20, 30][i]
"""
        assert_compile_failure(seq)


# ── Anonymous expressions in check statements ───────────────────────────


class TestAnonExprInCheck:
    """Check statements accept Fw.TimeIntervalValue for persist/period/timeout.
    Verify that anonymous struct syntax works in each position."""

    def test_check_anon_persist(self):
        """Anon struct for the persist clause."""
        seq = """
check_passed: bool = False
check True timeout Fw.TimeIntervalValue(1, 0) persist {seconds: 0, useconds: 0} period Fw.TimeIntervalValue(0, 100000):
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""

        assert_run_success(seq)

    def test_check_anon_freq(self):
        """Anon struct for the period clause."""
        seq = """
check_passed: bool = False
check True timeout Fw.TimeIntervalValue(1, 0) persist Fw.TimeIntervalValue(0, 0) period {seconds: 0, useconds: 100000}:
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(seq)

    def test_check_anon_timeout(self):
        """Anon struct for the timeout clause (as TimeIntervalValue added to now())."""
        seq = """
timed_out: bool = False
check False timeout {seconds: 0, useconds: 100000} persist Fw.TimeIntervalValue(0, 0) period Fw.TimeIntervalValue(0, 10000):
    assert False, 1
timeout:
    timed_out = True
assert timed_out
"""
        assert_run_success(seq)

    def test_check_all_anon(self):
        """All three clauses use anon struct syntax simultaneously."""
        seq = """
check_passed: bool = False
check True timeout {seconds: 1, useconds: 0} persist {seconds: 0, useconds: 0} period {seconds: 0, useconds: 100000}:
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(seq)

    def test_check_anon_persist_with_defaults(self):
        """Anon struct with defaults for persist (empty → {seconds:0, useconds:0})."""
        seq = """
check_passed: bool = False
check True timeout Fw.TimeIntervalValue(1, 0) persist {} period Fw.TimeIntervalValue(0, 100000):
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(seq)

    def test_check_anon_freq_partial(self):
        """Anon struct with partial members for period (useconds only, seconds defaults to 0)."""
        seq = """
check_passed: bool = False
check True timeout Fw.TimeIntervalValue(1, 0) persist Fw.TimeIntervalValue(0, 0) period {useconds: 100000}:
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(seq)


# ── Time arithmetic with anonymous structs ──────────────────────────────


class TestTimeArithmeticWithAnonStruct:
    """The + operator should work between Fw.Time and anonymous structs
    that are coercible to Fw.TimeIntervalValue."""

    def test_time_plus_anon_struct(self):
        """now() + {seconds, useconds} should compile and produce Fw.Time."""
        seq = """
t: Fw.Time = now() + {seconds: 1, useconds: 0}
"""
        assert_run_success(seq)

    def test_time_plus_anon_struct_partial(self):
        """now() + {useconds} (seconds defaults to 0) should compile."""
        seq = """
t: Fw.Time = now() + {useconds: 500000}
"""
        assert_run_success(seq)

    def test_time_plus_anon_struct_empty(self):
        """now() + {} (both default to 0) should compile."""
        seq = """
t: Fw.Time = now() + {}
"""
        assert_run_success(seq)

    def test_interval_plus_anon_struct(self):
        """TimeIntervalValue + anon struct should compile."""
        seq = """
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(1, 0) + {seconds: 2, useconds: 0}
assert interval.seconds == 3
assert interval.useconds == 0
"""
        assert_run_success(seq)

    def test_interval_minus_anon_struct(self):
        """TimeIntervalValue - anon struct should compile."""
        seq = """
interval: Fw.TimeIntervalValue = Fw.TimeIntervalValue(5, 0) - {seconds: 2, useconds: 0}
assert interval.seconds == 3
assert interval.useconds == 0
"""
        assert_run_success(seq)

    def test_anon_struct_plus_anon_struct_interval(self):
        """Two anon structs that are both coercible to TimeIntervalValue should add."""
        seq = """
interval: Fw.TimeIntervalValue = {seconds: 1, useconds: 0} + {seconds: 2, useconds: 0}
assert interval.seconds == 3
assert interval.useconds == 0
"""
        assert_run_success(seq)
