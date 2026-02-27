import pytest

from fpy.test_helpers import (
    assert_run_success,
    assert_compile_failure,
)


@pytest.fixture(name="fprime_test_api", scope="module")
def fprime_test_api_override(request):
    if request.config.getoption("--use-gds"):
        return request.getfixturevalue("fprime_test_api_session")
    return None


# ── Anonymous struct tests ──────────────────────────────────────────────

class TestAnonStructBasic:
    def test_anon_struct_all_members(self, fprime_test_api):
        """Anonymous struct with all members specified."""
        seq = """
val: Fw.TimeIntervalValue = {seconds: 10, useconds: 500}
assert val.seconds == 10
assert val.useconds == 500
"""
        assert_run_success(fprime_test_api, seq)

    def test_anon_struct_empty_with_defaults(self, fprime_test_api):
        """Empty anon struct {} should use all defaults."""
        seq = """
val: Ref.SignalPair = {}
assert val.time == 0.0
assert val.value == 0.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_anon_struct_partial_with_defaults(self, fprime_test_api):
        """Anon struct with some members, rest from defaults."""
        seq = """
val: Ref.SignalPair = {time: 42.0}
assert val.time == 42.0
assert val.value == 0.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_anon_struct_member_order_doesnt_matter(self, fprime_test_api):
        """Members can be specified in any order."""
        seq = """
val: Fw.TimeIntervalValue = {useconds: 999, seconds: 1}
assert val.seconds == 1
assert val.useconds == 999
"""
        assert_run_success(fprime_test_api, seq)

    def test_anon_struct_with_variable_values(self, fprime_test_api):
        """Anon struct with runtime variable values."""
        seq = """
s: U32 = 100
u: U32 = 200
val: Fw.TimeIntervalValue = {seconds: s, useconds: u}
assert val.seconds == 100
assert val.useconds == 200
"""
        assert_run_success(fprime_test_api, seq)


class TestAnonStructErrors:
    def test_anon_struct_unknown_member(self, fprime_test_api):
        """Anon struct with a member that doesn't exist in target type."""
        seq = """
val: Fw.TimeIntervalValue = {seconds: 1, nonexistent: 2}
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_anon_struct_duplicate_member(self, fprime_test_api):
        """Duplicate member names should fail."""
        seq = """
val: Fw.TimeIntervalValue = {seconds: 1, seconds: 2}
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_anon_struct_missing_required_member(self, fprime_test_api):
        """Anon struct missing a member with no default available."""
        seq = """
val: Fw.Time = {seconds: 1}
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_anon_struct_wrong_member_type(self, fprime_test_api):
        """Member value type incompatible with target member type."""
        seq = """
val: Fw.TimeIntervalValue = {seconds: True}
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_anon_struct_assigned_to_non_struct(self, fprime_test_api):
        """Anonymous struct cannot be coerced to a non-struct type."""
        seq = """
val: U32 = {seconds: 1}
"""
        assert_compile_failure(fprime_test_api, seq)


class TestAnonStructAdvanced:
    def test_anon_struct_as_func_arg(self, fprime_test_api):
        """Anonymous struct passed as a function argument."""
        seq = """
def check_time(t: Fw.TimeIntervalValue) -> bool:
    return t.seconds == 5

assert check_time({seconds: 5, useconds: 0})
"""
        assert_run_success(fprime_test_api, seq)

    def test_anon_struct_as_return_value(self, fprime_test_api):
        """Anonymous struct returned from a function."""
        seq = """
def make_interval() -> Fw.TimeIntervalValue:
    return {seconds: 42, useconds: 0}

val: Fw.TimeIntervalValue = make_interval()
assert val.seconds == 42
"""
        assert_run_success(fprime_test_api, seq)


# ── Anonymous array tests ──────────────────────────────────────────────

class TestAnonArrayBasic:
    def test_anon_array_simple(self, fprime_test_api):
        """Anonymous array with matching element count."""
        seq = """
val: Svc.ComQueueDepth = [111, 222]
assert val[0] == 111
assert val[1] == 222
"""
        assert_run_success(fprime_test_api, seq)

    def test_anon_array_with_variables(self, fprime_test_api):
        """Anonymous array with runtime variable values."""
        seq = """
a: U32 = 100
b: U32 = 200
val: Svc.ComQueueDepth = [a, b]
assert val[0] == 100
assert val[1] == 200
"""
        assert_run_success(fprime_test_api, seq)

    def test_anon_array_partial_with_defaults(self, fprime_test_api):
        """Anonymous array with fewer elements than target, rest from defaults."""
        seq = """
val: Svc.ComQueueDepth = [42]
assert val[0] == 42
assert val[1] == 0
"""
        assert_run_success(fprime_test_api, seq)

    def test_anon_array_empty_with_defaults(self, fprime_test_api):
        """Empty array [] should use all defaults."""
        seq = """
val: Svc.ComQueueDepth = []
assert val[0] == 0
assert val[1] == 0
"""
        assert_run_success(fprime_test_api, seq)


class TestAnonArrayErrors:
    def test_anon_array_too_many_elements(self, fprime_test_api):
        """Too many elements should fail."""
        seq = """
val: Svc.ComQueueDepth = [1, 2, 3]
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_anon_array_wrong_element_type(self, fprime_test_api):
        """Element type incompatible with target should fail."""
        seq = """
val: Svc.ComQueueDepth = [True, False]
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_anon_array_assigned_to_non_array(self, fprime_test_api):
        """Anonymous array cannot be coerced to a non-array type."""
        seq = """
val: U32 = [1, 2, 3]
"""
        assert_compile_failure(fprime_test_api, seq)


class TestAnonArrayAdvanced:
    def test_anon_array_as_func_arg(self, fprime_test_api):
        """Anonymous array passed as a function argument."""
        seq = """
def sum_arr(arr: Svc.ComQueueDepth) -> U64:
    return arr[0] + arr[1]

result: U64 = sum_arr([10, 20])
assert result == 30
"""
        assert_run_success(fprime_test_api, seq)

    def test_anon_array_as_return_value(self, fprime_test_api):
        """Anonymous array returned from a function."""
        seq = """
def make_arr() -> Svc.ComQueueDepth:
    return [99, 88]

val: Svc.ComQueueDepth = make_arr()
assert val[0] == 99
assert val[1] == 88
"""
        assert_run_success(fprime_test_api, seq)


# ── Nested anonymous expressions ───────────────────────────────────────

class TestAnonNested:
    def test_anon_array_of_anon_structs(self, fprime_test_api):
        """Array of anonymous structs: [{...}, {...}, ...]"""
        seq = """
val: Ref.SignalPairSet = [{time: 1.0, value: 2.0}, {time: 3.0, value: 4.0}, {time: 5.0, value: 6.0}, {time: 7.0, value: 8.0}]
assert val[0].time == 1.0
assert val[0].value == 2.0
assert val[1].time == 3.0
assert val[3].value == 8.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_anon_struct_containing_anon_array(self, fprime_test_api):
        """Struct with an array member set via anon array: {arr: [...]}"""
        seq = """
val: Ref.SignalInfo = {type: Ref.SignalType.TRIANGLE, history: [1.0, 2.0, 3.0, 4.0]}
assert val.history[0] == 1.0
assert val.history[3] == 4.0
"""
        assert_run_success(fprime_test_api, seq)


# ── Direct member/index access on anonymous literals ────────────────────

class TestAnonDirectAccess:
    def test_anon_struct_member_access(self, fprime_test_api):
        """Access a specific member from an anonymous struct literal."""
        seq = """
a: U32 = {x: 10, y: 20, z: 30}.y
assert a == 20
"""
        assert_run_success(fprime_test_api, seq)

    def test_anon_struct_member_access_nonexistent(self, fprime_test_api):
        """Accessing a non-existent member should fail."""
        seq = """
x: U32 = {xyz: 123}.abc
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_anon_array_index_access(self, fprime_test_api):
        """Index into an anonymous array literal with a constant index."""
        seq = """
x: U32 = [1, 2, 3][1]
assert x == 2
"""
        assert_run_success(fprime_test_api, seq)

    def test_anon_array_index_out_of_bounds(self, fprime_test_api):
        """Out-of-bounds index on anonymous array should fail."""
        seq = """
x: U32 = [1, 2, 3][3]
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_anon_struct_member_access_with_variable(self, fprime_test_api):
        """Access member of anon struct where member value is a runtime variable."""
        seq = """
y: U32 = 42
x: U32 = {a: y}.a
assert x == 42
"""
        assert_run_success(fprime_test_api, seq)

    def test_anon_array_dynamic_index_fails(self, fprime_test_api):
        """Dynamic (non-constant) indexing on anonymous array should fail."""
        seq = """
i: I64 = 1
x: U32 = [10, 20, 30][i]
"""
        assert_compile_failure(fprime_test_api, seq)
