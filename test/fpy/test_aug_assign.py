import pytest

from fpy.test_helpers import assert_compile_failure, assert_run_success


class TestAugmentedAssignment:
    """Augmented assignments (lhs op= rhs) are desugared into lhs = lhs op rhs
    before semantic analysis, so they follow the exact same typing and
    assignment-target rules as the plain form."""

    @pytest.mark.parametrize(
        "type_name, initial, operator, operand, expected",
        [
            ("I64", "7", "+", "3", "10"),
            ("I64", "7", "-", "3", "4"),
            ("I64", "7", "*", "3", "21"),
            ("I64", "17", "%", "5", "2"),
            ("I64", "17", "//", "5", "3"),
            ("I64", "-17", "//", "5", "-4"),
            ("F64", "9.0", "/", "2.0", "4.5"),
            ("F64", "4.5", "**", "2.0", "20.25"),
            ("F64", "7.5", "+", "0.25", "7.75"),
            ("F64", "7.5", "-", "0.25", "7.25"),
            ("F64", "7.5", "*", "2.0", "15.0"),
        ],
        ids=lambda v: str(v),
    )
    def test_operator(
        self, fprime_test_api, type_name, initial, operator, operand, expected
    ):
        seq = f"""
x: {type_name} = {initial}
x {operator}= {operand}
assert x == {expected}
"""
        assert_run_success(fprime_test_api, seq)

    def test_chained_ops(self, fprime_test_api):
        seq = """
x: I64 = 7
x += 3
x -= 2
x *= 5
x //= 4
x %= 7
assert x == 3
"""
        assert_run_success(fprime_test_api, seq)

    def test_rhs_precedence(self, fprime_test_api):
        # x += y * 2 must desugar to x = x + (y * 2), not x = (x + y) * 2
        seq = """
x: I64 = 1
y: I64 = 3
x += y * 2
assert x == 7
"""
        assert_run_success(fprime_test_api, seq)

    def test_member_aug_assign(self, fprime_test_api):
        seq = """
val: Ref.ScalarStruct = {}
val.i64 += 5
val.f64 += 0.5
assert val.i64 == 5
assert val.f64 == 0.5
"""
        assert_run_success(fprime_test_api, seq)

    def test_in_for_loop(self, fprime_test_api):
        seq = """
total: I64 = 0
for i in 0..5:
    total += i
assert total == 10
"""
        assert_run_success(fprime_test_api, seq)

    def test_in_function(self, fprime_test_api):
        seq = """
def add_one(a: I64) -> I64:
    a += 1
    return a

assert add_one(4) == 5
"""
        assert_run_success(fprime_test_api, seq)

    def test_undefined_var(self, fprime_test_api):
        seq = """
x += 1
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_literal_lhs(self, fprime_test_api):
        seq = """
1 += 1
"""
        assert_compile_failure(fprime_test_api, seq, match="Invalid assignment")

    def test_type_annotation_not_allowed(self, fprime_test_api):
        seq = """
x: U32 += 1
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_same_typing_as_plain_assign(self, fprime_test_api):
        # x += 3 on a U32 fails just like x = x + 3 does: the binary op
        # widens to a 64-bit intermediate which cannot narrow back implicitly
        seq = """
x: U32 = 7
x += 3
"""
        assert_compile_failure(fprime_test_api, seq, match="Expected U32, found U64")

    def test_element_aug_assign_same_typing_as_plain_assign(self, fprime_test_api):
        # same widening rule applies to array element targets, exactly as in
        # the plain form val[0] = val[0] + U32(5)
        seq = """
val: Svc.ComQueueDepth = [10, 0]
val[0] += U32(5)
"""
        assert_compile_failure(fprime_test_api, seq, match="Expected U32, found U64")
