from fpy.test_helpers import assert_compile_failure, assert_run_success


class TestAugmentedAssignment:
    """Augmented assignments (lhs op= rhs) are desugared into lhs = lhs op rhs
    before semantic analysis, so they follow the exact same typing and
    assignment-target rules as the plain form."""

    def test_add_assign(self):
        seq = """
x: I64 = 7
x += 3
assert x == 10
"""
        assert_run_success(seq)

    def test_sub_assign(self):
        seq = """
x: I64 = 7
x -= 3
assert x == 4
"""
        assert_run_success(seq)

    def test_mul_assign(self):
        seq = """
x: I64 = 7
x *= 3
assert x == 21
"""
        assert_run_success(seq)

    def test_div_assign(self):
        seq = """
f: F64 = 9.0
f /= 2.0
assert f == 4.5
"""
        assert_run_success(seq)

    def test_mod_assign(self):
        seq = """
x: I64 = 17
x %= 5
assert x == 2
"""
        assert_run_success(seq)

    def test_pow_assign(self):
        seq = """
f: F64 = 4.5
f **= 2.0
assert f == 20.25
"""
        assert_run_success(seq)

    def test_floor_div_assign(self):
        seq = """
x: I64 = 17
x //= 5
assert x == 3
"""
        assert_run_success(seq)

    def test_chained_ops(self):
        seq = """
x: I64 = 7
x += 3
x -= 2
x *= 5
x //= 4
x %= 7
assert x == 3
"""
        assert_run_success(seq)

    def test_rhs_precedence(self):
        # x += y * 2 must desugar to x = x + (y * 2), not x = (x + y) * 2
        seq = """
x: I64 = 1
y: I64 = 3
x += y * 2
assert x == 7
"""
        assert_run_success(seq)

    def test_member_aug_assign(self):
        seq = """
val: Ref.ScalarStruct = {}
val.i64 += 5
val.f64 += 0.5
assert val.i64 == 5
assert val.f64 == 0.5
"""
        assert_run_success(seq)

    def test_in_for_loop(self):
        seq = """
total: I64 = 0
for i in 0..5:
    total += i
assert total == 10
"""
        assert_run_success(seq)

    def test_in_function(self):
        seq = """
def add_one(a: I64) -> I64:
    a += 1
    return a

assert add_one(4) == 5
"""
        assert_run_success(seq)

    def test_undefined_var(self):
        seq = """
x += 1
"""
        assert_compile_failure(seq)

    def test_literal_lhs(self):
        seq = """
1 += 1
"""
        assert_compile_failure(seq, match="Invalid assignment")

    def test_type_annotation_not_allowed(self):
        seq = """
x: U32 += 1
"""
        assert_compile_failure(seq)

    def test_same_typing_as_plain_assign(self):
        # x += 3 on a U32 fails just like x = x + 3 does: the binary op
        # widens to a 64-bit intermediate which cannot narrow back implicitly
        seq = """
x: U32 = 7
x += 3
"""
        assert_compile_failure(seq, match="Expected U32, found U64")

    def test_element_aug_assign_same_typing_as_plain_assign(self):
        # same widening rule applies to array element targets, exactly as in
        # the plain form val[0] = val[0] + U32(5)
        seq = """
val: Svc.ComQueueDepth = [10, 0]
val[0] += U32(5)
"""
        assert_compile_failure(seq, match="Expected U32, found U64")
