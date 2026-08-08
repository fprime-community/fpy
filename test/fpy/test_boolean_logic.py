from fpy.test_helpers import assert_compile_failure, assert_run_success


class TestBooleanOperators:

    def test_or_expr(self):
        seq = """
if True or False:
    exit(0)
exit(1)
"""
        assert_run_success(seq)

    def test_not_expr(self):
        seq = """
if not False:
    exit(0)
exit(1)
"""
        assert_run_success(seq)

    def test_or_expr_with_vars(self):
        seq = """
var1: bool = True
var2: bool = False

if var1 or var2:
    exit(0)
exit(1)
"""
        assert_run_success(seq)

    def test_and_of_ors(self):
        seq = """
if True or False and True or True:
    exit(0)
exit(1)
"""

        assert_run_success(seq)

    def test_and_true_true(self):
        seq = """
if True and True:
    exit(0)
exit(1)
"""
        assert_run_success(seq)

    def test_and_true_false(self):
        seq = """
if True and False:
    exit(1)
exit(0)
"""
        assert_run_success(seq)

    def test_or_false_false(self):
        seq = """
if False or False:
    exit(1)
exit(0)
"""
        assert_run_success(seq)

    def test_not_true(self):
        seq = """
if not True:
    exit(1)
exit(0)
"""
        assert_run_success(seq)


class TestComplexExpressions:

    def test_complex_and_or_not(self):
        seq = """
if not False and (True or False):
    exit(0)
exit(1)
"""
        assert_run_success(seq)

    def test_nested_boolean_expressions(self):
        seq = """
if not (True and False or True and not False) and True:
    exit(1)  # Should not execute
exit(0)
"""
        assert_run_success(seq)

    def test_mixed_boolean_numeric_comparison(self):
        seq = """
val1: U8 = 1
val2: I8 = -1
if (val1 > 0) == True and (val2 < 0) == True:  # Compare boolean results
    if not ((val1 <= 0) == True or (val2 >= 0) == True):
        exit(0)
exit(1)
"""
        assert_run_success(seq)

    def test_complex_boolean_nesting(self):
        seq = """
if not not not not not True:  # Multiple not operators
    exit(1)
elif not (True and not (False or not True)):  # Complex nesting
    exit(1)
else:
    exit(0)
"""
        assert_run_success(seq)

    def test_bool_stack_value(self):
        seq = """
if (1 == 1) == True:
    exit(0)
exit(1)
"""
        assert_run_success(seq)


class TestShortCircuit:

    def test_and_short_circuit_skips_rhs(self):
        seq = """
def boom() -> bool:
    assert False
    return True

if False and boom():
    exit(1)
exit(0)
"""

        assert_run_success(seq)

    def test_or_short_circuit_skips_rhs(self):
        seq = """
def boom() -> bool:
    assert False
    return False

if True or boom():
    exit(0)
exit(1)
"""

        assert_run_success(seq)


class TestNonBoolOperands:

    def test_and_non_bool_operands(self):
        """Boolean 'and' requires bool operands; integers should be rejected."""
        seq = """
val: U32 = 1
if val and True:
    exit(0)
exit(1)
"""
        assert_compile_failure(seq)

    def test_or_non_bool_operands(self):
        """Boolean 'or' requires bool operands; integers should be rejected."""
        seq = """
val: U32 = 0
if val or False:
    exit(0)
exit(1)
"""
        assert_compile_failure(seq)

    def test_not_non_bool_operand(self):
        """Boolean 'not' requires a bool operand; integers should be rejected."""
        seq = """
val: U32 = 0
if not val:
    exit(0)
exit(1)
"""
        assert_compile_failure(seq)
