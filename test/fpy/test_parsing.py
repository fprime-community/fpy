import pytest

from fpy.types import U32

from fpy.test_helpers import assert_compile_failure, assert_run_success


class TestSourceStructure:

    def test_comment(self, fprime_test_api):
        seq = """
# test
"""

        assert_run_success(fprime_test_api, seq)

    def test_empty(self, fprime_test_api):
        seq = """"""

        assert_run_success(fprime_test_api, seq)

    def test_no_newline(self, fprime_test_api):
        seq = """# test"""

        assert_run_success(fprime_test_api, seq)

    def test_last_line_comment(self, fprime_test_api):
        seq = """
# test"""
        assert_run_success(fprime_test_api, seq)

    def test_two_stmts_on_same_line(self, fprime_test_api):
        # Two statements on the same line should fail to compile
        seq = """
0value: U8 = 0
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_no_trailing_newline(self, fprime_test_api):
        # Code without a trailing newline should still compile
        seq = "x: U32 = 1"  # No trailing newline
        assert_run_success(fprime_test_api, seq)

    @pytest.mark.xfail(reason="Support for non utf-8 characters should be added later")
    def test_non_utf_8(self, fprime_test_api):
        seq = """
val: F64 = 0.0 

CdhCore.cmdDisp.CMD_NO_OP_STRING("в")
"""
        assert_run_success(fprime_test_api, seq)

    def test_var_name_special_chars(self, fprime_test_api):
        # Variable names with invalid special characters should fail
        seq = """
@invalid: U8 = 0
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_newline_in_body(self, fprime_test_api):
        seq = """
if True:
    val: U8 = 0

    pass
"""

        assert_run_success(fprime_test_api, seq)

class TestLiterals:

    def test_int_literal(self, fprime_test_api):
        seq = """
var: I64 = 123_456
var = -123_456
var = +123_456
var = 000_00000_0
"""

        assert_run_success(fprime_test_api, seq)

    def test_bad_int_literal(self, fprime_test_api):
        seq = """
var: I64 = 0123_456

"""

        assert_compile_failure(fprime_test_api, seq)

    def test_float_literal(self, fprime_test_api):
        seq = """
var: F32 = 1.000e-5
var = .1
var = 2.123
var = 100.5e+10
var = -123.456
"""

        assert_run_success(fprime_test_api, seq)

    def test_bad_float_literal(self, fprime_test_api):
        seq = """
var: F32 = 1.
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_hex_literal(self, fprime_test_api):
        seq = """
var: U32 = 0xFF
assert var == 255
var = 0xDEADBEEF
assert var == 3735928559
var = 0x0
assert var == 0
var = 0X1A2B
assert var == 6699
"""

        assert_run_success(fprime_test_api, seq)

    def test_hex_literal_underscore(self, fprime_test_api):
        seq = """
var: U32 = 0xFF_FF
assert var == 65535
var = 0xDEAD_BEEF
assert var == 3735928559
var = 0x00_11_22_33
assert var == 1122867
"""

        assert_run_success(fprime_test_api, seq)

class TestExpressionStatements:

    def test_int_as_stmt(self, fprime_test_api):
        seq = """
2
"""

        assert_run_success(fprime_test_api, seq)

    def test_expr_as_stmt(self, fprime_test_api):
        seq = """
2 + 2
"""

        assert_run_success(fprime_test_api, seq)

    def test_str_as_stmt(self, fprime_test_api):
        seq = """
"test"
"""
        assert_run_success(fprime_test_api, seq)

    def test_complex_as_stmt(self, fprime_test_api):
        seq = """
CdhCore.cmdDisp.CMD_NO_OP
"""

        assert_compile_failure(fprime_test_api, seq)


class TestMultilineAndTrailingComma:
    """Expressions inside brackets/braces/parens can span multiple lines,
    and trailing commas are allowed in struct/array/function-call/parameter lists."""

    def test_multiline_anon_struct(self, fprime_test_api):
        """Anon struct split over multiple lines."""
        seq = """
val: Fw.TimeIntervalValue = {
    seconds: 10,
    useconds: 500
}
assert val.seconds == 10
assert val.useconds == 500
"""
        assert_run_success(fprime_test_api, seq)

    def test_multiline_anon_struct_trailing_comma(self, fprime_test_api):
        """Anon struct split over multiple lines with trailing comma."""
        seq = """
val: Fw.TimeIntervalValue = {
    seconds: 10,
    useconds: 500,
}
assert val.seconds == 10
assert val.useconds == 500
"""
        assert_run_success(fprime_test_api, seq)

    def test_trailing_comma_anon_struct_single_line(self, fprime_test_api):
        """Trailing comma on a single-line anon struct."""
        seq = """
val: Fw.TimeIntervalValue = {seconds: 10, useconds: 500,}
assert val.seconds == 10
assert val.useconds == 500
"""
        assert_run_success(fprime_test_api, seq)

    def test_multiline_anon_array(self, fprime_test_api):
        """Anon array split over multiple lines."""
        seq = """
x: U32 = [
    10,
    20,
    30
][1]
assert x == 20
"""
        assert_run_success(fprime_test_api, seq)

    def test_multiline_anon_array_trailing_comma(self, fprime_test_api):
        """Anon array split over multiple lines with trailing comma."""
        seq = """
x: U32 = [
    10,
    20,
    30,
][1]
assert x == 20
"""
        assert_run_success(fprime_test_api, seq)

    def test_trailing_comma_anon_array_single_line(self, fprime_test_api):
        """Trailing comma on a single-line anon array."""
        seq = """
x: U32 = [10, 20, 30,][1]
assert x == 20
"""
        assert_run_success(fprime_test_api, seq)

    def test_multiline_func_call(self, fprime_test_api):
        """Function call arguments split over multiple lines."""
        seq = """
val: Fw.TimeIntervalValue = Fw.TimeIntervalValue(
    10,
    500
)
assert val.seconds == 10
assert val.useconds == 500
"""
        assert_run_success(fprime_test_api, seq)

    def test_multiline_func_call_trailing_comma(self, fprime_test_api):
        """Function call arguments split over multiple lines with trailing comma."""
        seq = """
val: Fw.TimeIntervalValue = Fw.TimeIntervalValue(
    10,
    500,
)
assert val.seconds == 10
assert val.useconds == 500
"""
        assert_run_success(fprime_test_api, seq)

    def test_multiline_parenthesized_expr(self, fprime_test_api):
        """Parenthesized expression split over multiple lines."""
        seq = """
x: U32 = (
    10
    + 20
)
assert x == 30
"""
        assert_run_success(fprime_test_api, seq)

    def test_multiline_nested_braces_and_parens(self, fprime_test_api):
        """Nested multi-line expressions with braces inside parens."""
        seq = """
check_passed: bool = False
check True timeout time_add(
    now(),
    {
        seconds: 1,
        useconds: 0,
    },
) persist {seconds: 0, useconds: 0} freq {seconds: 0, useconds: 100000}:
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(fprime_test_api, seq)

    def test_multiline_check_clauses_with_anon_struct(self, fprime_test_api):
        """Check statement with multi-line anon struct in timeout position."""
        seq = """
check_passed: bool = False
check True timeout now() + {
    seconds: 1,
    useconds: 0,
} persist {
    seconds: 0,
    useconds: 0,
} freq {
    seconds: 0,
    useconds: 100000,
}:
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(fprime_test_api, seq)


class TestPythonLikeContinuation:
    """Tests inspired by CPython test_grammar.py – backslash continuation,
    implicit continuation inside brackets/parens/braces, trailing commas,
    comments inside multiline expressions, and edge cases."""

    # ── Backslash continuation (CPython test_backslash) ──────────────

    def test_backslash_continuation(self, fprime_test_api):
        """Backslash at end of line continues to the next (CPython test_backslash)."""
        seq = """\
x: U32 = 1 \\
+ 1
assert x == 2
"""
        assert_run_success(fprime_test_api, seq)

    def test_backslash_continuation_in_assignment(self, fprime_test_api):
        """Backslash continuation across an assignment expression."""
        seq = """\
x: U32 = \\
    42
assert x == 42
"""
        assert_run_success(fprime_test_api, seq)

    def test_backslash_continuation_multiple_lines(self, fprime_test_api):
        """Multiple successive backslash continuations."""
        seq = """\
x: U32 = 1 \\
    + 2 \\
    + 3
assert x == 6
"""
        assert_run_success(fprime_test_api, seq)

    # ── Trailing commas in function definitions

    def test_trailing_comma_one_param(self, fprime_test_api):
        """def f(a,): pass  — Python allows trailing comma in single param."""
        seq = """\
def f(
    a: U32,
) -> U32:
    return a
assert f(1) == 1
"""
        assert_run_success(fprime_test_api, seq)

    def test_trailing_comma_two_params(self, fprime_test_api):
        """def f(a, b,): pass  — trailing comma with two params."""
        seq = """\
def f(
    a: U64,
    b: U64,
) -> U64:
    return a + b
assert f(1, 2) == 3
"""
        assert_run_success(fprime_test_api, seq)

    # ── Trailing commas in function calls

    def test_trailing_comma_call_one_arg(self, fprime_test_api):
        """f(1,) — Python allows trailing comma in single-arg call."""
        seq = """\
def f(a: U32) -> U32:
    return a
assert f(1,) == 1
"""
        assert_run_success(fprime_test_api, seq)

    def test_trailing_comma_call_two_args(self, fprime_test_api):
        """f(1, 2,) — trailing comma with two args."""
        seq = """\
def f(a: U64, b: U64) -> U64:
    return a + b
assert f(1, 2,) == 3
"""
        assert_run_success(fprime_test_api, seq)

    def test_trailing_comma_call_many_args(self, fprime_test_api):
        """f(1, 2, 3,) — trailing comma with three args (cf. CPython v0/v1/v2)."""
        seq = """\
def f(a: U64, b: U64, c: U64) -> U64:
    return a + b + c
assert f(1, 2, 3,) == 6
"""
        assert_run_success(fprime_test_api, seq)

    # ── Multi-term arithmetic inside parens (CPython test_additive_ops) ─

    def test_paren_continuation_complex(self, fprime_test_api):
        """Deeply nested arithmetic inside parens across lines."""
        seq = """\
x: U32 = (
    1
    + 2
    + 3
    + 4
)
assert x == 10
"""
        assert_run_success(fprime_test_api, seq)

    def test_paren_continuation_with_operators(self, fprime_test_api):
        """Mixed operators inside parens across lines."""
        seq = """\
x: U32 = (
    2 * 3
    + 4
)
assert x == 10
"""
        assert_run_success(fprime_test_api, seq)

    # ── Comments inside continued expressions (CPython test_suite) ────

    def test_comment_inside_paren(self, fprime_test_api):
        """Comments inside parenthesized continuation (like Python)."""
        seq = """\
x: U32 = (
    # first term
    1
    # second term
    + 2
)
assert x == 3
"""
        assert_run_success(fprime_test_api, seq)

    def test_comment_inside_braces(self, fprime_test_api):
        """Comments inside struct literal."""
        seq = """\
val: Fw.TimeIntervalValue = {
    # the seconds field
    seconds: 10,
    # the useconds field
    useconds: 500,
}
assert val.seconds == 10
"""
        assert_run_success(fprime_test_api, seq)

    def test_comment_inside_func_call(self, fprime_test_api):
        """Comments inside a multiline function call."""
        seq = """\
val: Fw.TimeIntervalValue = Fw.TimeIntervalValue(
    # seconds
    10,
    # useconds
    500,
)
assert val.seconds == 10
"""
        assert_run_success(fprime_test_api, seq)

    # ── Nested continuation (CPython test_with_statement pattern) ─────

    def test_deeply_nested_continuation(self, fprime_test_api):
        """Three levels of nesting: parens > call > struct."""
        seq = """\
x: U64 = (
    Fw.TimeIntervalValue(
        10,
        500,
    ).seconds
    + 1
)
assert x == 11
"""
        assert_run_success(fprime_test_api, seq)

    # ── Empty lines inside continued expressions ──────────────────────

    def test_empty_line_inside_parens(self, fprime_test_api):
        """Empty lines inside parenthesized expression (Python allows this)."""
        seq = """\
x: U32 = (

    1

    + 2

)
assert x == 3
"""
        assert_run_success(fprime_test_api, seq)

    def test_empty_line_inside_braces(self, fprime_test_api):
        """Empty lines inside struct literal."""
        seq = """\
val: Fw.TimeIntervalValue = {

    seconds: 10,

    useconds: 500,

}
assert val.seconds == 10
"""
        assert_run_success(fprime_test_api, seq)

    # ── Multiline function definition

    def test_multiline_func_def_params(self, fprime_test_api):
        """Parameters each on their own line with trailing comma."""
        seq = """\
def add(
    a: U64,
    b: U64,
    c: U64,
) -> U64:
    return a + b + c
assert add(1, 2, 3) == 6
"""
        assert_run_success(fprime_test_api, seq)

    def test_multiline_func_def_and_call(self, fprime_test_api):
        """Both definition params and call args multiline (CPython common pattern)."""
        seq = """\
def add(
    a: U64,
    b: U64,
) -> U64:
    return a + b

x: U64 = add(
    10,
    20,
)
assert x == 30
"""
        assert_run_success(fprime_test_api, seq)

    # ── Named arguments multiline (CPython keyword arg patterns) ──────

    def test_multiline_named_args(self, fprime_test_api):
        """Named arguments across lines — cf. CPython d11(1, **{'b':2})."""
        seq = """\
val: Fw.TimeIntervalValue = Fw.TimeIntervalValue(
    seconds=10,
    useconds=500,
)
assert val.seconds == 10
assert val.useconds == 500
"""
        assert_run_success(fprime_test_api, seq)

    def test_mixed_positional_and_named_multiline(self, fprime_test_api):
        """Mix of positional and named args across lines."""
        seq = """\
val: Fw.TimeIntervalValue = Fw.TimeIntervalValue(
    10,
    useconds=500,
)
assert val.seconds == 10
assert val.useconds == 500
"""
        assert_run_success(fprime_test_api, seq)

    # ── Continuation with comparison / boolean operators ──────────────

    def test_multiline_comparison_in_parens(self, fprime_test_api):
        """Comparison across lines inside parens (CPython test_comparison)."""
        seq = """\
x: bool = (
    1
    == 1
)
assert x
"""
        assert_run_success(fprime_test_api, seq)

    def test_multiline_boolean_in_parens(self, fprime_test_api):
        """Boolean operators across lines (CPython test_test)."""
        seq = """\
x: bool = (
    True
    and True
    or False
)
assert x
"""
        assert_run_success(fprime_test_api, seq)

    # ── Multiline in control flow expressions ─────────────────────────

    def test_multiline_if_condition(self, fprime_test_api):
        """Parenthesized multiline condition in if (common Python pattern)."""
        seq = """\
x: U32 = 0
if (
    True
    and True
):
    x = 1
assert x == 1
"""
        assert_run_success(fprime_test_api, seq)

    def test_multiline_while_condition(self, fprime_test_api):
        """Parenthesized multiline condition in while."""
        seq = """\
x: U64 = 0
while (
    x
    < 3
):
    x = x + 1
assert x == 3
"""
        assert_run_success(fprime_test_api, seq)

    def test_multiline_assert(self, fprime_test_api):
        """Multiline expression in assert (parenthesized)."""
        seq = """\
assert (
    1
    + 1
    == 2
)
"""
        assert_run_success(fprime_test_api, seq)
