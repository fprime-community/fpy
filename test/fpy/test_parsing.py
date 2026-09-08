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

    def test_side_effecting_call_in_bare_expr_runs(self, fprime_test_api):
        """A bare expression statement is not a no-op just because its top-level
        node carries no side effects.

        Here the statement is an ``AstBinaryOp`` (``==``), which isn't a
        side-effecting node type -- but it embeds a call that *is*. The codegen
        once skipped bare statements by node type alone, dropping the call (and
        its side effect) entirely. ``bump`` sets the global ``hit``; if the call
        runs, the following assert holds.
        """
        seq = """
hit: U32 = 0
def bump() -> U32:
    hit = 1
    return 0
bump() == 0
assert hit == 1
"""
        assert_run_success(fprime_test_api, seq)

    def test_constant_bare_expr_is_noop(self, fprime_test_api):
        """The flip side: a *constant* bare expression has no side effects (const
        folding only folds pure expressions), so it is correctly skipped and the
        sequence runs cleanly to the end."""
        seq = """
1 + 2 * 3
hit: U32 = 5
assert hit == 5
"""
        assert_run_success(fprime_test_api, seq)


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
x: Ref.DpDemo.U32Array = [
    10,
    20,
    30
]
assert x[1] == 20
"""
        assert_run_success(fprime_test_api, seq)

    def test_multiline_anon_array_trailing_comma(self, fprime_test_api):
        """Anon array split over multiple lines with trailing comma."""
        seq = """
x: Ref.DpDemo.U32Array = [
    10,
    20,
    30,
]
assert x[1] == 20
"""
        assert_run_success(fprime_test_api, seq)

    def test_trailing_comma_anon_array_single_line(self, fprime_test_api):
        """Trailing comma on a single-line anon array."""
        seq = """
x: Ref.DpDemo.U32Array = [10, 20, 30,]
assert x[1] == 20
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
check True timeout time_interval_add(
    {
        seconds: 1,
        useconds: 0,
    },
    {seconds: 0, useconds: 0},
) persist {seconds: 0, useconds: 0} period {seconds: 0, useconds: 100000}:
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
check True timeout {
    seconds: 1,
    useconds: 0,
} persist {
    seconds: 0,
    useconds: 0,
} period {
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


class TestTrailingWhitespaceAtEndOfFile:
    """A file that ends *without* a final newline must still compile, even when
    the last line is a lone tab, trailing spaces, or an (over/under-indented)
    comment.

    Regression tests for https://github.com/fprime-community/fpy/issues/61
    ("Tabbed New Line does not compile at the end of a if/else statement"). The
    indenter derives INDENT/DEDENT from the whitespace each _NEWLINE token
    carries; with no trailing newline the last line's indentation used to be
    misread as a fresh indentation level at end-of-file, emitting a spurious
    INDENT (trailing tab / over-indented comment), a bogus dedent
    (under-indented comment), or a stray blank logical line inside a block.
    Python ignores such trailing blank/comment lines, and now so does fpy.

    The sequences are built with explicit ``\\n``/``\\t`` escapes (not
    triple-quoted literals) so the significant trailing whitespace survives
    editors and formatters.
    """

    def test_trailing_tab_after_statement(self, fprime_test_api):
        # A lone tab on the final line, no trailing newline.
        seq = "x: U32 = 1\nassert x == 1\n\t"
        assert_run_success(fprime_test_api, seq)

    def test_trailing_spaces_after_statement(self, fprime_test_api):
        seq = "x: U32 = 1\nassert x == 1\n    "
        assert_run_success(fprime_test_api, seq)

    def test_trailing_over_indented_comment(self, fprime_test_api):
        seq = "x: U32 = 1\nassert x == 1\n        # trailing comment"
        assert_run_success(fprime_test_api, seq)

    def test_trailing_tab_then_comment(self, fprime_test_api):
        # The exact shape called out in the issue: a tab, then a comment, at EOF.
        seq = "x: U32 = 1\nassert x == 1\n\t# there is a tab here"
        assert_run_success(fprime_test_api, seq)

    def test_trailing_comment_at_body_indent(self, fprime_test_api):
        # Comment aligned with the block body (used to leave a stray blank
        # logical line inside the block, breaking it).
        seq = "if True:\n    x: U32 = 1\n    # aligned comment"
        assert_run_success(fprime_test_api, seq)

    def test_trailing_under_indented_comment(self, fprime_test_api):
        # Comment indented between column 0 and the body (used to raise a
        # DedentError for dedenting to an unknown column).
        seq = "if True:\n    x: U32 = 1\n  # under-indented comment"
        assert_run_success(fprime_test_api, seq)

    def test_trailing_tab_after_if_elif(self, fprime_test_api):
        seq = "x: U32 = 0\nif True:\n    x = 1\nelif False:\n    x = 2\n\t"
        assert_run_success(fprime_test_api, seq)

    def test_trailing_tab_after_if_else(self, fprime_test_api):
        seq = "x: U32 = 0\nif True:\n    x = 1\nelse:\n    x = 2\n\t"
        assert_run_success(fprime_test_api, seq)

    def test_trailing_tab_after_for_body(self, fprime_test_api):
        seq = "s: I64 = 0\nfor i in 0 .. 3:\n    s = s + i\n\t"
        assert_run_success(fprime_test_api, seq)

    def test_trailing_tab_after_while_body(self, fprime_test_api):
        seq = "i: U64 = 0\nwhile i < 3:\n    i = i + 1\n\t"
        assert_run_success(fprime_test_api, seq)

    def test_trailing_tab_after_def_body(self, fprime_test_api):
        seq = "def f() -> U32:\n    return 1\n\t"
        assert_run_success(fprime_test_api, seq)

    def test_trailing_tab_after_check_body(self, fprime_test_api):
        seq = "x: bool = False\ncheck True timeout never:\n    x = True\n\t"
        assert_run_success(fprime_test_api, seq)

    def test_trailing_tabs_after_nested_blocks(self, fprime_test_api):
        # Several dedent levels to close at once, with an over-indented last line.
        seq = "x: U32 = 0\nif True:\n    if True:\n        x = 1\n\t\t\t"
        assert_run_success(fprime_test_api, seq)

    def test_trailing_crlf_then_tab(self, fprime_test_api):
        # CRLF line endings, trailing tab, no final newline.
        seq = "x: U32 = 1\r\nassert x == 1\r\n\t"
        assert_run_success(fprime_test_api, seq)

    def test_only_whitespace_file(self, fprime_test_api):
        # A file that is nothing but a tab must compile to an empty sequence.
        seq = "\t"
        assert_run_success(fprime_test_api, seq)

    def test_only_indented_comment_file(self, fprime_test_api):
        seq = "    # just a comment, indented, no newline"
        assert_run_success(fprime_test_api, seq)

    def test_issue_61_reported_example(self, fprime_test_api):
        # Faithful adaptation of the snippet in the issue: an if/elif whose final
        # branch is followed by a tab + comment at end of file, with a multiline
        # log() call (no backslash) inside a branch body.
        seq = (
            "temp: I64 = -1\n"
            "if temp < 0:\n"
            '    log("temperature sensor invalid reading",\n'
            "            Fw.LogSeverity.WARNING_HI)\n"
            "elif temp > 100:\n"
            '    log("temp high", Fw.LogSeverity.WARNING_HI)\n'
            "\t# there is a tab here"
        )
        assert_run_success(fprime_test_api, seq)


class TestMultilineConstructorsWithoutBackslash:
    """Multiline function calls / type constructors must not require a backslash
    at the end of each line: inside ``()``, ``[]`` or ``{}`` the expression
    continues implicitly, exactly like Python.

    Regression tests for https://github.com/fprime-community/fpy/issues/68
    ("Require fewer backslashes in multiline exprs"). Existing backslash
    continuation must keep working too.
    """

    def test_constructor_args_on_separate_lines(self, fprime_test_api):
        # The issue's shape: constructor arguments on separate lines, aligned
        # under the open paren, with no backslashes.
        seq = (
            "v: Fw.TimeIntervalValue = Fw.TimeIntervalValue(10,\n"
            "                                               500)\n"
            "assert v.seconds == 10\n"
            "assert v.useconds == 500\n"
        )
        assert_run_success(fprime_test_api, seq)

    def test_constructor_multiline_is_last_statement_no_newline(self, fprime_test_api):
        # Multiline constructor as the final statement, file ends with no
        # trailing newline (exercises issue #61 and #68 together).
        seq = "v: Fw.TimeIntervalValue = Fw.TimeIntervalValue(\n    10,\n    500\n)"
        assert_run_success(fprime_test_api, seq)

    def test_constructor_multiline_then_trailing_tab(self, fprime_test_api):
        seq = "v: Fw.TimeIntervalValue = Fw.TimeIntervalValue(\n    10,\n    500\n)\n\t"
        assert_run_success(fprime_test_api, seq)

    def test_constructor_multiline_with_backslash_still_works(self, fprime_test_api):
        # Redundant backslashes inside the parens must not break.
        seq = (
            "v: Fw.TimeIntervalValue = Fw.TimeIntervalValue(10, \\\n"
            "    500)\n"
            "assert v.useconds == 500\n"
        )
        assert_run_success(fprime_test_api, seq)

    def test_constructor_multiline_backslash_on_every_line(self, fprime_test_api):
        # The full "backslash at the end of each line" style users reached for;
        # it must remain valid alongside the now-unnecessary implicit form.
        seq = (
            "v: Fw.TimeIntervalValue = Fw.TimeIntervalValue( \\\n"
            "    10, \\\n"
            "    500 \\\n"
            ")\n"
            "assert v.seconds == 10\n"
        )
        assert_run_success(fprime_test_api, seq)

    def test_nested_multiline_constructor(self, fprime_test_api):
        seq = (
            "v: Fw.TimeIntervalValue = Fw.TimeIntervalValue(\n"
            "    Svc.ComQueueDepth(10, 20)[0],\n"
            "    500,\n"
            ")\n"
            "assert v.seconds == 10\n"
        )
        assert_run_success(fprime_test_api, seq)
