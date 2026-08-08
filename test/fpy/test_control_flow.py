from fpy.types import FpyValue, U32

from fpy.bytecode.errors import DirectiveErrorCode
from fpy.error import WarningType
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_failure,
    assert_run_success,
)


class TestExit:

    def test_exit_success(self):
        seq = """
exit(0)
"""
        assert_run_success(seq)

    def test_exit_failure(self):
        seq = """
exit(123)
"""
        assert_run_failure(seq, 123)


class TestIf:

    def test_simple_if(self):
        seq = """
var: bool = True

# use exit(0) if we want the sequence to succeed
# exit(1) if we want it to fail. helpful for testing.

if var:
    exit(0)
exit(1)
"""
        assert_run_success(seq)

    def test_large_elifs(self):
        seq = """
if CdhCore.cmdDisp.CommandsDispatched == 0:
    CdhCore.cmdDisp.CMD_NO_OP_STRING("0")
elif CdhCore.cmdDisp.CommandsDispatched == 1:
    CdhCore.cmdDisp.CMD_NO_OP_STRING("1")
elif CdhCore.cmdDisp.CommandsDispatched == 2:
    CdhCore.cmdDisp.CMD_NO_OP_STRING("2")
elif CdhCore.cmdDisp.CommandsDispatched == 3:
    CdhCore.cmdDisp.CMD_NO_OP_STRING("3")
elif CdhCore.cmdDisp.CommandsDispatched == 4:
    CdhCore.cmdDisp.CMD_NO_OP_STRING("4")
else:
    CdhCore.cmdDisp.CMD_NO_OP_STRING(">4")
"""

        assert_run_success(
            seq, {"CdhCore.cmdDisp.CommandsDispatched": FpyValue(U32, 4).serialize()}
        )

    def test_if_true(self):
        seq = """
if True:
    exit(0)
exit(1)
"""
        assert_run_success(seq)

    def test_if_false(self):
        seq = """
if False:
    exit(1)
exit(0)
"""
        assert_run_success(seq)

    def test_if_else_true(self):
        seq = """
if True:
    exit(0)
else:
    exit(1)
"""
        assert_run_success(seq)

    def test_if_else_false(self):
        seq = """
if False:
    exit(1)
else:
    exit(0)
"""
        assert_run_success(seq)

    def test_if_elif_else(self):
        seq = """
if False:
    exit(1)
elif True:
    exit(0)
else:
    exit(1)
"""
        assert_run_success(seq)


class TestBreakContinueErrors:

    def test_break_outside_loop(self):
        seq = """
break
"""

        assert_compile_failure(seq)

    def test_continue_outside_loop(self):
        seq = """
continue
"""

        assert_compile_failure(seq)

    def test_deeply_nested_loops_exhaust_recursion_depth(self):
        depth = 500
        loop_header_lines = [
            ("    " * level) + f"for i{level} in 0 .. 1:" for level in range(depth)
        ]
        seq = "\n" + "\n".join(loop_header_lines) + "\n" + ("    " * depth) + "pass\n"

        # Purposefully triggers RecursionError inside the compiler's parse transform.

        assert_compile_failure(seq)


class TestForLoops:

    def test_simple_for(self):
        seq = """
for i in 0..2:
    pass
"""

        assert_run_success(seq)

    def test_for_loop_break(self):
        seq = """
counter: I64 = 0
for i in 0 .. 10:
    counter = counter + 1
    if counter == 5:
        break
    counter = counter + 1
assert counter == 5
"""
        assert_run_success(seq)

    def test_for_loop_continue(self):
        seq = """
counter: I64 = 0
for i in 0 .. 10:
    counter = counter + 1
    continue
    counter = counter + 1
assert counter == 10
"""
        assert_run_success(seq)

    def test_slightly_more_complex_for(self):
        seq = """
counter: U8 = 0
for i in 0 .. 2:
    if i > 2:
        exit(1)
    counter = U8(counter + 1)


assert counter == 2
"""

        assert_run_success(seq)

    def test_nested_for_loops(self):
        seq = """
counter: U64 = 0
z: U8 = 123
for i in 0 .. 7:
    for y in 20 .. 30:
        assert i < 8
        assert y >= 20 and y < 30
        assert z == 123
        counter = counter + 1
assert counter == 70
"""

        assert_run_success(seq)

    def test_nested_for_loops_break_inner(self):
        seq = """
outer_count: I64 = 0
inner_count: I64 = 0
for i in 0 .. 10:
    for j in 0 .. 5:
        inner_count = inner_count + 1
        break
    outer_count = outer_count + 1
assert outer_count == 10
assert inner_count == 10
"""
        assert_run_success(seq)

    def test_nested_for_loops_break_outer(self):
        seq = """
for i in 0 .. 10:
    for j in 0 .. 5:
        break
    break
"""
        assert_run_success(seq)

    def test_nested_for_while_break(self):
        seq = """
counter: I64 = 0
for i in 0 .. 10:
    while True:
        break
    counter = counter + 1
assert counter == 10
"""
        assert_run_success(seq)

    def test_for_break_in_if(self):
        seq = """
for i in 0 .. 100:
    if True:
        break
    exit(1)
"""

        assert_run_success(seq)

    def test_for_continue_in_if(self):
        """Test that continue both skips rest of body AND continues to next iteration."""
        seq = """
sum: U64 = 0
for i in 0 .. 100:
    sum = sum + 1
    if True:
        continue
    exit(1)  # should be skipped by continue

# Verify loop ran all 100 iterations
assert sum == 100
"""

        assert_run_success(seq)

    def test_two_fors_same_loop_var(self):
        seq = """
for i in 0 .. 7:
    assert i >= 0 and i < 7
for i in 0 .. 7:
    assert i >= 0 and i < 7
"""
        assert_run_success(seq)

    def test_empty_range(self):
        seq = """
for i in 7..0:
    exit(1)
"""
        assert_run_success(seq, expected_warnings={WarningType.EMPTY_RANGE})


class TestLoopVariableScoping:

    def test_loop_var_outside_loop_after(self):
        seq = """
for i in 0 .. 7:
    pass
assert i == 7
"""
        # i is scoped to the for loop body; not visible after
        assert_compile_failure(seq)

    def test_loop_var_outside_loop_before(self):
        seq = """
i = 123
for i in 0 .. 7:
    pass
"""

        assert_compile_failure(seq)

    def test_loop_var_redeclare_right_type(self):
        # The loop var `i` shadows the outer `i` (warns).
        seq = """
i: I64 = 123
for i in 0 .. 7:
    assert i >= 0 and i < 7
assert i == 123
"""
        assert_run_success(seq, expected_warnings={WarningType.SHADOW_VALUE})

    def test_loop_var_redeclare_right_type_after(self):
        seq = """
for i in 0 .. 7:
    assert i >= 0 and i < 7

i: I64 = 123
assert i == 123
"""
        assert_run_success(seq)

    def test_loop_var_redeclare_in_inner_scope_func(self):
        # The loop var `i` inside the function shadows the global `i` (warns).
        seq = """
def test():
    for i in 0 .. 7:
        assert i >= 0 and i < 7

i: I64 = 123

assert i == 123

test()
"""
        assert_run_success(seq, expected_warnings={WarningType.SHADOW_VALUE})

    def test_loop_var_redeclare_in_inner_scope_after(self):
        seq = """
def test():
    for i in 0 .. 7:
        pass

    # After block scoping, this is fine: i is scoped to the for body
    i: I64 = 123
    assert i == 123
"""
        assert_run_success(seq)

    def test_loop_var_redeclare_wrong_type(self):
        # With block scoping, the for loop var shadows the outer i (warns). No conflict.
        seq = """
i: U16 = 123
for i in 0 .. 7:
    pass
assert i == 123
"""

        assert_run_success(seq, expected_warnings={WarningType.SHADOW_VALUE})

    def test_for_loop_declare_var_bad(self):
        seq = """
for x.y in 0 .. 7:
    pass
"""

        assert_compile_failure(seq)

    def test_use_loop_var_in_bounds(self):
        seq = """
for i in i .. 8:
    pass
"""

        assert_compile_failure(seq)


class TestWhileLoops:

    def test_while_break_in_if(self):
        seq = """
while True:
    if True:
        break
    exit(1)
"""

        assert_run_success(seq)

    def test_while_continue_in_if(self):
        """Test that continue both skips rest of body AND loops back to condition."""
        seq = """
i: U64 = 0
while i < 5:
    i = i + 1
    if True:
        continue
    exit(1)  # should be skipped by continue

# Verify loop ran all 5 iterations (not just 1 like a break would)
assert i == 5
"""

        assert_run_success(seq)


class TestNonBoolConditions:

    def test_if_non_bool_condition(self):
        """If condition must be bool; an integer should be rejected."""
        seq = """
val: U32 = 1
if val:
    exit(0)
exit(1)
"""
        assert_compile_failure(seq)

    def test_while_non_bool_condition(self):
        """While condition must be bool; an integer should be rejected."""
        seq = """
val: U32 = 1
while val:
    val = 0
"""
        assert_compile_failure(seq)
