from fpy.types import FpyValue, U32

from fpy.model import DirectiveErrorCode
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_failure,
    assert_run_success,
)


class TestExit:

    def test_exit_success(self, fprime_test_api):
        seq = """
exit(0)
"""
        assert_run_success(fprime_test_api, seq)

    def test_exit_failure(self, fprime_test_api):
        seq = """
exit(123)
"""
        assert_run_failure(fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR)

class TestIf:

    def test_simple_if(self, fprime_test_api):
        seq = """
var: bool = True

# use exit(0) if we want the sequence to succeed
# exit(1) if we want it to fail. helpful for testing.

if var:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_large_elifs(self, fprime_test_api):
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
            fprime_test_api,
            seq,
            {"CdhCore.cmdDisp.CommandsDispatched": FpyValue(U32, 4).serialize()},
        )

    def test_if_true(self, fprime_test_api):
        seq = """
if True:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_if_false(self, fprime_test_api):
        seq = """
if False:
    exit(1)
exit(0)
"""
        assert_run_success(fprime_test_api, seq)

    def test_if_else_true(self, fprime_test_api):
        seq = """
if True:
    exit(0)
else:
    exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_if_else_false(self, fprime_test_api):
        seq = """
if False:
    exit(1)
else:
    exit(0)
"""
        assert_run_success(fprime_test_api, seq)

    def test_if_elif_else(self, fprime_test_api):
        seq = """
if False:
    exit(1)
elif True:
    exit(0)
else:
    exit(1)
"""
        assert_run_success(fprime_test_api, seq)

class TestBreakContinueErrors:

    def test_break_outside_loop(self, fprime_test_api):
        seq = """
break
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_continue_outside_loop(self, fprime_test_api):
        seq = """
continue
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_deeply_nested_loops_exhaust_recursion_depth(self, fprime_test_api):
        depth = 500
        loop_header_lines = [
            ("    " * level) + f"for i{level} in 0 .. 1:" for level in range(depth)
        ]
        seq = "\n" + "\n".join(loop_header_lines) + "\n" + ("    " * depth) + "pass\n"

        # Purposefully triggers RecursionError inside the compiler's parse transform.

        assert_compile_failure(fprime_test_api, seq)

class TestForLoops:

    def test_simple_for(self, fprime_test_api):
        seq = """
for i in 0..2:
    pass
"""

        assert_run_success(fprime_test_api, seq)

    def test_for_loop_break(self, fprime_test_api):
        seq = """
counter: I64 = 0
for i in 0 .. 10:
    counter = counter + 1
    if counter == 5:
        break
    counter = counter + 1
assert counter == 5
"""
        assert_run_success(fprime_test_api, seq)

    def test_for_loop_continue(self, fprime_test_api):
        seq = """
counter: I64 = 0
for i in 0 .. 10:
    counter = counter + 1
    continue
    counter = counter + 1
assert counter == 10
"""
        assert_run_success(fprime_test_api, seq)

    def test_slightly_more_complex_for(self, fprime_test_api):
        seq = """
counter: U8 = 0
for i in 0 .. 2:
    if i > 2:
        exit(1)
    counter = U8(counter + 1)


assert counter == 2
"""

        assert_run_success(fprime_test_api, seq)

    def test_nested_for_loops(self, fprime_test_api):
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

        assert_run_success(fprime_test_api, seq)

    def test_nested_for_loops_break_inner(self, fprime_test_api):
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
        assert_run_success(fprime_test_api, seq)

    def test_nested_for_loops_break_outer(self, fprime_test_api):
        seq = """
for i in 0 .. 10:
    for j in 0 .. 5:
        break
    break
"""
        assert_run_success(fprime_test_api, seq)

    def test_nested_for_while_break(self, fprime_test_api):
        seq = """
counter: I64 = 0
for i in 0 .. 10:
    while True:
        break
    counter = counter + 1
assert counter == 10
"""
        assert_run_success(fprime_test_api, seq)

    def test_for_break_in_if(self, fprime_test_api):
        seq = """
for i in 0 .. 100:
    if True:
        break
    exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_for_continue_in_if(self, fprime_test_api):
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

        assert_run_success(fprime_test_api, seq)

    def test_two_fors_same_loop_var(self, fprime_test_api):
        seq = """
for i in 0 .. 7:
    assert i >= 0 and i < 7
for i in 0 .. 7:
    assert i >= 0 and i < 7
"""
        assert_run_success(fprime_test_api, seq)

    def test_empty_range(self, fprime_test_api):
        seq = """
for i in 7..0:
    exit(1)
"""
        assert_run_success(fprime_test_api, seq)

class TestLoopVariableScoping:

    def test_loop_var_outside_loop_after(self, fprime_test_api):
        seq = """
for i in 0 .. 7:
    pass
assert i == 7
"""
        # i is scoped to the for loop body; not visible after
        assert_compile_failure(fprime_test_api, seq)

    def test_loop_var_outside_loop_before(self, fprime_test_api):
        seq = """
i = 123
for i in 0 .. 7:
    pass
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_loop_var_redeclare_right_type(self, fprime_test_api):
        seq = """
i: I64 = 123
for i in 0 .. 7:
    assert i >= 0 and i < 7
assert i == 123
"""
        assert_run_success(fprime_test_api, seq)

    def test_loop_var_redeclare_right_type_after(self, fprime_test_api):
        seq = """
for i in 0 .. 7:
    assert i >= 0 and i < 7

i: I64 = 123
assert i == 123
"""
        assert_run_success(fprime_test_api, seq)

    def test_loop_var_redeclare_in_inner_scope_func(self, fprime_test_api):
        seq = """
def test():
    for i in 0 .. 7:
        assert i >= 0 and i < 7

i: I64 = 123

assert i == 123

test()
"""
        assert_run_success(fprime_test_api, seq)

    def test_loop_var_redeclare_in_inner_scope_after(self, fprime_test_api):
        seq = """
def test():
    for i in 0 .. 7:
        pass

    # After block scoping, this is fine: i is scoped to the for body
    i: I64 = 123
    assert i == 123
"""
        assert_run_success(fprime_test_api, seq)

    def test_loop_var_redeclare_wrong_type(self, fprime_test_api):
        # With block scoping, the for loop var shadows the outer i. No conflict.
        seq = """
i: U16 = 123
for i in 0 .. 7:
    pass
assert i == 123
"""

        assert_run_success(fprime_test_api, seq)

    def test_for_loop_declare_var_bad(self, fprime_test_api):
        seq = """
for x.y in 0 .. 7:
    pass
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_use_loop_var_in_bounds(self, fprime_test_api):
        seq = """
for i in i .. 8:
    pass
"""

        assert_compile_failure(fprime_test_api, seq)

class TestWhileLoops:

    def test_while_break_in_if(self, fprime_test_api):
        seq = """
while True:
    if True:
        break
    exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_while_continue_in_if(self, fprime_test_api):
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

        assert_run_success(fprime_test_api, seq)
