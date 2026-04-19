from fpy.test_helpers import assert_compile_failure, assert_run_success


class TestCheckBehavior:

    def test_check_condition_true_immediately(self, fprime_test_api):
        """Test that check succeeds immediately when condition is true with zero persist."""
        seq = """
check_passed: bool = False
check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) persist Fw.TimeIntervalValue(0, 0) period Fw.TimeIntervalValue(0, 100000):
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_timeout_when_always_false(self, fprime_test_api):
        """Test that check times out when condition is always false."""
        seq = """
timed_out: bool = False
check False timeout time_add(now(), Fw.TimeIntervalValue(0, 100000)) persist Fw.TimeIntervalValue(0, 0) period Fw.TimeIntervalValue(0, 10000):
    assert False, 1
timeout:
    timed_out = True
assert timed_out
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_condition_evaluated_multiple_times(self, fprime_test_api):
        """Test that check evaluates condition multiple times until it becomes true."""
        seq = """
# Count how many times condition is evaluated
eval_count: I64 = 0

def check_condition() -> bool:
    eval_count = eval_count + 1
    # Return true on 3rd evaluation
    return eval_count >= 3

check check_condition() timeout time_add(now(), Fw.TimeIntervalValue(5, 0)) persist Fw.TimeIntervalValue(0, 0) period Fw.TimeIntervalValue(0, 10000):
    pass
timeout:
    assert False, 1

# Should have been evaluated at least 3 times
assert eval_count >= 3
"""
        assert_run_success(fprime_test_api, seq, timeout_s=6)

    def test_check_condition_must_persist(self, fprime_test_api):
        """Test that condition must remain true for the full persist duration.

    If condition becomes false before persist duration, the timer resets.
    """
        seq = """
# Condition returns true twice, then false, then true forever
call_count: I64 = 0

def flaky_condition() -> bool:
    call_count = call_count + 1
    # True for calls 1-2, false for call 3, true thereafter
    if call_count <= 2:
        return True
    if call_count == 3:
        return False
    return True

# Calls 1-2 are true but call 3 is false, resetting the persist timer
# Call 4+ are true, so it should eventually succeed
check flaky_condition() timeout time_add(now(), Fw.TimeIntervalValue(10, 0)) persist Fw.TimeIntervalValue(3, 0) period Fw.TimeIntervalValue(1, 0):
    pass
timeout:
    assert False, 1

# Should have been called more than 3 times since persist timer was reset
assert call_count > 3
"""
        assert_run_success(fprime_test_api, seq, timeout_s=20)

    def test_check_zero_persist_true_once_enough(self, fprime_test_api):
        """Test that with zero persist, condition being true once is enough."""
        seq = """
# Return true only once, then false forever
returned_true: bool = False

def return_true_once() -> bool:
    if not returned_true:
        returned_true = True
        return True
    return False

check return_true_once() timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) persist Fw.TimeIntervalValue(0, 0) period Fw.TimeIntervalValue(0, 10000):
    exit(0)
timeout:
    assert False, 1
"""
        assert_run_success(fprime_test_api, seq, timeout_s=6)

class TestCheckBodies:

    def test_check_body_runs_on_success(self, fprime_test_api):
        """Test that check body runs when condition succeeds."""
        seq = """
body_ran: bool = False
check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) persist Fw.TimeIntervalValue(0, 0) period Fw.TimeIntervalValue(0, 10000):
    body_ran = True
assert body_ran
"""
        assert_run_success(fprime_test_api, seq, timeout_s=6)

    def test_check_timeout_body_runs_on_timeout(self, fprime_test_api):
        """Test that timeout body runs when check times out."""
        seq = """
timeout_body_ran: bool = False
check False timeout time_add(now(), Fw.TimeIntervalValue(0, 50000)) persist Fw.TimeIntervalValue(0, 0) period Fw.TimeIntervalValue(0, 10000):
    assert False, 1
timeout:
    timeout_body_ran = True
assert timeout_body_ran
"""
        assert_run_success(fprime_test_api, seq)

class TestCheckClauses:

    def test_check_no_timeout_clause(self, fprime_test_api):
        """Test check without timeout clause (runs indefinitely until success)."""
        seq = """
# Without timeout, check runs until condition succeeds
call_count: I64 = 0

def eventually_true() -> bool:
    call_count = call_count + 1
    return call_count >= 3

check eventually_true():
    exit(0)
"""
        assert_run_success(fprime_test_api, seq, timeout_s=10)

    def test_check_only_timeout_specified(self, fprime_test_api):
        """Test check with only timeout specified (uses default persist=0 and period=1s)."""
        seq = """
check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
    exit(0)
timeout:
    assert False, 1
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_absolute_timeout(self, fprime_test_api):
        """Test that check works with absolute Fw.Time timeout."""
        seq = """
# Create an absolute timeout 100ms from now
abs_timeout: Fw.Time = time_add(now(), Fw.TimeIntervalValue(0, 100000))
result: bool = False
check True timeout abs_timeout persist Fw.TimeIntervalValue(0, 0) period Fw.TimeIntervalValue(0, 10000):
    result = True
timeout:
    pass
assert result
"""
        assert_run_success(fprime_test_api, seq)

class TestCheckNesting:

    def test_check_nested_in_function(self, fprime_test_api):
        """Test that check statements work inside functions."""
        seq = """
def do_check() -> bool:
    result: bool = False
    check True timeout time_add(now(), Fw.TimeIntervalValue(0, 100000)) persist Fw.TimeIntervalValue(0, 0) period Fw.TimeIntervalValue(0, 10000):
        result = True
    timeout:
        pass
    return result

assert do_check()
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_modifies_outer_scope(self, fprime_test_api):
        """Test that check body can modify variables in outer scope."""
        seq = """
outer_var: I32 = 0

check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) persist Fw.TimeIntervalValue(0, 0) period Fw.TimeIntervalValue(0, 10000):
    outer_var = 42

assert outer_var == 42
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_nested(self, fprime_test_api):
        """Test nested check statements."""
        seq = """
outer_passed: bool = False
inner_passed: bool = False

check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
    outer_passed = True
    check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
        inner_passed = True

assert outer_passed
assert inner_passed
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_nested_timeout_body(self, fprime_test_api):
        """Test nested check with inner check timing out."""
        seq = """
outer_passed: bool = False
inner_timed_out: bool = False

check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
    outer_passed = True
    check False timeout time_add(now(), Fw.TimeIntervalValue(0, 100000)) period Fw.TimeIntervalValue(0, 10000):
        pass
    timeout:
        inner_timed_out = True

assert outer_passed
assert inner_timed_out
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_inside_while_loop(self, fprime_test_api):
        """Test check inside a while loop."""
        seq = """
iterations: I64 = 0
checks_passed: I64 = 0

while iterations < 3:
    check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
        checks_passed = checks_passed + 1
    iterations = iterations + 1

assert iterations == 3
assert checks_passed == 3
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_inside_for_loop(self, fprime_test_api):
        """Test check inside a for loop."""
        seq = """
checks_passed: I64 = 0

for i in 0..3:
    check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
        checks_passed = checks_passed + 1

assert checks_passed == 3
"""
        assert_run_success(fprime_test_api, seq)

class TestCheckTypeErrors:

    def test_check_timeout_wrong_type(self, fprime_test_api):
        """Test that check timeout with wrong type gives compile error."""
        seq = """
check True timeout 123 persist Fw.TimeIntervalValue(0, 0) period Fw.TimeIntervalValue(0, 10000):
    pass
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_check_condition_wrong_type(self, fprime_test_api):
        """Test that check condition must be bool, not int."""
        seq = """
check 123 timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) persist Fw.TimeIntervalValue(0, 0) period Fw.TimeIntervalValue(0, 10000):
    pass
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_check_condition_wrong_type_string(self, fprime_test_api):
        """Test that check condition must be bool, not string."""
        seq = """
check "hello" timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) persist Fw.TimeIntervalValue(0, 0) period Fw.TimeIntervalValue(0, 10000):
    pass
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_check_persist_wrong_type(self, fprime_test_api):
        """Test that check persist must be TimeIntervalValue, not int."""
        seq = """
check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) persist 123 period Fw.TimeIntervalValue(0, 10000):
    pass
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_check_persist_wrong_type_time(self, fprime_test_api):
        """Test that check persist must be TimeIntervalValue, not Fw.Time."""
        seq = """
check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) persist now() period Fw.TimeIntervalValue(0, 10000):
    pass
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_check_freq_wrong_type(self, fprime_test_api):
        """Test that check period must be TimeIntervalValue, not int."""
        seq = """
check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) persist Fw.TimeIntervalValue(0, 0) period 123:
    pass
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_check_freq_wrong_type_time(self, fprime_test_api):
        """Test that check period must be TimeIntervalValue, not Fw.Time."""
        seq = """
check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) persist Fw.TimeIntervalValue(0, 0) period now():
    pass
"""
        assert_compile_failure(fprime_test_api, seq)

class TestCheckDuplicateClauses:

    def test_check_duplicate_timeout(self, fprime_test_api):
        """Test that duplicate timeout clauses cause a compile error."""
        seq = """
check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) timeout time_add(now(), Fw.TimeIntervalValue(2, 0)):
    pass
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_check_duplicate_persist(self, fprime_test_api):
        """Test that duplicate persist clauses cause a compile error."""
        seq = """
check True persist Fw.TimeIntervalValue(1, 0) persist Fw.TimeIntervalValue(2, 0):
    pass
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_check_duplicate_freq(self, fprime_test_api):
        """Test that duplicate period clauses cause a compile error."""
        seq = """
check True period Fw.TimeIntervalValue(1, 0) period Fw.TimeIntervalValue(2, 0):
    pass
"""
        assert_compile_failure(fprime_test_api, seq)

class TestCheckMultilineSyntax:

    def test_check_multiline_timeout_only(self, fprime_test_api):
        """Test multi-line check with only timeout clause."""
        seq = """
check_passed: bool = False
check True
    timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_multiline_all_clauses(self, fprime_test_api):
        """Test multi-line check with all three clauses."""
        seq = """
check_passed: bool = False
check True
    timeout time_add(now(), Fw.TimeIntervalValue(1, 0))
    persist Fw.TimeIntervalValue(0, 0)
    period Fw.TimeIntervalValue(0, 100000):
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_multiline_different_order(self, fprime_test_api):
        """Test multi-line check with clauses in different order (period, persist, timeout)."""
        seq = """
check_passed: bool = False
check True
    period Fw.TimeIntervalValue(0, 100000)
    persist Fw.TimeIntervalValue(0, 0)
    timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_multiline_persist_timeout_order(self, fprime_test_api):
        """Test multi-line check with persist before timeout."""
        seq = """
check_passed: bool = False
check True
    persist Fw.TimeIntervalValue(0, 0)
    timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_singleline_any_order_persist_first(self, fprime_test_api):
        """Test single-line check with persist before timeout."""
        seq = """
check_passed: bool = False
check True persist Fw.TimeIntervalValue(0, 0) timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_singleline_any_order_freq_first(self, fprime_test_api):
        """Test single-line check with period before other clauses."""
        seq = """
check_passed: bool = False
check True period Fw.TimeIntervalValue(0, 100000) persist Fw.TimeIntervalValue(0, 0) timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_singleline_only_persist(self, fprime_test_api):
        """Test single-line check with only persist clause."""
        seq = """
check_passed: bool = False
check True persist Fw.TimeIntervalValue(0, 0):
    check_passed = True
assert check_passed
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_singleline_only_freq(self, fprime_test_api):
        """Test single-line check with only period clause."""
        seq = """
check_passed: bool = False
check True period Fw.TimeIntervalValue(0, 100000):
    check_passed = True
assert check_passed
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_multiline_timeout_occurs(self, fprime_test_api):
        """Test multi-line check that times out."""
        seq = """
timed_out: bool = False
check False
    timeout time_add(now(), Fw.TimeIntervalValue(0, 100000))
    period Fw.TimeIntervalValue(0, 10000):
    assert False, 1
timeout:
    timed_out = True
assert timed_out
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_mixed_inline_and_multiline(self, fprime_test_api):
        """Test check with some clauses inline and some on indented lines."""
        seq = """
check_passed: bool = False
check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0))
    persist Fw.TimeIntervalValue(0, 0)
    period Fw.TimeIntervalValue(0, 100000):
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_mixed_two_inline_one_multiline(self, fprime_test_api):
        """Test check with two clauses inline and one on indented line."""
        seq = """
check_passed: bool = False
check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) persist Fw.TimeIntervalValue(0, 0)
    period Fw.TimeIntervalValue(0, 100000):
    check_passed = True
timeout:
    assert False, 1
assert check_passed
"""
        assert_run_success(fprime_test_api, seq)

class TestCheckControlFlow:

    def test_check_break_not_allowed(self, fprime_test_api):
        """Test that break inside check body is not allowed when check is not in a loop."""
        seq = """
check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
    break
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_check_continue_not_allowed(self, fprime_test_api):
        """Test that continue inside check body is not allowed when check is not in a loop."""
        seq = """
check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
    continue
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_check_break_allowed_in_loop(self, fprime_test_api):
        """Test that break inside check body IS allowed when check is inside a loop."""
        seq = """
loop_ran: bool = False
while True:
    check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
        loop_ran = True
        break
assert loop_ran
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_continue_allowed_in_loop(self, fprime_test_api):
        """Test that continue inside check body IS allowed when check is inside a loop."""
        seq = """
iterations: I64 = 0
while iterations < 3:
    iterations = iterations + 1
    check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
        continue
assert iterations == 3
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_return_in_function(self, fprime_test_api):
        """Test that return inside check body works correctly in a function."""
        seq = """
def check_and_return() -> I64:
    check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
        return 42
    return 0

result: I64 = check_and_return()
assert result == 42
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_return_from_timeout_body(self, fprime_test_api):
        """Test that return from timeout body works correctly."""
        seq = """
def check_timeout_return() -> I64:
    check False timeout time_add(now(), Fw.TimeIntervalValue(0, 100000)) period Fw.TimeIntervalValue(0, 10000):
        return 1
    timeout:
        return 42
    return 0

result: I64 = check_timeout_return()
assert result == 42
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_both_branches_return(self, fprime_test_api):
        """Test that a function with check where both bodies return doesn't need trailing return."""
        seq = """
def check_returns() -> I64:
    check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
        return 42
    timeout:
        return 0

result: I64 = check_returns()
assert result == 42
"""
        assert_run_success(fprime_test_api, seq)

    def test_check_no_timeout_body_return_required(self, fprime_test_api):
        """Test that a function with check but no timeout body still needs trailing return."""
        seq = """
def check_needs_return() -> I64:
    check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)):
        return 42
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_check_no_timeout_clause_still_needs_return(self, fprime_test_api):
        """Test that a check with no timeout clause still needs a trailing return
    because the desugared if/else has an implicit else branch that doesn't return."""
        seq = """
def check_no_timeout() -> I64:
    check True:
        return 42
"""
        # The check desugars to: while True:...; if result: <body> else: <implicit>
        # The implicit else doesn't return, so we need a trailing return
        assert_compile_failure(fprime_test_api, seq)


class TestBodylessCheck:

    def test_bodyless_check_inline_true(self, fprime_test_api):
        """Body-less check with True condition proceeds immediately."""
        seq = """
done: bool = False
check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) period Fw.TimeIntervalValue(0, 100000)
done = True
assert done
"""
        assert_run_success(fprime_test_api, seq)

    def test_bodyless_check_inline_with_timeout(self, fprime_test_api):
        """Body-less check that times out (no timeout body, just proceeds)."""
        seq = """
check False timeout time_add(now(), Fw.TimeIntervalValue(0, 50000)) period Fw.TimeIntervalValue(0, 10000)
"""
        assert_run_success(fprime_test_api, seq)

    def test_bodyless_check_inline_no_clauses(self, fprime_test_api):
        """Body-less check with no clauses uses defaults (True passes immediately)."""
        seq = """
done: bool = False
check True
done = True
assert done
"""
        assert_run_success(fprime_test_api, seq)

    def test_bodyless_check_inline_only_timeout(self, fprime_test_api):
        """Body-less check with only a timeout clause."""
        seq = """
done: bool = False
check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0))
done = True
assert done
"""
        assert_run_success(fprime_test_api, seq)

    def test_bodyless_check_inline_only_period(self, fprime_test_api):
        """Body-less check with only a period clause."""
        seq = """
done: bool = False
check True period Fw.TimeIntervalValue(0, 100000)
done = True
assert done
"""
        assert_run_success(fprime_test_api, seq)

    def test_bodyless_check_multiline_clauses(self, fprime_test_api):
        """Body-less check with multi-line clauses (no colon on last clause)."""
        seq = """
done: bool = False
check True
    timeout time_add(now(), Fw.TimeIntervalValue(1, 0))
    period Fw.TimeIntervalValue(0, 100000)
done = True
assert done
"""
        assert_run_success(fprime_test_api, seq)

    def test_bodyless_check_multiline_with_timeout_body(self, fprime_test_api):
        """Body-less check with multi-line clauses and a timeout body."""
        seq = """
timed_out: bool = False
check False
    timeout time_add(now(), Fw.TimeIntervalValue(0, 50000))
    period Fw.TimeIntervalValue(0, 10000)
timeout:
    timed_out = True
assert timed_out
"""
        assert_run_success(fprime_test_api, seq)

    def test_bodyless_check_waits_for_condition(self, fprime_test_api):
        """Body-less check actually waits until the condition becomes true."""
        seq = """
counter: I64 = 0
def counting_condition() -> bool:
    counter = counter + 1
    return counter >= 3

check counting_condition() timeout time_add(now(), Fw.TimeIntervalValue(5, 0)) period Fw.TimeIntervalValue(0, 10000)
assert counter >= 3
"""
        assert_run_success(fprime_test_api, seq)

    def test_bodyless_check_inside_block(self, fprime_test_api):
        """Body-less check inside an if block."""
        seq = """
done: bool = False
if True:
    check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) period Fw.TimeIntervalValue(0, 100000)
    done = True
assert done
"""
        assert_run_success(fprime_test_api, seq)

    def test_bodyless_check_in_function(self, fprime_test_api):
        """Body-less check inside a function, followed by more statements."""
        seq = """
def wait_and_return() -> I64:
    check True timeout time_add(now(), Fw.TimeIntervalValue(1, 0)) period Fw.TimeIntervalValue(0, 100000)
    return 42

result: I64 = wait_and_return()
assert result == 42
"""
        assert_run_success(fprime_test_api, seq)
