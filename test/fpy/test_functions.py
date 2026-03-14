from fpy.types import U32

from fpy.test_helpers import assert_compile_failure, assert_run_success

def test_func_bad_type(fprime_test_api):
    seq = """
var: U32 = 1
(var + 1)(3)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_simple_func_def(fprime_test_api):
    seq = """
def test():
    pass
"""
    assert_run_success(fprime_test_api, seq)


def test_def_with_args(fprime_test_api):
    seq = """
def test(arg: U8):
    pass
"""
    assert_run_success(fprime_test_api, seq)


def test_return_outside_func(fprime_test_api):
    seq = """
return
"""

    assert_compile_failure(fprime_test_api, seq)


def test_simple_return(fprime_test_api):
    seq = """
def test():
    return
"""

    assert_run_success(fprime_test_api, seq)


def test_return_val(fprime_test_api):
    seq = """
def test() -> U8:
    return 1

assert test() == 1
"""

    assert_run_success(fprime_test_api, seq)


def test_wrong_return_type(fprime_test_api):
    seq = """
def test() -> U8:
    return 1.0
"""

    assert_compile_failure(fprime_test_api, seq)


def test_wrong_arg_type(fprime_test_api):
    seq = """
def test(arg: U8):
    pass

test(1.0)
"""

    assert_compile_failure(fprime_test_api, seq)


def test_break_in_func_in_loop(fprime_test_api):
    seq = """
for i in 0..2:
    def test(arg: U8):
        break
"""

    assert_compile_failure(fprime_test_api, seq)


def test_get_outside_var_in_func(fprime_test_api):
    seq = """
i: U8 = 1
def test():
    assert i == 1
test()
"""

    assert_run_success(fprime_test_api, seq)


def test_get_outside_struct_member_in_func(fprime_test_api):
    seq = """
var: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
def test():
    assert var.priority == 3
test()
"""

    assert_run_success(fprime_test_api, seq)


def test_get_outside_array_element_const_index_in_func(fprime_test_api):
    """Test that functions can access array elements of global variables with a constant index"""
    seq = """
arr: Svc.ComQueueDepth = Svc.ComQueueDepth(123, 456)
def test():
    assert arr[0] == 123
    assert arr[1] == 456
test()
"""

    assert_run_success(fprime_test_api, seq)


def test_get_outside_array_element_non_const_index_in_func(fprime_test_api):
    """Test that functions can access array elements of global variables with a non-constant index"""
    seq = """
arr: Svc.ComQueueDepth = Svc.ComQueueDepth(123, 456)
def test():
    idx: I64 = 1
    assert arr[idx] == 456
test()
"""

    assert_run_success(fprime_test_api, seq)


def test_modify_global_var_in_func(fprime_test_api):
    """Test that functions can modify top-level (global) variables"""
    seq = """
i: I64 = 0
def increment():
    i = i + 1
increment()
increment()
assert i == 2
"""

    assert_run_success(fprime_test_api, seq)


def test_modify_global_var_in_func_before_definition(fprime_test_api):
    """Test that functions can modify top-level (global) variables, declared after the function definition"""
    seq = """
increment()

def increment():
    assert i == 0
    i = i + 1

i: I64 = 123

assert i == 123
"""

    assert_run_success(fprime_test_api, seq)


def test_use_lvar_from_func_outside_func(fprime_test_api):
    seq = """
def test():
    i: U8 = 0
assert i == 0
"""

    assert_compile_failure(fprime_test_api, seq)


def test_use_arg_from_func_outside_func(fprime_test_api):
    seq = """
def test(arg: U8):
    pass
if arg == 0:
    pass
"""

    assert_compile_failure(fprime_test_api, seq)


def test_func_in_func(fprime_test_api):
    seq = """
def test(arg: U8):
    def test2() -> U8:
        return 0
    assert test2() == 0
"""

    assert_compile_failure(fprime_test_api, seq)


def test_func_in_if(fprime_test_api):
    """Function definitions are only allowed at the top level, not inside if blocks"""
    seq = """
if True:
    def test():
        pass
"""

    assert_compile_failure(fprime_test_api, seq)


def test_func_in_for(fprime_test_api):
    """Function definitions are only allowed at the top level, not inside for loops"""
    seq = """
for i in 0..10:
    def test():
        pass
"""

    assert_compile_failure(fprime_test_api, seq)


def test_func_in_while(fprime_test_api):
    """Function definitions are only allowed at the top level, not inside while loops"""
    seq = """
while True:
    def test():
        pass
"""

    assert_compile_failure(fprime_test_api, seq)


def test_use_func_outside_scope(fprime_test_api):
    # This test previously tested nested function scoping, but nested functions
    # are now disallowed at the syntax level. Keep as a syntax error test.
    seq = """
def test(arg: U8):
    def test2() -> U8:
        return 0

assert test2() == 0
"""

    assert_compile_failure(fprime_test_api, seq)


def test_func_call_func(fprime_test_api):
    seq = """
def test() -> U8:
    return 1

def test2() -> U8:
    return test()

assert test2() == 1
"""

    assert_run_success(fprime_test_api, seq)


def test_func_call_func_before_defined(fprime_test_api):
    seq = """

def test2() -> U8:
    return test()

def test() -> U8:
    return 1

assert test2() == 1
"""

    assert_run_success(fprime_test_api, seq)


def test_two_func_args_same_name(fprime_test_api):
    seq = """

def test(arg: U8, arg: U8):
    pass
"""

    assert_compile_failure(fprime_test_api, seq)


def test_redeclare_func(fprime_test_api):
    seq = """

def test():
    pass
def test():
    pass
"""

    assert_compile_failure(fprime_test_api, seq)


def test_redeclare_func_from_var(fprime_test_api):
    # Functions and variables are in separate scopes, so this is allowed
    seq = """

test: U8 = 0
def test():
    pass
"""

    assert_run_success(fprime_test_api, seq)


def test_redeclare_var_from_func(fprime_test_api):
    # Functions and variables are in separate scopes, so this is allowed
    seq = """

def test():
    pass
test: U8 = 0
"""

    assert_run_success(fprime_test_api, seq)


def test_fib(fprime_test_api):
    seq = """
def fib(a: U64) -> U64:
    if a < 2:
        return 1
    return fib(a - 1) + fib(a - 2)

assert fib(0) == 1
assert fib(1) == 1
assert fib(2) == 2
assert fib(4) == 5
"""

    assert_run_success(fprime_test_api, seq)


def test_var_in_func(fprime_test_api):
    seq = """
def test():
    var: U32 = 123
    assert var == 123

test()
"""

    assert_run_success(fprime_test_api, seq)


def test_missing_return(fprime_test_api):
    seq = """
def test() -> U32:
    pass
"""

    assert_compile_failure(fprime_test_api, seq)


def test_if_elif_return(fprime_test_api):
    seq = """
def test() -> U32:
    if True:
        return 1
    elif False:
        return 3
    else:
        return 2
    


assert test() == 1
"""

    assert_run_success(fprime_test_api, seq)


def test_if_elif_return_missing(fprime_test_api):
    seq = """
def test() -> U32:
    if True:
        return 1
    elif False:
        pass
    else:
        return 2
"""

    assert_compile_failure(fprime_test_api, seq)


def test_return_in_for(fprime_test_api):
    seq = """
def test() -> U32:
    for i in 0..0: # this fails because we can't guarantee that the loop body executes
        return 0
"""

    assert_compile_failure(fprime_test_api, seq)


def test_func_if_without_else_missing_return(fprime_test_api):
    seq = """
def missing_return(flag: bool) -> U32:
    if flag:
        return 1
"""

    assert_compile_failure(fprime_test_api, seq)


def test_func_with_multiple_args(fprime_test_api):
    seq = """
def add_vals(lhs: U32, rhs: U32) -> U64:
    return lhs + rhs

assert add_vals(1, 2) == 3
"""

    assert_run_success(fprime_test_api, seq)


def test_nested_func_capture_outer_local(fprime_test_api):
    seq = """
def outer() -> U32:
    local_val: U32 = 42
    def inner() -> U32:
        return local_val
    return inner()
"""

    assert_compile_failure(fprime_test_api, seq)


def test_nested_func_capture_outer_arg(fprime_test_api):
    seq = """
def outer(value: U32) -> U32:
    def inner() -> U32:
        return value
    return inner()
"""

    assert_compile_failure(fprime_test_api, seq)


def test_void_function_without_explicit_return(fprime_test_api):
    seq = """
def noop():
    pass

noop()
"""

    assert_run_success(fprime_test_api, seq)


def test_param_name_matching_type(fprime_test_api):
    seq = """
def echo(U32: U32) -> U32:
    return U32

assert echo(5) == 5
"""

    assert_run_success(fprime_test_api, seq)


def test_default_arg_simple(fprime_test_api):
    seq = """
def test(a: U64, b: U64 = 5) -> U64:
    return a + b

assert test(1) == 6
assert test(1, 2) == 3
"""

    assert_run_success(fprime_test_api, seq)


def test_default_arg_before_non_default(fprime_test_api):
    seq = """
def test(a: U64 = 5, b: U64):
    pass
"""

    assert_compile_failure(fprime_test_api, seq)


def test_default_arg_multiple(fprime_test_api):
    seq = """
def test(a: U64, b: U64 = 2, c: U64 = 3) -> U64:
    return a + b + c

assert test(1) == 6
assert test(1, 5) == 9
assert test(1, 5, 10) == 16
"""

    assert_run_success(fprime_test_api, seq)


def test_default_arg_too_few_args(fprime_test_api):
    seq = """
def test(a: U64, b: U64 = 5) -> U64:
    return a + b

test()
"""

    assert_compile_failure(fprime_test_api, seq)


def test_default_arg_too_many_args(fprime_test_api):
    seq = """
def test(a: U64, b: U64 = 5) -> U64:
    return a + b

test(1, 2, 3)
"""

    assert_compile_failure(fprime_test_api, seq)


def test_default_arg_all_defaults(fprime_test_api):
    seq = """
def test(a: U64 = 1, b: U64 = 2) -> U64:
    return a + b

assert test() == 3
assert test(10) == 12
assert test(10, 20) == 30
"""

    assert_run_success(fprime_test_api, seq)


def test_default_arg_float_coercion(fprime_test_api):
    """Float literal coercion for default value."""
    seq = """
def test(x: F64 = 2.5) -> F64:
    return x + 1.0

# 2.5 is an arbitrary precision Float literal that gets coerced to F64
assert test() == 3.5
assert test(10.0) == 11.0
"""

    assert_run_success(fprime_test_api, seq)


def test_default_arg_expression(fprime_test_api):
    """Default value can be an expression."""
    seq = """
def test(a: U64 = 2 + 3) -> U64:
    return a

assert test() == 5
"""

    assert_run_success(fprime_test_api, seq)


def test_default_arg_incompatible_type(fprime_test_api):
    """Default value type must be compatible with parameter type."""
    seq = """
def test(a: U64 = -5) -> U64:
    return a
"""

    # Should fail because -5 can't be assigned to U64
    assert_compile_failure(fprime_test_api, seq)


def test_default_arg_wrong_type(fprime_test_api):
    """Default value type must be assignable to parameter type."""
    seq = """
def test(a: bool = 5) -> bool:
    return a
"""

    # Should fail because int literal is not a bool
    assert_compile_failure(fprime_test_api, seq)


def test_default_arg_with_cast(fprime_test_api):
    """Default value can be a cast expression."""
    seq = """
def test(a: U8 = U8(255)) -> U8:
    return a

assert test() == 255
assert test(U8(10)) == 10
"""

    assert_run_success(fprime_test_api, seq)


def test_default_arg_with_narrowing_cast(fprime_test_api):
    """Default value can be a narrowing cast expression."""
    seq = """
def test(a: U8 = U8(1000)) -> U8:
    return a

# 1000 truncated to U8 is 232 (1000 & 0xFF)
assert test() == 232
"""

    assert_run_success(fprime_test_api, seq)


def test_default_arg_with_tlm(fprime_test_api):
    """Default value cannot be a runtime value like telemetry - must be const expr."""
    seq = """
def test(a: U32 = CdhCore.cmdDisp.CommandsDispatched) -> U32:
    return a
"""

    # Should fail because telemetry is not a const expression
    assert_compile_failure(fprime_test_api, seq)


def test_default_arg_with_function_call(fprime_test_api):
    """Default value cannot be a function call - must be const expr."""
    seq = """
def helper() -> U64:
    return 42

def test(a: U64 = helper()) -> U64:
    return a
"""

    # Should fail because function call is not a const expression
    assert_compile_failure(fprime_test_api, seq)


def test_default_arg_forward_called_function(fprime_test_api):
    """Default arg const values are calculated even for forward-called functions.

    This tests that when a function is called before it's defined in the source,
    the default argument's const value is still properly available.
    """
    seq = """
def caller() -> U64:
    # Call test() before it's defined, using default arg
    return test()

def test(a: U64 = 42) -> U64:
    return a

assert caller() == 42
"""

    assert_run_success(fprime_test_api, seq)


def test_default_arg_with_variable(fprime_test_api):
    """Default value cannot reference a variable - must be const expr."""
    seq = """
x: U64 = 10

def test(a: U64 = x) -> U64:
    return test()
"""

    # Should fail because variable is not a const expression
    assert_compile_failure(fprime_test_api, seq)


def test_default_arg_nested_func_cannot_access_outer_local(fprime_test_api):
    """Default value cannot reference a variable - must be const expr."""
    seq = """
def outer() -> U64:
    x: U64 = 10
    
    def inner(a: U64 = x) -> U64:
        return a
    
    return inner()
"""

    # Should fail because x is not a const expression
    assert_compile_failure(fprime_test_api, seq)


def test_default_arg_forward_reference(fprime_test_api):
    """Default value cannot reference a variable declared after the function."""
    seq = """
def test(a: U64 = x) -> U64:
    return a

x: U64 = 10
"""

    # Should fail: "'x' used before declared"
    assert_compile_failure(fprime_test_api, seq)


def test_default_arg_undefined_variable(fprime_test_api):
    """Default value cannot reference an undefined variable."""
    seq = """
def test(a: U64 = undefined_var) -> U64:
    return a
"""

    # Should fail: "Unknown value"
    assert_compile_failure(fprime_test_api, seq)


def test_named_arg_simple(fprime_test_api):
    """Basic named argument usage."""
    seq = """
def test(a: U64, b: U64) -> U64:
    return a + b

assert test(a=1, b=2) == 3
"""

    assert_run_success(fprime_test_api, seq)


def test_named_arg_reorder(fprime_test_api):
    """Named arguments can be in any order."""
    seq = """
def test(a: U64, b: U64, c: U64) -> U64:
    return a * 100 + b * 10 + c

assert test(c=3, a=1, b=2) == 123
"""

    assert_run_success(fprime_test_api, seq)


def test_named_arg_mixed_positional(fprime_test_api):
    """Positional args followed by named args."""
    seq = """
def test(a: U64, b: U64, c: U64) -> U64:
    return a * 100 + b * 10 + c

assert test(1, c=3, b=2) == 123
"""

    assert_run_success(fprime_test_api, seq)


def test_named_arg_with_defaults(fprime_test_api):
    """Named arguments with default values."""
    seq = """
def test(a: U64, b: U64 = 5, c: U64 = 10) -> U64:
    return a + b + c

# Only provide first and last, middle uses default
assert test(a=1, c=20) == 26
"""

    assert_run_success(fprime_test_api, seq)


def test_named_arg_all_with_defaults(fprime_test_api):
    """Named arguments for all parameters, some with defaults."""
    seq = """
def test(a: U64 = 1, b: U64 = 2, c: U64 = 3) -> U64:
    return a * 100 + b * 10 + c

# Override middle one only
assert test(b=5) == 153
"""

    assert_run_success(fprime_test_api, seq)


def test_named_arg_unknown_name(fprime_test_api):
    """Error when using unknown argument name."""
    seq = """
def test(a: U64, b: U64) -> U64:
    return a + b

test(a=1, c=2)
"""

    assert_compile_failure(fprime_test_api, seq)


def test_named_arg_duplicate(fprime_test_api):
    """Error when same argument specified twice."""
    seq = """
def test(a: U64, b: U64) -> U64:
    return a + b

test(a=1, a=2)
"""

    assert_compile_failure(fprime_test_api, seq)


def test_named_arg_positional_and_named(fprime_test_api):
    """Error when same argument specified by position and by name."""
    seq = """
def test(a: U64, b: U64) -> U64:
    return a + b

test(1, a=2)
"""

    assert_compile_failure(fprime_test_api, seq)


def test_named_arg_positional_after_named(fprime_test_api):
    """Error when positional argument follows named argument."""
    seq = """
def test(a: U64, b: U64, c: U64) -> U64:
    return a + b + c

test(a=1, 2, 3)
"""

    assert_compile_failure(fprime_test_api, seq)


def test_named_arg_missing_required(fprime_test_api):
    """Error when required argument is missing."""
    seq = """
def test(a: U64, b: U64, c: U64) -> U64:
    return a + b + c

test(a=1, c=3)
"""

    assert_compile_failure(fprime_test_api, seq)


def test_named_arg_builtin(fprime_test_api):
    """Named arguments work with builtin functions."""
    seq = """
sleep(useconds=1000, seconds=1)
"""

    assert_run_success(fprime_test_api, seq)


def test_named_arg_builtin_single(fprime_test_api):
    """Named arguments work with single-arg builtins."""
    seq = """
exit(exit_code=0)
"""

    assert_run_success(fprime_test_api, seq)


def test_func_modify_param(fprime_test_api):
    seq = """
def test(arg: U8):
    arg = 1
    assert arg == 1

val: U8 = 123
test(val)
assert val == 123
"""
    assert_run_success(fprime_test_api, seq)


def test_return_nothing_expr_in_void_func(fprime_test_api):
    seq = """
def test():
    return Fw
"""

    assert_compile_failure(fprime_test_api, seq)


def test_use_loop_var_in_func_before_declared(fprime_test_api):
    # loop_var is scoped to the for body; functions can't access it
    seq = """
def fun():
    assert loop_var == 2

for loop_var in 0..2:
    pass

fun()
"""

    assert_compile_failure(fprime_test_api, seq)
