from fpy.types import U32

from fpy.test_helpers import assert_compile_failure, assert_run_success


class TestDefinition:

    def test_func_bad_type(self, fprime_test_api):
        seq = """
var: U32 = 1
(var + 1)(3)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_simple_func_def(self, fprime_test_api):
        seq = """
def test():
    pass
"""
        assert_run_success(fprime_test_api, seq)

    def test_def_with_args(self, fprime_test_api):
        seq = """
def test(arg: U8):
    pass
"""
        assert_run_success(fprime_test_api, seq)

    def test_two_func_args_same_name(self, fprime_test_api):
        seq = """

def test(arg: U8, arg: U8):
    pass
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_redeclare_func(self, fprime_test_api):
        seq = """

def test():
    pass
def test():
    pass
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_redeclare_func_from_var(self, fprime_test_api):
        # Functions and variables are in separate scopes, so this is allowed
        seq = """

test: U8 = 0
def test():
    pass
"""

        assert_run_success(fprime_test_api, seq)

    def test_redeclare_var_from_func(self, fprime_test_api):
        # Functions and variables are in separate scopes, so this is allowed
        seq = """

def test():
    pass
test: U8 = 0
"""

        assert_run_success(fprime_test_api, seq)

    def test_param_name_matching_type(self, fprime_test_api):
        seq = """
def echo(U32: U32) -> U32:
    return U32

assert echo(5) == 5
"""

        assert_run_success(fprime_test_api, seq)

    def test_func_modify_param(self, fprime_test_api):
        seq = """
def test(arg: U8):
    arg = 1
    assert arg == 1

val: U8 = 123
test(val)
assert val == 123
"""
        assert_run_success(fprime_test_api, seq)

class TestReturns:

    def test_return_outside_func(self, fprime_test_api):
        seq = """
return
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_simple_return(self, fprime_test_api):
        seq = """
def test():
    return
"""

        assert_run_success(fprime_test_api, seq)

    def test_return_val(self, fprime_test_api):
        seq = """
def test() -> U8:
    return 1

assert test() == 1
"""

        assert_run_success(fprime_test_api, seq)

    def test_wrong_return_type(self, fprime_test_api):
        seq = """
def test() -> U8:
    return 1.0
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_missing_return(self, fprime_test_api):
        seq = """
def test() -> U32:
    pass
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_if_elif_return(self, fprime_test_api):
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

    def test_if_elif_return_missing(self, fprime_test_api):
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

    def test_return_in_for(self, fprime_test_api):
        seq = """
def test() -> U32:
    for i in 0..0: # this fails because we can't guarantee that the loop body executes
        return 0
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_func_if_without_else_missing_return(self, fprime_test_api):
        seq = """
def missing_return(flag: bool) -> U32:
    if flag:
        return 1
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_return_nothing_expr_in_void_func(self, fprime_test_api):
        seq = """
def test():
    return Fw
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_void_function_without_explicit_return(self, fprime_test_api):
        seq = """
def noop():
    pass

noop()
"""

        assert_run_success(fprime_test_api, seq)

    def test_return_value_in_void_func(self, fprime_test_api):
        """A void function must not return a value."""
        seq = """
def test():
    return 42
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_bare_return_in_typed_func(self, fprime_test_api):
        """A function with a return type must not use a bare return."""
        seq = """
def test() -> U32:
    return
"""
        assert_compile_failure(fprime_test_api, seq)

class TestCalls:

    def test_wrong_arg_type(self, fprime_test_api):
        seq = """
def test(arg: U8):
    pass

test(1.0)
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_func_call_func(self, fprime_test_api):
        seq = """
def test() -> U8:
    return 1

def test2() -> U8:
    return test()

assert test2() == 1
"""

        assert_run_success(fprime_test_api, seq)

    def test_func_call_func_before_defined(self, fprime_test_api):
        seq = """

def test2() -> U8:
    return test()

def test() -> U8:
    return 1

assert test2() == 1
"""

        assert_run_success(fprime_test_api, seq)

    def test_func_with_multiple_args(self, fprime_test_api):
        seq = """
def add_vals(lhs: U32, rhs: U32) -> U64:
    return lhs + rhs

assert add_vals(1, 2) == 3
"""

        assert_run_success(fprime_test_api, seq)

    def test_fib(self, fprime_test_api):
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

    def test_use_void_function_result(self, fprime_test_api):
        """Using the result of a void function in an expression should fail."""
        seq = """
def noop():
    pass

val: U32 = noop()
"""
        assert_compile_failure(fprime_test_api, seq)

class TestScoping:

    def test_break_in_func_in_loop(self, fprime_test_api):
        seq = """
for i in 0..2:
    def test(arg: U8):
        break
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_get_outside_var_in_func(self, fprime_test_api):
        seq = """
i: U8 = 1
def test():
    assert i == 1
test()
"""

        assert_run_success(fprime_test_api, seq)

    def test_get_outside_struct_member_in_func(self, fprime_test_api):
        seq = """
var: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
def test():
    assert var.priority == 3
test()
"""

        assert_run_success(fprime_test_api, seq)

    def test_get_outside_array_element_const_index_in_func(self, fprime_test_api):
        """Test that functions can access array elements of global variables with a constant index"""
        seq = """
arr: Svc.ComQueueDepth = Svc.ComQueueDepth(123, 456)
def test():
    assert arr[0] == 123
    assert arr[1] == 456
test()
"""

        assert_run_success(fprime_test_api, seq)

    def test_get_outside_array_element_non_const_index_in_func(self, fprime_test_api):
        """Test that functions can access array elements of global variables with a non-constant index"""
        seq = """
arr: Svc.ComQueueDepth = Svc.ComQueueDepth(123, 456)
def test():
    idx: I64 = 1
    assert arr[idx] == 456
test()
"""

        assert_run_success(fprime_test_api, seq)

    def test_modify_global_var_in_func(self, fprime_test_api):
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

    def test_modify_global_var_in_func_before_definition(self, fprime_test_api):
        """Calling a function that reads a global before that global is declared
        is an error, even though the function itself may be defined earlier."""
        seq = """
increment()

def increment():
    assert i == 0
    i = i + 1

i: I64 = 123

assert i == 123
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_call_after_global_defined(self, fprime_test_api):
        """A function defined before its global is fine as long as the call
        comes after the global's declaration."""
        seq = """
def increment():
    i = i + 1

i: I64 = 0
increment()
assert i == 1
"""

        assert_run_success(fprime_test_api, seq)

    def test_call_reads_global_transitively_before_definition(
        self, fprime_test_api
    ):
        """Calling a function that reads a global only through a function it
        calls is still an error if the global isn't defined yet."""
        seq = """
def reader():
    assert i == 0

def caller():
    reader()

caller()

i: I64 = 5
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_call_reads_global_transitively_after_definition(
        self, fprime_test_api
    ):
        """The transitive case succeeds once the global is declared before the
        top-level call."""
        seq = """
def reader():
    assert i == 5

def caller():
    reader()

i: I64 = 5
caller()
"""

        assert_run_success(fprime_test_api, seq)

    def test_recursive_func_reads_global_before_definition(
        self, fprime_test_api
    ):
        """A recursive function that reads a global, called before that global
        is declared, is an error (the self-call must not loop the fixpoint)."""
        seq = """
def countdown(n: I64):
    if n == 0:
        assert g == 1
        return
    countdown(n - 1)

countdown(3)
g: I64 = 1
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_recursive_func_reads_global_after_definition(
        self, fprime_test_api
    ):
        """The same recursive function succeeds when the global is declared
        before the call."""
        seq = """
def countdown(n: I64):
    if n == 0:
        assert g == 1
        return
    countdown(n - 1)

g: I64 = 1
countdown(3)
"""

        assert_run_success(fprime_test_api, seq)

    def test_mutually_recursive_funcs_read_global_before_definition(
        self, fprime_test_api
    ):
        """Mutual recursion: a global read only in one of two mutually
        recursive functions must still be defined before either is called."""
        seq = """
def ping(n: I64):
    if n == 0:
        return
    pong(n - 1)

def pong(n: I64):
    assert g == 1
    ping(n - 1)

ping(3)
g: I64 = 1
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_call_in_own_initializer_reads_global(self, fprime_test_api):
        """A global whose initializer calls a function that reads that same
        global is an error: the global isn't defined until the assignment
        completes."""
        seq = """
def read_g() -> I64:
    return g

g: I64 = read_g()
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_use_lvar_from_func_outside_func(self, fprime_test_api):
        seq = """
def test():
    i: U8 = 0
assert i == 0
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_use_arg_from_func_outside_func(self, fprime_test_api):
        seq = """
def test(arg: U8):
    pass
if arg == 0:
    pass
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_var_in_func(self, fprime_test_api):
        seq = """
def test():
    var: U32 = 123
    assert var == 123

test()
"""

        assert_run_success(fprime_test_api, seq)

class TestNestedFunctions:

    def test_func_in_func(self, fprime_test_api):
        seq = """
def test(arg: U8):
    def test2() -> U8:
        return 0
    assert test2() == 0
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_func_in_if(self, fprime_test_api):
        """Function definitions are only allowed at the top level, not inside if blocks"""
        seq = """
if True:
    def test():
        pass
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_func_in_for(self, fprime_test_api):
        """Function definitions are only allowed at the top level, not inside for loops"""
        seq = """
for i in 0..10:
    def test():
        pass
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_func_in_while(self, fprime_test_api):
        """Function definitions are only allowed at the top level, not inside while loops"""
        seq = """
while True:
    def test():
        pass
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_use_loop_var_in_func_before_declared(self, fprime_test_api):
        # loop_var is scoped to the for body; functions can't access it
        seq = """
def fun():
    assert loop_var == 2

for loop_var in 0..2:
    pass

fun()
"""

        assert_compile_failure(fprime_test_api, seq)

class TestDefaultArguments:

    def test_default_arg_simple(self, fprime_test_api):
        seq = """
def test(a: U64, b: U64 = 5) -> U64:
    return a + b

assert test(1) == 6
assert test(1, 2) == 3
"""

        assert_run_success(fprime_test_api, seq)

    def test_default_arg_before_non_default(self, fprime_test_api):
        seq = """
def test(a: U64 = 5, b: U64):
    pass
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_default_arg_multiple(self, fprime_test_api):
        seq = """
def test(a: U64, b: U64 = 2, c: U64 = 3) -> U64:
    return a + b + c

assert test(1) == 6
assert test(1, 5) == 9
assert test(1, 5, 10) == 16
"""

        assert_run_success(fprime_test_api, seq)

    def test_default_arg_too_few_args(self, fprime_test_api):
        seq = """
def test(a: U64, b: U64 = 5) -> U64:
    return a + b

test()
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_default_arg_too_many_args(self, fprime_test_api):
        seq = """
def test(a: U64, b: U64 = 5) -> U64:
    return a + b

test(1, 2, 3)
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_default_arg_all_defaults(self, fprime_test_api):
        seq = """
def test(a: U64 = 1, b: U64 = 2) -> U64:
    return a + b

assert test() == 3
assert test(10) == 12
assert test(10, 20) == 30
"""

        assert_run_success(fprime_test_api, seq)

    def test_default_arg_float_coercion(self, fprime_test_api):
        """Float literal coercion for default value."""
        seq = """
def test(x: F64 = 2.5) -> F64:
    return x + 1.0

# 2.5 is an arbitrary precision Float literal that gets coerced to F64
assert test() == 3.5
assert test(10.0) == 11.0
"""

        assert_run_success(fprime_test_api, seq)

    def test_default_arg_expression(self, fprime_test_api):
        """Default value can be an expression."""
        seq = """
def test(a: U64 = 2 + 3) -> U64:
    return a

assert test() == 5
"""

        assert_run_success(fprime_test_api, seq)

    def test_default_arg_incompatible_type(self, fprime_test_api):
        """Default value type must be compatible with parameter type."""
        seq = """
def test(a: U64 = -5) -> U64:
    return a
"""

        # Should fail because -5 can't be assigned to U64
        assert_compile_failure(fprime_test_api, seq)

    def test_default_arg_wrong_type(self, fprime_test_api):
        """Default value type must be assignable to parameter type."""
        seq = """
def test(a: bool = 5) -> bool:
    return a
"""

        # Should fail because int literal is not a bool
        assert_compile_failure(fprime_test_api, seq)

    def test_default_arg_with_cast(self, fprime_test_api):
        """Default value can be a cast expression."""
        seq = """
def test(a: U8 = U8(255)) -> U8:
    return a

assert test() == 255
assert test(U8(10)) == 10
"""

        assert_run_success(fprime_test_api, seq)

    def test_default_arg_with_narrowing_cast(self, fprime_test_api):
        """Default value can be a narrowing cast expression."""
        seq = """
def test(a: U8 = U8(1000)) -> U8:
    return a

# 1000 truncated to U8 is 232 (1000 & 0xFF)
assert test() == 232
"""

        assert_run_success(fprime_test_api, seq)

    def test_default_arg_with_tlm(self, fprime_test_api):
        """Default value cannot be a runtime value like telemetry - must be const expr."""
        seq = """
def test(a: U32 = CdhCore.cmdDisp.CommandsDispatched) -> U32:
    return a
"""

        # Should fail because telemetry is not a const expression
        assert_compile_failure(fprime_test_api, seq)

    def test_default_arg_with_function_call(self, fprime_test_api):
        """Default value cannot be a function call - must be const expr."""
        seq = """
def helper() -> U64:
    return 42

def test(a: U64 = helper()) -> U64:
    return a
"""

        # Should fail because function call is not a const expression
        assert_compile_failure(fprime_test_api, seq)

    def test_default_arg_forward_called_function(self, fprime_test_api):
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

    def test_default_arg_with_variable(self, fprime_test_api):
        """Default value cannot reference a variable - must be const expr."""
        seq = """
x: U64 = 10

def test(a: U64 = x) -> U64:
    return test()
"""

        # Should fail because variable is not a const expression
        assert_compile_failure(fprime_test_api, seq)

    def test_default_arg_forward_reference(self, fprime_test_api):
        """Default value cannot reference a variable declared after the function."""
        seq = """
def test(a: U64 = x) -> U64:
    return a

x: U64 = 10
"""

        # Should fail: "'x' used before declared"
        assert_compile_failure(fprime_test_api, seq)

    def test_default_arg_undefined_variable(self, fprime_test_api):
        """Default value cannot reference an undefined variable."""
        seq = """
def test(a: U64 = undefined_var) -> U64:
    return a
"""

        # Should fail: "Unknown value"
        assert_compile_failure(fprime_test_api, seq)

class TestNamedArguments:

    def test_named_arg_simple(self, fprime_test_api):
        """Basic named argument usage."""
        seq = """
def test(a: U64, b: U64) -> U64:
    return a + b

assert test(a=1, b=2) == 3
"""

        assert_run_success(fprime_test_api, seq)

    def test_named_arg_reorder(self, fprime_test_api):
        """Named arguments can be in any order."""
        seq = """
def test(a: U64, b: U64, c: U64) -> U64:
    return a * 100 + b * 10 + c

assert test(c=3, a=1, b=2) == 123
"""

        assert_run_success(fprime_test_api, seq)

    def test_named_arg_mixed_positional(self, fprime_test_api):
        """Positional args followed by named args."""
        seq = """
def test(a: U64, b: U64, c: U64) -> U64:
    return a * 100 + b * 10 + c

assert test(1, c=3, b=2) == 123
"""

        assert_run_success(fprime_test_api, seq)

    def test_named_arg_with_defaults(self, fprime_test_api):
        """Named arguments with default values."""
        seq = """
def test(a: U64, b: U64 = 5, c: U64 = 10) -> U64:
    return a + b + c

# Only provide first and last, middle uses default
assert test(a=1, c=20) == 26
"""

        assert_run_success(fprime_test_api, seq)

    def test_named_arg_all_with_defaults(self, fprime_test_api):
        """Named arguments for all parameters, some with defaults."""
        seq = """
def test(a: U64 = 1, b: U64 = 2, c: U64 = 3) -> U64:
    return a * 100 + b * 10 + c

# Override middle one only
assert test(b=5) == 153
"""

        assert_run_success(fprime_test_api, seq)

    def test_named_arg_unknown_name(self, fprime_test_api):
        """Error when using unknown argument name."""
        seq = """
def test(a: U64, b: U64) -> U64:
    return a + b

test(a=1, c=2)
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_named_arg_duplicate(self, fprime_test_api):
        """Error when same argument specified twice."""
        seq = """
def test(a: U64, b: U64) -> U64:
    return a + b

test(a=1, a=2)
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_named_arg_positional_and_named(self, fprime_test_api):
        """Error when same argument specified by position and by name."""
        seq = """
def test(a: U64, b: U64) -> U64:
    return a + b

test(1, a=2)
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_named_arg_positional_after_named(self, fprime_test_api):
        """Error when positional argument follows named argument."""
        seq = """
def test(a: U64, b: U64, c: U64) -> U64:
    return a + b + c

test(a=1, 2, 3)
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_named_arg_missing_required(self, fprime_test_api):
        """Error when required argument is missing."""
        seq = """
def test(a: U64, b: U64, c: U64) -> U64:
    return a + b + c

test(a=1, c=3)
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_named_arg_builtin(self, fprime_test_api):
        """Named arguments work with builtin functions."""
        seq = """
sleep(useconds=1000, seconds=1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_named_arg_builtin_single(self, fprime_test_api):
        """Named arguments work with single-arg builtins."""
        seq = """
exit(exit_code=0)
"""

        assert_run_success(fprime_test_api, seq)
