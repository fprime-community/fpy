import pytest

from fpy.test_helpers import (
    assert_compile_failure,
    assert_compile_success,
    assert_run_success,
    assert_run_failure,
)
from fpy.model import DirectiveErrorCode
from fpy.types import U8, U32, F64, BOOL, FpyValue


# When --use-gds is NOT passed (the default), override fprime_test_api with None
# so tests run against the Python model instead of a live GDS.
# When --use-gds IS passed, delegate to the fprime-gds plugin's session fixture
# so tests run against the real deployment.
@pytest.fixture(name="fprime_test_api", scope="module")
def fprime_test_api_override(request):
    if request.config.getoption("--use-gds"):
        return request.getfixturevalue("fprime_test_api_session")
    return None


def test_empty_sequence(fprime_test_api):
    """Test that sequence() with no parameters compiles successfully."""
    seq = """
sequence()
"""
    assert_compile_success(fprime_test_api, seq)


def test_sequence_with_single_parameter(fprime_test_api):
    """Test that sequence(arg: U32) compiles successfully."""
    seq = """
sequence(arg: U32)
"""
    assert_compile_success(fprime_test_api, seq)


def test_sequence_with_multiple_parameters(fprime_test_api):
    """Test that sequence() with multiple parameters compiles successfully."""
    seq = """
sequence(arg1: U8, arg2: U32)
"""
    assert_compile_success(fprime_test_api, seq)


def test_sequence_with_trailing_comma(fprime_test_api):
    """Test that sequence() with trailing comma compiles successfully."""
    seq = """
sequence(arg1: U8, arg2: U32,)
"""
    assert_compile_success(fprime_test_api, seq)


def test_sequence_parameter_as_variable(fprime_test_api):
    """Test that sequence parameters can be used as variables in the sequence."""
    seq = """
sequence(arg1: U32)
# arg1 should be available as a variable
x: U32 = arg1
"""
    assert_compile_success(fprime_test_api, seq)


def test_sequence_parameter_reassignment(fprime_test_api):
    """Test that sequence parameters can be reassigned."""
    seq = """
sequence(arg1: U32)
arg1 = 42
assert arg1 == 42
"""
    # Note: This will only run successfully if parameters are properly initialized
    # For now, just test that it compiles
    assert_compile_success(fprime_test_api, seq)


def test_sequence_multiple_parameters_usage(fprime_test_api):
    """Test using multiple sequence parameters."""
    seq = """
sequence(arg1: U8, arg2: U32, arg3: I64)
result: I64 = I64(arg1) + I64(arg2) + arg3
"""
    assert_compile_success(fprime_test_api, seq)


def test_sequence_with_struct_type(fprime_test_api):
    """Test that sequence parameters can have struct types."""
    seq = """
sequence(record: Svc.DpRecord)
x: Svc.DpRecord = record
"""
    assert_compile_success(fprime_test_api, seq)


def test_sequence_with_array_type(fprime_test_api):
    """Test that sequence parameters can have array types."""
    seq = """
sequence(arr: Ref.DpDemo.U32Array)
x: Ref.DpDemo.U32Array = arr
"""
    assert_compile_success(fprime_test_api, seq)


def test_duplicate_sequence_statement(fprime_test_api):
    """Test that duplicate sequence statements fail to compile."""
    seq = """
sequence(arg1: U32)
sequence(arg2: U32)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_duplicate_parameter_names(fprime_test_api):
    """Test that duplicate parameter names fail to compile."""
    seq = """
sequence(arg1: U32, arg1: U8)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_sequence_parameter_conflicts_with_variable(fprime_test_api):
    """Test that a sequence parameter with the same name as a variable causes an error."""
    seq = """
sequence(x: U32)
x: U32 = 5
"""
    # This should fail because x is already defined as a parameter
    assert_compile_failure(fprime_test_api, seq)


def test_sequence_with_invalid_type(fprime_test_api):
    """Test that sequence parameters with invalid types fail to compile."""
    seq = """
sequence(arg1: NonExistentType)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_sequence_after_statement(fprime_test_api):
    """Test that sequence cannot appear after other statements."""
    seq = """
x: U32 = 1
sequence(arg1: U32)
"""
    # This may or may not be enforced - adjust based on requirements
    assert_compile_failure(fprime_test_api, seq)


def test_sequence_with_all_basic_types(fprime_test_api):
    """Test sequence parameters with all basic types."""
    seq = """
sequence(
    u8_val: U8,
    u16_val: U16,
    u32_val: U32,
    u64_val: U64,
    i8_val: I8,
    i16_val: I16,
    i32_val: I32,
    i64_val: I64,
    f32_val: F32,
    f64_val: F64,
    bool_val: bool
)
"""
    assert_compile_success(fprime_test_api, seq)


def test_sequence_parameter_in_function(fprime_test_api):
    """Test that sequence parameters can be accessed inside functions."""
    seq = """
sequence(arg1: U32)

def test_func():
    x: U32 = arg1

test_func()
"""
    assert_compile_success(fprime_test_api, seq)


def test_sequence_with_enum_type(fprime_test_api):
    """Test that sequence parameters can have enum types."""
    seq = """
sequence(enabled: Fw.Enabled)
x: Fw.Enabled = enabled
"""
    assert_compile_success(fprime_test_api, seq)


def test_sequence_parameters_in_expressions(fprime_test_api):
    """Test that sequence parameters can be used in complex expressions."""
    seq = """
sequence(a: U32, b: U32)
result: U64 = (a + b) * 2
assert result == (a + b) * 2
"""
    assert_compile_success(fprime_test_api, seq)


def test_sequence_parameter_in_control_flow(fprime_test_api):
    """Test sequence parameters in if statements."""
    seq = """
sequence(value: I32)
if value > 0:
    x: U32 = 1
elif value < 0:
    x: U32 = 2
else:
    x: U32 = 0
"""
    assert_compile_success(fprime_test_api, seq)


def test_sequence_parameter_in_loops(fprime_test_api):
    """Test sequence parameters in loops."""
    seq = """
sequence(max_val: I64)
sum: I64 = 0
for i in 0..max_val:
    sum = sum + i
"""
    assert_compile_success(fprime_test_api, seq)

def test_sequence_literal_as_type(fprime_test_api):
    """Test that a literal as a type annotation fails."""
    seq = """
sequence(x: 5)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_sequence_bool_as_type(fprime_test_api):
    """Test that a bool as a type annotation fails."""
    seq = """
sequence(x: True)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_sequence_string_type_parameter(fprime_test_api):
    """Test that string-typed parameters are rejected (not constant-sized)."""
    seq = """
sequence(s: Ref.DpDemo.StringAlias)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_sequence_struct_with_string_member(fprime_test_api):
    """Test that structs containing strings are rejected."""
    seq = """
sequence(s: Ref.DpDemo.StructWithStringMembers)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_sequence_param_as_default_arg(fprime_test_api):
    """Test that sequence parameters cannot be used as default argument values."""
    seq = """
sequence(x: U32)
def foo(y: U32 = x):
    pass
"""
    assert_compile_failure(fprime_test_api, seq)


def test_sequence_param_same_name_as_func(fprime_test_api):
    """Test that a sequence parameter can coexist with a function of the same name."""
    seq = """
sequence(foo: U32)
def foo():
    pass
foo()
"""
    assert_compile_success(fprime_test_api, seq)


def test_sequence_param_shadowed_by_loop_var(fprime_test_api):
    """Test that a for-loop variable can shadow a sequence parameter."""
    seq = """
sequence(i: U32)
for i in 0..10:
    pass
"""
    assert_compile_success(fprime_test_api, seq)


def test_sequence_param_shadowed_by_func_param(fprime_test_api):
    """Test that a function parameter can shadow a sequence parameter."""
    seq = """
sequence(x: U32)
def foo(x: I32):
    pass
foo(5)
"""
    assert_compile_success(fprime_test_api, seq)


def test_defining_sequence_in_function(fprime_test_api):
    """Test defining sequence in a function."""
    seq = """
def test_func():
    sequence(max_val: I64)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_defining_sequence_in_loop(fprime_test_api):
    """Test defining sequence in a loop."""
    seq = """
for i in 0..5:
    sequence(max_val: I64)
"""
    assert_compile_failure(fprime_test_api, seq)

def test_defining_sequence_in_if_stmt(fprime_test_api):
    """Test defining sequence in an if stmt."""
    seq = """
value: bool = True
if value:
    sequence(max_val: I64)
"""
    assert_compile_failure(fprime_test_api, seq)


# ---- End-to-end arg passing tests ----

def test_run_sequence_with_u32_arg(fprime_test_api):
    """Run a sequence that takes a U32 arg and uses it."""
    seq = """
sequence(x: U32)
result: U32 = x
"""
    assert_run_success(fprime_test_api, seq, args=[FpyValue(U32, 42)])


def test_run_sequence_with_multiple_args(fprime_test_api):
    """Run a sequence that takes multiple args of different types."""
    seq = """
sequence(a: U32, b: U32)
sum: U64 = a + b
"""
    assert_run_success(
        fprime_test_api, seq,
        args=[FpyValue(U32, 10), FpyValue(U32, 32)],
    )


def test_run_sequence_no_args_expected_none_provided(fprime_test_api):
    """Running a sequence with no args declared and no args passed should work."""
    seq = """
sequence()
result: U32 = 1
"""
    assert_run_success(fprime_test_api, seq)


def test_run_sequence_args_wrong_size(fprime_test_api):
    """Passing wrong-size args should raise an error."""
    seq = """
sequence(x: U32)
"""
    assert_run_failure(fprime_test_api, seq, validation_error=True, args=[FpyValue(U8, 0)])


def test_run_sequence_args_expected_but_missing(fprime_test_api):
    """Declaring args but not providing them should raise an error."""
    seq = """
sequence(x: U32)
"""
    assert_run_failure(fprime_test_api, seq, validation_error=True)


# ---- Value assertion tests ----


def test_arg_value_u32(fprime_test_api):
    """Assert on the value of a U32 arg."""
    seq = """
sequence(x: U32)
assert x == 42
"""
    assert_run_success(fprime_test_api, seq, args=[FpyValue(U32, 42)])


def test_arg_value_bool_true(fprime_test_api):
    """Assert on a bool arg being True."""
    seq = """
sequence(flag: bool)
assert flag
"""
    assert_run_success(fprime_test_api, seq, args=[FpyValue(BOOL, True)])


def test_arg_value_bool_false(fprime_test_api):
    """Assert on a bool arg being False."""
    seq = """
sequence(flag: bool)
assert not flag
"""
    assert_run_success(fprime_test_api, seq, args=[FpyValue(BOOL, False)])


def test_arg_value_arithmetic(fprime_test_api):
    """Use arg in arithmetic and assert the result."""
    seq = """
sequence(a: U32, b: U32)
assert a + b == 100
"""
    assert_run_success(fprime_test_api, seq, args=[FpyValue(U32, 60), FpyValue(U32, 40)])


def test_arg_value_in_if(fprime_test_api):
    """Use arg as a condition and verify control flow."""
    seq = """
sequence(flag: bool)
result: U32 = 0
if flag:
    result = 1
assert result == 1
"""
    assert_run_success(fprime_test_api, seq, args=[FpyValue(BOOL, True)])


def test_arg_wrong_value_fails_assert(fprime_test_api):
    """Passing a value that doesn't satisfy the assert should fail."""
    seq = """
sequence(x: U32)
assert x == 99
"""
    assert_run_failure(
        fprime_test_api, seq,
        error_code=DirectiveErrorCode.EXIT_WITH_ERROR,
        args=[FpyValue(U32, 1)],
    )


def test_multiple_args_correct_offsets(fprime_test_api):
    """Each arg should be at the correct frame offset and readable independently."""
    seq = """
sequence(a: U32, b: U32, c: U32)
assert a == 1
assert b == 2
assert c == 3
"""
    assert_run_success(
        fprime_test_api, seq,
        args=[FpyValue(U32, 1), FpyValue(U32, 2), FpyValue(U32, 3)],
    )


def test_args_and_locals_coexist(fprime_test_api):
    """Args and local variables should both be accessible."""
    seq = """
sequence(x: U32)
y: U64 = x + 10
assert x == 5
assert y == 15
"""
    assert_run_success(fprime_test_api, seq, args=[FpyValue(U32, 5)])


def test_arg_passed_to_function(fprime_test_api):
    """Sequence arg can be passed to a user-defined function."""
    seq = """
sequence(x: U32)

def double(v: U32) -> U64:
    return v + v

assert double(x) == 84
"""
    assert_run_success(fprime_test_api, seq, args=[FpyValue(U32, 42)])


def test_arg_returned_from_function(fprime_test_api):
    """Function can return a value derived from an arg, and the arg is still valid after."""
    seq = """
sequence(x: U32, y: U32)

def add(a: U32, b: U32) -> U64:
    return a + b

result: U64 = add(x, y)
assert result == 30
assert x == 10
assert y == 20
"""
    assert_run_success(
        fprime_test_api, seq,
        args=[FpyValue(U32, 10), FpyValue(U32, 20)],
    )


def test_arg_with_flags_modification(fprime_test_api):
    """Modifying flags should not corrupt sequence args."""
    seq = """
sequence(x: U32)
assert x == 7
flags.assert_cmd_success = False
assert x == 7
flags.assert_cmd_success = True
assert x == 7
"""
    assert_run_success(fprime_test_api, seq, args=[FpyValue(U32, 7)])


def test_arg_with_flags_and_locals(fprime_test_api):
    """Args, flags, and locals should all coexist without stack corruption."""
    seq = """
sequence(a: U32, b: U32)
flags.assert_cmd_success = False
y: U64 = a + b
assert a == 3
assert b == 4
assert y == 7
assert flags.assert_cmd_success == False
flags.assert_cmd_success = True
"""
    assert_run_success(
        fprime_test_api, seq,
        args=[FpyValue(U32, 3), FpyValue(U32, 4)],
    )


def test_arg_survives_function_call(fprime_test_api):
    """Args should not be corrupted after a function call modifies the stack."""
    seq = """
sequence(x: U32)

def work(v: U32) -> U64:
    a: U64 = v + 100
    b: U64 = a + 200
    return b

result: U64 = work(x)
assert x == 5
assert result == 305
"""
    assert_run_success(fprime_test_api, seq, args=[FpyValue(U32, 5)])


def test_modify_arg(fprime_test_api):
    """Sequence args can be reassigned."""
    seq = """
sequence(x: U32)
assert x == 10
x = 20
assert x == 20
"""
    assert_run_success(fprime_test_api, seq, args=[FpyValue(U32, 10)])


def test_modify_arg_does_not_affect_other_args(fprime_test_api):
    """Modifying one arg doesn't corrupt another."""
    seq = """
sequence(a: U32, b: U32)
assert a == 1
assert b == 2
a = 99
assert a == 99
assert b == 2
"""
    assert_run_success(
        fprime_test_api, seq,
        args=[FpyValue(U32, 1), FpyValue(U32, 2)],
    )


def test_too_many_parameters(fprime_test_api):
    """Sequences with more than 255 parameters should fail to compile."""
    params = ", ".join(f"a{i}: U8" for i in range(256))
    seq = f"sequence({params})\n"
    assert_compile_failure(fprime_test_api, seq)
