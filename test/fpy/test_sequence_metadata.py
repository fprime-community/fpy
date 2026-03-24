import pytest

from fpy.test_helpers import (
    assert_compile_failure,
    assert_compile_success,
)


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