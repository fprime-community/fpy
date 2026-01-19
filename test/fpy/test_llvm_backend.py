"""
Tests for the LLVM backend.
"""
import pytest
from pathlib import Path

# Path to the test dictionary
DICTIONARY_PATH = str(Path(__file__).parent / "RefTopologyDictionary.json")


class TestLLVMBackend:
    """Test class for LLVM backend functionality."""
    
    def test_import(self):
        """Test that we can import the LLVM backend."""
        from fpy.llvm_backend import (
            LLVMCodeGenerator,
            LLVMEmitter,
            compile_fpy_to_llvm,
            fpy_type_to_llvm,
            JITCompiler,
            run_fpy,
        )
    
    def test_simple_variable_bool(self):
        """Test simple boolean variable declaration."""
        from fpy.llvm_backend import compile_fpy_to_llvm
        
        code = """
x: bool = True
y: bool = False
"""
        module, state = compile_fpy_to_llvm(code, DICTIONARY_PATH)
        ir_str = str(module)
        
        # Check that the module was generated (LLVM quotes function names)
        assert 'define i32 @"main"()' in ir_str
        # Check for alloca instructions for variables
        assert "alloca i1" in ir_str
        # Check for store instructions
        assert "store i1" in ir_str
        
    def test_simple_variable_u32(self):
        """Test simple U32 variable declaration."""
        from fpy.llvm_backend import compile_fpy_to_llvm
        
        code = """
x: U32 = 42
"""
        module, state = compile_fpy_to_llvm(code, DICTIONARY_PATH)
        ir_str = str(module)
        
        # Check that the module was generated (LLVM quotes function names)
        assert 'define i32 @"main"()' in ir_str
        # Check for 32-bit integer allocation
        assert "alloca i32" in ir_str
        # Check for store of constant 42
        assert "store i32 42" in ir_str
    
    def test_variable_reassignment(self):
        """Test variable reassignment."""
        from fpy.llvm_backend import compile_fpy_to_llvm
        
        code = """
x: U32 = 1
x = 2
"""
        module, state = compile_fpy_to_llvm(code, DICTIONARY_PATH)
        ir_str = str(module)
        
        # Should have one alloca and two stores
        assert "alloca i32" in ir_str
        assert ir_str.count("store i32") >= 2
    
    def test_simple_if(self):
        """Test simple if statement."""
        from fpy.llvm_backend import compile_fpy_to_llvm
        
        code = """
x: U32 = 1
if x == U32(1):
    x = 2
"""
        module, state = compile_fpy_to_llvm(code, DICTIONARY_PATH)
        ir_str = str(module)
        
        # Should have conditional branch
        assert "br i1" in ir_str
        # Should have then and ifcont blocks
        assert "then:" in ir_str
        assert "ifcont:" in ir_str
    
    def test_if_else(self):
        """Test if-else statement."""
        from fpy.llvm_backend import compile_fpy_to_llvm
        
        code = """
x: U32 = 1
if x == U32(1):
    x = 2
else:
    x = 3
"""
        module, state = compile_fpy_to_llvm(code, DICTIONARY_PATH)
        ir_str = str(module)
        
        # Should have conditional branch
        assert "br i1" in ir_str
        # Should have then, else, and ifcont blocks
        assert "then:" in ir_str
        assert "else:" in ir_str
        assert "ifcont:" in ir_str
    
    def test_comparison_operators(self):
        """Test various comparison operators."""
        from fpy.llvm_backend import compile_fpy_to_llvm
        
        code = """
a: U32 = 10
b: U32 = 20
eq: bool = a == b
ne: bool = a != b
"""
        module, state = compile_fpy_to_llvm(code, DICTIONARY_PATH)
        ir_str = str(module)
        
        # Should have icmp instructions
        assert "icmp eq" in ir_str or "icmp ne" in ir_str
    
    def test_llvm_verification(self):
        """Test that generated LLVM IR is valid."""
        from fpy.llvm_backend import compile_fpy_to_llvm, LLVMCodeGenerator
        from fpy.compiler import text_to_ast, get_base_compile_state
        from fpy.llvm_backend import run_semantic_passes
        
        code = """
x: U32 = 42
y: bool = True
if y:
    x = 100
"""
        ast = text_to_ast(code)
        state = get_base_compile_state(DICTIONARY_PATH, {})
        state.root = ast
        state = run_semantic_passes(ast, state)
        
        generator = LLVMCodeGenerator()
        module = generator.compile_module(ast, state)
        
        # This should not raise
        assert generator.verify()
    
    def test_type_mapping(self):
        """Test that fpy types map to correct LLVM types."""
        from fpy.llvm_backend import fpy_type_to_llvm
        from llvmlite import ir
        from fprime_gds.common.models.serialize.bool_type import BoolType as BoolValue
        from fprime_gds.common.models.serialize.numerical_types import (
            U8Type as U8Value,
            U16Type as U16Value,
            U32Type as U32Value,
            U64Type as U64Value,
            I8Type as I8Value,
            I16Type as I16Value,
            I32Type as I32Value,
            I64Type as I64Value,
            F32Type as F32Value,
            F64Type as F64Value,
        )
        
        assert fpy_type_to_llvm(BoolValue) == ir.IntType(1)
        assert fpy_type_to_llvm(U8Value) == ir.IntType(8)
        assert fpy_type_to_llvm(U16Value) == ir.IntType(16)
        assert fpy_type_to_llvm(U32Value) == ir.IntType(32)
        assert fpy_type_to_llvm(U64Value) == ir.IntType(64)
        assert fpy_type_to_llvm(I8Value) == ir.IntType(8)
        assert fpy_type_to_llvm(I16Value) == ir.IntType(16)
        assert fpy_type_to_llvm(I32Value) == ir.IntType(32)
        assert fpy_type_to_llvm(I64Value) == ir.IntType(64)
        assert isinstance(fpy_type_to_llvm(F32Value), ir.FloatType)
        assert isinstance(fpy_type_to_llvm(F64Value), ir.DoubleType)


class TestJITExecution:
    """Test JIT compilation and execution."""
    
    def test_jit_simple_variable(self):
        """Test JIT execution with simple variable."""
        from fpy.llvm_backend import run_fpy
        
        code = """
x: U32 = 42
"""
        result = run_fpy(code, DICTIONARY_PATH)
        # Main returns 0 for success
        assert result == 0
    
    def test_jit_if_else(self):
        """Test JIT execution with if-else."""
        from fpy.llvm_backend import run_fpy
        
        code = """
x: U32 = 1
if x == U32(1):
    x = 2
else:
    x = 3
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_boolean_operations(self):
        """Test JIT execution with boolean operations."""
        from fpy.llvm_backend import run_fpy
        
        code = """
a: bool = True
b: bool = False
c: bool = a == b
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_nested_if(self):
        """Test JIT execution with nested if statements."""
        from fpy.llvm_backend import run_fpy
        
        code = """
x: U32 = 10
y: U32 = 20
if x < y:
    if x == U32(10):
        x = 100
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_compiler_get_ir(self):
        """Test that JITCompiler can return IR."""
        from fpy.llvm_backend import JITCompiler
        
        compiler = JITCompiler()
        compiler.compile("x: U32 = 42", DICTIONARY_PATH)
        
        ir_str = compiler.get_ir()
        assert "define i32" in ir_str
        assert "alloca i32" in ir_str
        assert "store i32 42" in ir_str
    
    def test_jit_multiple_runs(self):
        """Test that we can compile and run multiple programs."""
        from fpy.llvm_backend import JITCompiler
        
        compiler1 = JITCompiler()
        result1 = compiler1.run("x: U32 = 1", DICTIONARY_PATH)
        
        compiler2 = JITCompiler()
        result2 = compiler2.run("y: bool = True", DICTIONARY_PATH)
        
        assert result1 == 0
        assert result2 == 0
    
    def test_jit_exit_success(self):
        """Test exit(0) returns 0."""
        from fpy.llvm_backend import run_fpy
        
        code = """
exit(0)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_exit_failure(self):
        """Test exit(non-zero) returns that value."""
        from fpy.llvm_backend import run_fpy
        
        code = """
exit(42)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 42
    
    def test_jit_while_loop(self):
        """Test while loop execution."""
        from fpy.llvm_backend import run_fpy
        
        # Note: fpy uses integer literals (not casted), and types are inferred from
        # the variable declaration. U64 is the default for integer literals.
        code = """
x: U64 = 0
count: U64 = 0
while count < 5:
    x = x + 1
    count = count + 1
# x should be 5 now
if x == 5:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_while_break(self):
        """Test while loop with break."""
        from fpy.llvm_backend import run_fpy
        
        code = """
x: U64 = 0
while x < 100:
    x = x + 1
    if x == 3:
        break
# x should be 3 after break
if x == 3:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_while_continue(self):
        """Test while loop with continue."""
        from fpy.llvm_backend import run_fpy
        
        # Count up to 5, incrementing by 1 each iteration
        code = """
x: U64 = 0
iterations: U64 = 0
while x < 5:
    iterations = iterations + 1
    x = x + 1
    continue
# iterations should be 5
if iterations == 5:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_assert_pass(self):
        """Test assert that passes."""
        from fpy.llvm_backend import run_fpy
        
        code = """
x: U32 = 5
assert x == U32(5)
exit(0)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_assert_fail(self):
        """Test assert that fails - returns 1."""
        from fpy.llvm_backend import run_fpy
        
        code = """
x: U32 = 5
assert x == U32(10)
exit(0)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        # assert failure should return 1
        assert result == 1
    
    def test_jit_unary_not(self):
        """Test unary not operator."""
        from fpy.llvm_backend import run_fpy
        
        code = """
x: bool = True
y: bool = not x
if y == False:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_unary_minus(self):
        """Test unary minus operator."""
        from fpy.llvm_backend import run_fpy
        
        code = """
x: I64 = 5
y: I64 = -x
if y == -5:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_modulo(self):
        """Test modulo operator."""
        from fpy.llvm_backend import run_fpy
        
        code = """
x: U64 = 17
y: U64 = x % 5
if y == 2:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_floor_division(self):
        """Test floor division operator."""
        from fpy.llvm_backend import run_fpy
        
        code = """
x: U64 = 17
y: U64 = x // 5
if y == 3:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_arithmetic(self):
        """Test arithmetic operations."""
        from fpy.llvm_backend import run_fpy
        
        # Note: In fpy, / is float division, // is integer division
        code = """
a: U64 = 10
b: U64 = 3
add_result: U64 = a + b
sub_result: U64 = a - b
mul_result: U64 = a * b
div_result: U64 = a // b
mod_result: U64 = a % b
# 10 + 3 = 13, 10 - 3 = 7, 10 * 3 = 30, 10 // 3 = 3, 10 % 3 = 1
if add_result == 13:
    if sub_result == 7:
        if mul_result == 30:
            if div_result == 3:
                if mod_result == 1:
                    exit(0)
exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_comparison_chain(self):
        """Test various comparison operators."""
        from fpy.llvm_backend import run_fpy
        
        code = """
a: U32 = 5
b: U32 = 10
c: U32 = 5

if a < b:
    if b > a:
        if a <= c:
            if c >= a:
                if a == c:
                    if a != b:
                        exit(0)
exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_simple_function(self):
        """Test simple function definition and call."""
        from fpy.llvm_backend import run_fpy
        
        code = """
def get_five() -> U8:
    return 5

result: U8 = get_five()
if result == 5:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_function_with_args(self):
        """Test function with parameters."""
        from fpy.llvm_backend import run_fpy
        
        code = """
def add_vals(lhs: U64, rhs: U64) -> U64:
    return lhs + rhs

result: U64 = add_vals(10, 20)
if result == 30:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_function_call_function(self):
        """Test function calling another function."""
        from fpy.llvm_backend import run_fpy
        
        code = """
def inner() -> U8:
    return 1

def outer() -> U8:
    return inner()

if outer() == 1:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_for_loop(self):
        """Test for loop (desugared to while)."""
        from fpy.llvm_backend import run_fpy
        
        code = """
sum: I64 = 0
for i in 0 .. 5:
    sum = sum + i
# 0 + 1 + 2 + 3 + 4 = 10
if sum == 10:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_for_loop_with_break(self):
        """Test for loop with break."""
        from fpy.llvm_backend import run_fpy
        
        code = """
sum: I64 = 0
for i in 0 .. 100:
    sum = sum + 1
    if sum == 5:
        break
# Should have summed 5 iterations
if sum == 5:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_nested_function_calls(self):
        """Test nested function calls in expressions."""
        from fpy.llvm_backend import run_fpy
        
        code = """
def double(x: U64) -> U64:
    return x * 2

def triple(x: U64) -> U64:
    return x * 3

result: U64 = double(triple(5))
# triple(5) = 15, double(15) = 30
if result == 30:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_float_arithmetic(self):
        """Test floating point operations."""
        from fpy.llvm_backend import run_fpy
        
        code = """
a: F64 = 10.0
b: F64 = 3.0
add_result: F64 = a + b
sub_result: F64 = a - b
mul_result: F64 = a * b
div_result: F64 = a / b

# Verify results (with tolerance for floating point)
# 10.0 + 3.0 = 13.0
# 10.0 - 3.0 = 7.0
# 10.0 * 3.0 = 30.0
# 10.0 / 3.0 ≈ 3.333...

if add_result > 12.9:
    if add_result < 13.1:
        if sub_result > 6.9:
            if sub_result < 7.1:
                if mul_result > 29.9:
                    if mul_result < 30.1:
                        if div_result > 3.3:
                            if div_result < 3.4:
                                exit(0)
exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_function_with_locals(self):
        """Test function with local variables."""
        from fpy.llvm_backend import run_fpy
        
        code = """
def calculate(x: U64) -> U64:
    doubled: U64 = x * 2
    tripled: U64 = x * 3
    return doubled + tripled

# calculate(5) = 10 + 15 = 25
if calculate(5) == 25:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_recursive_function(self):
        """Test recursive function call."""
        from fpy.llvm_backend import run_fpy
        
        code = """
def factorial(n: U64) -> U64:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# factorial(5) = 120
if factorial(5) == 120:
    exit(0)
else:
    exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_boolean_and(self):
        """Test boolean and operator."""
        from fpy.llvm_backend import run_fpy
        
        code = """
a: bool = True
b: bool = True
c: bool = False

# True and True = True
if a and b:
    # True and False = False
    if not (a and c):
        exit(0)
exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
    
    def test_jit_boolean_or(self):
        """Test boolean or operator."""
        from fpy.llvm_backend import run_fpy
        
        code = """
a: bool = True
b: bool = False
c: bool = False

# True or False = True
if a or b:
    # False or False = False  
    if not (b or c):
        exit(0)
exit(1)
"""
        result = run_fpy(code, DICTIONARY_PATH)
        assert result == 0
