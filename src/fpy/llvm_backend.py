"""
LLVM backend for fpy using llvmlite.

This module provides an LLVM code generator for fpy, enabling native compilation
of fpy programs. It reuses the existing semantics and type checking from the
fpy compiler.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union

from llvmlite import ir, binding

from fpy.types import (
    CompileState,
    Emitter,
    FppType,
    VariableSymbol,
    CastSymbol,
    BuiltinFuncSymbol,
    FunctionSymbol,
    CommandSymbol,
    TypeCtorSymbol,
    FieldAccess,
    SymbolTable,
    is_instance_compat,
    NothingValue,
    FpyIntegerValue,
    FpyFloatValue,
)
from fprime_gds.common.templates.ch_template import ChTemplate
from fpy.syntax import (
    Ast,
    AstAssert,
    AstAssign,
    AstBinaryOp,
    AstBlock,
    AstBoolean,
    AstBreak,
    AstContinue,
    AstDef,
    AstFor,
    AstFuncCall,
    AstGetAttr,
    AstIf,
    AstIndexExpr,
    AstIdent,
    AstNumber,
    AstPass,
    AstReturn,
    AstString,
    AstStmtList,
    AstUnaryOp,
    AstWhile,
)
from fprime_gds.common.models.serialize.bool_type import BoolType as BoolValue
from fprime_gds.common.models.serialize.enum_type import EnumType as EnumValue
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
from fprime_gds.common.models.serialize.serializable_type import (
    SerializableType as StructValue,
)
from fprime_gds.common.models.serialize.array_type import ArrayType as ArrayValue


# No need to call binding.initialize() in newer versions of llvmlite - it's automatic


# Cache for LLVM struct types to avoid recreating them
_llvm_struct_type_cache: dict[type, ir.Type] = {}


def get_llvm_struct_type(typ: FppType) -> ir.Type:
    """Get or create an LLVM struct type for an fpy struct type.
    
    The struct is laid out sequentially based on MEMBER_LIST.
    """
    if typ in _llvm_struct_type_cache:
        return _llvm_struct_type_cache[typ]
    
    member_types = []
    for member_name, member_type, _, _ in typ.MEMBER_LIST:
        member_llvm_type = fpy_type_to_llvm(member_type)
        member_types.append(member_llvm_type)
    
    # Create a packed struct to match fprime's serialization layout
    struct_type = ir.LiteralStructType(member_types, packed=True)
    _llvm_struct_type_cache[typ] = struct_type
    return struct_type


def get_llvm_array_type(typ: FppType) -> ir.Type:
    """Get or create an LLVM array type for an fpy array type."""
    if typ in _llvm_struct_type_cache:
        return _llvm_struct_type_cache[typ]
    
    element_type = fpy_type_to_llvm(typ.MEMBER_TYPE)
    array_type = ir.ArrayType(element_type, typ.LENGTH)
    _llvm_struct_type_cache[typ] = array_type
    return array_type


def get_struct_member_index(typ: FppType, member_name: str) -> int | None:
    """Get the index of a member in a struct type's MEMBER_LIST."""
    for i, (name, _, _, _) in enumerate(typ.MEMBER_LIST):
        if name == member_name:
            return i
    return None


@dataclass
class LLVMCompileState:
    """State for LLVM code generation."""
    module: ir.Module
    builder: ir.IRBuilder
    variables: dict[str, ir.AllocaInst] = field(default_factory=dict)
    """Map of variable names to their stack allocations (function-local)"""
    global_variables: dict[str, ir.GlobalVariable] = field(default_factory=dict)
    """Map of variable names to their global variables (module-level)"""
    loop_stack: list[tuple[ir.Block, ir.Block]] = field(default_factory=list)
    """Stack of (condition_block, end_block) for break/continue"""
    functions: dict[str, ir.Function] = field(default_factory=dict)
    """Map of function names to their LLVM function objects"""
    current_function: ir.Function = None
    """The current function being compiled (for returns)"""
    entry_block: ir.Block = None
    """The entry block for the current function (for placing allocas)"""
    in_function: bool = False
    """True when compiling inside a user-defined function (not main)"""
    
    # Reference to the semantic compile state for type info
    compile_state: CompileState = None
    
    # Telemetry database for JIT execution
    # Maps channel name -> serialized value bytes
    tlm_db: dict[str, bytes] = field(default_factory=dict)


def fpy_type_to_llvm(typ: FppType, allow_complex: bool = False) -> ir.Type:
    """Convert an fpy type to an LLVM IR type.
    
    Args:
        typ: The fpy type to convert
        allow_complex: If True, allows struct/array types and returns their proper LLVM types
    """
    if typ is None:
        raise NotImplementedError("None type not supported in LLVM backend")
    if typ == BoolValue:
        return ir.IntType(1)
    elif typ == U8Value or typ == I8Value:
        return ir.IntType(8)
    elif typ == U16Value or typ == I16Value:
        return ir.IntType(16)
    elif typ == U32Value or typ == I32Value:
        return ir.IntType(32)
    elif typ == U64Value or typ == I64Value:
        return ir.IntType(64)
    elif typ == F32Value:
        return ir.FloatType()
    elif typ == F64Value:
        return ir.DoubleType()
    elif typ == FpyIntegerValue:
        # Arbitrary precision integer literals - use 64-bit
        return ir.IntType(64)
    elif typ == FpyFloatValue:
        # Arbitrary precision float literals - use double
        return ir.DoubleType()
    elif issubclass(typ, EnumValue):
        # Enum types are stored as integers - use their actual size
        enum_size = typ.getMaxSize()
        return ir.IntType(enum_size * 8)
    elif issubclass(typ, ArrayValue):
        # Array type - create LLVM array type
        return get_llvm_array_type(typ)
    elif issubclass(typ, StructValue):
        # Struct type - create LLVM struct type
        return get_llvm_struct_type(typ)
    elif allow_complex:
        # For unknown complex types, return a placeholder
        return ir.IntType(64)
    else:
        raise NotImplementedError(f"Type {typ} not supported in LLVM backend")


def is_signed_type(typ: FppType) -> bool:
    """Check if a type is signed."""
    return typ in (I8Value, I16Value, I32Value, I64Value, FpyIntegerValue)


def is_float_type(typ: FppType) -> bool:
    """Check if a type is floating point."""
    return typ in (F32Value, F64Value, FpyFloatValue)


class LLVMEmitter(Emitter):
    """Emits LLVM IR from fpy AST nodes."""
    
    def __init__(self, llvm_state: LLVMCompileState):
        super().__init__()
        self.llvm_state = llvm_state
    
    def create_entry_block_alloca(self, llvm_type: ir.Type, name: str) -> ir.AllocaInstr:
        """Create an alloca at the function entry block.
        
        This ensures proper dominance - the alloca dominates all uses
        because it's at the start of the function.
        """
        entry_block = self.llvm_state.entry_block
        if entry_block is None:
            # Fallback: create at current position
            return self.llvm_state.builder.alloca(llvm_type, name=name)
        
        # Save current insertion point
        current_block = self.llvm_state.builder.block
        
        # Position at start of entry block
        # Find first non-alloca instruction (or end of block)
        insert_before = None
        for instr in entry_block.instructions:
            if instr.opname != 'alloca':
                insert_before = instr
                break
        
        if insert_before is not None:
            # Position before the first non-alloca instruction
            self.llvm_state.builder.position_before(insert_before)
        else:
            # Position at end of entry block (only allocas so far)
            self.llvm_state.builder.position_at_end(entry_block)
        
        # Create the alloca
        alloca = self.llvm_state.builder.alloca(llvm_type, name=name)
        
        # Restore insertion point
        self.llvm_state.builder.position_at_end(current_block)
        
        return alloca
    
    def emit(self, node: Ast, state: CompileState) -> ir.Value | None:
        """Override emit to return LLVM values instead of directives."""
        return self.emitters[type(node)](node, state)
    
    def emit_AstBlock(self, node: AstBlock, state: CompileState) -> ir.Value | None:
        """Emit code for a block of statements."""
        result = None
        for stmt in node.stmts:
            # Don't emit statements after a terminator (return, break, continue)
            if self.llvm_state.builder.block.is_terminated:
                break
            result = self.emit(stmt, state)
        return result
    
    def emit_AstStmtList(self, node: AstStmtList, state: CompileState) -> ir.Value | None:
        """Emit code for a statement list."""
        result = None
        for stmt in node.stmts:
            # Don't emit statements after a terminator (return, break, continue)
            if self.llvm_state.builder.block.is_terminated:
                break
            result = self.emit(stmt, state)
        return result
    
    def emit_AstPass(self, node: AstPass, state: CompileState) -> ir.Value | None:
        """Pass statement does nothing."""
        return None
    
    def emit_AstNumber(self, node: AstNumber, state: CompileState) -> ir.Value:
        """Emit a numeric literal."""
        typ = state.contextual_types[node]
        llvm_type = fpy_type_to_llvm(typ)
        
        if is_float_type(typ):
            return ir.Constant(llvm_type, float(node.value))
        else:
            return ir.Constant(llvm_type, int(node.value))
    
    def emit_AstBoolean(self, node: AstBoolean, state: CompileState) -> ir.Value:
        """Emit a boolean literal."""
        return ir.Constant(ir.IntType(1), 1 if node.value else 0)
    
    def emit_AstFuncCall(self, node: AstFuncCall, state: CompileState) -> ir.Value:
        """Emit a function call.
        
        Handles type casts like U32(x), builtin functions like exit(), and user-defined functions.
        """
        builder = self.llvm_state.builder
        
        # Get the resolved symbol for the function
        func_sym = state.resolved_symbols.get(node.func)
        
        if is_instance_compat(func_sym, CastSymbol):
            # This is a type cast like U32(1)
            assert len(node.args) == 1, "Cast should have exactly one argument"
            arg = node.args[0]
            
            # Emit the argument
            arg_value = self.emit(arg, state)
            
            # Get source and destination types
            src_type = state.contextual_types[arg]
            dst_type = func_sym.to_type
            
            # Convert types
            return self.convert_value(arg_value, src_type, dst_type)
        elif is_instance_compat(func_sym, BuiltinFuncSymbol):
            # Handle builtin functions
            builtin_name = func_sym.name
            if builtin_name == "exit":
                # exit(code) - return from main with the given exit code
                assert len(node.args) == 1, "exit() takes exactly one argument"
                exit_code = self.emit(node.args[0], state)
                # Convert to i32 (standard exit code type)
                if exit_code.type != ir.IntType(32):
                    exit_code = builder.zext(exit_code, ir.IntType(32))
                builder.ret(exit_code)
                # Create a new block for any unreachable code after exit
                func = builder.function
                unreachable_block = func.append_basic_block("unreachable")
                builder.position_at_end(unreachable_block)
                # Return a dummy value (this code is unreachable)
                return ir.Constant(ir.IntType(32), 0)
            elif builtin_name == "print":
                # For now, print is a no-op in LLVM (we'd need runtime support)
                # Just evaluate the arguments for side effects
                for arg in node.args:
                    self.emit(arg, state)
                return ir.Constant(ir.IntType(1), 0)  # Return dummy value
            elif builtin_name == "abs":
                # abs(x) - absolute value
                assert len(node.args) == 1, "abs() takes exactly one argument"
                arg = self.emit(node.args[0], state)
                arg_type = state.contextual_types[node.args[0]]
                if is_float_type(arg_type):
                    # Use fabs intrinsic for floats
                    module = self.llvm_state.module
                    if arg.type == ir.FloatType():
                        fabs = module.declare_intrinsic('llvm.fabs', [ir.FloatType()])
                    else:
                        fabs = module.declare_intrinsic('llvm.fabs', [ir.DoubleType()])
                    return builder.call(fabs, [arg])
                else:
                    # For integers: x < 0 ? -x : x
                    zero = ir.Constant(arg.type, 0)
                    neg = builder.sub(zero, arg)
                    is_neg = builder.icmp_signed('<', arg, zero)
                    return builder.select(is_neg, neg, arg)
            elif builtin_name == "min":
                # min(a, b)
                assert len(node.args) == 2, "min() takes exactly two arguments"
                a = self.emit(node.args[0], state)
                b = self.emit(node.args[1], state)
                arg_type = state.contextual_types[node.args[0]]
                if is_float_type(arg_type):
                    cmp = builder.fcmp_ordered('<', a, b)
                elif is_signed_type(arg_type):
                    cmp = builder.icmp_signed('<', a, b)
                else:
                    cmp = builder.icmp_unsigned('<', a, b)
                return builder.select(cmp, a, b)
            elif builtin_name == "max":
                # max(a, b)
                assert len(node.args) == 2, "max() takes exactly two arguments"
                a = self.emit(node.args[0], state)
                b = self.emit(node.args[1], state)
                arg_type = state.contextual_types[node.args[0]]
                if is_float_type(arg_type):
                    cmp = builder.fcmp_ordered('>', a, b)
                elif is_signed_type(arg_type):
                    cmp = builder.icmp_signed('>', a, b)
                else:
                    cmp = builder.icmp_unsigned('>', a, b)
                return builder.select(cmp, a, b)
            elif builtin_name == "log":
                # log(x) - natural logarithm
                assert len(node.args) == 1, "log() takes exactly one argument"
                arg = self.emit(node.args[0], state)
                arg_type = state.contextual_types[node.args[0]]
                module = self.llvm_state.module
                if is_float_type(arg_type):
                    llvm_type = fpy_type_to_llvm(arg_type)
                    log_func = module.declare_intrinsic('llvm.log', [llvm_type])
                    return builder.call(log_func, [arg], name="log")
                else:
                    # Convert int to double, compute log, return double
                    float_arg = builder.sitofp(arg, ir.DoubleType())
                    log_func = module.declare_intrinsic('llvm.log', [ir.DoubleType()])
                    return builder.call(log_func, [float_arg], name="log")
            elif builtin_name == "sqrt":
                # sqrt(x)
                assert len(node.args) == 1, "sqrt() takes exactly one argument"
                arg = self.emit(node.args[0], state)
                arg_type = state.contextual_types[node.args[0]]
                module = self.llvm_state.module
                if is_float_type(arg_type):
                    llvm_type = fpy_type_to_llvm(arg_type)
                    sqrt_func = module.declare_intrinsic('llvm.sqrt', [llvm_type])
                    return builder.call(sqrt_func, [arg], name="sqrt")
                else:
                    float_arg = builder.sitofp(arg, ir.DoubleType())
                    sqrt_func = module.declare_intrinsic('llvm.sqrt', [ir.DoubleType()])
                    return builder.call(sqrt_func, [float_arg], name="sqrt")
            elif builtin_name == "wait_rel" or builtin_name == "wait_abs":
                # wait_rel and wait_abs - for JIT these are no-ops
                # Just evaluate the argument
                for arg in node.args:
                    self.emit(arg, state)
                return ir.Constant(ir.IntType(32), 0)
            else:
                # For any other builtins, try to treat as no-op
                # This allows JIT to compile code with unsupported builtins
                for arg in node.args:
                    self.emit(arg, state)
                return ir.Constant(ir.IntType(64), 0)
        elif is_instance_compat(func_sym, FunctionSymbol):
            # User-defined function call
            func_name = func_sym.definition.name.name
            if isinstance(func_name, str):
                pass
            else:
                func_name = str(func_name)
            
            # Get the LLVM function
            if func_name not in self.llvm_state.functions:
                raise RuntimeError(f"Function '{func_name}' not defined")
            
            llvm_func = self.llvm_state.functions[func_name]
            
            # Evaluate arguments
            # Use resolved_func_args if available (includes default args)
            resolved_args = state.resolved_func_args.get(node, node.args)
            args = []
            for i, arg in enumerate(resolved_args):
                arg_val = self.emit(arg, state)
                # Convert type if needed
                expected_type = llvm_func.function_type.args[i]
                if arg_val.type != expected_type:
                    arg_val = self.convert_value_types(arg_val, expected_type)
                args.append(arg_val)
            
            # Call the function
            return builder.call(llvm_func, args)
        elif is_instance_compat(func_sym, CommandSymbol):
            # Command calls - evaluate arguments but treat as no-op for JIT
            # In real execution, these would send commands to the hardware
            for arg in node.args:
                self.emit(arg, state)
            # Return a dummy value (commands don't return useful values in JIT)
            return ir.Constant(ir.IntType(32), 0)
        elif is_instance_compat(func_sym, TypeCtorSymbol):
            # Type constructor (struct/enum/array construction)
            return self._construct_type(node, func_sym, state)
        else:
            raise NotImplementedError(f"Function calls not yet supported: {func_sym}")
    
    def _construct_type(self, node: AstFuncCall, func_sym: TypeCtorSymbol, state: CompileState) -> ir.Value:
        """Emit code for constructing a struct or array value."""
        builder = self.llvm_state.builder
        typ = func_sym.type
        
        # Get resolved arguments (with defaults filled in)
        resolved_args = state.resolved_func_args.get(node, node.args)
        
        if issubclass(typ, StructValue):
            # Construct a struct
            llvm_struct_type = get_llvm_struct_type(typ)
            
            # Create an alloca for the struct at entry block
            struct_alloca = self.create_entry_block_alloca(llvm_struct_type, "struct_tmp")
            
            # Initialize each member
            for i, (member_info, arg_expr) in enumerate(zip(typ.MEMBER_LIST, resolved_args)):
                member_name, member_type, _, _ = member_info
                arg_value = self.emit(arg_expr, state)
                
                # Convert the arg value to the member type if needed
                expected_llvm_type = fpy_type_to_llvm(member_type)
                if arg_value.type != expected_llvm_type:
                    arg_fpy_type = state.contextual_types.get(arg_expr) or state.synthesized_types.get(arg_expr)
                    if arg_fpy_type is not None:
                        arg_value = self.convert_value(arg_value, arg_fpy_type, member_type)
                    else:
                        arg_value = self.convert_value_types(arg_value, expected_llvm_type)
                
                # Get pointer to member and store the value
                member_ptr = builder.gep(struct_alloca, [
                    ir.Constant(ir.IntType(32), 0),
                    ir.Constant(ir.IntType(32), i)
                ], inbounds=True, name=f"struct.{member_name}")
                builder.store(arg_value, member_ptr)
            
            # Load and return the constructed struct
            return builder.load(struct_alloca, name="struct_val")
        
        elif issubclass(typ, ArrayValue):
            # Construct an array
            llvm_array_type = get_llvm_array_type(typ)
            
            # Create an alloca for the array at entry block
            array_alloca = self.create_entry_block_alloca(llvm_array_type, "array_tmp")
            
            # Initialize each element
            for i, arg_expr in enumerate(resolved_args):
                arg_value = self.emit(arg_expr, state)
                
                # Convert the arg value to the element type if needed
                expected_llvm_type = fpy_type_to_llvm(typ.MEMBER_TYPE)
                if arg_value.type != expected_llvm_type:
                    arg_fpy_type = state.contextual_types.get(arg_expr) or state.synthesized_types.get(arg_expr)
                    if arg_fpy_type is not None:
                        arg_value = self.convert_value(arg_value, arg_fpy_type, typ.MEMBER_TYPE)
                    else:
                        arg_value = self.convert_value_types(arg_value, expected_llvm_type)
                
                # Get pointer to element and store the value
                elem_ptr = builder.gep(array_alloca, [
                    ir.Constant(ir.IntType(32), 0),
                    ir.Constant(ir.IntType(32), i)
                ], inbounds=True, name=f"array.{i}")
                builder.store(arg_value, elem_ptr)
            
            # Load and return the constructed array
            return builder.load(array_alloca, name="array_val")
        
        elif issubclass(typ, EnumValue):
            # Enum construction - just return the integer value
            assert len(resolved_args) == 1, "Enum constructor takes exactly one argument"
            arg_value = self.emit(resolved_args[0], state)
            return arg_value
        
        else:
            # Unknown type - fall back to evaluating args and returning placeholder
            for arg in resolved_args:
                self.emit(arg, state)
            return ir.Constant(ir.IntType(64), 0)
    
    def convert_value(self, value: ir.Value, src_type: FppType, dst_type: FppType) -> ir.Value:
        """Convert a value from one type to another."""
        builder = self.llvm_state.builder
        
        if src_type == dst_type:
            return value
        
        src_llvm = fpy_type_to_llvm(src_type)
        dst_llvm = fpy_type_to_llvm(dst_type)
        
        src_is_float = is_float_type(src_type)
        dst_is_float = is_float_type(dst_type)
        src_is_signed = is_signed_type(src_type)
        dst_is_signed = is_signed_type(dst_type)
        
        # Float to float
        if src_is_float and dst_is_float:
            if src_type == F32Value and dst_type == F64Value:
                return builder.fpext(value, dst_llvm, name="fpext")
            else:
                return builder.fptrunc(value, dst_llvm, name="fptrunc")
        
        # Int to float
        if not src_is_float and dst_is_float:
            if src_is_signed:
                return builder.sitofp(value, dst_llvm, name="sitofp")
            else:
                return builder.uitofp(value, dst_llvm, name="uitofp")
        
        # Float to int
        if src_is_float and not dst_is_float:
            if dst_is_signed:
                return builder.fptosi(value, dst_llvm, name="fptosi")
            else:
                return builder.fptoui(value, dst_llvm, name="fptoui")
        
        # Int to int
        src_bits = src_llvm.width
        dst_bits = dst_llvm.width
        
        if src_bits < dst_bits:
            if src_is_signed:
                return builder.sext(value, dst_llvm, name="sext")
            else:
                return builder.zext(value, dst_llvm, name="zext")
        elif src_bits > dst_bits:
            return builder.trunc(value, dst_llvm, name="trunc")
        else:
            # Same size, might be signed/unsigned conversion - no LLVM op needed
            return value

    def emit_AstIdent(self, node: AstIdent, state: CompileState) -> ir.Value:
        """Emit a variable reference (load)."""
        sym = state.resolved_symbols.get(node)
        
        if is_instance_compat(sym, VariableSymbol):
            # Get the alloca for this variable - check local first, then global
            alloca = self.llvm_state.variables.get(sym.name)
            if alloca is None:
                # Check global variables
                global_var = self.llvm_state.global_variables.get(sym.name)
                if global_var is None:
                    raise RuntimeError(f"Variable '{sym.name}' not found")
                # Load from global variable
                return self.llvm_state.builder.load(global_var, name=sym.name)
            # Load the value
            return self.llvm_state.builder.load(alloca, name=sym.name)
        elif is_instance_compat(sym, SymbolTable):
            # This is a namespace reference (e.g., "Svc" in "Svc.DpRecord")
            # Return a placeholder - the real value comes from the attribute access
            return ir.Constant(ir.IntType(64), 0)
        elif is_instance_compat(sym, TypeCtorSymbol):
            # Type constructor being referenced as a value (e.g., for type annotation resolution)
            return ir.Constant(ir.IntType(64), 0)
        elif is_instance_compat(sym, CastSymbol):
            # Cast symbol - return placeholder
            return ir.Constant(ir.IntType(64), 0)
        elif sym is None:
            # Unresolved symbol - might be a constant or enum
            return ir.Constant(ir.IntType(64), 0)
        else:
            raise NotImplementedError(f"Symbol type {type(sym)} not supported")
    
    def emit_AstAssign(self, node: AstAssign, state: CompileState) -> ir.Value | None:
        """Emit an assignment statement."""
        # Emit the RHS expression first
        rhs_value = self.emit(node.rhs, state)
        
        # Check if this is an array element assignment (arr[i] = x)
        if is_instance_compat(node.lhs, AstIndexExpr):
            return self._emit_array_element_assign(node, rhs_value, state)
        
        # Get the symbol for the LHS
        assert is_instance_compat(node.lhs, AstIdent), "Only simple variable assignments supported"
        
        sym = state.resolved_symbols.get(node.lhs)
        assert is_instance_compat(sym, VariableSymbol), f"Expected variable, got {type(sym)}"
        
        # Check if this is a new variable declaration or assignment to existing
        if node.type_ann is not None:
            # New variable declaration
            var_type = state.contextual_types.get(node.lhs)
            if var_type is None:
                # Try synthesized type
                var_type = state.synthesized_types.get(node.lhs)
            if var_type is None:
                # Try to get from the symbol's type
                var_type = sym.fpy_type if hasattr(sym, 'fpy_type') else None
            if var_type is None:
                # Fall back to inferring from RHS
                var_type = state.contextual_types.get(node.rhs) or state.synthesized_types.get(node.rhs)
            if var_type is None:
                # Ultimate fallback - use the LLVM type from the RHS value
                llvm_type = rhs_value.type
            else:
                # Allow complex types (structs, arrays) - proper LLVM types are now generated
                llvm_type = fpy_type_to_llvm(var_type)
            
            # If we're at the top level (not inside a user-defined function),
            # create a global variable so it can be accessed from functions
            if not self.llvm_state.in_function:
                # Create global variable
                global_var = ir.GlobalVariable(self.llvm_state.module, llvm_type, name=f"global.{sym.name}")
                global_var.linkage = "internal"
                # Initialize to zero/undef
                if isinstance(llvm_type, (ir.IntType, ir.FloatType, ir.DoubleType)):
                    global_var.initializer = ir.Constant(llvm_type, 0)
                elif isinstance(llvm_type, ir.LiteralStructType):
                    global_var.initializer = ir.Constant(llvm_type, [ir.Constant(t, 0) for t in llvm_type.elements])
                elif isinstance(llvm_type, ir.ArrayType):
                    global_var.initializer = ir.Constant(llvm_type, [ir.Constant(llvm_type.element, 0)] * llvm_type.count)
                else:
                    global_var.initializer = ir.Constant(llvm_type, None)
                
                self.llvm_state.global_variables[sym.name] = global_var
                # Store the initial value - convert type if needed
                if rhs_value.type != llvm_type:
                    if isinstance(llvm_type, (ir.LiteralStructType, ir.ArrayType)):
                        pass  # Complex types should already match
                    else:
                        rhs_value = self.convert_value_types(rhs_value, llvm_type)
                self.llvm_state.builder.store(rhs_value, global_var)
                return None
            else:
                # Inside a function - create local alloca
                alloca = self.create_entry_block_alloca(llvm_type, sym.name)
                self.llvm_state.variables[sym.name] = alloca
        else:
            # Assignment to existing variable - check local first, then global
            alloca = self.llvm_state.variables.get(sym.name)
            if alloca is None:
                # Check global variables
                global_var = self.llvm_state.global_variables.get(sym.name)
                if global_var is not None:
                    # Convert RHS to match the global's type if needed
                    global_pointee = global_var.type.pointee
                    if rhs_value.type != global_pointee:
                        if isinstance(global_pointee, (ir.LiteralStructType, ir.ArrayType)):
                            pass  # Complex types should already match
                        else:
                            rhs_value = self.convert_value_types(rhs_value, global_pointee)
                    self.llvm_state.builder.store(rhs_value, global_var)
                    return None
                raise RuntimeError(f"Variable '{sym.name}' not declared")
        
        # Convert RHS to match the variable's type if needed
        # Get the pointee type from the alloca
        alloca_pointee = alloca.type.pointee
        if rhs_value.type != alloca_pointee:
            # For struct/array types, types should match; for primitives, convert
            if isinstance(alloca_pointee, (ir.LiteralStructType, ir.ArrayType)):
                # Complex types should already match
                pass
            else:
                rhs_value = self.convert_value_types(rhs_value, alloca_pointee)
        
        # Store the value
        self.llvm_state.builder.store(rhs_value, alloca)
        return None
    
    def _emit_array_element_assign(self, node: AstAssign, rhs_value: ir.Value, state: CompileState) -> ir.Value | None:
        """Emit an array element assignment (arr[i] = x).
        
        This stores a value into an array element, with runtime bounds checking.
        """
        builder = self.llvm_state.builder
        func = builder.function
        
        lhs = node.lhs  # AstIndexExpr
        assert is_instance_compat(lhs, AstIndexExpr)
        
        # Get the parent array type and variable
        parent_type = state.contextual_types.get(lhs.parent) or state.synthesized_types.get(lhs.parent)
        assert parent_type is not None and issubclass(parent_type, ArrayValue), \
            f"Expected array type, got {parent_type}"
        
        # Get the array variable's alloca or global - the parent should be an AstIdent
        assert is_instance_compat(lhs.parent, AstIdent), "Array element assignment only works on variables"
        parent_sym = state.resolved_symbols.get(lhs.parent)
        assert is_instance_compat(parent_sym, VariableSymbol), f"Expected variable, got {type(parent_sym)}"
        
        # Check local variables first, then global
        array_ptr = self.llvm_state.variables.get(parent_sym.name)
        if array_ptr is None:
            array_ptr = self.llvm_state.global_variables.get(parent_sym.name)
        if array_ptr is None:
            raise RuntimeError(f"Variable '{parent_sym.name}' not found")
        
        # Emit the index expression
        index_value = self.emit(lhs.item, state)
        
        # Get array length for bounds check
        array_length = parent_type.LENGTH
        
        # Check if we have a constant index (can skip runtime bounds check if already verified)
        index_const = state.contextual_values.get(lhs.item)
        
        if index_const is not None:
            # Constant index - bounds checking should have been done at compile time
            # Use extractvalue/insertvalue semantics (but we need to load, modify, store)
            idx = int(index_const._val)
            
            # Load current array value
            array_value = builder.load(array_ptr, name="array")
            
            # Convert RHS to match element type if needed
            element_type = parent_type.MEMBER_TYPE
            element_llvm_type = fpy_type_to_llvm(element_type)
            if rhs_value.type != element_llvm_type:
                rhs_value = self.convert_value_types(rhs_value, element_llvm_type)
            
            # Insert the new element value
            new_array_value = builder.insert_value(array_value, rhs_value, idx, name="array.updated")
            
            # Store back
            builder.store(new_array_value, array_ptr)
        else:
            # Dynamic index - need runtime bounds check
            # Convert index to i64 for comparison
            idx64 = index_value
            if index_value.type != ir.IntType(64):
                if index_value.type.width < 64:
                    idx64 = builder.sext(index_value, ir.IntType(64), name="idx64")
                else:
                    idx64 = builder.trunc(index_value, ir.IntType(64), name="idx64")
            
            # Create bounds check blocks
            oob_block = func.append_basic_block(name="array.oob")
            ok_block = func.append_basic_block(name="array.ok")
            
            # Check: 0 <= index < length
            length_const = ir.Constant(ir.IntType(64), array_length)
            zero_const = ir.Constant(ir.IntType(64), 0)
            
            # index >= 0
            ge_zero = builder.icmp_signed(">=", idx64, zero_const, name="ge_zero")
            # index < length
            lt_len = builder.icmp_signed("<", idx64, length_const, name="lt_len")
            # Combined
            in_bounds = builder.and_(ge_zero, lt_len, name="in_bounds")
            
            builder.cbranch(in_bounds, ok_block, oob_block)
            
            # OOB block - return error
            builder.position_at_end(oob_block)
            expected_return_type = func.function_type.return_type
            if isinstance(expected_return_type, ir.IntType):
                exit_code = ir.Constant(expected_return_type, 1)
            elif isinstance(expected_return_type, (ir.FloatType, ir.DoubleType)):
                exit_code = ir.Constant(expected_return_type, 1.0)
            else:
                exit_code = ir.Constant(ir.IntType(32), 1)
            builder.ret(exit_code)
            
            # OK block - perform the assignment
            builder.position_at_end(ok_block)
            
            # Convert index to i32 for GEP
            if index_value.type != ir.IntType(32):
                if index_value.type.width > 32:
                    idx32 = builder.trunc(index_value, ir.IntType(32), name="idx32")
                else:
                    idx32 = builder.zext(index_value, ir.IntType(32), name="idx32")
            else:
                idx32 = index_value
            
            # Use GEP to get pointer to the element
            element_ptr = builder.gep(array_ptr, [
                ir.Constant(ir.IntType(32), 0),
                idx32
            ], inbounds=True, name="elem_ptr")
            
            # Convert RHS to match element type if needed
            element_type = parent_type.MEMBER_TYPE
            element_llvm_type = fpy_type_to_llvm(element_type)
            if rhs_value.type != element_llvm_type:
                rhs_value = self.convert_value_types(rhs_value, element_llvm_type)
            
            # Store the value
            builder.store(rhs_value, element_ptr)
        
        return None
    
    def _is_complex_type(self, typ: FppType) -> bool:
        """Check if a type is a complex type (struct or array)."""
        if typ is None:
            return False
        return issubclass(typ, (StructValue, ArrayValue))
    
    def _compare_complex_values(self, lhs: ir.Value, rhs: ir.Value, lhs_type: FppType, op: str) -> ir.Value:
        """Compare two struct or array values for equality/inequality."""
        builder = self.llvm_state.builder
        
        # For struct/array comparison, we need to compare element by element
        if issubclass(lhs_type, StructValue):
            # Compare all struct members
            result = ir.Constant(ir.IntType(1), 1)  # Start with true (all equal)
            for i, (member_name, member_type, _, _) in enumerate(lhs_type.MEMBER_LIST):
                lhs_member = builder.extract_value(lhs, i, name=f"lhs.{member_name}")
                rhs_member = builder.extract_value(rhs, i, name=f"rhs.{member_name}")
                
                if self._is_complex_type(member_type):
                    # Recursive comparison for nested complex types
                    member_eq = self._compare_complex_values(lhs_member, rhs_member, member_type, "==")
                elif is_float_type(member_type):
                    member_eq = builder.fcmp_ordered("==", lhs_member, rhs_member, name=f"eq.{member_name}")
                else:
                    member_eq = builder.icmp_signed("==", lhs_member, rhs_member, name=f"eq.{member_name}")
                
                result = builder.and_(result, member_eq, name="and_eq")
            
            if op == "!=":
                result = builder.xor(result, ir.Constant(ir.IntType(1), 1), name="ne")
            return result
        
        elif issubclass(lhs_type, ArrayValue):
            # Compare all array elements
            result = ir.Constant(ir.IntType(1), 1)  # Start with true (all equal)
            for i in range(lhs_type.LENGTH):
                lhs_elem = builder.extract_value(lhs, i, name=f"lhs.{i}")
                rhs_elem = builder.extract_value(rhs, i, name=f"rhs.{i}")
                
                if self._is_complex_type(lhs_type.MEMBER_TYPE):
                    # Recursive comparison for nested complex types
                    elem_eq = self._compare_complex_values(lhs_elem, rhs_elem, lhs_type.MEMBER_TYPE, "==")
                elif is_float_type(lhs_type.MEMBER_TYPE):
                    elem_eq = builder.fcmp_ordered("==", lhs_elem, rhs_elem, name=f"eq.{i}")
                else:
                    elem_eq = builder.icmp_signed("==", lhs_elem, rhs_elem, name=f"eq.{i}")
                
                result = builder.and_(result, elem_eq, name="and_eq")
            
            if op == "!=":
                result = builder.xor(result, ir.Constant(ir.IntType(1), 1), name="ne")
            return result
        
        else:
            # Fallback for unknown complex types
            return ir.Constant(ir.IntType(1), 0)
    
    def emit_AstBinaryOp(self, node: AstBinaryOp, state: CompileState) -> ir.Value:
        """Emit a binary operation."""
        builder = self.llvm_state.builder
        
        lhs = self.emit(node.lhs, state)
        rhs = self.emit(node.rhs, state)
        
        # Get the types of the operands for signedness
        lhs_type = state.synthesized_types.get(node.lhs)
        rhs_type = state.synthesized_types.get(node.rhs)
        
        # Get the intermediate type for the operation
        intermediate_type = state.op_intermediate_types.get(node)
        result_type = state.contextual_types[node]
        
        op = node.op
        
        # Handle complex type comparisons (struct/array)
        if op in ("==", "!=") and self._is_complex_type(lhs_type):
            return self._compare_complex_values(lhs, rhs, lhs_type, op)
        
        # Coerce types to match intermediate type if needed
        if intermediate_type:
            target_type = fpy_type_to_llvm(intermediate_type)
            intermediate_is_signed = is_signed_type(intermediate_type)
            if lhs.type != target_type:
                lhs_is_signed = is_signed_type(lhs_type) if lhs_type else intermediate_is_signed
                lhs = self.convert_value_types(lhs, target_type, lhs_is_signed)
            if rhs.type != target_type:
                rhs_is_signed = is_signed_type(rhs_type) if rhs_type else intermediate_is_signed
                rhs = self.convert_value_types(rhs, target_type, rhs_is_signed)
        elif lhs.type != rhs.type:
            # Fallback: coerce to match each other
            target_type = lhs.type
            rhs_is_signed = is_signed_type(rhs_type) if rhs_type else False
            rhs = self.convert_value_types(rhs, target_type, rhs_is_signed)
        
        # Comparison operations
        if op == "==":
            if is_float_type(intermediate_type):
                return builder.fcmp_ordered("==", lhs, rhs, name="eq")
            else:
                return builder.icmp_signed("==", lhs, rhs, name="eq")
        elif op == "!=":
            if is_float_type(intermediate_type):
                return builder.fcmp_ordered("!=", lhs, rhs, name="ne")
            else:
                return builder.icmp_signed("!=", lhs, rhs, name="ne")
        elif op == "<":
            if is_float_type(intermediate_type):
                return builder.fcmp_ordered("<", lhs, rhs, name="lt")
            elif is_signed_type(intermediate_type):
                return builder.icmp_signed("<", lhs, rhs, name="lt")
            else:
                return builder.icmp_unsigned("<", lhs, rhs, name="lt")
        elif op == "<=":
            if is_float_type(intermediate_type):
                return builder.fcmp_ordered("<=", lhs, rhs, name="le")
            elif is_signed_type(intermediate_type):
                return builder.icmp_signed("<=", lhs, rhs, name="le")
            else:
                return builder.icmp_unsigned("<=", lhs, rhs, name="le")
        elif op == ">":
            if is_float_type(intermediate_type):
                return builder.fcmp_ordered(">", lhs, rhs, name="gt")
            elif is_signed_type(intermediate_type):
                return builder.icmp_signed(">", lhs, rhs, name="gt")
            else:
                return builder.icmp_unsigned(">", lhs, rhs, name="gt")
        elif op == ">=":
            if is_float_type(intermediate_type):
                return builder.fcmp_ordered(">=", lhs, rhs, name="ge")
            elif is_signed_type(intermediate_type):
                return builder.icmp_signed(">=", lhs, rhs, name="ge")
            else:
                return builder.icmp_unsigned(">=", lhs, rhs, name="ge")
        
        # Arithmetic operations
        elif op == "+":
            if is_float_type(intermediate_type):
                return builder.fadd(lhs, rhs, name="add")
            else:
                return builder.add(lhs, rhs, name="add")
        elif op == "-":
            if is_float_type(intermediate_type):
                return builder.fsub(lhs, rhs, name="sub")
            else:
                return builder.sub(lhs, rhs, name="sub")
        elif op == "*":
            if is_float_type(intermediate_type):
                return builder.fmul(lhs, rhs, name="mul")
            else:
                return builder.mul(lhs, rhs, name="mul")
        elif op == "/":
            if is_float_type(intermediate_type):
                return builder.fdiv(lhs, rhs, name="div")
            elif is_signed_type(intermediate_type):
                return builder.sdiv(lhs, rhs, name="div")
            else:
                return builder.udiv(lhs, rhs, name="div")
        
        # Boolean operations
        elif op == "and":
            return builder.and_(lhs, rhs, name="and")
        elif op == "or":
            return builder.or_(lhs, rhs, name="or")
        
        # Modulo (Python semantics: result has same sign as divisor)
        elif op == "%":
            if is_float_type(intermediate_type):
                return builder.frem(lhs, rhs, name="mod")
            elif is_signed_type(intermediate_type):
                # Python-style modulo: a % b has same sign as b
                # Formula: a % b = a - b * floor(a / b)
                # Or: if srem != 0 and signs of a and b differ, add b to srem result
                c_mod = builder.srem(lhs, rhs, name="cmod")
                
                # Check if result is non-zero and signs differ
                zero = ir.Constant(lhs.type, 0)
                mod_is_nonzero = builder.icmp_signed("!=", c_mod, zero, name="mod_nz")
                lhs_is_neg = builder.icmp_signed("<", lhs, zero, name="lhs_neg")
                rhs_is_neg = builder.icmp_signed("<", rhs, zero, name="rhs_neg")
                signs_differ = builder.xor(lhs_is_neg, rhs_is_neg, name="signs_diff")
                needs_adjust = builder.and_(mod_is_nonzero, signs_differ, name="needs_adj")
                
                # If needs adjustment, add rhs to c_mod
                adjusted = builder.add(c_mod, rhs, name="adjusted")
                return builder.select(needs_adjust, adjusted, c_mod, name="pymod")
            else:
                return builder.urem(lhs, rhs, name="mod")
        
        # Floor division
        elif op == "//":
            if is_float_type(intermediate_type):
                # For float floor division: divide then floor
                div_result = builder.fdiv(lhs, rhs, name="div")
                # Call floor intrinsic - for now just truncate to int and back
                int_result = builder.fptosi(div_result, ir.IntType(64), name="toint")
                return builder.sitofp(int_result, fpy_type_to_llvm(intermediate_type), name="tofloat")
            elif is_signed_type(intermediate_type):
                return builder.sdiv(lhs, rhs, name="floordiv")
            else:
                return builder.udiv(lhs, rhs, name="floordiv")
        
        # Power operator
        elif op == "**":
            # llvm.pow requires both arguments to be the same float type
            # Always convert to double for safety and precision
            module = self.llvm_state.module
            pow_func = module.declare_intrinsic('llvm.pow', [ir.DoubleType()])
            
            # Get original operand types to determine signedness
            # Use synthesized_types first since contextual_types may be promoted
            lhs_type = state.synthesized_types.get(node.lhs) or state.contextual_types.get(node.lhs)
            rhs_type = state.synthesized_types.get(node.rhs) or state.contextual_types.get(node.rhs)
            lhs_is_signed = is_signed_type(lhs_type) if lhs_type else True  # Default to signed
            rhs_is_signed = is_signed_type(rhs_type) if rhs_type else True
            
            # Convert lhs to double
            if isinstance(lhs.type, ir.IntType):
                float_lhs = builder.sitofp(lhs, ir.DoubleType()) if lhs_is_signed else builder.uitofp(lhs, ir.DoubleType())
            elif isinstance(lhs.type, ir.FloatType):
                float_lhs = builder.fpext(lhs, ir.DoubleType())
            else:
                float_lhs = lhs  # Already double
            
            # Convert rhs to double
            if isinstance(rhs.type, ir.IntType):
                float_rhs = builder.sitofp(rhs, ir.DoubleType()) if rhs_is_signed else builder.uitofp(rhs, ir.DoubleType())
            elif isinstance(rhs.type, ir.FloatType):
                float_rhs = builder.fpext(rhs, ir.DoubleType())
            else:
                float_rhs = rhs  # Already double
            
            float_result = builder.call(pow_func, [float_lhs, float_rhs], name="pow")
            
            # Convert result back to the expected type
            if is_float_type(intermediate_type):
                result_llvm_type = fpy_type_to_llvm(intermediate_type)
                if isinstance(result_llvm_type, ir.FloatType):
                    return builder.fptrunc(float_result, result_llvm_type)
                else:
                    return float_result
            else:
                # Integer power - convert back to int
                result_type = lhs.type
                if lhs_is_signed:
                    return builder.fptosi(float_result, result_type, name="pow_int")
                else:
                    return builder.fptoui(float_result, result_type, name="pow_int")
        
        else:
            raise NotImplementedError(f"Binary operator '{op}' not implemented")
    
    def emit_AstUnaryOp(self, node: AstUnaryOp, state: CompileState) -> ir.Value:
        """Emit a unary operation."""
        builder = self.llvm_state.builder
        val = self.emit(node.val, state)
        val_type = state.contextual_types[node.val]
        
        op = node.op
        
        if op == "not":
            # Boolean not - XOR with 1
            return builder.xor(val, ir.Constant(ir.IntType(1), 1), name="not")
        elif op == "-":
            if is_float_type(val_type):
                return builder.fneg(val, name="neg")
            else:
                # Use the actual LLVM type of the value, not the contextual type
                zero = ir.Constant(val.type, 0)
                return builder.sub(zero, val, name="neg")
        elif op == "+":
            # Unary plus is a no-op
            return val
        else:
            raise NotImplementedError(f"Unary operator '{op}' not implemented")
    
    def emit_AstString(self, node: AstString, state: CompileState) -> ir.Value:
        """Emit a string literal.
        
        For now, strings are not fully supported in LLVM backend.
        We just return a null pointer placeholder.
        """
        # Strings need more complex handling (global constants, etc.)
        # For now, return a placeholder
        return ir.Constant(ir.IntType(8).as_pointer(), None)
    
    def emit_AstGetAttr(self, node: AstGetAttr, state: CompileState) -> ir.Value:
        """Emit an attribute access (e.g., struct.field, tlm.value).
        
        Handles telemetry access, struct member access, and namespace references.
        """
        builder = self.llvm_state.builder
        
        # Check if this is resolved to a FieldAccess or other symbol
        resolved = state.resolved_symbols.get(node)
        
        # Check if this is telemetry access (ChTemplate)
        if is_instance_compat(resolved, ChTemplate):
            # Telemetry channel access
            chan_name = resolved.get_full_name()
            # Use the channel's native type for deserialization
            channel_type = resolved.get_type_obj()
            # Get the target type for the result (may be different due to type coercion)
            result_type = state.contextual_types.get(node) or state.synthesized_types.get(node)
            
            # Look up the value in the telemetry database
            tlm_bytes = self.llvm_state.tlm_db.get(chan_name)
            if tlm_bytes is not None:
                # Handle complex types (struct/array)
                if issubclass(channel_type, StructValue):
                    return self._deserialize_tlm_struct(tlm_bytes, channel_type)
                elif issubclass(channel_type, ArrayValue):
                    return self._deserialize_tlm_array(tlm_bytes, channel_type)
                else:
                    # Primitive type - deserialize using the channel's native type
                    value = self._deserialize_tlm_value(tlm_bytes, channel_type)
                    # Return constant with the result type (may be coerced)
                    llvm_type = fpy_type_to_llvm(result_type)
                    if isinstance(llvm_type, ir.IntType):
                        return ir.Constant(llvm_type, int(value))
                    elif isinstance(llvm_type, (ir.FloatType, ir.DoubleType)):
                        return ir.Constant(llvm_type, float(value))
            
            # No telemetry value available - return placeholder
            try:
                llvm_type = fpy_type_to_llvm(result_type)
                if isinstance(llvm_type, ir.IntType):
                    return ir.Constant(llvm_type, 0)
                elif isinstance(llvm_type, (ir.FloatType, ir.DoubleType)):
                    return ir.Constant(llvm_type, 0.0)
                elif isinstance(llvm_type, ir.LiteralStructType):
                    # Return an undefined struct value 
                    return ir.Constant(llvm_type, ir.Undefined)
            except (NotImplementedError, TypeError):
                pass
            return ir.Constant(ir.IntType(64), 0)
        
        if is_instance_compat(resolved, FieldAccess):
            # Struct member access
            if resolved.is_struct_member:
                # Get the parent struct value
                parent_value = self.emit(resolved.parent_expr, state)
                parent_type = state.contextual_types.get(resolved.parent_expr) or state.synthesized_types.get(resolved.parent_expr)
                
                # Get the member index
                member_idx = get_struct_member_index(parent_type, resolved.name)
                assert member_idx is not None, f"Member {resolved.name} not found in {parent_type}"
                
                # Use extractvalue to get the struct member
                member_value = builder.extract_value(parent_value, member_idx, name=f"get.{resolved.name}")
                
                # Convert type if needed
                result_type = state.contextual_types.get(node) or state.synthesized_types.get(node)
                field_type = resolved.type
                if result_type != field_type:
                    member_value = self.convert_value(member_value, field_type, result_type)
                
                return member_value
            elif resolved.is_array_element:
                # This shouldn't happen for GetAttr - array access goes through IndexExpr
                assert False, "Array element access via GetAttr is not supported"
            else:
                # Generic field access - try to handle
                field_type = resolved.type
                try:
                    llvm_type = fpy_type_to_llvm(field_type)
                    if isinstance(llvm_type, ir.IntType):
                        return ir.Constant(llvm_type, 0)
                    elif isinstance(llvm_type, (ir.FloatType, ir.DoubleType)):
                        return ir.Constant(llvm_type, 0.0)
                    else:
                        return ir.Constant(ir.IntType(64), 0)
                except NotImplementedError:
                    return ir.Constant(ir.IntType(64), 0)
        
        # Could be namespace reference or other qualified name
        # For namespaces (SymbolTable), just return a placeholder
        if is_instance_compat(resolved, SymbolTable):
            return ir.Constant(ir.IntType(64), 0)
        
        # Emit the parent and return a placeholder
        self.emit(node.parent, state)
        
        # Try to get the result type
        result_type = state.contextual_types.get(node) or state.synthesized_types.get(node)
        if result_type is not None:
            try:
                llvm_type = fpy_type_to_llvm(result_type)
                if isinstance(llvm_type, ir.IntType):
                    return ir.Constant(llvm_type, 0)
                elif isinstance(llvm_type, (ir.FloatType, ir.DoubleType)):
                    return ir.Constant(llvm_type, 0.0)
            except NotImplementedError:
                pass
        
        return ir.Constant(ir.IntType(64), 0)
    
    def _deserialize_tlm_struct(self, data: bytes, typ: FppType) -> ir.Value:
        """Deserialize telemetry bytes to a struct LLVM value."""
        builder = self.llvm_state.builder
        llvm_type = get_llvm_struct_type(typ)
        
        # Create an alloca for the struct
        struct_alloca = self.create_entry_block_alloca(llvm_type, "tlm_struct")
        
        # Deserialize each member
        offset = 0
        for i, (member_name, member_type, _, _) in enumerate(typ.MEMBER_LIST):
            member_size = member_type.getMaxSize()
            member_data = data[offset:offset + member_size]
            
            if issubclass(member_type, StructValue):
                # Nested struct - recursively deserialize
                member_value = self._deserialize_tlm_struct(member_data, member_type)
            elif issubclass(member_type, ArrayValue):
                # Array - deserialize element by element
                member_value = self._deserialize_tlm_array(member_data, member_type)
            else:
                # Primitive type
                numeric_value = self._deserialize_tlm_value(member_data, member_type)
                member_llvm_type = fpy_type_to_llvm(member_type)
                if isinstance(member_llvm_type, ir.IntType):
                    member_value = ir.Constant(member_llvm_type, int(numeric_value))
                elif isinstance(member_llvm_type, (ir.FloatType, ir.DoubleType)):
                    member_value = ir.Constant(member_llvm_type, float(numeric_value))
                else:
                    member_value = ir.Constant(ir.IntType(64), 0)
            
            # Store to struct member
            member_ptr = builder.gep(struct_alloca, [
                ir.Constant(ir.IntType(32), 0),
                ir.Constant(ir.IntType(32), i)
            ], inbounds=True)
            builder.store(member_value, member_ptr)
            
            offset += member_size
        
        # Load and return the struct
        return builder.load(struct_alloca, name="tlm_struct_val")
    
    def _deserialize_tlm_array(self, data: bytes, typ: FppType) -> ir.Value:
        """Deserialize telemetry bytes to an array LLVM value."""
        builder = self.llvm_state.builder
        llvm_type = get_llvm_array_type(typ)
        
        # Create an alloca for the array
        array_alloca = self.create_entry_block_alloca(llvm_type, "tlm_array")
        
        element_size = typ.MEMBER_TYPE.getMaxSize()
        for i in range(typ.LENGTH):
            elem_data = data[i * element_size:(i + 1) * element_size]
            
            if issubclass(typ.MEMBER_TYPE, StructValue):
                elem_value = self._deserialize_tlm_struct(elem_data, typ.MEMBER_TYPE)
            elif issubclass(typ.MEMBER_TYPE, ArrayValue):
                elem_value = self._deserialize_tlm_array(elem_data, typ.MEMBER_TYPE)
            else:
                numeric_value = self._deserialize_tlm_value(elem_data, typ.MEMBER_TYPE)
                elem_llvm_type = fpy_type_to_llvm(typ.MEMBER_TYPE)
                if isinstance(elem_llvm_type, ir.IntType):
                    elem_value = ir.Constant(elem_llvm_type, int(numeric_value))
                elif isinstance(elem_llvm_type, (ir.FloatType, ir.DoubleType)):
                    elem_value = ir.Constant(elem_llvm_type, float(numeric_value))
                else:
                    elem_value = ir.Constant(ir.IntType(64), 0)
            
            # Store to array element
            elem_ptr = builder.gep(array_alloca, [
                ir.Constant(ir.IntType(32), 0),
                ir.Constant(ir.IntType(32), i)
            ], inbounds=True)
            builder.store(elem_value, elem_ptr)
        
        # Load and return the array
        return builder.load(array_alloca, name="tlm_array_val")
    
    def _deserialize_tlm_value(self, data: bytes, typ: FppType) -> int | float:
        """Deserialize telemetry bytes to a numeric value."""
        import struct
        
        if typ == BoolValue:
            return 1 if data[0] != 0 else 0
        elif typ == U8Value:
            return struct.unpack('>B', data[:1])[0]
        elif typ == I8Value:
            return struct.unpack('>b', data[:1])[0]
        elif typ == U16Value:
            return struct.unpack('>H', data[:2])[0]
        elif typ == I16Value:
            return struct.unpack('>h', data[:2])[0]
        elif typ == U32Value:
            return struct.unpack('>I', data[:4])[0]
        elif typ == I32Value:
            return struct.unpack('>i', data[:4])[0]
        elif typ == U64Value:
            return struct.unpack('>Q', data[:8])[0]
        elif typ == I64Value:
            return struct.unpack('>q', data[:8])[0]
        elif typ == F32Value:
            return struct.unpack('>f', data[:4])[0]
        elif typ == F64Value:
            return struct.unpack('>d', data[:8])[0]
        elif issubclass(typ, EnumValue):
            # Enum values - deserialize based on their actual size (usually 4 bytes)
            enum_size = typ.getMaxSize()
            if enum_size == 1:
                return struct.unpack('>B', data[:1])[0]
            elif enum_size == 2:
                return struct.unpack('>H', data[:2])[0]
            elif enum_size == 4:
                return struct.unpack('>I', data[:4])[0]
            else:
                return struct.unpack('>Q', data[:8])[0]
        else:
            # Unknown type - return 0
            return 0
    
    def emit_AstIndexExpr(self, node: AstIndexExpr, state: CompileState) -> ir.Value:
        """Emit an array index expression (e.g., array[i]).
        
        Handles array element access by computing the correct element from the array value.
        """
        builder = self.llvm_state.builder
        
        # Get the parent array type
        parent_type = state.contextual_types.get(node.parent) or state.synthesized_types.get(node.parent)
        
        # Emit the parent array expression
        array_value = self.emit(node.parent, state)
        
        # Emit the index expression
        index_value = self.emit(node.item, state)
        
        # Get the result (element) type
        result_type = state.contextual_types.get(node) or state.synthesized_types.get(node)
        
        # Check if we can use a constant index (simpler code generation)
        index_const = state.contextual_values.get(node.item)
        
        if index_const is not None:
            # Constant index - use extractvalue
            idx = int(index_const._val)
            element_value = builder.extract_value(array_value, idx, name=f"array.{idx}")
        else:
            # Dynamic index - need to store array to memory and use GEP
            # Create a temporary alloca for the array
            array_type = array_value.type
            array_alloca = self.create_entry_block_alloca(array_type, "array_tmp")
            builder.store(array_value, array_alloca)
            
            # Convert index to i32 for GEP (LLVM requires i32/i64 for array indices)
            if index_value.type != ir.IntType(32):
                index_value = builder.trunc(index_value, ir.IntType(32), name="idx32") if index_value.type.width > 32 else builder.zext(index_value, ir.IntType(32), name="idx32")
            
            # Use GEP to get pointer to the element
            element_ptr = builder.gep(array_alloca, [
                ir.Constant(ir.IntType(32), 0),
                index_value
            ], inbounds=True, name="elem_ptr")
            
            # Load the element
            element_value = builder.load(element_ptr, name="elem")
        
        # Convert type if needed
        if parent_type is not None and issubclass(parent_type, ArrayValue):
            element_type = parent_type.MEMBER_TYPE
            if result_type != element_type:
                element_value = self.convert_value(element_value, element_type, result_type)
        
        return element_value
    
    def emit_AstWhile(self, node: AstWhile, state: CompileState) -> ir.Value | None:
        """Emit a while loop.
        
        For while loops that were desugared from for loops, we need special handling:
        - The last statement in the body is the loop variable increment
        - `continue` should jump to the increment block, not the condition
        """
        builder = self.llvm_state.builder
        func = builder.function
        
        # Check if this is a desugared for loop
        is_for_loop = node in state.desugared_for_loops
        
        # Create basic blocks
        cond_block = func.append_basic_block(name="while.cond")
        body_block = func.append_basic_block(name="while.body")
        end_block = func.append_basic_block(name="while.end")
        
        # For desugared for loops, create an increment block
        if is_for_loop and len(node.body.stmts) > 0:
            inc_block = func.append_basic_block(name="while.inc")
            # Store loop blocks for break/continue (continue goes to inc_block)
            self.llvm_state.loop_stack.append((inc_block, end_block))
        else:
            inc_block = None
            # Store loop blocks for break/continue (continue goes to cond_block)
            self.llvm_state.loop_stack.append((cond_block, end_block))
        
        # Jump to condition check
        builder.branch(cond_block)
        
        # Condition block
        builder.position_at_end(cond_block)
        cond = self.emit(node.condition, state)
        builder.cbranch(cond, body_block, end_block)
        
        # Body block
        builder.position_at_end(body_block)
        
        if is_for_loop and inc_block is not None:
            # Emit all statements except the last one (the increment)
            for stmt in node.body.stmts[:-1]:
                if builder.block.is_terminated:
                    break
                self.emit(stmt, state)
            
            # Jump to increment block if not terminated
            if not builder.block.is_terminated:
                builder.branch(inc_block)
            
            # Increment block - emit the last statement (the increment)
            builder.position_at_end(inc_block)
            if len(node.body.stmts) > 0:
                self.emit(node.body.stmts[-1], state)
            if not builder.block.is_terminated:
                builder.branch(cond_block)
        else:
            # Regular while loop
            self.emit(node.body, state)
            if not builder.block.is_terminated:
                builder.branch(cond_block)
        
        # Pop loop from stack
        self.llvm_state.loop_stack.pop()
        
        # Continue after loop
        builder.position_at_end(end_block)
        return None
    
    def emit_AstBreak(self, node: AstBreak, state: CompileState) -> ir.Value | None:
        """Emit a break statement."""
        if not self.llvm_state.loop_stack:
            raise RuntimeError("break outside of loop")
        _, end_block = self.llvm_state.loop_stack[-1]
        self.llvm_state.builder.branch(end_block)
        return None
    
    def emit_AstContinue(self, node: AstContinue, state: CompileState) -> ir.Value | None:
        """Emit a continue statement."""
        if not self.llvm_state.loop_stack:
            raise RuntimeError("continue outside of loop")
        cond_block, _ = self.llvm_state.loop_stack[-1]
        self.llvm_state.builder.branch(cond_block)
        return None
    
    def emit_AstAssert(self, node: AstAssert, state: CompileState) -> ir.Value | None:
        """Emit an assert statement.
        
        Assert checks the condition and exits with error code if false.
        """
        builder = self.llvm_state.builder
        func = builder.function
        
        # Emit condition
        cond = self.emit(node.condition, state)
        
        # Create blocks
        fail_block = func.append_basic_block(name="assert.fail")
        cont_block = func.append_basic_block(name="assert.cont")
        
        # Branch based on condition
        builder.cbranch(cond, cont_block, fail_block)
        
        # Fail block - return error code
        builder.position_at_end(fail_block)
        
        # Get the expected return type of the current function
        expected_return_type = func.function_type.return_type
        
        if node.exit_code is not None:
            exit_code = self.emit(node.exit_code, state)
            # Convert to expected return type
            if exit_code.type != expected_return_type:
                exit_code = self.convert_value_types(exit_code, expected_return_type)
        else:
            # Default exit code - use 1 for int types, 0.0 for float types
            if isinstance(expected_return_type, ir.IntType):
                exit_code = ir.Constant(expected_return_type, 1)
            elif isinstance(expected_return_type, (ir.FloatType, ir.DoubleType)):
                exit_code = ir.Constant(expected_return_type, 1.0)
            else:
                exit_code = ir.Constant(ir.IntType(32), 1)
        builder.ret(exit_code)
        
        # Continue block
        builder.position_at_end(cont_block)
        return None

    def emit_AstIf(self, node: AstIf, state: CompileState) -> ir.Value | None:
        """Emit an if statement."""
        builder = self.llvm_state.builder
        func = builder.function
        
        # Emit the condition
        cond = self.emit(node.condition, state)
        
        # Create basic blocks
        then_block = func.append_basic_block(name="then")
        merge_block = func.append_basic_block(name="ifcont")
        
        # Handle elif chains first
        elif_blocks = []
        if len(node.elifs) > 0:
            for i, elif_ in enumerate(node.elifs):
                elif_blocks.append(func.append_basic_block(name=f"elif{i}"))
        
        # Create else block if needed
        else_block = None
        if node.els is not None:
            else_block = func.append_basic_block(name="else")
        
        # Determine what block to jump to if condition is false
        if len(elif_blocks) > 0:
            false_target = elif_blocks[0]
        elif else_block is not None:
            false_target = else_block
        else:
            false_target = merge_block
        
        builder.cbranch(cond, then_block, false_target)
        
        # Emit then block
        builder.position_at_end(then_block)
        self.emit(node.body, state)
        if not builder.block.is_terminated:
            builder.branch(merge_block)
        
        # Handle elif chains
        if len(node.elifs) > 0:
            for i, (elif_, elif_block) in enumerate(zip(node.elifs, elif_blocks)):
                builder.position_at_end(elif_block)
                elif_cond = self.emit(elif_.condition, state)
                
                elif_then = func.append_basic_block(name=f"elif{i}_then")
                if i + 1 < len(elif_blocks):
                    next_block = elif_blocks[i + 1]
                elif else_block is not None:
                    next_block = else_block
                else:
                    next_block = merge_block
                
                builder.cbranch(elif_cond, elif_then, next_block)
                
                builder.position_at_end(elif_then)
                self.emit(elif_.body, state)
                if not builder.block.is_terminated:
                    builder.branch(merge_block)
        
        # Emit else block if present
        if else_block is not None:
            builder.position_at_end(else_block)
            self.emit(node.els, state)
            if not builder.block.is_terminated:
                builder.branch(merge_block)
        
        # Continue at merge block
        builder.position_at_end(merge_block)
        return None
    
    def emit_AstDef(self, node: AstDef, state: CompileState) -> ir.Value | None:
        """Emit a function definition."""
        builder = self.llvm_state.builder
        module = self.llvm_state.module
        
        # Get the function name
        func_name = str(node.name.name)
        
        # Get return type
        if node.return_type is not None:
            return_type_sym = state.resolved_symbols.get(node.return_type)
            if return_type_sym is not None:
                llvm_return_type = fpy_type_to_llvm(return_type_sym)
            else:
                # Default to i32 if can't resolve
                llvm_return_type = ir.IntType(32)
        else:
            # No return type means void/nothing - but for simplicity use i32
            llvm_return_type = ir.IntType(32)
        
        # Get parameter types
        param_types = []
        param_names = []
        if node.parameters:
            for param_name, param_type, default_val in node.parameters:
                param_type_sym = state.resolved_symbols.get(param_type)
                if param_type_sym is not None:
                    param_types.append(fpy_type_to_llvm(param_type_sym))
                else:
                    param_types.append(ir.IntType(64))  # Default
                param_names.append(str(param_name.name))
        
        # Create function type and function
        func_type = ir.FunctionType(llvm_return_type, param_types)
        func = ir.Function(module, func_type, name=func_name)
        
        # Store function in state
        self.llvm_state.functions[func_name] = func
        
        # Save current builder state
        old_builder = self.llvm_state.builder
        old_function = self.llvm_state.current_function
        old_variables = self.llvm_state.variables.copy()
        old_in_function = self.llvm_state.in_function
        old_entry_block = self.llvm_state.entry_block
        
        # Create entry block for the function
        entry_block = func.append_basic_block(name="entry")
        new_builder = ir.IRBuilder(entry_block)
        
        # Set up new state for function body
        self.llvm_state.builder = new_builder
        self.llvm_state.current_function = func
        self.llvm_state.in_function = True
        self.llvm_state.entry_block = entry_block
        # Clear local variables but keep access to globals via global_variables dict
        self.llvm_state.variables = {}
        
        # Allocate space for parameters and store them
        for i, (param_name, arg) in enumerate(zip(param_names, func.args)):
            arg.name = param_name
            # Allocate stack space for the parameter
            alloca = new_builder.alloca(arg.type, name=param_name)
            new_builder.store(arg, alloca)
            self.llvm_state.variables[param_name] = alloca
        
        # Emit function body
        self.emit(node.body, state)
        
        # Add default return if needed
        if not new_builder.block.is_terminated:
            if isinstance(llvm_return_type, ir.IntType):
                new_builder.ret(ir.Constant(llvm_return_type, 0))
            elif isinstance(llvm_return_type, (ir.FloatType, ir.DoubleType)):
                new_builder.ret(ir.Constant(llvm_return_type, 0.0))
            else:
                new_builder.ret_void()
        
        # Restore old state
        self.llvm_state.builder = old_builder
        self.llvm_state.current_function = old_function
        self.llvm_state.variables = old_variables
        self.llvm_state.in_function = old_in_function
        self.llvm_state.entry_block = old_entry_block
        
        return None
    
    def emit_AstReturn(self, node: AstReturn, state: CompileState) -> ir.Value | None:
        """Emit a return statement."""
        builder = self.llvm_state.builder
        
        if node.value is not None:
            value = self.emit(node.value, state)
            # Get expected return type
            func = self.llvm_state.current_function
            expected_type = func.function_type.return_type
            # Convert if needed
            if value.type != expected_type:
                value = self.convert_value_types(value, expected_type)
            builder.ret(value)
        else:
            # Return without a value - if the function has a return type, return default
            func = self.llvm_state.current_function
            expected_type = func.function_type.return_type
            if isinstance(expected_type, ir.IntType):
                builder.ret(ir.Constant(expected_type, 0))
            elif isinstance(expected_type, (ir.FloatType, ir.DoubleType)):
                builder.ret(ir.Constant(expected_type, 0.0))
            elif isinstance(expected_type, ir.VoidType):
                builder.ret_void()
            else:
                # Default - try to return 0 as i32
                builder.ret(ir.Constant(ir.IntType(32), 0))
        
        return None
    
    def convert_value_types(self, value: ir.Value, target_type: ir.Type, is_signed: bool = False) -> ir.Value:
        """Convert LLVM value to target LLVM type.
        
        Args:
            value: The LLVM value to convert
            target_type: The target LLVM type
            is_signed: If True, use sign extension for int widening; if False, use zero extension
        """
        builder = self.llvm_state.builder
        
        if value.type == target_type:
            return value
        
        # Int to int conversions
        if isinstance(value.type, ir.IntType) and isinstance(target_type, ir.IntType):
            if value.type.width < target_type.width:
                if is_signed:
                    return builder.sext(value, target_type)
                else:
                    return builder.zext(value, target_type)
            else:
                return builder.trunc(value, target_type)
        
        # Float to float
        if isinstance(value.type, (ir.FloatType, ir.DoubleType)) and isinstance(target_type, (ir.FloatType, ir.DoubleType)):
            if isinstance(value.type, ir.FloatType) and isinstance(target_type, ir.DoubleType):
                return builder.fpext(value, target_type)
            else:
                return builder.fptrunc(value, target_type)
        
        # Int to float
        if isinstance(value.type, ir.IntType) and isinstance(target_type, (ir.FloatType, ir.DoubleType)):
            if is_signed:
                return builder.sitofp(value, target_type)
            else:
                return builder.uitofp(value, target_type)
        
        # Float to int
        if isinstance(value.type, (ir.FloatType, ir.DoubleType)) and isinstance(target_type, ir.IntType):
            if is_signed:
                return builder.fptosi(value, target_type)
            else:
                return builder.fptoui(value, target_type)
        
        return value
    
    def emit_AstFor(self, node: AstFor, state: CompileState) -> ir.Value | None:
        """Emit a for loop.
        
        Note: For loops are normally desugared to while loops before reaching here.
        This is a fallback implementation.
        """
        # For loops should be desugared to while loops by the desugaring pass
        # If we get here, emit as a while loop manually
        raise NotImplementedError(
            "For loops should be desugared to while loops before LLVM code generation. "
            "If you see this error, the desugaring pass may not have run."
        )


class LLVMCodeGenerator:
    """Main class for generating LLVM IR from fpy AST."""
    
    def __init__(self, module_name: str = "fpy_module"):
        self.module = ir.Module(name=module_name)
        self.module.triple = binding.get_default_triple()
        
    def compile_module(self, ast: AstBlock, state: CompileState) -> ir.Module:
        """Compile an fpy module to LLVM IR."""
        # Create the main function
        func_type = ir.FunctionType(ir.IntType(32), [])
        main_func = ir.Function(self.module, func_type, name="main")
        
        # Create entry block
        entry_block = main_func.append_basic_block(name="entry")
        builder = ir.IRBuilder(entry_block)
        
        # Create LLVM compile state
        llvm_state = LLVMCompileState(
            module=self.module,
            builder=builder,
            compile_state=state,
            entry_block=entry_block,
        )
        
        # Create emitter and emit code
        emitter = LLVMEmitter(llvm_state)
        emitter.emit(ast, state)
        
        # Add return statement
        if not builder.block.is_terminated:
            builder.ret(ir.Constant(ir.IntType(32), 0))
        
        return self.module
    
    def get_ir_string(self) -> str:
        """Get the LLVM IR as a string."""
        return str(self.module)
    
    def verify(self) -> bool:
        """Verify the LLVM module."""
        try:
            llvm_mod = binding.parse_assembly(str(self.module))
            llvm_mod.verify()
            return True
        except Exception as e:
            print(f"Verification failed: {e}")
            return False
    
    def compile_to_object(self) -> bytes:
        """Compile the module to native object code."""
        llvm_mod = binding.parse_assembly(str(self.module))
        
        # Create target machine
        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        
        # Compile to object code
        return target_machine.emit_object(llvm_mod)
    
    def optimize(self, level: int = 2):
        """Run optimization passes on the module."""
        llvm_mod = binding.parse_assembly(str(self.module))
        
        # Create pass manager
        pm_builder = binding.PassManagerBuilder()
        pm_builder.opt_level = level
        
        pm = binding.ModulePassManager()
        pm_builder.populate(pm)
        pm.run(llvm_mod)
        
        # Update our module with optimized version
        self.module = ir.Module(name=self.module.name)
        # Note: we'd need to rebuild from the optimized llvm_mod
        # For now, just return - optimization is available but needs more work


def run_semantic_passes(body, state: CompileState) -> CompileState:
    """Run semantic analysis passes on the AST."""
    from fpy.semantics import (
        AssignIds,
        CreateFunctionScopes,
        CalculateConstExprValues,
        CalculateDefaultArgConstValues,
        CheckBreakAndContinueInLoop,
        CheckConstArrayAccesses,
        CheckFunctionReturns,
        CheckReturnInFunc,
        CheckUseBeforeDefine,
        CreateVariablesAndFuncs,
        PickTypesAndResolveMembersAndElements,
        ResolveQualifiedNames,
        UpdateTypesAndFuncs,
        WarnRangesAreNotEmpty,
    )
    from fpy.desugaring import DesugarDefaultArgs, DesugarForLoops, DesugarCheckStatements
    
    pre_semantic_desugaring_passes = [
        DesugarCheckStatements()
    ]
    
    semantics_passes = [
        AssignIds(),
        CreateFunctionScopes(),
        CreateVariablesAndFuncs(),
        CheckBreakAndContinueInLoop(),
        CheckReturnInFunc(),
        ResolveQualifiedNames(),
        UpdateTypesAndFuncs(),
        CheckUseBeforeDefine(),
        PickTypesAndResolveMembersAndElements(),
        CalculateDefaultArgConstValues(),
        CalculateConstExprValues(),
        CheckFunctionReturns(),
        CheckConstArrayAccesses(),
        WarnRangesAreNotEmpty(),
    ]
    
    desugaring_passes = [
        DesugarDefaultArgs(),
        DesugarForLoops(),
    ]
    
    for compile_pass in pre_semantic_desugaring_passes:
        compile_pass.run(body, state)
        if state.errors:
            return state
    
    for compile_pass in semantics_passes:
        compile_pass.run(body, state)
        if state.errors:
            return state
    
    for compile_pass in desugaring_passes:
        compile_pass.run(body, state)
        if state.errors:
            return state
    
    return state


def compile_fpy_to_llvm(text: str, dictionary: str) -> tuple[ir.Module, CompileState]:
    """
    Compile fpy source text to LLVM IR.
    
    Args:
        text: The fpy source code
        dictionary: Path to the fprime dictionary
    
    Returns:
        Tuple of (LLVM module, compile state)
    """
    from fpy.compiler import text_to_ast, get_base_compile_state
    
    ast = text_to_ast(text)
    if ast is None:
        raise RuntimeError("Failed to parse fpy source")
    
    state = get_base_compile_state(dictionary, {})
    state.root = ast
    state = run_semantic_passes(ast, state)
    
    if state.errors:
        for err in state.errors:
            print(err)
        raise RuntimeError("Semantic analysis failed")
    
    generator = LLVMCodeGenerator()
    module = generator.compile_module(ast, state)
    
    return module, state


def compile_fpy_to_ir_string(text: str, dictionary: str) -> str:
    """Compile fpy source to LLVM IR string."""
    module, _ = compile_fpy_to_llvm(text, dictionary)
    return str(module)


class JITCompiler:
    """JIT compiles and executes fpy code using LLVM's MCJIT engine."""
    
    def __init__(self, module_name: str = "fpy_module"):
        self.module_name = module_name
        self.ir_module = None
        self.engine = None
        self._initialized = False
        self._tlm_db: dict[str, bytes] = {}
        
    def _ensure_initialized(self):
        """Initialize LLVM native target (only once)."""
        if not self._initialized:
            binding.initialize_native_target()
            binding.initialize_native_asmprinter()
            self._initialized = True
    
    def compile(self, text: str, dictionary: str, tlm: dict[str, bytes] = None) -> None:
        """
        Compile fpy source code to native machine code.
        
        Args:
            text: The fpy source code
            dictionary: Path to the fprime dictionary
            tlm: Optional telemetry database mapping channel names to serialized values
        """
        from fpy.compiler import text_to_ast, get_base_compile_state
        
        self._ensure_initialized()
        
        # Store telemetry for use during compilation
        self._tlm_db = tlm or {}
        
        # Parse and analyze
        ast = text_to_ast(text)
        if ast is None:
            raise RuntimeError("Failed to parse fpy source")
        
        state = get_base_compile_state(dictionary, {})
        state.root = ast
        state = run_semantic_passes(ast, state)
        
        if state.errors:
            for err in state.errors:
                print(err)
            raise RuntimeError("Semantic analysis failed")
        
        # Create LLVM module
        self.ir_module = ir.Module(name=self.module_name)
        self.ir_module.triple = binding.get_default_triple()
        
        # Create the main function
        func_type = ir.FunctionType(ir.IntType(32), [])
        main_func = ir.Function(self.ir_module, func_type, name="main")
        
        # Create entry block
        entry_block = main_func.append_basic_block(name="entry")
        builder = ir.IRBuilder(entry_block)
        
        # Create LLVM compile state
        llvm_state = LLVMCompileState(
            module=self.ir_module,
            builder=builder,
            compile_state=state,
            tlm_db=self._tlm_db,
            entry_block=entry_block,
        )
        
        # Emit code
        emitter = LLVMEmitter(llvm_state)
        emitter.emit(ast, state)
        
        # Add return statement if not terminated
        if not builder.block.is_terminated:
            builder.ret(ir.Constant(ir.IntType(32), 0))
        
        # Create JIT engine
        self._create_engine()
    
    def _create_engine(self):
        """Create the MCJIT execution engine."""
        # Parse the IR
        llvm_ir = str(self.ir_module)
        llvm_mod = binding.parse_assembly(llvm_ir)
        llvm_mod.verify()
        
        # Create target machine
        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        
        # Create execution engine with an empty backing module
        backing_mod = binding.parse_assembly("")
        self.engine = binding.create_mcjit_compiler(backing_mod, target_machine)
        
        # Add our module
        self.engine.add_module(llvm_mod)
        self.engine.finalize_object()
    
    def get_ir(self) -> str:
        """Get the LLVM IR as a string."""
        if self.ir_module is None:
            raise RuntimeError("No module compiled yet")
        return str(self.ir_module)
    
    def run(self, text: str = None, dictionary: str = None, tlm: dict[str, bytes] = None) -> int:
        """
        Compile (if needed) and run the code.
        
        Args:
            text: Optional fpy source code (if not already compiled)
            dictionary: Optional path to fprime dictionary (required if text provided)
            tlm: Optional telemetry database mapping channel names to serialized values
            
        Returns:
            The return value of the main function
        """
        if text is not None:
            if dictionary is None:
                raise ValueError("dictionary is required when providing source text")
            self.compile(text, dictionary, tlm)
        
        if self.engine is None:
            raise RuntimeError("No code compiled - call compile() first")
        
        return self._execute()
    
    def _execute(self) -> int:
        """Execute the compiled code."""
        import ctypes
        
        # Get function pointer for main
        func_ptr = self.engine.get_function_address("main")
        if func_ptr == 0:
            raise RuntimeError("Could not find 'main' function")
        
        # Create a callable using ctypes
        cfunc = ctypes.CFUNCTYPE(ctypes.c_int)(func_ptr)
        
        # Call and return the result
        return cfunc()


def run_fpy(text: str, dictionary: str) -> int:
    """
    Compile and run fpy source code using JIT compilation.
    
    Args:
        text: The fpy source code
        dictionary: Path to the fprime dictionary
        
    Returns:
        The return value of the main function (0 for success)
    """
    compiler = JITCompiler()
    return compiler.run(text, dictionary)
