from __future__ import annotations
import inspect
from dataclasses import fields
from typing import Callable, Union, get_args, get_origin
import typing

# In Python 3.10+, the `|` operator creates a `types.UnionType`.
# We need to handle this for forward compatibility, but it won't exist in 3.9.
try:
    from types import UnionType

    UNION_TYPES = (Union, UnionType)
except ImportError:
    UNION_TYPES = (Union,)

from fpy.error import BackendError
from fpy.ir import Ir, IrGoto, IrIf, IrLabel, IrPushLabelOffset
from fpy.model import DirectiveErrorCode, STACK_FRAME_HEADER_SIZE
from fpy.types import (
    SIGNED_INTEGER_TYPES,
    SPECIFIC_NUMERIC_TYPES,
    UNSIGNED_INTEGER_TYPES,
    FpyType,
    FpyValue,
    INTEGER,
    FLOAT,
    INTERNAL_STRING,
    TypeKind,
    NOTHING,
    NOTHING_VALUE,
    BOOL,
    U8,
    U64,
    I64,
    F32,
    F64,
    is_instance_compat,
)
from fpy.state import (
    BuiltinFuncSymbol,
    CastSymbol,
    CommandSymbol,
    CompileState,
    FieldAccess,
    FunctionSymbol,
    TypeCtorSymbol,
    VariableSymbol,
)
from fpy.state import ChDef, PrmDef
from fpy.visitors import (
    STOP_DESCENT,
    Emitter,
    TopDownVisitor,
    Visitor,
)

from fpy.bytecode.directives import (
    BINARY_STACK_OPS,
    UNARY_STACK_OPS,
    AllocateDirective,
    ArrayIndexType,
    BinaryStackOp,
    CallDirective,
    ConstCmdDirective,
    DiscardDirective,
    ExitDirective,
    FloatDivideDirective,
    FloatExtendDirective,
    FloatToSignedIntDirective,
    FloatToUnsignedIntDirective,
    FloatTruncateDirective,
    FwOpcodeType,
    GotoDirective,
    IfDirective,
    IntegerSignedExtend16To64Directive,
    IntegerSignedExtend32To64Directive,
    IntegerSignedExtend8To64Directive,
    IntegerTruncate64To16Directive,
    IntegerTruncate64To8Directive,
    IntegerZeroExtend16To64Directive,
    IntegerZeroExtend32To64Directive,
    IntegerZeroExtend8To64Directive,
    OrDirective,
    PeekDirective,
    FloatMultiplyDirective,
    GetFieldDirective,
    IntAddDirective,
    IntMultiplyDirective,
    LoadRelDirective,
    LoadAbsDirective,
    MemCompareDirective,
    NoOpDirective,
    IntegerTruncate64To32Directive,
    ReturnDirective,
    SignedGreaterThanOrEqualDirective,
    SignedIntToFloatDirective,
    SignedLessThanDirective,
    StackCmdDirective,
    Directive,
    NotDirective,
    PushValDirective,
    SignedStackSizeType,
    StackSizeType,
    StoreRelConstOffsetDirective,
    StoreAbsConstOffsetDirective,
    StoreRelDirective,
    StoreAbsDirective,
    PushPrmDirective,
    PushTlmValDirective,
    UnaryStackOp,
    UnsignedIntToFloatDirective,
)
from fpy.syntax import (
    Ast,
    AstAssert,
    AstBinaryOp,
    AstBreak,
    AstContinue,
    AstDef,
    AstExpr,
    AstFor,
    AstGetAttr,
    AstIndexExpr,
    AstLiteral,
    AstNodeWithSideEffects,
    AstReturn,
    AstBlock,
    AstBlock,
    AstIf,
    AstAssign,
    AstFuncCall,
    AstUnaryOp,
    AstIdent,
    AstWhile,
)


class CollectUsedFunctions(Visitor):
    """Collects the set of functions that are called anywhere in the code.
    
    Any function that is called (even from within other functions) will be
    marked as used and have code generated for it.
    """

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        func = state.resolved_symbols.get(node.func)
        if not is_instance_compat(func, FunctionSymbol):
            return
        state.used_funcs.add(func.definition)


class CalculateFrameSizes(TopDownVisitor):
    """Assigns frame offsets to variables before code generation.

    Each instance handles one frame (global or function). Visits blocks
    top-down, assigning sequential offsets to variables. At function
    boundaries, spawns a fresh instance for the function's frame and
    returns STOP_DESCENT to isolate frames from each other.

    This must run before GenerateFunctions so that global variable offsets
    are known when generating function bodies that access them.
    """

    def __init__(self):
        super().__init__()
        self.offset = 0

    def run(self, start: Ast, state: CompileState):
        super().run(start, state)
        state.frame_sizes[start] = self.offset

    def visit_AstBlock(self, node: AstBlock, state: CompileState):
        scope = state.enclosing_value_scope.get(node)
        if scope is None:
            return
        for _name, sym in scope.items():
            if is_instance_compat(sym, VariableSymbol) and sym.frame_offset is None:
                sym.frame_offset = self.offset
                self.offset += sym.type.max_size

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # Assign argument offsets (negative offsets before frame start)
        func = state.resolved_symbols[node.name]
        if func.args:
            arg_offset = -STACK_FRAME_HEADER_SIZE
            for arg in reversed(func.args):
                arg_name, arg_type, _ = arg
                arg_var = state.enclosing_value_scope[node.body][arg_name]
                arg_offset -= arg_type.max_size
                arg_var.frame_offset = arg_offset
        # Assign body variable offsets in a fresh frame
        CalculateFrameSizes().run(node.body, state)
        return STOP_DESCENT


class GenerateFunctionEntryPoints(Visitor):
    def visit_AstDef(self, node: AstDef, state: CompileState):
        if node not in state.used_funcs:
            # Function is never called, skip it
            return
        entry_label = IrLabel(node, "entry")
        state.func_entry_labels[node] = entry_label


class GenerateFunctions(Visitor):
    def visit_AstDef(self, node: AstDef, state: CompileState):
        if node not in state.used_funcs:
            # Function is never called, skip generating code for it
            return
        entry_label = state.func_entry_labels[node]
        code = [entry_label]
        
        # Allocate space for local variables
        lvar_array_size_bytes = state.frame_sizes[node.body]
        if lvar_array_size_bytes > 0:
            code.append(AllocateDirective(lvar_array_size_bytes))
        
        code.extend(GenerateFunctionBody().emit(node.body, state))
        func = state.resolved_symbols[node.name]
        if func.return_type is NOTHING and not state.does_return[node.body]:
            # implicit empty return
            arg_bytes = sum(arg[1].max_size for arg in (func.args or []))
            code.append(ReturnDirective(0, arg_bytes))
        state.generated_funcs[node] = code


class GenerateFunctionBody(Emitter):
    # Flag indicating we're generating code inside a function body
    # This affects how we access global variables (need GLOBAL directives)
    in_function = True

    def try_emit_expr_as_const(
        self, node: AstExpr, state: CompileState
    ) -> Union[list[Directive | Ir], None]:
        """if the expr has a compile time const value, emit that as a PUSH_VAL"""
        expr_value = state.const_expr_values.get(node)

        if expr_value is None:
            # no const value
            return None

        assert isinstance(expr_value, FpyValue) and expr_value.type not in (
            INTEGER, INTERNAL_STRING, FLOAT
        ), expr_value

        if expr_value is NOTHING_VALUE:
            # nothing type has no value
            return []

        # it has a constant value at compile time
        serialized_expr_value = expr_value.serialize()

        # push it to the stack
        return [PushValDirective(serialized_expr_value)]

    def discard_expr_result(self, node: Ast, state: CompileState) -> list[Directive]:
        """if the node is an expr, generate code to discard its stack value"""
        if not is_instance_compat(node, AstExpr):
            # nothing to discard
            return []

        result_type = state.contextual_types[node]
        if result_type == NOTHING:
            return []
        if result_type.max_size > 0:
            return [DiscardDirective(result_type.max_size)]
        return []

    def get_64_bit_numeric_type(self, type: FpyType) -> FpyType:
        """return the 64 bit version of the input numeric type"""
        assert type in SPECIFIC_NUMERIC_TYPES, type
        return (
            I64
            if type in SIGNED_INTEGER_TYPES
            else U64 if type in UNSIGNED_INTEGER_TYPES else F64
        )

    def convert_numeric_type(
        self, from_type: FpyType, to_type: FpyType
    ) -> list[Directive]:
        """
        return a list of dirs needed to convert a numeric stack value of from_type to a stack value of to_type
        """
        if from_type == to_type:
            return []

        # only valid runtime type conversion is between two numeric types
        assert (
            from_type in SPECIFIC_NUMERIC_TYPES and to_type in SPECIFIC_NUMERIC_TYPES
        ), (
            from_type,
            to_type,
        )

        dirs = []
        # first go to 64 bit width
        dirs.extend(self.extend_numeric_type_to_64_bits(from_type))
        from_64_bit = self.get_64_bit_numeric_type(from_type)
        to_64_bit = self.get_64_bit_numeric_type(to_type)

        # now convert between int and float if necessary
        if from_64_bit == U64 and to_64_bit == F64:
            dirs.append(UnsignedIntToFloatDirective())
            from_64_bit = F64
        elif from_64_bit == I64 and to_64_bit == F64:
            dirs.append(SignedIntToFloatDirective())
            from_64_bit = F64
        elif from_64_bit == U64 or from_64_bit == I64:
            assert to_64_bit == U64 or to_64_bit == I64
            # conversion from signed to unsigned int is implicit, doesn't need code gen
            from_64_bit = to_64_bit
        elif from_64_bit == F64 and to_64_bit == I64:
            dirs.append(FloatToSignedIntDirective())
            from_64_bit = I64
        elif from_64_bit == F64 and to_64_bit == U64:
            dirs.append(FloatToUnsignedIntDirective())
            from_64_bit = U64

        assert from_64_bit == to_64_bit, (from_64_bit, to_64_bit)

        # now truncate back down to desired size
        dirs.extend(
            self.truncate_numeric_type_from_64_bits(to_64_bit, to_type.max_size)
        )
        return dirs

    def truncate_numeric_type_from_64_bits(
        self, from_type: FpyType, new_size: int
    ) -> list[Directive]:

        assert new_size in (1, 2, 4, 8), new_size
        assert from_type.max_size == 8, from_type.max_size

        if new_size == 8:
            # already correct size
            return []

        if from_type == F64:
            # only one option for float trunc
            assert new_size == 4, new_size
            return [FloatTruncateDirective()]

        # must be an int
        assert from_type.is_integer, from_type

        if new_size == 1:
            return [IntegerTruncate64To8Directive()]
        elif new_size == 2:
            return [IntegerTruncate64To16Directive()]

        return [IntegerTruncate64To32Directive()]

    def extend_numeric_type_to_64_bits(self, type: FpyType) -> list[Directive]:
        if type.max_size == 8:
            # already 8 bytes
            return []
        if type == F32:
            return [FloatExtendDirective()]

        # must be an int
        assert type.is_integer, type

        from_size = type.max_size
        assert from_size in (1, 2, 4, 8), from_size

        if type in SIGNED_INTEGER_TYPES:
            if from_size == 1:
                return [IntegerSignedExtend8To64Directive()]
            elif from_size == 2:
                return [IntegerSignedExtend16To64Directive()]
            else:
                return [IntegerSignedExtend32To64Directive()]
        else:
            if from_size == 1:
                return [IntegerZeroExtend8To64Directive()]
            elif from_size == 2:
                return [IntegerZeroExtend16To64Directive()]
            else:
                return [IntegerZeroExtend32To64Directive()]

    def calc_lvar_offset_of_array_element(
        self, node: Ast, idx_expr: AstExpr, array_type: FpyType, state: CompileState
    ) -> list[Directive | Ir]:
        """generates code to push to stack the U64 byte offset in the array for an array access, while performing an array oob
        check. idx_expr is the expression to calculate the index, and dest is the FieldSymbol containing info about the
        dest array"""
        dirs = []
        # let's push the offset of base lvar first, then
        # calculate the offset in base type, then add

        # push the index to the stack, do a bounds check,
        dirs.extend(self.emit(idx_expr, state))
        # okay now let's do an array oob check
        # we want to peek the index so we can consume it for the oob check
        # byte count
        dirs.append(
            PushValDirective(FpyValue(StackSizeType, ArrayIndexType.max_size).serialize())
        )
        # offset
        dirs.append(PushValDirective(FpyValue(StackSizeType, 0).serialize()))
        dirs.append(PeekDirective())  # duplicate the index
        # convert idx to i64
        dirs.extend(self.convert_numeric_type(ArrayIndexType, I64))
        dirs.append(
            PushValDirective(FpyValue(I64, array_type.length).serialize())
        )  # push the length as I64
        # check if idx >= length
        dirs.append(SignedGreaterThanOrEqualDirective())
        # okay now dupe index again to check < 0
        # byte count
        dirs.append(
            PushValDirective(FpyValue(StackSizeType, ArrayIndexType.max_size).serialize())
        )
        # offset is 1 because we currently have the result of the last check on stack
        dirs.append(PushValDirective(FpyValue(StackSizeType, 1).serialize()))
        dirs.append(PeekDirective())  # duplicate the index
        # convert idx to i64
        dirs.extend(self.convert_numeric_type(ArrayIndexType, I64))
        dirs.append(
            PushValDirective(FpyValue(I64, 0).serialize())
        )  # push 0 as i64
        # check if idx < 0
        dirs.append(SignedLessThanDirective())
        # or both checks together
        dirs.append(OrDirective())
        # if either true, fail with error code, otherwise go to after check
        oob_check_end_label = IrLabel(node, "oob_check_end")
        dirs.append(IrIf(oob_check_end_label))
        # push the error code we should fail with if false
        dirs.append(
            PushValDirective(
                FpyValue(U8, DirectiveErrorCode.ARRAY_OUT_OF_BOUNDS.value).serialize()
            )
        )
        dirs.append(ExitDirective())
        dirs.append(oob_check_end_label)
        # okay we're good. should still have the idx on the stack

        # multiply the index by the member type size
        dirs.append(PushValDirective(FpyValue(U64, array_type.elem_type.max_size).serialize()))
        dirs.append(IntMultiplyDirective())
        return dirs

    def emit_AstBlock(self, node: AstBlock, state: CompileState):
        dirs = []
        for stmt in node.stmts:
            if not is_instance_compat(stmt, AstNodeWithSideEffects):
                # if the stmt can't do anything on its own, ignore it
                # TODO warn
                continue
            dirs.extend(self.emit(stmt, state))
            # discard stack value if it was an expr
            dirs.extend(self.discard_expr_result(stmt, state))
        return dirs

    def emit_AstIf(self, node: AstIf, state: CompileState):
        dirs = []

        cases: list[tuple[AstExpr, AstBlock]] = []

        cases.append((node.condition, node.body))

        for case in node.elifs:
            cases.append((case.condition, case.body))

        if_end_label = IrLabel(node, "end")

        for case in cases:
            case_end_label = IrLabel(case[1], "end")
            case_dirs = []
            # put the conditional on top of stack
            case_dirs.extend(self.emit(case[0], state))
            # include if stmt (update the end idx later)
            if_dir = IrIf(case_end_label)

            case_dirs.append(if_dir)
            # include body
            case_dirs.extend(self.emit(case[1], state))
            # once we've finished executing the body:
            # include a goto end of if
            case_dirs.append(IrGoto(if_end_label))
            case_dirs.append(case_end_label)

            dirs.extend(case_dirs)

        if node.els is not None:
            dirs.extend(self.emit(node.els, state))

        dirs.append(if_end_label)

        return dirs

    def emit_AstWhile(self, node: AstWhile, state: CompileState):
        # start by creating labels. store them in dicts so that break/continue
        # can use them
        while_start_label = IrLabel(node, "start")
        while_end_label = IrLabel(node, "end")
        for_loop_increment_label = None
        state.while_loop_start_labels[node] = while_start_label
        state.while_loop_end_labels[node] = while_end_label
        # if this used to be a for loop:
        if node in state.desugared_for_loops:
            # there should be at least one stmt in a for loop's body (the inc stmt)
            for_loop_increment_label = IrLabel(node, "increment")
            state.for_loop_inc_labels[node] = for_loop_increment_label

        dirs = [while_start_label]
        # push the condition to the stack
        dirs.extend(self.emit(node.condition, state))
        # if the cond is true, fall thru, otherwise go to end
        dirs.append(IrIf(while_end_label))
        # run body

        for stmt_idx, stmt in enumerate(node.body.stmts):
            if not is_instance_compat(stmt, AstNodeWithSideEffects):
                # if the stmt can't do anything on its own, ignore it
                continue
            # we're going to manually emit the body's stmts instead
            # of just emitting the body, because A) it doesn't matter
            # and B) we need the index of the last statement in the body
            # if we're a for loop, because that's where the continue stmt
            # needs to go
            if (
                stmt_idx == len(node.body.stmts) - 1
                and for_loop_increment_label is not None
            ):
                # last stmt, it must be the inc stmt, add the label before it
                dirs.append(for_loop_increment_label)
            dirs.extend(self.emit(stmt, state))
            # discard stack value if it was an expr
            dirs.extend(self.discard_expr_result(stmt, state))
        # go back to condition check
        dirs.append(IrGoto(while_start_label))
        dirs.append(while_end_label)

        return dirs

    def emit_AstBreak(self, node: AstBreak, state: CompileState):
        enclosing_loop = state.enclosing_loops[node]
        loop_end = state.while_loop_end_labels[enclosing_loop]
        return [IrGoto(loop_end)]

    def emit_AstContinue(self, node: AstContinue, state: CompileState):
        enclosing_loop = state.enclosing_loops[node]
        if enclosing_loop in state.desugared_for_loops:
            loop_start = state.for_loop_inc_labels[enclosing_loop]
        else:
            loop_start = state.while_loop_start_labels[enclosing_loop]
        return [IrGoto(loop_start)]

    def emit_AstDef(self, node: AstDef, state: CompileState):
        # don't generate other functions, just do this one
        return []

    def emit_AstReturn(self, node: AstReturn, state: CompileState):
        enclosing_func = state.enclosing_funcs[node]
        enclosing_func = state.resolved_symbols[enclosing_func.name]
        func_args_size = sum(arg[1].max_size for arg in enclosing_func.args)

        if node.value is not None:
            dirs = self.emit(node.value, state)
            value_size = state.contextual_types[node.value].max_size
        else:
            dirs = []
            value_size = 0
        dirs.append(ReturnDirective(value_size, func_args_size))

        return dirs

    def emit_AstFor(self, node: AstFor, state: CompileState):
        # should have been desugared out
        assert False, node

    def emit_AstIndexExpr(self, node: AstIndexExpr, state: CompileState):
        const_dirs = self.try_emit_expr_as_const(node, state)
        if const_dirs is not None:
            return const_dirs
        sym = state.resolved_symbols[node]

        assert is_instance_compat(sym, FieldAccess), sym

        # use the unconverted for this expr for now, because we haven't run conversion
        unconverted_type = state.synthesized_types[node]
        # however, for parent, use converted because conversion has been run
        parent_type = state.contextual_types[node.parent]

        assert parent_type.kind == TypeKind.ARRAY
        assert unconverted_type == parent_type.elem_type, (
            parent_type.elem_type,
            unconverted_type,
        )

        # okay, we want to get an element from an array on the stack

        # TODO optimization: leave it in the lvar array instead of pushing the whole thing to stack
        # for now we push the whole thing
        dirs = self.emit(node.parent, state)

        # calculate the offset in the parent array
        dirs.extend(
            self.calc_lvar_offset_of_array_element(node, node.item, parent_type, state)
        )
        # truncate back to StackSizeType which is what get field uses
        dirs.extend(self.convert_numeric_type(U64, StackSizeType))

        # get the member from the stack at this offset, discard the rest of
        # the parent
        dirs.append(
            GetFieldDirective(
                parent_type.max_size, parent_type.elem_type.max_size
            )
        )

        # now convert the type if necessary
        converted_type = state.contextual_types[node]
        if unconverted_type != converted_type:
            dirs.extend(self.convert_numeric_type(unconverted_type, converted_type))

        return dirs

    def emit_AstIdent(self, node: AstIdent, state: CompileState):
        const_dirs = self.try_emit_expr_as_const(node, state)
        if const_dirs is not None:
            return const_dirs

        sym = state.resolved_symbols.get(node)

        assert is_instance_compat(sym, VariableSymbol), sym

        # Use global directives only when inside a function AND accessing a global variable
        # At top level, stack_frame_start = 0, so local and global offsets are the same
        use_global = self.in_function and sym.is_global
        if use_global:
            dirs = [LoadAbsDirective(sym.frame_offset, sym.type.max_size)]
        else:
            dirs = [LoadRelDirective(sym.frame_offset, sym.type.max_size)]

        unconverted_type = state.synthesized_types[node]
        converted_type = state.contextual_types[node]
        if unconverted_type != converted_type:
            dirs.extend(self.convert_numeric_type(unconverted_type, converted_type))

        return dirs

    def emit_AstGetAttr(self, node: AstGetAttr, state: CompileState):
        const_dirs = self.try_emit_expr_as_const(node, state)
        if const_dirs is not None:
            return const_dirs

        sym = state.resolved_symbols.get(node)

        if is_instance_compat(sym, dict):
            # don't generate code for it, it's a reference to a scope and
            # doesn't have a value
            return []

        # start with the unconverted type, because we haven't applied runtime type conversion yet
        unconverted_type = state.synthesized_types[node]

        dirs = []

        if is_instance_compat(sym, ChDef):
            dirs.append(PushTlmValDirective(sym.ch_id))
        elif is_instance_compat(sym, PrmDef):
            dirs.append(PushPrmDirective(sym.prm_id))
        elif is_instance_compat(sym, VariableSymbol):
            # Use global directives only when inside a function AND accessing a global variable
            use_global = self.in_function and sym.is_global
            if use_global:
                dirs.append(
                    LoadAbsDirective(sym.frame_offset, sym.type.max_size)
                )
            else:
                dirs.append(LoadRelDirective(sym.frame_offset, sym.type.max_size))
        elif is_instance_compat(sym, FieldAccess):
            # okay, put parent dirs in first
            dirs.extend(self.emit(sym.parent_expr, state))
            assert sym.local_offset is not None
            # use the converted type of parent
            parent_type = state.contextual_types[sym.parent_expr]
            # push the offset to the stack
            dirs.append(PushValDirective(FpyValue(StackSizeType, sym.local_offset).serialize()))
            dirs.append(
                GetFieldDirective(
                    parent_type.max_size, unconverted_type.max_size
                )
            )
        else:
            assert (
                False
            ), sym  # sym should either be impossible to put on stack or should have a compile time val

        converted_type = state.contextual_types[node]
        if converted_type != unconverted_type:
            dirs.extend(self.convert_numeric_type(unconverted_type, converted_type))

        return dirs

    def emit_AstBinaryOp(self, node: AstBinaryOp, state: CompileState):
        const_dirs = self.try_emit_expr_as_const(node, state)
        if const_dirs is not None:
            return const_dirs

        if node.op in (BinaryStackOp.AND, BinaryStackOp.OR):
            dirs = self.generate_short_circuit_boolean(node, state)
        else:
            # push lhs and rhs to stack
            dirs = self.emit(node.lhs, state)
            dirs.extend(self.emit(node.rhs, state))

            intermediate_type = state.op_intermediate_types[node]

            if (
                node.op == BinaryStackOp.EQUAL or node.op == BinaryStackOp.NOT_EQUAL
            ) and intermediate_type not in SPECIFIC_NUMERIC_TYPES:
                lhs_type = state.contextual_types[node.lhs]
                rhs_type = state.contextual_types[node.rhs]
                assert lhs_type == rhs_type, (lhs_type, rhs_type)
                dirs.append(MemCompareDirective(lhs_type.max_size))
                if node.op == BinaryStackOp.NOT_EQUAL:
                    dirs.append(NotDirective())
            elif (
                node.op == BinaryStackOp.FLOOR_DIVIDE and intermediate_type == F64
            ):
                # for float floor division, do float division, then convert to int, then
                # back to float
                dirs.append(FloatDivideDirective())
                dirs.append(FloatToSignedIntDirective())
                dirs.append(SignedIntToFloatDirective())
            else:

                dir = BINARY_STACK_OPS[node.op][intermediate_type]
                if dir != NoOpDirective:
                    # don't include no op
                    dirs.append(dir())

            # The VM operates on 64-bit values, so after the op we have a 64-bit result.
            # Convert from the 64-bit intermediate type to the synthesized result type.
            synthesized_type = state.synthesized_types[node]
            if intermediate_type in SPECIFIC_NUMERIC_TYPES and synthesized_type in SPECIFIC_NUMERIC_TYPES:
                dirs.extend(self.convert_numeric_type(intermediate_type, synthesized_type))

        # and convert the result of the op into the desired result of this expr
        unconverted_type = state.synthesized_types[node]
        converted_type = state.contextual_types[node]
        if unconverted_type != converted_type:
            dirs.extend(self.convert_numeric_type(unconverted_type, converted_type))

        return dirs

    def generate_short_circuit_boolean(
        self, node: AstBinaryOp, state: CompileState
    ) -> list[Directive | Ir]:
        dirs: list[Directive | Ir] = []
        end_label = IrLabel(node, "bool_end")

        if node.op == BinaryStackOp.AND:
            short_label = IrLabel(node, "and_short")
            dirs.extend(self.emit(node.lhs, state))
            # jump to short circuit when lhs is false
            dirs.append(IrIf(short_label))
            dirs.extend(self.emit(node.rhs, state))
            dirs.append(IrGoto(end_label))
            dirs.append(short_label)
            dirs.append(PushValDirective(FpyValue(BOOL, False).serialize()))
        else:
            rhs_label = IrLabel(node, "or_rhs")
            dirs.extend(self.emit(node.lhs, state))
            # only evaluate rhs if lhs is false
            dirs.append(IrIf(rhs_label))
            dirs.append(PushValDirective(FpyValue(BOOL, True).serialize()))
            dirs.append(IrGoto(end_label))
            dirs.append(rhs_label)
            dirs.extend(self.emit(node.rhs, state))

        dirs.append(end_label)
        return dirs

    def emit_AstUnaryOp(self, node: AstUnaryOp, state: CompileState):
        const_dirs = self.try_emit_expr_as_const(node, state)
        if const_dirs is not None:
            return const_dirs

        # push val to stack
        dirs = self.emit(node.val, state)

        # generate the actual op itself
        # which dir should we use?
        intermediate_type = state.op_intermediate_types[node]
        dir = UNARY_STACK_OPS[node.op][intermediate_type]

        if node.op == UnaryStackOp.NEGATE:
            # in this case, we also need to push -1
            if dir == FloatMultiplyDirective:
                dirs.append(PushValDirective(FpyValue(F64, -1).serialize()))
            elif dir == IntMultiplyDirective:
                dirs.append(PushValDirective(FpyValue(I64, -1).serialize()))

        dirs.append(dir())

        # The VM operates on 64-bit values, so after the op we have a 64-bit result.
        # Convert from the 64-bit intermediate type to the synthesized result type.
        synthesized_type = state.synthesized_types[node]
        if intermediate_type in SPECIFIC_NUMERIC_TYPES and synthesized_type in SPECIFIC_NUMERIC_TYPES:
            dirs.extend(self.convert_numeric_type(intermediate_type, synthesized_type))

        # and convert the result of the op into the desired result of this expr
        unconverted_type = state.synthesized_types[node]
        converted_type = state.contextual_types[node]
        if unconverted_type != converted_type:
            dirs.extend(self.convert_numeric_type(unconverted_type, converted_type))

        return dirs

    def emit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        const_dirs = self.try_emit_expr_as_const(node, state)
        if const_dirs is not None:
            return const_dirs

        node_args = node.args if node.args is not None else []
        func = state.resolved_symbols[node.func]
        dirs = []
        if is_instance_compat(func, CommandSymbol):
            const_args = not any(
                state.const_expr_values[arg_node] is None for arg_node in node_args
            )
            if const_args:
                # can just hardcode this cmd
                arg_bytes = bytes()
                for arg_node in node_args:
                    arg_value = state.const_expr_values[arg_node]
                    arg_bytes += arg_value.serialize()
                dirs.append(ConstCmdDirective(func.cmd.opcode, arg_bytes))
            else:
                arg_byte_count = 0
                # push all args to the stack
                # keep track of how many bytes total we have pushed
                for arg_node in node_args:
                    dirs.extend(self.emit(arg_node, state))
                    arg_converted_type = state.contextual_types[arg_node]
                    arg_byte_count += arg_converted_type.max_size
                # then push cmd opcode to stack as u32
                dirs.append(
                    PushValDirective(FpyValue(FwOpcodeType, func.cmd.opcode).serialize())
                )
                # now that all args are pushed to the stack, pop them and opcode off the stack
                # as a command
                dirs.append(StackCmdDirective(arg_byte_count))
        elif is_instance_compat(func, BuiltinFuncSymbol):
            # collect compile-time constant args (not pushed to stack)
            const_arg_values: dict[int, FpyValue] = {}
            for i in func.const_arg_indices:
                const_val = state.const_expr_values.get(node_args[i])
                assert const_val is not None, f"const arg {i} of {func.name} should have been validated by semantics"
                const_arg_values[i] = const_val

            # put non-const arg values on stack
            for i, arg_node in enumerate(node_args):
                if i not in func.const_arg_indices:
                    dirs.extend(self.emit(arg_node, state))

            dirs.extend(func.generate(node, const_arg_values))
        elif is_instance_compat(func, TypeCtorSymbol):
            # put arg values onto stack in correct order for serialization
            for arg_node in node_args:
                dirs.extend(self.emit(arg_node, state))
        elif is_instance_compat(func, CastSymbol):
            # just putting the arg value on the stack should be good enough, the
            # conversion will happen below
            dirs.extend(self.emit(node_args[0], state))
        elif is_instance_compat(func, FunctionSymbol):
            # script-defined function
            # okay.. calling convention says we're going to put the args on the stack
            for arg_node in node_args:
                dirs.extend(self.emit(arg_node, state))
            # okay, args are on the stack. now we're going to generate CALL
            func_entry_label = state.func_entry_labels[func.definition]
            # push the offset of the func
            dirs.append(IrPushLabelOffset(func_entry_label))
            # pop it off the stack and perform func call
            dirs.append(CallDirective())
        else:
            assert False, func

        # perform type conversion if called for
        unconverted_type = state.synthesized_types[node]
        converted_type = state.contextual_types[node]
        if unconverted_type != converted_type:
            dirs.extend(self.convert_numeric_type(unconverted_type, converted_type))

        return dirs

    def emit_AstAssign(self, node: AstAssign, state: CompileState):
        lhs = state.resolved_symbols[node.lhs]

        const_frame_offset = -1
        is_global_var = False
        if is_instance_compat(lhs, VariableSymbol):
            const_frame_offset = lhs.frame_offset
            is_global_var = lhs.is_global
            assert const_frame_offset is not None, lhs
        else:
            # okay now push the lvar arr offset to stack
            assert is_instance_compat(lhs, FieldAccess), lhs
            assert is_instance_compat(lhs.base_sym, VariableSymbol), lhs.base_sym
            is_global_var = lhs.base_sym.is_global

            # is the lvar array offset a constant?
            # okay, are we assigning to a member or an element?
            if lhs.is_struct_member:
                # if it's a struct, then the lvar offset is always constant
                const_frame_offset = lhs.base_offset + lhs.base_sym.frame_offset
            else:
                assert lhs.is_array_element
                # again, offset is the offset in base type + offset of base lvar

                # however, because array idx can be variable, we might not know at compile time
                # the offset in base type.

                # check if we have a value for it
                const_idx_expr_value = state.const_expr_values.get(lhs.idx_expr)
                if const_idx_expr_value is not None:
                    assert isinstance(const_idx_expr_value, FpyValue) and const_idx_expr_value.type == ArrayIndexType
                    # okay, so we have a constant value index
                    lhs_parent_type = state.contextual_types[lhs.parent_expr]
                    const_frame_offset = (
                        lhs.base_sym.frame_offset
                        + const_idx_expr_value.val
                        * lhs_parent_type.elem_type.max_size
                    )
                # otherwise, the array idx is unknown at compile time. we will have to calculate it

        # Use global directives only when inside a function AND accessing a global variable
        use_global = self.in_function and is_global_var

        # start with rhs on stack
        dirs = self.emit(node.rhs, state)

        if const_frame_offset != -1:
            # in this case, we can use StoreConstOffset
            if use_global:
                dirs.append(
                    StoreAbsConstOffsetDirective(
                        const_frame_offset, lhs.type.max_size
                    )
                )
            else:
                dirs.append(
                    StoreRelConstOffsetDirective(
                        const_frame_offset, lhs.type.max_size
                    )
                )
        else:
            # okay we don't know the offset at compile time
            # only one case where that can be:
            assert is_instance_compat(lhs, FieldAccess) and lhs.is_array_element, lhs

            # we need to calculate absolute offset in lvar array
            # == (parent offset) + (offset in parent)

            # offset in parent:
            lhs_parent_type = state.contextual_types[lhs.parent_expr]
            dirs.extend(
                self.calc_lvar_offset_of_array_element(
                    node, lhs.idx_expr, lhs_parent_type, state
                )
            )

            # parent offset:
            dirs.append(
                PushValDirective(FpyValue(U64, lhs.base_sym.frame_offset).serialize())
            )

            # add them
            dirs.append(IntAddDirective())

            # and now convert the u64 back into the SignedStackSizeType that store expects
            dirs.extend(self.convert_numeric_type(U64, SignedStackSizeType))

            # now that lvar array offset is pushed, use it to store in lvar array
            if use_global:
                dirs.append(StoreAbsDirective(lhs.type.max_size))
            else:
                dirs.append(StoreRelDirective(lhs.type.max_size))

        return dirs

    def emit_AstLiteral(self, node: AstLiteral, state: CompileState):
        const_dirs = self.try_emit_expr_as_const(node, state)
        assert const_dirs is not None
        return const_dirs

    def emit_AstAssert(self, node: AstAssert, state: CompileState):
        dirs = self.emit(node.condition, state)
        # invert the condition, we want to continue to exit if fail
        dirs.append(NotDirective())
        end_label = IrLabel(node, f"pass")
        dirs.append(IrIf(end_label))
        # push the error code we should use if false, if one was given
        if node.exit_code is not None:
            dirs.extend(self.emit(node.exit_code, state))
        else:
            # otherwise just use the default EXIT_WITH_ERROR error code
            dirs.append(
                PushValDirective(
                    FpyValue(U8, DirectiveErrorCode.EXIT_WITH_ERROR.value).serialize()
                )
            )
        dirs.append(ExitDirective())
        dirs.append(end_label)

        return dirs


class GenerateModule(Emitter):

    def emit_AstBlock(self, node: AstBlock, state: CompileState):
        if node is not state.root:
            return []

        # generate the main function using GenerateTopLevel (not in a function context)
        main_body = []
        
        # Allocate space for top-level local variables
        lvar_array_size_bytes = state.frame_sizes[node]
        if lvar_array_size_bytes > 0:
            main_body.append(AllocateDirective(lvar_array_size_bytes))
        
        main_body.extend(GenerateTopLevel().emit(node, state))

        # if there are functions, emit them at the top with a goto to skip past them
        if state.generated_funcs:
            funcs_code = []
            func_code_end_label = IrLabel(node, "main")
            funcs_code.append(IrGoto(func_code_end_label))
            for func, code in state.generated_funcs.items():
                funcs_code.extend(code)
            funcs_code.append(func_code_end_label)
            return funcs_code + main_body

        return main_body


class GenerateTopLevel(GenerateFunctionBody):
    """Generates top-level (main) code, not inside any function.
    At top level, stack_frame_start = 0, so local and global offsets are equivalent.
    All variables use LOCAL directives."""

    in_function = False


class IrPass:
    def run(
        self, ir: list[Directive | Ir], state: CompileState
    ) -> Union[list[Directive | Ir], BackendError]:
        pass


class ResolveLabels(IrPass):
    def run(self, ir, state: CompileState):
        labels: dict[str, int] = {}
        idx = 0
        dirs = []
        for dir in ir:
            if is_instance_compat(dir, IrLabel):
                if dir.name in labels:
                    return BackendError(f"Label {dir.name} already exists")
                labels[dir.name] = idx
                continue
            idx += 1

        # okay, we have all the labels
        for dir in ir:
            if is_instance_compat(dir, IrLabel):
                # drop these from the result
                continue
            elif is_instance_compat(dir, IrGoto):
                label = dir.label.name
                if label not in labels:
                    return BackendError(f"Unknown label {label}")
                dirs.append(GotoDirective(labels[label]))
            elif is_instance_compat(dir, IrIf):
                label = dir.goto_if_false_label.name
                if label not in labels:
                    return BackendError(f"Unknown label {label}")
                dirs.append(IfDirective(labels[label]))
            elif is_instance_compat(dir, IrPushLabelOffset):
                label = dir.label.name
                if label not in labels:
                    return BackendError(f"Unknown label {label}")
                dirs.append(PushValDirective(FpyValue(StackSizeType, labels[label]).serialize()))
            else:
                dirs.append(dir)

        return dirs


class FinalChecks(IrPass):
    def run(self, ir, state):
        if len(ir) > state.max_directives_count:
            return BackendError(
                f"Too many directives in sequence (expected less than {state.max_directives_count}, had {len(ir)})"
            )

        for dir in ir:
            # double check we've got rid of all the IR
            assert is_instance_compat(dir, Directive), dir

        return ir
