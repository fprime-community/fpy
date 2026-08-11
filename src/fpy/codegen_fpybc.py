from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union

from fpy.state import BackendState, CompileState

# In Python 3.10+, the `|` operator creates a `types.UnionType`.
# We need to handle this for forward compatibility, but it won't exist in 3.9.
try:
    from types import UnionType

    UNION_TYPES = (Union, UnionType)
except ImportError:
    UNION_TYPES = (Union,)

from fpy.error import BackendError
from fpy.ir import Ir, IrGoto, IrIf, IrLabel, IrPushLabelOffset
from fpy.bytecode.directives import DirectiveErrorCode, STACK_FRAME_HEADER_SIZE
from fpy.semantics import is_cmd_and_response_unhandled
from fpy.types import (
    SIGNED_INTEGER_TYPES,
    SPECIFIC_NUMERIC_TYPES,
    UNSIGNED_INTEGER_TYPES,
    CMD_RESPONSE,
    FpyType,
    FpyValue,
    INTEGER,
    FLOAT,
    INTERNAL_STRING,
    TypeKind,
    NOTHING,
    NOTHING_VALUE,
    BOOL,
    U64,
    I32,
    I64,
    F32,
    F64,
    SEQ_ARGS,
    is_instance_compat,
)
from fpy.symbols import (
    BuiltinFuncSymbol,
    CastSymbol,
    CommandSymbol,
    FieldAccess,
    FunctionSymbol,
    NameGroup,
    TypeCtorSymbol,
    VariableSymbol,
)
from fpy.types import ChDef, PrmDef
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
    FloatFloorDirective,
    FloatToSignedIntDirective,
    FloatToUnsignedIntDirective,
    FloatTruncateDirective,
    FwOpcodeType,
    FwPacketDescriptorType,
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
    PopSerializableDirective,
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
    AstAnonStruct,
    AstAnonArray,
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


@dataclass
class FpybcBackendState(BackendState):
    """The fpybc backend's view of a program: how its variables are laid out,
    and what it has emitted so far."""

    frame_offsets: dict[VariableSymbol, int] = field(default_factory=dict)
    """variable to the offset of its storage within its frame"""

    frame_sizes: dict[Ast, int] = field(default_factory=dict)
    """the block that owns a frame, to the total size in bytes of that frame's
    locals"""

    used_funcs: set[AstDef] = field(default_factory=set)
    """the function definitions that are called, and so need code generated"""

    func_entry_labels: dict[AstDef, IrLabel] = field(default_factory=dict)
    """function definition to the label at its entry point"""

    generated_funcs: dict[AstDef, list[Directive | Ir]] = field(default_factory=dict)
    """function definition to the code generated for its body"""

    while_loop_start_labels: dict[AstWhile, IrLabel] = field(default_factory=dict)
    """while loop to the label just before its conditional"""

    while_loop_end_labels: dict[AstWhile, IrLabel] = field(default_factory=dict)
    """while loop to the label at the end of the loop"""

    # keyed by while, because for loops are desugared to while loops
    for_loop_inc_labels: dict[AstWhile, IrLabel] = field(default_factory=dict)
    """desugared for loop to the label at its increment stmt"""


class CollectUsedFunctions(Visitor):
    """Collects the set of functions that are called anywhere in the code.

    Any function that is called (even from within other functions) will be
    marked as used and have code generated for it.
    """

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        func = state.resolved_symbols.get(node.func)
        if not is_instance_compat(func, FunctionSymbol):
            return
        state.backend.used_funcs.add(func.definition)


class _LayOutFrameLocals(TopDownVisitor):
    """Walk one frame's blocks, giving each not-yet-placed local variable the
    next offset. Does not descend into nested function bodies -- each of those
    owns its own frame.

    `offset` starts where the frame's first local goes: past the main frame's
    reserved slots (the sequence args, then the flags struct), or at 0 for a
    function frame, which reserves none. It ends past the last local, i.e. at
    the frame size."""

    def __init__(self, offset: int):
        super().__init__()
        self.offset = offset

    def visit_AstBlock(self, node: AstBlock, state: CompileState):
        frame_offsets = state.backend.frame_offsets
        for sym in state.enclosing_scope[node].group(NameGroup.VALUE).values():
            if is_instance_compat(sym, VariableSymbol) and sym not in frame_offsets:
                frame_offsets[sym] = self.offset
                self.offset += sym.type.max_size

    def visit_AstDef(self, node: AstDef, state: CompileState):
        return STOP_DESCENT


class AssignFrameOffsets(Visitor):
    """Assign every local variable its offset within its stack frame, and record
    each frame's total size.

    A frame is owned by a block: the main block (the global frame) or a
    function body. Its locals are the variables declared in that block and its
    nested blocks, except nested function bodies, which own their own frames.

    The main frame reserves slots below its locals for the sequence arguments,
    then the flags struct. A function frame reserves none -- its locals start at
    offset 0, and its formal parameters sit below the frame start at negative
    offsets, past the header CALL leaves there.

    We walk the whole tree only to find the frame owners: run() lays out the
    main frame, then visit_AstDef lays out each function's frame.

    This must run before GenerateFunctions so global variable offsets are known
    when generating function bodies that access them."""

    def run(self, start: Ast, state: CompileState):
        self._layout_main_frame(state)
        # The rest of the walk just finds the function definitions; each lays out
        # its own frame in visit_AstDef.
        super().run(start, state)

    def visit_AstDef(self, node: AstDef, state: CompileState):
        self._layout_function_frame(node, state)

    def _layout_main_frame(self, state: CompileState):
        # Sequence args arrive on the stack first, then the flags slot -- which
        # lives in the base scope but occupies a slot in the main frame here.
        frame_offsets = state.backend.frame_offsets
        offset = 0
        for name, arg_type in state.this_seq_arg_specs:
            arg_var = state.main_scope.group(NameGroup.VALUE)[name]
            frame_offsets[arg_var] = offset
            offset += arg_type.max_size
        frame_offsets[state.flags_var] = offset
        offset += state.flags_var.type.max_size

        state.backend.frame_sizes[state.main_block] = self._layout_locals(
            state.main_block, offset, state
        )

    def _layout_function_frame(self, node: AstDef, state: CompileState):
        # FIXME you can inline this func
        # Formal parameters sit before the frame start, at negative offsets.
        frame_offsets = state.backend.frame_offsets
        func = state.resolved_symbols[node.name]
        body_values = state.enclosing_scope[node.body].group(NameGroup.VALUE)
        arg_offset = -STACK_FRAME_HEADER_SIZE
        for arg_name, arg_type, _default in reversed(func.args):
            arg_offset -= arg_type.max_size
            frame_offsets[body_values[arg_name]] = arg_offset

        state.backend.frame_sizes[node.body] = self._layout_locals(node.body, 0, state)

    def _layout_locals(self, frame_block: AstBlock, offset: int, state) -> int:
        """Lay out every local in *frame_block*'s frame, starting at *offset*,
        and return the offset past the last one (the frame's total size)."""
        layout = _LayOutFrameLocals(offset)
        layout.run(frame_block, state)
        return layout.offset


class GenerateFunctionEntryPoints(Visitor):
    def visit_AstDef(self, node: AstDef, state: CompileState):
        if node not in state.backend.used_funcs:
            # Function is never called, skip it
            return
        entry_label = IrLabel(node, "entry")
        state.backend.func_entry_labels[node] = entry_label


class GenerateFunctions(Visitor):
    def visit_AstDef(self, node: AstDef, state: CompileState):
        if node not in state.backend.used_funcs:
            # Function is never called, skip generating code for it
            return
        entry_label = state.backend.func_entry_labels[node]
        code = [entry_label]

        # Allocate space for local variables
        frame_size_bytes = state.backend.frame_sizes[node.body]
        if frame_size_bytes > 0:
            code.append(AllocateDirective(frame_size_bytes))

        code.extend(GenerateFunctionBody().emit(node.body, state))
        func = state.resolved_symbols[node.name]
        if func.return_type is NOTHING and not state.does_return[node.body]:
            # implicit empty return
            arg_bytes = sum(arg[1].max_size for arg in (func.args or []))
            code.append(ReturnDirective(0, arg_bytes))
        state.backend.generated_funcs[node] = code


class EmitterWithNodeInfo(Emitter):
    """Stamps each emitted directive with the AST node that produced it, so
    errors raised on a directive can point at a source line."""

    def emit(self, node: Ast, state: CompileState) -> list[Directive | Ir]:
        dirs = super().emit(node, state)
        # Nested emit calls run first, so an existing stamp is the more
        # specific node. (Ir instances are frozen and are skipped; the
        # directives they become carry no arguments worth locating.)
        for dir in dirs:
            if isinstance(dir, Directive) and dir.source_node is None:
                dir.source_node = node
        return dirs


class GenerateFunctionBody(EmitterWithNodeInfo):
    # Flag indicating we're generating code inside a function body.
    # This affects how we access global variables: their frame offsets are
    # relative to the main frame, so from inside a function they can only be
    # reached by absolute offset (LOAD_ABS/STORE_ABS).
    in_function = True

    def _emit_func_arg(self, arg, state: CompileState) -> list[Directive | Ir]:
        """Emit code to push a function argument onto the stack.

        If *arg* is an FpyValue (a default value from a builtin or dictionary
        type constructor), serialize it directly.  Otherwise delegate to the
        normal AST emitter."""
        if isinstance(arg, FpyValue):
            return [PushValDirective(arg.serialize())]
        return self.emit(arg, state)

    def _emit_cmd_arg(
        self, arg, state: CompileState
    ) -> tuple[list[Directive | Ir], int]:
        """Emit a command argument, returning (directives, actual_byte_count).

        For compile-time constants the actual serialized size may be smaller
        than the type's max_size (e.g. strings serialize compactly).
        The caller must use the returned byte count for StackCmdDirective
        accounting so it matches the bytes actually placed on the stack.
        """
        const_val = (
            arg if isinstance(arg, FpyValue) else state.const_expr_values.get(arg)
        )
        if const_val is not None:
            serialized = const_val.serialize()
            return [PushValDirective(serialized)], len(serialized)
        dirs = self.emit(arg, state)
        return dirs, state.contextual_types[arg].max_size

    def _emit_seq_run_cmd(
        self,
        node: AstFuncCall,
        func: CommandSymbol,
        state: CompileState,
    ) -> list[Directive | Ir]:
        """Emit a sequence-run command.

        Serializes the two fixed args (fileName, blockState) plus a SeqArgs
        struct containing the vararg values packed into its buffer.
        """
        resolved_args = state.resolved_args[node]

        # Split resolved args into fixed (command) args and seq args.
        # ResolveSequenceDependencies extended func.args to include the
        # target sequence's parameters.
        bin_name = resolved_args[0].value
        seq_dep = state.called_seq_arg_specs[bin_name]
        seq_arg_types = [t for _, t in seq_dep]
        n_fixed = len(func.args) - len(seq_dep)
        fixed_args = resolved_args[:n_fixed]
        seq_args = resolved_args[n_fixed:]

        # Compute the actual data size in the SeqArgs buffer
        # vararg data guaranteed no strings
        vararg_data_size = sum(t.max_size for t in seq_arg_types)
        buffer_size = SEQ_ARGS.members[1].type.length
        padding_size = buffer_size - vararg_data_size
        size_type = SEQ_ARGS.members[0].type
        size_bytes = FpyValue(size_type, vararg_data_size).serialize()

        # Check if all args (fixed + seq) are compile-time constants
        # fixed args may have strings (almost certainly does of course)
        # but we don't need to know the actual byte count. just push as a byte array
        # as part of const cmd
        all_fixed_const = all(
            isinstance(a, FpyValue) or state.const_expr_values.get(a) is not None
            for a in fixed_args
        )
        all_seq_const = all(
            isinstance(a, FpyValue) or state.const_expr_values.get(a) is not None
            for a in seq_args
        )

        if all_fixed_const and all_seq_const:
            # All constant: build the full command payload at compile time
            arg_bytes = bytes()
            # Fixed args
            for a in fixed_args:
                val = a if isinstance(a, FpyValue) else state.const_expr_values[a]
                arg_bytes += val.serialize()
            # SeqArgs struct: $size + buffer (seq arg data + padding)
            arg_bytes += size_bytes
            for a in seq_args:
                val = a if isinstance(a, FpyValue) else state.const_expr_values[a]
                arg_bytes += val.serialize()
            arg_bytes += bytes(padding_size)
            return [ConstCmdDirective(func.cmd.opcode, arg_bytes)]
        else:
            dirs = []
            arg_byte_count = 0

            # Push fixed args
            # okay, for StackCmd we actually need to know at compile time the size of
            # the args we pushed. so we use this emit cmd arg func which tells us the
            # size (if the arg is a const value, which is always the case for strings rn, which
            # are the only type which are not always their max_size when serialized)
            for a in fixed_args:
                arg_dirs, actual_size = self._emit_cmd_arg(a, state)
                dirs.extend(arg_dirs)
                arg_byte_count += actual_size

            # Push SeqArgs struct: $size field
            # size_bytes is guaranteed to represent the real size, because the
            # sequence args cannot contain strings
            dirs.append(PushValDirective(size_bytes))
            arg_byte_count += size_type.max_size

            # Push seq arg values
            for a in seq_args:
                arg_dirs, actual_size = self._emit_cmd_arg(a, state)
                dirs.extend(arg_dirs)
                arg_byte_count += actual_size

            # Push zero padding to fill the rest of the buffer
            if padding_size > 0:
                dirs.append(AllocateDirective(size=padding_size))
                arg_byte_count += padding_size

            # Push opcode, then emit stack command
            dirs.append(
                PushValDirective(FpyValue(FwOpcodeType, func.cmd.opcode).serialize())
            )
            stack_cmd = StackCmdDirective(arg_byte_count)
            stack_cmd.cmd_opcode = func.cmd.opcode
            dirs.append(stack_cmd)
            return dirs

    def try_emit_expr_as_const(
        self, node: AstExpr, state: CompileState
    ) -> Union[list[Directive | Ir], None]:
        """if the expr has a compile time const value, emit that as a PUSH_VAL"""
        expr_value = state.const_expr_values.get(node)

        if expr_value is None:
            # no const value
            return None

        assert isinstance(expr_value, FpyValue) and expr_value.type not in (
            INTEGER,
            INTERNAL_STRING,
            FLOAT,
        ), expr_value

        if expr_value is NOTHING_VALUE:
            # nothing type has no value
            return []

        # it has a constant value at compile time
        serialized_expr_value = expr_value.serialize()

        # push it to the stack
        return [PushValDirective(serialized_expr_value)]

    def _emit_discard_expr_result(
        self, node: Ast, state: CompileState
    ) -> list[Directive]:
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

    def _emit_assert_cmd_response_ok(
        self, node: AstFuncCall, state: CompileState
    ) -> list[Directive | Ir]:
        """For a bare command call, emit code to check the response and exit if
        it is not OK and the flags.assert_cmd_success variable is set."""
        dirs: list[Directive | Ir] = []
        end_label = IrLabel(node, "cmd_ok")
        # compare response on stack to Fw.CmdResponse.OK
        dirs.append(
            PushValDirective(
                FpyValue(CMD_RESPONSE, CMD_RESPONSE.enum_dict["OK"]).serialize()
            )
        )
        dirs.append(MemCompareDirective(CMD_RESPONSE.max_size))
        # now stack has True if response == OK
        # if response was OK, skip to end, otherwise go to "cmd_not_ok"
        not_ok_label = IrLabel(node, "cmd_not_ok")
        dirs.append(IrIf(not_ok_label))
        dirs.append(IrGoto(end_label))

        dirs.append(not_ok_label)
        # response was not OK — read flags.assert_cmd_success from the stack
        # assert_cmd_success is at offset 0 within the flags struct
        flag_offset = state.backend.frame_offsets[state.flags_var]
        dirs.append(LoadAbsDirective(flag_offset, BOOL.max_size))
        # if flag is false, skip to end (don't exit)
        dirs.append(IrIf(end_label))
        # flag is true and response was not OK — exit with error
        dirs.append(
            PushValDirective(
                FpyValue(I32, DirectiveErrorCode.CMD_FAIL.value).serialize()
            )
        )
        dirs.append(ExitDirective())
        dirs.append(end_label)
        return dirs

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

    def _emit_array_element_offset(
        self, node: Ast, idx_expr: AstExpr, array_type: FpyType, state: CompileState
    ) -> list[Directive | Ir]:
        """generates code to bounds check the index *idx_expr* into an array of
        *array_type* and push its element's U64 byte offset within that array.
        The offset is relative to the array's own start, so a caller that wants
        a frame offset must add the array's."""
        dirs = []
        # push the index to the stack, do a bounds check,
        dirs.extend(self.emit(idx_expr, state))
        # okay now let's do an array oob check
        # we want to peek the index so we can consume it for the oob check
        # byte count
        dirs.append(
            PushValDirective(
                FpyValue(StackSizeType, ArrayIndexType.max_size).serialize()
            )
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
            PushValDirective(
                FpyValue(StackSizeType, ArrayIndexType.max_size).serialize()
            )
        )
        # offset is 1 because we currently have the result of the last check on stack
        dirs.append(PushValDirective(FpyValue(StackSizeType, 1).serialize()))
        dirs.append(PeekDirective())  # duplicate the index
        # convert idx to i64
        dirs.extend(self.convert_numeric_type(ArrayIndexType, I64))
        dirs.append(PushValDirective(FpyValue(I64, 0).serialize()))  # push 0 as i64
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
                FpyValue(I32, DirectiveErrorCode.ARRAY_OUT_OF_BOUNDS.value).serialize()
            )
        )
        dirs.append(ExitDirective())
        dirs.append(oob_check_end_label)
        # okay we're good. should still have the idx on the stack

        # multiply the index by the member type size
        dirs.append(
            PushValDirective(FpyValue(U64, array_type.elem_type.max_size).serialize())
        )
        dirs.append(IntMultiplyDirective())
        return dirs

    def _should_lower_stmt(self, stmt: Ast, state: CompileState) -> bool:
        """Whether a statement needs code generated for it.

        Constants are skipped, and this is required, not just an optimization: a
        bare statement gives its expression no type context, so a folded literal
        keeps its *abstract* type (Integer/Float/InternalString), which has no
        serialized representation -- emitting `2 + 2` would assert in
        try_emit_expr_as_const / FpyValue.serialize. They're also pure (const
        folding only folds pure expressions), so dropping them changes nothing.
        """
        if is_instance_compat(stmt, AstNodeWithSideEffects):
            return True
        if is_instance_compat(stmt, AstExpr):
            return state.const_expr_values.get(stmt) is None
        return False

    def emit_AstBlock(self, node: AstBlock, state: CompileState):
        dirs = []
        for stmt in node.stmts:
            if is_instance_compat(stmt, AstBlock):
                # a sub block. this is only possible if it is an imported sequence
                # emit its statements inline in this frame
                dirs.extend(self.emit(stmt, state))
                continue
            if not self._should_lower_stmt(stmt, state):
                continue
            dirs.extend(self.emit(stmt, state))
            if is_cmd_and_response_unhandled(stmt, state):
                dirs.extend(self._emit_assert_cmd_response_ok(stmt, state))
            else:
                # discard stack value if it was an expr
                dirs.extend(self._emit_discard_expr_result(stmt, state))
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
        state.backend.while_loop_start_labels[node] = while_start_label
        state.backend.while_loop_end_labels[node] = while_end_label
        # if this used to be a for loop:
        if node in state.desugared_for_loops:
            # there should be at least one stmt in a for loop's body (the inc stmt)
            for_loop_increment_label = IrLabel(node, "increment")
            state.backend.for_loop_inc_labels[node] = for_loop_increment_label

        dirs = [while_start_label]
        # push the condition to the stack
        dirs.extend(self.emit(node.condition, state))
        # if the cond is true, fall thru, otherwise go to end
        dirs.append(IrIf(while_end_label))
        # run body

        for stmt_idx, stmt in enumerate(node.body.stmts):
            if not self._should_lower_stmt(stmt, state):
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
            if is_cmd_and_response_unhandled(stmt, state):
                dirs.extend(self._emit_assert_cmd_response_ok(stmt, state))
            else:
                # discard stack value if it was an expr
                dirs.extend(self._emit_discard_expr_result(stmt, state))
        # go back to condition check
        dirs.append(IrGoto(while_start_label))
        dirs.append(while_end_label)

        return dirs

    def emit_AstBreak(self, node: AstBreak, state: CompileState):
        enclosing_loop = state.enclosing_loops[node]
        loop_end = state.backend.while_loop_end_labels[enclosing_loop]
        return [IrGoto(loop_end)]

    def emit_AstContinue(self, node: AstContinue, state: CompileState):
        enclosing_loop = state.enclosing_loops[node]
        if enclosing_loop in state.desugared_for_loops:
            loop_start = state.backend.for_loop_inc_labels[enclosing_loop]
        else:
            loop_start = state.backend.while_loop_start_labels[enclosing_loop]
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

        if is_instance_compat(node.parent, AstAnonArray):
            # Direct index access on anonymous array literal.
            # The index must be a compile-time constant.
            idx_value = state.const_expr_values.get(node.item)
            assert (
                idx_value is not None
            ), "Dynamic indexing on anonymous array literals is not supported"
            idx = idx_value.val
            assert 0 <= idx < len(node.parent.elements), f"Index {idx} out of bounds"
            dirs = self.emit(node.parent.elements[idx], state)
            converted_type = state.contextual_types[node]
            if unconverted_type != converted_type:
                dirs.extend(self.convert_numeric_type(unconverted_type, converted_type))
            return dirs

        # however, for parent, use converted because conversion has been run
        parent_type = state.contextual_types[node.parent]

        assert parent_type.kind == TypeKind.ARRAY
        assert unconverted_type == parent_type.elem_type, (
            parent_type.elem_type,
            unconverted_type,
        )

        # okay, we want to get an element from an array on the stack

        # TODO optimization: read the element in place at its frame offset
        # instead of copying the whole array to the top of the stack
        # for now we push the whole thing
        dirs = self.emit(node.parent, state)

        # calculate the offset in the parent array
        dirs.extend(
            self._emit_array_element_offset(node, node.item, parent_type, state)
        )
        # truncate back to StackSizeType which is what get field uses
        dirs.extend(self.convert_numeric_type(U64, StackSizeType))

        # get the member from the stack at this offset, discard the rest of
        # the parent
        dirs.append(
            GetFieldDirective(parent_type.max_size, parent_type.elem_type.max_size)
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

        # Use the absolute directives only when inside a function AND accessing
        # a global variable. At top level, stack_frame_start = 0, so a
        # frame-relative offset is already the absolute one.
        use_abs = self.in_function and sym.is_global
        offset = state.backend.frame_offsets[sym]
        if use_abs:
            dirs = [LoadAbsDirective(offset, sym.type.max_size)]
        else:
            dirs = [LoadRelDirective(offset, sym.type.max_size)]

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

        # A qualified name can't denote a variable (an imported sequence may
        # not declare a top-level variable), so sym is never a VariableSymbol.
        if is_instance_compat(sym, ChDef):
            dirs.append(PushTlmValDirective(sym.ch_id))
        elif is_instance_compat(sym, PrmDef):
            dirs.append(PushPrmDirective(sym.prm_id))
        elif is_instance_compat(sym, FieldAccess):
            if is_instance_compat(sym.parent_expr, AstAnonStruct):
                # Direct member access on anonymous struct literal.
                # Emit just the accessed member expression (skip the struct build).
                for name, value_expr in sym.parent_expr.members:
                    if name == node.attr:
                        dirs.extend(self.emit(value_expr, state))
                        break
                else:
                    assert False, f"Member {node.attr} not found in anon struct"
            else:
                # okay, put parent dirs in first
                dirs.extend(self.emit(sym.parent_expr, state))
                assert sym.local_offset is not None
                # use the converted type of parent
                parent_type = state.contextual_types[sym.parent_expr]
                # push the offset to the stack
                dirs.append(
                    PushValDirective(
                        FpyValue(StackSizeType, sym.local_offset).serialize()
                    )
                )
                dirs.append(
                    GetFieldDirective(parent_type.max_size, unconverted_type.max_size)
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
            elif node.op == BinaryStackOp.FLOOR_DIVIDE and intermediate_type == F64:
                # float floor division: divide, then floor toward -inf
                dirs.append(FloatDivideDirective())
                dirs.append(FloatFloorDirective())
            else:

                dir = BINARY_STACK_OPS[node.op][intermediate_type]
                if dir != NoOpDirective:
                    # don't include no op
                    dirs.append(dir())

            # The VM operates on 64-bit values, so after the op we have a 64-bit result.
            # Convert from the 64-bit intermediate type to the synthesized result type.
            synthesized_type = state.synthesized_types[node]
            if (
                intermediate_type in SPECIFIC_NUMERIC_TYPES
                and synthesized_type in SPECIFIC_NUMERIC_TYPES
            ):
                dirs.extend(
                    self.convert_numeric_type(intermediate_type, synthesized_type)
                )

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
        if (
            intermediate_type in SPECIFIC_NUMERIC_TYPES
            and synthesized_type in SPECIFIC_NUMERIC_TYPES
        ):
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
        if is_instance_compat(func, CommandSymbol) and func.is_seq_run_with_args:
            dirs = self._emit_seq_run_cmd(node, func, state)
        elif is_instance_compat(func, CommandSymbol):
            const_args = all(
                isinstance(arg_node, FpyValue)
                or (state.const_expr_values[arg_node] is not None)
                for arg_node in node_args
            )
            if const_args:
                # can just hardcode this cmd
                arg_bytes = bytes()
                for arg_node in node_args:
                    arg_value = (
                        arg_node
                        if isinstance(arg_node, FpyValue)
                        else state.const_expr_values[arg_node]
                    )
                    arg_bytes += arg_value.serialize()
                dirs.append(ConstCmdDirective(func.cmd.opcode, arg_bytes))
            else:
                arg_byte_count = 0
                # push all args to the stack
                # keep track of how many bytes total we have pushed
                for arg_node in node_args:
                    arg_dirs, actual_size = self._emit_cmd_arg(arg_node, state)
                    dirs.extend(arg_dirs)
                    arg_byte_count += actual_size
                # then push cmd opcode to stack as u32
                dirs.append(
                    PushValDirective(
                        FpyValue(FwOpcodeType, func.cmd.opcode).serialize()
                    )
                )
                # now that all args are pushed to the stack, pop them and opcode off the stack
                # as a command
                stack_cmd = StackCmdDirective(arg_byte_count)
                stack_cmd.cmd_opcode = func.cmd.opcode
                dirs.append(stack_cmd)
        elif is_instance_compat(func, BuiltinFuncSymbol):
            # collect compile-time constant args (not pushed to stack)
            const_arg_values: dict[int, FpyValue] = {}
            for i in func.const_arg_indices:
                arg = node_args[i]
                const_val = (
                    arg
                    if isinstance(arg, FpyValue)
                    else state.const_expr_values.get(arg)
                )
                assert (
                    const_val is not None
                ), f"const arg {i} of {func.name} should have been validated by semantics"
                const_arg_values[i] = const_val

            # write_to_port's generate_fpybc is empty; the directive is built
            # here instead, because only this backend knows the port index and
            # the value's serialized size. Its argument types (SerialPortIndex
            # and SIZED) already did the validating.
            if func.name == "write_to_port":
                value_arg = node_args[1]
                # Push the value for the directive to pop and send.
                dirs.extend(self._emit_func_arg(value_arg, state))
                # Value is coerced to a concrete sized type, so max_size is the exact size to pop.
                size = state.contextual_types[value_arg].max_size
                # Port is a const dictionary SerialPortIndex enum; .val is the constant name, resolve to its int index.
                port_val = const_arg_values[0]
                assert isinstance(port_val.val, str), port_val
                port_index = port_val.type.enum_dict[port_val.val]
                dirs.append(PopSerializableDirective(portIndex=port_index, size=size))
            else:
                # put non-const arg values on stack
                for i, arg_node in enumerate(node_args):
                    if i not in func.const_arg_indices:
                        dirs.extend(self._emit_func_arg(arg_node, state))

                dirs.extend(func.generate_fpybc(node, const_arg_values))
        elif is_instance_compat(func, TypeCtorSymbol):
            # put arg values onto stack in correct order for serialization
            for arg_node in node_args:
                dirs.extend(self._emit_func_arg(arg_node, state))
        elif is_instance_compat(func, CastSymbol):
            # just putting the arg value on the stack should be good enough, the
            # conversion will happen below
            dirs.extend(self.emit(node_args[0], state))
        elif is_instance_compat(func, FunctionSymbol):
            # script-defined function
            # okay.. calling convention says we're going to put the args on the stack
            for arg_node in node_args:
                dirs.extend(self._emit_func_arg(arg_node, state))
            # okay, args are on the stack. now we're going to generate CALL
            func_entry_label = state.backend.func_entry_labels[func.definition]
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

    def _compute_field_access_offset(
        self, lhs: FieldAccess, state: CompileState
    ) -> tuple[int, list[tuple[AstExpr, FpyType]]]:
        """Walk the FieldAccess chain to compute offset components.

        Returns (const_offset, dynamic_components) where:
        - const_offset is the sum of all constant byte offsets from the base
          variable (struct member offsets + constant-index array element offsets)
        - dynamic_components is a list of (idx_expr, array_type) for each
          array element access whose index is not known at compile time

        The total offset is the sum of these two.
        """
        const_offset = 0
        dynamic_components = []

        current = lhs
        while is_instance_compat(current, FieldAccess):
            if current.is_struct_member:
                assert current.local_offset is not None
                const_offset += current.local_offset
            elif current.is_array_element:
                parent_type = state.contextual_types[current.parent_expr]
                const_idx = state.const_expr_values.get(current.idx_expr)
                if const_idx is not None:
                    assert (
                        isinstance(const_idx, FpyValue)
                        and const_idx.type == ArrayIndexType
                    )
                    const_offset += const_idx.val * parent_type.elem_type.max_size
                else:
                    dynamic_components.append((current.idx_expr, parent_type))
            current = state.resolved_symbols.get(current.parent_expr)

        return const_offset, dynamic_components

    def emit_AstAssign(self, node: AstAssign, state: CompileState):
        lhs = state.resolved_symbols[node.lhs]

        is_global_var = False
        # field_const_offset: the constant byte offset within the variable's
        # storage (from struct member offsets and constant-index array elements).
        # dynamic_components: variable-index array accesses whose offsets can
        # only be computed at runtime.
        field_const_offset = 0
        dynamic_components = []

        if is_instance_compat(lhs, VariableSymbol):
            base_frame_offset = state.backend.frame_offsets[lhs]
            is_global_var = lhs.is_global
        else:
            assert is_instance_compat(lhs, FieldAccess), lhs
            assert is_instance_compat(lhs.base_sym, VariableSymbol), lhs.base_sym
            base_frame_offset = state.backend.frame_offsets[lhs.base_sym]
            is_global_var = lhs.base_sym.is_global

            # Walk the field access chain to compute the total offset.
            field_const_offset, dynamic_components = self._compute_field_access_offset(
                lhs, state
            )

        # Use the absolute directives only when inside a function AND accessing
        # a global variable
        use_abs = self.in_function and is_global_var

        # start with rhs on stack
        dirs = self.emit(node.rhs, state)

        if not dynamic_components:
            # The full frame offset is known at compile time.
            frame_offset = base_frame_offset + field_const_offset
            if use_abs:
                dirs.append(
                    StoreAbsConstOffsetDirective(frame_offset, lhs.type.max_size)
                )
            else:
                dirs.append(
                    StoreRelConstOffsetDirective(frame_offset, lhs.type.max_size)
                )
        else:
            # At least one array index in the access chain is not known at
            # compile time.  Compute the total offset dynamically.
            assert is_instance_compat(lhs, FieldAccess), lhs
            assert dynamic_components, lhs

            # Emit code to compute each dynamic offset component
            # (idx * elem_size) and sum them together on the stack.
            for i, (idx_expr, parent_type) in enumerate(dynamic_components):
                dirs.extend(
                    self._emit_array_element_offset(node, idx_expr, parent_type, state)
                )
                if i > 0:
                    dirs.append(IntAddDirective())

            # Add the constant part: base variable's frame offset +
            # accumulated constant field offsets.
            const_part = base_frame_offset + field_const_offset
            dirs.append(PushValDirective(FpyValue(U64, const_part).serialize()))
            dirs.append(IntAddDirective())

            # and now convert the u64 back into the SignedStackSizeType that store expects
            dirs.extend(self.convert_numeric_type(U64, SignedStackSizeType))

            # now that the frame offset is pushed, use it to store into the frame
            if use_abs:
                dirs.append(StoreAbsDirective(lhs.type.max_size))
            else:
                dirs.append(StoreRelDirective(lhs.type.max_size))

        return dirs

    def emit_AstLiteral(self, node: AstLiteral, state: CompileState):
        const_dirs = self.try_emit_expr_as_const(node, state)
        assert const_dirs is not None
        return const_dirs

    def emit_AstAnonStruct(self, node: AstAnonStruct, state: CompileState):
        # Try to emit as a constant first
        const_dirs = self.try_emit_expr_as_const(node, state)
        if const_dirs is not None:
            return const_dirs

        # Emit each resolved member value in target struct order
        dirs = []
        resolved_members = state.resolved_args[node]
        for member_expr in resolved_members:
            dirs.extend(self._emit_func_arg(member_expr, state))
        return dirs

    def emit_AstAnonArray(self, node: AstAnonArray, state: CompileState):
        # Try to emit as a constant first
        const_dirs = self.try_emit_expr_as_const(node, state)
        if const_dirs is not None:
            return const_dirs

        # Emit each element value
        dirs = []
        resolved_elements = state.resolved_args[node]
        for elem_expr in resolved_elements:
            dirs.extend(self._emit_func_arg(elem_expr, state))
        return dirs

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
                    FpyValue(I32, DirectiveErrorCode.EXIT_WITH_ERROR.value).serialize()
                )
            )
        dirs.append(ExitDirective())
        dirs.append(end_label)

        return dirs


class GenerateSequence(EmitterWithNodeInfo):

    def emit_AstBlock(self, node: AstBlock, state: CompileState):
        if node is not state.main_block:
            return []

        main_body = []

        # the structure of the lvar section of the main stack frame is:
        # (sequence args) (flags struct) (user-defined lvars)

        # sequence args will be pushed to stack before the first instruction is
        # executed. flags struct, we will push a value (won't just write zeroes)
        # for user-defined lvars, we will have to write zeroes with Allocate

        flags_type = state.flags_var.type
        args_size = sum(t.max_size for _, t in state.this_seq_arg_specs)
        assert state.backend.frame_offsets[state.flags_var] == args_size
        flags_default = FpyValue(flags_type, dict(flags_type.member_defaults))
        main_body.append(PushValDirective(flags_default.serialize()))

        # we can calc how much space the user-defined lvars take by subtracting
        # the sequence args size, and the flags size, from the frame size

        remaining = state.backend.frame_sizes[node] - flags_type.max_size - args_size
        assert remaining >= 0, remaining

        # allocate space for local variables
        if remaining > 0:
            main_body.append(AllocateDirective(remaining))

        # generate the main body using GenerateTopLevel (not in a function context)
        main_body.extend(GenerateTopLevel().emit(node, state))

        # if there are functions, emit them at the top with a goto to skip past them
        if state.backend.generated_funcs:
            funcs_code = []
            func_code_end_label = IrLabel(node, "main")
            funcs_code.append(IrGoto(func_code_end_label))
            for func, code in state.backend.generated_funcs.items():
                funcs_code.extend(code)
            funcs_code.append(func_code_end_label)
            return funcs_code + main_body

        return main_body


class GenerateTopLevel(GenerateFunctionBody):
    """Generates top-level (main) code, not inside any function.
    At top level, stack_frame_start = 0, so frame-relative and absolute offsets
    are equivalent. All variables use the relative directives."""

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
                dirs.append(
                    PushValDirective(FpyValue(StackSizeType, labels[label]).serialize())
                )
            else:
                dirs.append(dir)

        return dirs


class FinalChecks(IrPass):
    def run(self, ir, state):
        if len(ir) > state.max_directives_count:
            return BackendError(
                f"Too many directives in sequence (expected at most {state.max_directives_count}, had {len(ir)})"
            )

        for dir in ir:
            # double check we've got rid of all the IR
            assert is_instance_compat(dir, Directive), dir

            # mirrors the sequencer's statement deserialization limit
            # (Svc.Fpy.MAX_DIRECTIVE_SIZE)
            dir_size = len(dir.serialize())
            if dir_size > state.max_directive_size:
                return BackendError(
                    f"Directive {dir.opcode.name} in sequence too large (expected at most "
                    f"{state.max_directive_size} bytes, was {dir_size})",
                    dir.source_node,
                )

            # commands are serialized into an Fw::ComBuffer as
            # (packet descriptor, opcode, args) and their args are copied into
            # an Fw::CmdArgBuffer by the command dispatcher; a sequence whose
            # commands exceed either capacity always fails at runtime
            if is_instance_compat(dir, ConstCmdDirective):
                cmd_args_size = len(dir.args)
            elif is_instance_compat(dir, StackCmdDirective):
                # codegen stamps the opcode; only codegen output reaches here
                assert dir.cmd_opcode is not None
                cmd_args_size = dir.args_size
            else:
                continue
            cmd_desc = f"Command {state.cmd_names_by_opcode.get(dir.cmd_opcode, hex(dir.cmd_opcode))}"

            if (
                state.cmd_arg_buffer_max_size is not None
                and cmd_args_size > state.cmd_arg_buffer_max_size
            ):
                return BackendError(
                    f"{cmd_desc} has {cmd_args_size} bytes of arguments, which "
                    f"exceeds FW_CMD_ARG_BUFFER_MAX_SIZE ({state.cmd_arg_buffer_max_size})",
                    dir.source_node,
                )

            cmd_packet_size = (
                FwPacketDescriptorType.max_size + FwOpcodeType.max_size + cmd_args_size
            )
            if (
                state.com_buffer_max_size is not None
                and cmd_packet_size > state.com_buffer_max_size
            ):
                return BackendError(
                    f"{cmd_desc} serializes to a {cmd_packet_size} byte packet "
                    f"(packet descriptor + opcode + {cmd_args_size} bytes of arguments), "
                    f"which exceeds FW_COM_BUFFER_MAX_SIZE ({state.com_buffer_max_size})",
                    dir.source_node,
                )

        return ir
