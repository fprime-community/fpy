from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from fpy.error import BackendError

try:
    from llvmlite import ir
    import llvmlite.binding as llvm
except ImportError as exc:
    raise BackendError(
        "the 'llvmlite' package is required by the LLVM backend; install it "
        "with 'pip install fprime-fpy[wasm]'"
    ) from exc

from fpy.bytecode.directives import DirectiveErrorCode
from fpy.state import BackendState, CompileState
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
from fpy.syntax import (
    Ast,
    AstAnonArray,
    AstAnonStruct,
    AstAssert,
    AstAssign,
    AstBinaryOp,
    AstBlock,
    AstBreak,
    AstContinue,
    AstDef,
    AstExpr,
    AstFuncCall,
    AstGetAttr,
    AstIdent,
    AstIf,
    AstIndexExpr,
    AstNodeWithSideEffects,
    AstReturn,
    AstUnaryOp,
    AstWhile,
    BinaryStackOp,
    COMPARISON_OPS,
    UnaryStackOp,
)
from fpy.bytecode.directives import FwOpcodeType
from fpy.semantics import is_cmd_and_response_unhandled
import fpy.types
from fpy.types import (
    CMD_RESPONSE,
    ChDef,
    FpyType,
    FpyValue,
    PrmDef,
    TypeKind,
    is_instance_compat,
)
from fpy.visitors import STOP_DESCENT, Emitter, TopDownVisitor, Visitor
from fpy.wasm_host import (
    ERROR_CODE_TYPE,
    HOST_CMD_FUNC_NAME,
    HOST_EXIT_FUNC_NAME,
    HOST_PANIC_FUNC_NAME,
    declare_host_imports,
)

LLVM_TRIPLE = "wasm32-unknown-unknown"

# Target the original WebAssembly 1.0 MVP (the W3C Core Spec 1.0)
LLVM_CPU = "mvp"
WASM_VERSION = "1.0 (MVP)"

# TODO enable custom page sizes
# TODO strip custom sections from the wasm / optimize for size (wasm-opt)?
# TODO could just start with a .a and a header??


@dataclass
class CommandBuffer:
    """The buffer one command call dispatches: the big-endian serialized
    FwOpcodeType followed by the fprime-serialized arguments, in linear
    memory."""

    buf: "ir.GlobalVariable"
    size: int
    runtime_args: list[tuple[int, AstExpr, FpyType]]
    """the arguments not known at compile time, as (byte offset of the zeroed
    slot the buffer leaves for it, argument expression, argument type)"""


@dataclass
class LlvmBackendState(BackendState):
    """The LLVM backend's view of a program: the module it is building, and
    where in that module everything the program stores lives."""

    module: "ir.Module"

    variable_ptrs: dict[VariableSymbol, "ir.Value"] = field(default_factory=dict)
    """variable to the pointer to its storage"""

    temp_slots: dict[AstExpr, "ir.Value"] = field(default_factory=dict)
    """expression that is addressed but denotes no location of its own, to the
    stack slot holding its value"""

    cmd_buffers: dict[AstFuncCall, CommandBuffer] = field(default_factory=dict)
    """command call to the buffer it dispatches"""

    funcs: dict[AstDef, "ir.Function"] = field(default_factory=dict)
    """used function definition to the LLVM function it lowers to"""


def is_addressable(expr: AstExpr, state: CompileState) -> bool:
    """True when *expr* denotes a location rather than a computed value, so
    that a pointer to it can be formed: *expr* is a variable, or a
    member/element access. An access into an anonymous literal is the
    exception -- that literal is never built, so the access reduces to the
    accessed member's own expression instead."""
    sym = state.resolved_symbols.get(expr)
    if is_instance_compat(sym, VariableSymbol):
        return True
    return is_instance_compat(sym, FieldAccess) and not is_instance_compat(
        sym.parent_expr, (AstAnonStruct, AstAnonArray)
    )


class EmitLlvmExpr(Emitter):
    """Lowers a single Fpy arithmetic/comparison expression into LLVM IR.

    Each ``emit_*`` returns the ``ir.Value`` holding the computed result at the
    node's *synthesized* type; emit() then converts it to the node's contextual
    (coerced) type. (Constants are stored already at their contextual type, so
    they skip that conversion.)
    """

    def __init__(self, builder: ir.IRBuilder):
        super().__init__()
        self.builder: ir.IRBuilder = builder

    def emit(self, node, state: CompileState) -> ir.Value:
        value = state.const_expr_values.get(node)
        if value is not None:
            if not value.type.is_concrete:
                # Only a bare expression statement leaves a constant at an
                # abstract type (Integer/Float/InternalString/anon literal):
                # every other position coerces its expression to a concrete
                # contextual type. An abstract value has no machine
                # representation, and a constant is pure (const folding only
                # folds pure expressions), so there is nothing to emit.
                return None
            # Const values are stored at the node's contextual type already.
            return value.llvm_value
        emitter = self.emitters.get(type(node))
        if emitter is None:
            raise BackendError(
                f"LLVM backend can't lower expression {type(node).__name__} yet"
            )
        result = emitter(node, state)
        if result is None:
            # NOTHING-typed expr, nothing to convert.
            return None
        synthesized = state.synthesized_types[node]
        contextual = state.contextual_types[node]
        if synthesized != contextual:
            result = self.convert_numeric_type(result, synthesized, contextual)
        return result

    def emit_AstBinaryOp(self, node: AstBinaryOp, state: CompileState) -> ir.Value:
        # `and`/`or` short-circuit, so they must branch rather than eagerly
        # evaluate both operands.
        if node.op in (BinaryStackOp.AND, BinaryStackOp.OR):
            return self._emit_short_circuit(node, state)

        intermediate_type = state.op_intermediate_types[node]
        is_float = intermediate_type.is_float
        is_signed = intermediate_type.is_signed
        b = self.builder

        # Operands are coerced to the intermediate type during semantics, so the
        # emitted values are already at that type.
        lhs = self.emit(node.lhs, state)
        rhs = self.emit(node.rhs, state)
        op = node.op

        # -- arithmetic: result is the (numeric) intermediate type ------------
        if op == BinaryStackOp.ADD:
            return b.fadd(lhs, rhs) if is_float else b.add(lhs, rhs)
        if op == BinaryStackOp.SUBTRACT:
            return b.fsub(lhs, rhs) if is_float else b.sub(lhs, rhs)
        if op == BinaryStackOp.MULTIPLY:
            return b.fmul(lhs, rhs) if is_float else b.mul(lhs, rhs)
        if op == BinaryStackOp.DIVIDE:
            # `/` always computes over floats (semantics widens to F64).
            assert is_float, intermediate_type
            return b.fdiv(lhs, rhs)
        if op == BinaryStackOp.MODULUS:
            return self._emit_modulo(lhs, rhs, is_float, is_signed)
        if op == BinaryStackOp.FLOOR_DIVIDE:
            return self._emit_floor_divide(lhs, rhs, is_float, is_signed)
        if op == BinaryStackOp.EXPONENT:
            # `**` always computes over floats (semantics widens to F64).
            assert is_float, intermediate_type
            # assume that the host provides a pow func
            pow_fn = b.module.declare_intrinsic(
                "llvm.pow", [intermediate_type.llvm_type]
            )
            return b.call(pow_fn, [lhs, rhs])

        assert op in COMPARISON_OPS, op
        if is_float:
            # IEEE `!=` is the negation of `==` and is therefore true when
            # either operand is NaN (une, wasm's f64.ne, Python's !=). Every
            # other comparison is ordered: false on NaN, like Python's.
            if op == "!=":
                return b.fcmp_unordered(op, lhs, rhs)
            return b.fcmp_ordered(op, lhs, rhs)
        # Enums and bools lower to integers too, so any integer-typed value
        # (not just numeric types) compares with icmp; aggregates don't.
        if isinstance(lhs.type, ir.IntType):
            return (
                b.icmp_signed(op, lhs, rhs)
                if is_signed
                else b.icmp_unsigned(op, lhs, rhs)
            )
        raise BackendError(
            f"LLVM backend can't compare values of type "
            f"'{intermediate_type.display_name}' yet"
        )

    def _emit_floor_divide(
        self, lhs: ir.Value, rhs: ir.Value, is_float: bool, is_signed: bool
    ) -> ir.Value:
        """Emit `lhs // rhs`, flooring toward -inf (Python `//`; spec "Floor
        division semantics").

        Floats floor the quotient directly via llvm.floor (a native f64.floor on
        wasm). Integer sdiv/udiv truncate toward zero, which differs from floor
        only when the operands have opposite signs and the division is inexact;
        in that case we subtract one.
        """
        b = self.builder
        if is_float:
            # Floats have a real floor: divide, then floor the quotient. The
            # llvm.floor intrinsic lowers to a native f64.floor on wasm (no
            # libcall), so e.g. -5.5 / 2.0 = -2.75 floors to -3.0. A zero
            # divisor needs no guard: float division is IEEE (inf/nan), and
            # floor passes that through ("an infinite or NaN quotient is
            # unchanged" per the spec).
            quotient = b.fdiv(lhs, rhs)
            floor_fn = b.module.declare_intrinsic("llvm.floor", [quotient.type])
            return b.call(floor_fn, [quotient])
        # The spec makes an integer zero divisor a DOMAIN_ERROR fault; wasm's
        # integer div instructions would instead trap uncatchably, so guard
        # first.
        self._emit_zero_divisor_check(rhs, is_float=False)
        if not is_signed:
            # Unsigned operands are non-negative, so the exact quotient is too;
            # there's nothing below zero to floor toward, so udiv (which
            # truncates) already gives the floored result.
            return b.udiv(lhs, rhs)

        # Signed integers have no floor instruction: sdiv truncates toward zero.
        # Truncation and floor agree except when the exact quotient is negative
        # and non-integer -- i.e. the operands have opposite signs (negative
        # quotient) AND the division leaves a remainder (non-integer). There,
        # truncation rounds *up* toward zero, so it overshoots the floor by one
        # and we subtract one to correct it.
        #
        #   -7 // 2:  sdiv = -3, srem = -1 -> opposite signs, inexact -> -3-1 = -4
        #   -6 // 2:  sdiv = -3, srem =  0 -> exact, no adjust          -> -3
        #    7 // 2:  sdiv =  3, srem =  1 -> same signs, no adjust      ->  3
        quotient = b.sdiv(lhs, rhs)
        rem = b.srem(lhs, rhs)
        zero = ir.Constant(lhs.type, 0)
        # Non-integer quotient: the remainder is nonzero.
        inexact = b.icmp_signed("!=", rem, zero)
        # Negative quotient: lhs and rhs have opposite signs, which in two's
        # complement is exactly when their xor has the sign bit set (is < 0).
        opposite_signs = b.icmp_signed("<", b.xor(lhs, rhs), zero)
        adjust = b.and_(inexact, opposite_signs)
        return b.select(adjust, b.sub(quotient, ir.Constant(lhs.type, 1)), quotient)

    def _emit_modulo(
        self, lhs: ir.Value, rhs: ir.Value, is_float: bool, is_signed: bool
    ) -> ir.Value:
        """Emit `lhs % rhs` with *floored* semantics (Python `%`; spec "Modulus
        semantics"): the result takes the sign of the divisor.

        The IR remainder ops (srem/frem) are *truncated* -- the result takes the
        sign of the dividend -- so we correct it by adding the divisor back when
        the remainder is nonzero and its sign differs from the divisor's. (frem
        lowers to an fmod libcall on wasm, hence the imported env.fmod.)
        """
        b = self.builder
        # The spec makes a zero divisor a DOMAIN_ERROR fault for *every*
        # modulo, including floats (unlike float division, which is IEEE):
        # wasm's integer rem instructions would trap uncatchably, and the
        # host fmod would return NaN, so guard first.
        self._emit_zero_divisor_check(rhs, is_float)
        if not is_float and not is_signed:
            # Unsigned operands are non-negative, so floored == truncated.
            return b.urem(lhs, rhs)

        zero = ir.Constant(lhs.type, 0)
        if is_float:
            rem = b.frem(lhs, rhs)
            nonzero = b.fcmp_ordered("!=", rem, zero)
            signs_differ = b.xor(
                b.fcmp_ordered("<", rem, zero), b.fcmp_ordered("<", rhs, zero)
            )
            corrected = b.fadd(rem, rhs)
        else:
            rem = b.srem(lhs, rhs)
            nonzero = b.icmp_signed("!=", rem, zero)
            # rem and rhs have differing signs iff their xor is negative.
            signs_differ = b.icmp_signed("<", b.xor(rem, rhs), zero)
            corrected = b.add(rem, rhs)
        return b.select(b.and_(nonzero, signs_differ), corrected, rem)

    def _emit_zero_divisor_check(self, rhs: ir.Value, is_float: bool) -> None:
        """Fault with DOMAIN_ERROR when the divisor *rhs* is zero, per the
        spec's modulus and floor-division semantics. (fcmp `==` treats -0.0
        as zero, so a -0.0 divisor faults too.)"""
        b = self.builder
        zero = ir.Constant(rhs.type, 0)
        # A hardware float compare of a NaN operand against anything has no
        # meaningful true/false answer, so every fcmp predicate must pick one
        # up front, and that's the whole ordered/unordered split: *ordered*
        # predicates answer false when an operand is NaN, *unordered* ones
        # answer true. Concretely, with rhs = NaN:
        #   fcmp_ordered("==", NaN, 0.0)   -> false  (what we want: no fault)
        #   fcmp_unordered("==", NaN, 0.0) -> true   (would fault on NaN!)
        # A NaN divisor must not fault here: the spec faults only a *zero*
        # divisor and defines `lhs % nan` as nan -- which is exactly what
        # frem/fmod return.
        # On the int side, signedness doesn't exist for equality: LLVM has a
        # single `icmp eq` (signed/unsigned variants exist only for order
        # predicates like slt/ult), so icmp_signed("==") and
        # icmp_unsigned("==") emit the same instruction and unsigned operands
        # are fine here.
        is_zero = (
            b.fcmp_ordered("==", rhs, zero)
            if is_float
            else b.icmp_signed("==", rhs, zero)
        )
        fail_block = b.function.append_basic_block("div_zero")
        ok_block = b.function.append_basic_block("div_ok")
        b.cbranch(is_zero, fail_block, ok_block)
        b.position_at_end(fail_block)
        b.call(
            b.module.globals[HOST_PANIC_FUNC_NAME],
            [ir.Constant(ERROR_CODE_TYPE, DirectiveErrorCode.DOMAIN_ERROR.value)],
        )
        b.unreachable()
        b.position_at_end(ok_block)

    def _emit_short_circuit(self, node: AstBinaryOp, state: CompileState) -> ir.Value:
        """Lower ``and``/``or`` with short-circuit evaluation."""
        b = self.builder
        bool_type = ir.IntType(1)

        lhs = self.emit(node.lhs, state)
        lhs_block = b.block  # the block the branch on lhs lives in
        rhs_block = b.append_basic_block("bool_rhs")
        end_block = b.append_basic_block("bool_end")

        if node.op == BinaryStackOp.AND:
            # lhs true -> evaluate rhs; lhs false -> short-circuit to False.
            b.cbranch(lhs, rhs_block, end_block)
            short_value = ir.Constant(bool_type, 0)
        else:
            # lhs true -> short-circuit to True; lhs false -> evaluate rhs.
            b.cbranch(lhs, end_block, rhs_block)
            short_value = ir.Constant(bool_type, 1)

        b.position_at_end(rhs_block)
        rhs = self.emit(node.rhs, state)
        rhs_end_block = b.block  # rhs may itself have added blocks
        b.branch(end_block)

        b.position_at_end(end_block)
        phi = b.phi(bool_type, name="bool_result")
        phi.add_incoming(short_value, lhs_block)
        phi.add_incoming(rhs, rhs_end_block)
        return phi

    def emit_AstUnaryOp(self, node: AstUnaryOp, state: CompileState) -> ir.Value:
        intermediate_type = state.op_intermediate_types[node]
        b = self.builder
        val = self.emit(node.val, state)

        if node.op == UnaryStackOp.IDENTITY:
            # `+x` is a no-op.
            return val
        if node.op == UnaryStackOp.NOT:
            # `not x` flips a bool (i1).
            return b.not_(val)

        # The only remaining unary op is `-x`: float negation, or 0 - x for
        # integers.
        assert node.op == UnaryStackOp.NEGATE, node.op
        if intermediate_type.is_float:
            return b.fneg(val)
        return b.neg(val)

    def emit_AstIdent(self, node: AstIdent, state: CompileState) -> ir.Value:
        sym = state.resolved_symbols[node]
        assert isinstance(sym, VariableSymbol), sym
        # Load the variable at its stored (declared) type, which is the ident's
        # synthesized type; emit() handles any widening to the contextual type.
        return self.builder.load(state.backend.variable_ptrs[sym], name=str(node.name))

    def emit_AstGetAttr(self, node: AstGetAttr, state: CompileState) -> ir.Value:
        sym = state.resolved_symbols[node]
        if is_instance_compat(sym, (ChDef, PrmDef)):
            raise BackendError(
                "LLVM backend can't read telemetry channels or parameters yet"
            )
        # A qualified name can't denote a variable (an imported sequence may
        # not declare a top-level variable), so what remains is member access.
        assert is_instance_compat(sym, FieldAccess), sym
        if is_addressable(node, state):
            return self.builder.load(self._emit_ptr(node, state), name=node.attr)
        # Member access on an anonymous struct literal: the struct is never
        # built; emit just the accessed member's expression.
        assert is_instance_compat(sym.parent_expr, AstAnonStruct), sym.parent_expr
        for name, value_expr in sym.parent_expr.members:
            if name == node.attr:
                return self.emit(value_expr, state)
        assert False, f"member {node.attr} not found in anon struct"

    def emit_AstIndexExpr(self, node: AstIndexExpr, state: CompileState) -> ir.Value:
        sym = state.resolved_symbols[node]
        assert is_instance_compat(sym, FieldAccess), sym
        if is_addressable(node, state):
            return self.builder.load(self._emit_ptr(node, state))
        # Element access on an anonymous array literal: the array is never
        # built; the index must be a compile-time constant, so emit just that
        # element's expression.
        assert is_instance_compat(sym.parent_expr, AstAnonArray), sym.parent_expr
        idx_value = state.const_expr_values.get(node.item)
        assert (
            idx_value is not None
        ), "Dynamic indexing on anonymous array literals is not supported"
        idx = idx_value.val
        assert 0 <= idx < len(sym.parent_expr.elements), f"Index {idx} out of bounds"
        return self.emit(sym.parent_expr.elements[idx], state)

    def _emit_ptr(self, expr: AstExpr, state: CompileState) -> ir.Value:
        """Emit a pointer to *expr*'s value, without loading it."""
        b = self.builder
        i32 = ir.IntType(32)
        if not is_addressable(expr, state):
            return self._emit_to_temp_slot(expr, state)
        sym = state.resolved_symbols[expr]
        if is_instance_compat(sym, VariableSymbol):
            return state.backend.variable_ptrs[sym]
        assert is_instance_compat(sym, FieldAccess), sym
        parent_ptr = self._emit_ptr(sym.parent_expr, state)
        parent_type = state.contextual_types[sym.parent_expr]
        if sym.is_struct_member:
            # gep(p, [0, i]) points at member i of the struct behind p: the
            # leading 0 stays on the struct p itself points at (a GEP's first
            # index strides in whole pointee units), and i then selects the
            # member. A struct's member index must be a constant, since each
            # member has its own type and the result type must be known
            # statically. `inbounds` promises the address stays inside the
            # parent object, which a valid member index always does.
            member_idx = next(
                i for i, m in enumerate(parent_type.members) if m.name == sym.name
            )
            return b.gep(
                parent_ptr,
                [ir.Constant(i32, 0), ir.Constant(i32, member_idx)],
                inbounds=True,
                name=sym.name,
            )
        assert sym.is_array_element, sym
        idx = self.emit(sym.idx_expr, state)
        self._emit_array_bounds_check(idx, parent_type.length)
        return b.gep(parent_ptr, [ir.Constant(i32, 0), idx], inbounds=True)

    def _emit_to_temp_slot(self, expr: AstExpr, state: CompileState) -> ir.Value:
        """Emit *expr* as a value into the stack slot it was given, so that a
        value that denotes no location has an address to GEP into."""
        slot = state.backend.temp_slots[expr]
        self.builder.store(self.emit(expr, state), slot)
        return slot

    def _emit_array_bounds_check(self, idx: ir.Value, length: int) -> None:
        """Fault with ARRAY_OUT_OF_BOUNDS unless 0 <= idx < length, the
        runtime bounds check the spec gives element accesses. ArrayIndexType
        is signed, so a negative index must be caught explicitly."""
        b = self.builder
        too_low = b.icmp_signed("<", idx, ir.Constant(idx.type, 0))
        too_high = b.icmp_signed(">=", idx, ir.Constant(idx.type, length))
        oob = b.or_(too_low, too_high)
        fail_block = b.function.append_basic_block("idx_oob")
        ok_block = b.function.append_basic_block("idx_ok")
        b.cbranch(oob, fail_block, ok_block)
        b.position_at_end(fail_block)
        b.call(
            b.module.globals[HOST_PANIC_FUNC_NAME],
            [
                ir.Constant(
                    ERROR_CODE_TYPE, DirectiveErrorCode.ARRAY_OUT_OF_BOUNDS.value
                )
            ],
        )
        b.unreachable()
        b.position_at_end(ok_block)

    def emit_AstFuncCall(
        self, node: AstFuncCall, state: CompileState
    ) -> ir.Value | None:
        node_args = node.args if node.args is not None else []
        func = state.resolved_symbols[node.func]
        if is_instance_compat(func, CommandSymbol):
            if func.is_seq_run_with_args:
                raise BackendError(
                    "LLVM backend can't lower sequence-run commands with "
                    "arguments yet"
                )
            command = state.backend.cmd_buffers[node]
            values = [
                self.emit(arg, state) for _offset, arg, _type in command.runtime_args
            ]
            return self._emit_command_dispatch(command, values)
        elif is_instance_compat(func, BuiltinFuncSymbol):
            return self._emit_builtin_call(node_args, func, state)
        elif is_instance_compat(func, TypeCtorSymbol):
            # A ctor call with all-constant args folds before reaching here.
            raise BackendError(
                "LLVM backend can't lower runtime type-constructor calls yet"
            )
        elif is_instance_compat(func, CastSymbol):
            # the actual conversion happens already as part of the
            # synthesized -> contextual conversion in emit()
            return self.emit(node_args[0], state)
        elif is_instance_compat(func, FunctionSymbol):
            fn = state.backend.funcs[func.definition]
            # The args are already positional with defaults filled in
            # (DesugarDefaultArgs) and coerced to the parameter types, so each
            # emits at its parameter's LLVM type. A script function's default
            # is always an AstExpr, never a bare FpyValue.
            assert all(is_instance_compat(arg, Ast) for arg in node_args), node_args
            args = [self.emit(arg, state) for arg in node_args]
            return self.builder.call(fn, args)
        else:
            assert False, func

    def _emit_builtin_call(
        self, node_args: list, func: BuiltinFuncSymbol, state: CompileState
    ) -> ir.Value | None:
        # Pass each argument as (emitted ir.Value, its constant FpyValue or None
        # if it isn't a compile-time constant). The builtin's generate_llvm picks
        # whichever it needs. Args the builtin requires to be compile-time
        # constants are not emitted at all -- they may have no machine
        # representation (e.g. log's InternalString message) -- so they arrive
        # as (None, value).
        args: list[tuple[ir.Value | None, FpyValue | None]] = []
        for i, arg in enumerate(node_args):
            const_val = (
                arg
                if is_instance_compat(arg, FpyValue)
                else state.const_expr_values.get(arg)
            )
            if i in func.const_arg_indices:
                assert (
                    const_val is not None
                ), f"const arg {i} of {func.name} should have been validated by semantics"
                args.append((None, const_val))
            elif is_instance_compat(arg, FpyValue):  # a filled-in default argument
                args.append((arg.llvm_value, const_val))
            else:
                args.append((self.emit(arg, state), const_val))
        return func.generate_llvm(self.builder, args)

    def _emit_command_dispatch(
        self, command: CommandBuffer, values: list[ir.Value]
    ) -> ir.Value:
        """Write *values* -- *command*'s runtime arguments, already evaluated,
        in runtime_args order -- into its buffer and dispatch it through the
        host cmd import, returning the host's Fw.CmdResponse."""
        b = self.builder
        assert len(values) == len(command.runtime_args), (values, command)
        for (offset, _arg, arg_type), value in zip(command.runtime_args, values):
            written = self._emit_store_big_endian(value, arg_type, command.buf, offset)
            assert written == arg_type.max_size, (arg_type, written)

        # The host returns the response widened to i32; narrow it back to
        # Fw.CmdResponse's type.
        response = b.call(
            b.module.globals[HOST_CMD_FUNC_NAME],
            [
                b.bitcast(command.buf, ir.IntType(8).as_pointer()),
                ir.Constant(ir.IntType(32), command.size),
            ],
        )
        # assumption: the response is a valid cmd response code
        return b.trunc(response, CMD_RESPONSE.llvm_type)

    def _emit_store_big_endian(
        self, value: ir.Value, fpy_type: FpyType, base_ptr: ir.Value, offset: int
    ) -> int:
        """Store *value* at byte *offset* inside the [N x i8] buffer at
        *base_ptr* in fprime wire format: big-endian, tightly packed, bools as
        the FW_SERIALIZE truth bytes, aggregates member by member. Returns the
        number of bytes written (fpy_type's serialized size)."""
        b = self.builder
        i8 = ir.IntType(8)
        i32 = ir.IntType(32)
        if fpy_type.kind == TypeKind.STRUCT:
            start = offset
            for i, member in enumerate(fpy_type.members):
                offset += self._emit_store_big_endian(
                    b.extract_value(value, i), member.type, base_ptr, offset
                )
            assert offset - start == fpy_type.max_size, fpy_type
            return fpy_type.max_size
        if fpy_type.kind == TypeKind.ARRAY:
            start = offset
            for i in range(fpy_type.length):
                offset += self._emit_store_big_endian(
                    b.extract_value(value, i), fpy_type.elem_type, base_ptr, offset
                )
            assert offset - start == fpy_type.max_size, fpy_type
            return fpy_type.max_size
        if fpy_type.kind == TypeKind.BOOL:
            # The truth bytes are dictionary-configurable module globals, so
            # read them through the module, never a from-import.
            value = b.select(
                value,
                ir.Constant(i8, fpy.types.FW_SERIALIZE_TRUE_VALUE),
                ir.Constant(i8, fpy.types.FW_SERIALIZE_FALSE_VALUE),
            )
            width = 1
        else:
            # A runtime string can't exist (semantics rejects it), so what's
            # left is an integer, an enum at its rep type, or a float.
            assert not fpy_type.is_string, fpy_type
            width = fpy_type.max_size
            if fpy_type.is_float:
                value = b.bitcast(value, ir.IntType(width * 8))
            if width > 1:
                # wasm memory is little-endian; the wire format is big-endian.
                swap = b.module.declare_intrinsic("llvm.bswap", [value.type])
                value = b.call(swap, [value])
        ptr = b.gep(
            base_ptr, [ir.Constant(i32, 0), ir.Constant(i32, offset)], inbounds=True
        )
        if width > 1:
            ptr = b.bitcast(ptr, value.type.as_pointer())
        # The buffer is packed, so wider slots aren't naturally aligned.
        b.store(value, ptr, align=1)
        return width

    def _emit_load_big_endian(
        self, fpy_type: FpyType, base_ptr: ir.Value, offset: int
    ) -> ir.Value:
        """Load a value of *fpy_type* from byte *offset* inside the [N x i8]
        buffer at *base_ptr*, the inverse of _emit_store_big_endian: the bytes
        are fprime wire format (big-endian, tightly packed, aggregates member
        by member). If the bytes cannot be deserialized to the fpy_type,
        raise an error. Returns the value at fpy_type's LLVM type."""
        b = self.builder
        i32 = ir.IntType(32)
        if fpy_type.kind == TypeKind.STRUCT:
            value = ir.Constant(fpy_type.llvm_type, ir.Undefined)
            start = offset
            for i, member in enumerate(fpy_type.members):
                value = b.insert_value(
                    value, self._emit_load_big_endian(member.type, base_ptr, offset), i
                )
                offset += member.type.max_size
            assert offset - start == fpy_type.max_size, fpy_type
            return value
        if fpy_type.kind == TypeKind.ARRAY:
            value = ir.Constant(fpy_type.llvm_type, ir.Undefined)
            start = offset
            for i in range(fpy_type.length):
                value = b.insert_value(
                    value,
                    self._emit_load_big_endian(fpy_type.elem_type, base_ptr, offset),
                    i,
                )
                offset += fpy_type.elem_type.max_size
            assert offset - start == fpy_type.max_size, fpy_type
            return value
        assert not fpy_type.is_string, fpy_type
        width = fpy_type.max_size
        ptr = b.gep(
            base_ptr, [ir.Constant(i32, 0), ir.Constant(i32, offset)], inbounds=True
        )
        if fpy_type.kind == TypeKind.BOOL:
            # The truth bytes are dictionary-configurable module globals, so
            # read them through the module, never a from-import.
            byte = b.load(ptr, align=1)
            is_true = b.icmp_unsigned(
                "==", byte, ir.Constant(byte.type, fpy.types.FW_SERIALIZE_TRUE_VALUE)
            )
            is_false = b.icmp_unsigned(
                "==", byte, ir.Constant(byte.type, fpy.types.FW_SERIALIZE_FALSE_VALUE)
            )
            fail_block = b.function.append_basic_block("bool_bad")
            ok_block = b.function.append_basic_block("bool_ok")
            b.cbranch(b.or_(is_true, is_false), ok_block, fail_block)
            b.position_at_end(fail_block)
            b.call(
                b.module.globals[HOST_PANIC_FUNC_NAME],
                [
                    ir.Constant(
                        ERROR_CODE_TYPE,
                        DirectiveErrorCode.DESERIALIZE_ERROR_INVALID_BOOL.value,
                    )
                ],
            )
            b.unreachable()
            b.position_at_end(ok_block)
            return is_true
        if width > 1:
            ptr = b.bitcast(ptr, ir.IntType(width * 8).as_pointer())
        # The buffer is packed, so wider slots aren't naturally aligned.
        value = b.load(ptr, align=1)
        if width > 1:
            # wasm memory is little-endian; the wire format is big-endian.
            swap = b.module.declare_intrinsic("llvm.bswap", [value.type])
            value = b.call(swap, [value])
        if fpy_type.is_float:
            value = b.bitcast(value, fpy_type.llvm_type)
        return value

    def convert_numeric_type(self, value: ir.Value, from_type, to_type) -> ir.Value:
        """Convert a scalar numeric value between two concrete numeric types."""
        assert from_type.is_numerical and to_type.is_numerical, (from_type, to_type)
        target = to_type.llvm_type

        if from_type.is_integer and to_type.is_integer:
            if to_type.bits > from_type.bits:
                # Widen: sign-extend signed sources, zero-extend unsigned ones.
                extend = self.builder.sext if from_type.is_signed else self.builder.zext
                return extend(value, target)
            if to_type.bits < from_type.bits:
                return self.builder.trunc(value, target)
            # Same width: signedness isn't part of an LLVM integer type.
            return value
        if from_type.is_integer:  # int -> float
            to_float = (
                self.builder.sitofp if from_type.is_signed else self.builder.uitofp
            )
            return to_float(value, target)
        if to_type.is_integer:  # float -> int
            return self._emit_fp_to_int_saturating(value, to_type)
        # float -> float
        if to_type.bits > from_type.bits:
            return self.builder.fpext(value, target)
        if to_type.bits < from_type.bits:
            return self.builder.fptrunc(value, target)
        return value

    def _emit_fp_to_int_saturating(self, value: ir.Value, to_type) -> ir.Value:
        """Convert a float to an integer with saturating semantics:
        an out-of-range value clamps to the target type's min/max and NaN maps to
        0, rather than producing a poison value"""
        base = "llvm.fptosi.sat" if to_type.is_signed else "llvm.fptoui.sat"
        result_type = to_type.llvm_type
        fn = self.builder.module.declare_intrinsic(
            base, (result_type, value.type), ir.FunctionType(result_type, [value.type])
        )
        return self.builder.call(fn, [value])


# The symbol and wasm export name of the user entrypoint: `void main()`, no
# arguments and no return value. Every failure (and explicit exit) reports
# its code through the exit/fault host imports and never comes back, so
# returning at all means success and there is no status to return. This
# deliberately diverges from the Basic C ABI's `int main` user-entrypoint
# shape -- and the void return is also what keeps LLVM's wasm backend
# (WebAssemblyFixFunctionBitcasts) from touching the function: that pass only
# rewrites a zero-arg `main` that returns i32, renaming it `__original_main`
# and synthesizing a canonical main(argc, argv) wrapper around it.
FPY_ENTRY_POINT = "main"


class AssignAddresses(TopDownVisitor):
    """Gives an address to everything one function's code addresses -- its
    variables, each expression that is addressed without denoting a location of
    its own, and each command call's buffer -- before any of that function is
    lowered, so that an emitter only ever looks an address up."""

    def __init__(self, builder: ir.IRBuilder):
        super().__init__()
        self.builder: ir.IRBuilder = builder

    def visit_AstBlock(self, node: AstBlock, state: CompileState):
        # A block's scope holds every variable declared in it, including the
        # ones no assignment declares (a loop's variable and its bound).
        for sym in state.enclosing_scope[node].group(NameGroup.VALUE).values():
            if is_instance_compat(sym, VariableSymbol):
                self._create_variable_slot(sym, state)

    def visit_AstDef(self, node: AstDef, state: CompileState):
        return STOP_DESCENT  # a def is a function of its own, with its own storage

    def visit_AstGetAttr(self, node: AstGetAttr, state: CompileState):
        self._create_temp_slot(node, state)

    def visit_AstIndexExpr(self, node: AstIndexExpr, state: CompileState):
        self._create_temp_slot(node, state)

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        func = state.resolved_symbols.get(node.func)
        # A sequence-run command with arguments has no buffer layout yet;
        # lowering the call is what reports that.
        if is_instance_compat(func, CommandSymbol) and not func.is_seq_run_with_args:
            self._create_command_buffer(node, func, state)

    def _create_variable_slot(self, sym: VariableSymbol, state: CompileState) -> None:
        """Create *sym*'s slot: a module-level global when it is a global
        variable, so that a function reaches the same slot the entry function
        does, and an entry-block alloca otherwise."""
        ptrs = state.backend.variable_ptrs
        assert sym not in ptrs, sym
        if sym.is_global:
            gvar = ir.GlobalVariable(
                state.backend.module, sym.type.llvm_type, name=sym.name
            )
            gvar.linkage = "internal"
            # Zero-initialized; the declaring assignment writes the real value.
            gvar.initializer = ir.Constant(sym.type.llvm_type, None)
            ptrs[sym] = gvar
        else:
            ptrs[sym] = self.builder.alloca(sym.type.llvm_type, name=sym.name)

    def _create_temp_slot(self, access: AstExpr, state: CompileState) -> None:
        """Give the expression the member/element access *access* addresses
        into -- its parent -- a slot to be copied into, when the parent does not
        already denote a location. A folded constructor call's constant
        aggregate is the case that arises: a runtime index has to GEP into it.

        Only the parent one level up is considered, because every link of a
        chain like ``Ctor(...)[i].m`` is itself an access this pass visits, so
        each link creates its own parent's slot when its turn comes.
        """
        # _emit_ptr is what takes the parent's address, and it is only ever
        # reached for an access that survives folding and is addressable -- the
        # same two tests the emitter makes before it calls _emit_ptr.
        if state.const_expr_values.get(access) is not None:
            return  # folded to a constant, so no address is taken
        if not is_addressable(access, state):
            return  # a qualified name, or a never-built anonymous literal

        # _emit_ptr copies its argument to a slot on exactly this condition.
        parent = state.resolved_symbols[access].parent_expr
        if is_addressable(parent, state):
            return

        slots = state.backend.temp_slots
        if parent not in slots:
            # The AST can share one expression between two places (a default
            # argument), so a parent may be reached more than once.
            slots[parent] = self.builder.alloca(
                state.contextual_types[parent].llvm_type, name="temp"
            )

    def _create_command_buffer(
        self, node: AstFuncCall, func: CommandSymbol, state: CompileState
    ) -> None:
        """Lay out the buffer *node* dispatches: the opcode, then each argument
        in order. An argument known at compile time is serialized straight into
        the buffer at its actual size -- a string packs to its real length, not
        its declared capacity. Every other gets a zeroed max_size slot."""
        contents = bytearray(FpyValue(FwOpcodeType, func.cmd.opcode).serialize())
        runtime_args = []  # (byte offset of the arg's slot, arg expr, arg type)
        for arg in node.args if node.args is not None else []:
            const_val = (
                arg
                if is_instance_compat(arg, FpyValue)
                else state.const_expr_values.get(arg)
            )
            if const_val is not None:
                contents += const_val.serialize()
            else:
                arg_type = state.contextual_types[arg]
                runtime_args.append((len(contents), arg, arg_type))
                contents += bytes(arg_type.max_size)

        module = state.backend.module
        buf_type = ir.ArrayType(ir.IntType(8), len(contents))
        buf = ir.GlobalVariable(
            module, buf_type, name=module.get_unique_name("cmd_buf")
        )
        buf.linkage = "private"
        buf.global_constant = not runtime_args
        buf.initializer = ir.Constant(buf_type, contents)
        state.backend.cmd_buffers[node] = CommandBuffer(
            buf, len(contents), runtime_args
        )


class EmitLlvmStmt(Emitter):
    """Lowers Fpy statements into LLVM IR via a shared builder."""

    def __init__(self, builder: ir.IRBuilder):
        super().__init__()
        self.builder: ir.IRBuilder = builder
        self.expr: EmitLlvmExpr = EmitLlvmExpr(builder)
        self.loop_continue_blocks: dict[AstWhile, ir.Block] = {}
        """loop to the block its `continue` statements jump to"""
        self.loop_break_blocks: dict[AstWhile, ir.Block] = {}
        """loop to the block its `break` statements jump to"""

    def emit(self, node, state: CompileState) -> None:
        emitter = self.emitters.get(type(node))
        if emitter is None:
            raise BackendError(
                f"LLVM backend doesn't handle statement " f"{type(node).__name__} yet"
            )
        return emitter(node, state)

    def emit_AstBlock(self, node: AstBlock, state: CompileState) -> None:
        """Lower the statements of *block* into the current basic block(s)."""
        self._emit_stmts(node.stmts, state)

    def _emit_stmts(self, stmts: list[Ast], state: CompileState) -> None:
        for stmt in stmts:
            if self.builder.block.is_terminated:
                # A return/break/continue ended this block; the remaining
                # statements are unreachable.
                break
            if is_instance_compat(stmt, AstExpr):
                # A constant statement emits no code: emit() maps it to a bare
                # ir.Constant no instruction uses (or to None when it has an
                # abstract type, which only a bare statement can leave it at).
                result = self.expr.emit(stmt, state)
                if is_cmd_and_response_unhandled(stmt, state):
                    self._emit_assert_cmd_response_ok(result, state)
            elif is_instance_compat(stmt, AstNodeWithSideEffects):
                # A non-expression statement (assign, if, assert, def, ...).
                self.emit(stmt, state)
            # else: a non-expression statement that can't affect the program on
            # its own (e.g. `pass`, sequence metadata) -- nothing to lower.

    def _emit_assert_cmd_response_ok(
        self, response: ir.Value, state: CompileState
    ) -> None:
        """For a bare command call, exit with CMD_FAIL if the response is not
        OK and flags.assert_cmd_success is set."""
        b = self.builder
        i32 = ir.IntType(32)
        ok = b.icmp_unsigned(
            "==", response, ir.Constant(response.type, CMD_RESPONSE.enum_dict["OK"])
        )
        not_ok_block = b.function.append_basic_block("cmd_not_ok")
        fail_block = b.function.append_basic_block("cmd_fail")
        end_block = b.function.append_basic_block("cmd_ok")
        b.cbranch(ok, end_block, not_ok_block)

        b.position_at_end(not_ok_block)
        flags_type = state.flags_var.type
        member_idx = next(
            i
            for i, m in enumerate(flags_type.members)
            if m.name == "assert_cmd_success"
        )
        flag_ptr = b.gep(
            state.backend.variable_ptrs[state.flags_var],
            [ir.Constant(i32, 0), ir.Constant(i32, member_idx)],
            inbounds=True,
        )
        b.cbranch(b.load(flag_ptr), fail_block, end_block)

        b.position_at_end(fail_block)
        b.call(
            b.module.globals[HOST_EXIT_FUNC_NAME],
            [ir.Constant(ERROR_CODE_TYPE, DirectiveErrorCode.CMD_FAIL.value)],
        )
        b.unreachable()
        b.position_at_end(end_block)

    def emit_AstDef(self, node: AstDef, state: CompileState) -> None:
        # A function definition emits nothing where it is written; the used
        # definitions each become an LLVM function of their own.
        return

    def emit_AstReturn(self, node: AstReturn, state: CompileState) -> None:
        if node.value is None:
            self.builder.ret_void()
            return
        # The value is coerced to the function's return type by semantics.
        self.builder.ret(self.expr.emit(node.value, state))

    def emit_AstWhile(self, node: AstWhile, state: CompileState) -> None:
        b = self.builder
        func = b.function
        cond_block = func.append_basic_block("while_cond")
        body_block = func.append_basic_block("while_body")
        end_block = func.append_basic_block("while_end")
        # A desugared for loop's body ends with its increment statement;
        # `continue` must run the increment before re-testing the condition.
        is_for = node in state.desugared_for_loops
        inc_block = func.append_basic_block("for_inc") if is_for else None
        self.loop_continue_blocks[node] = inc_block if is_for else cond_block
        self.loop_break_blocks[node] = end_block

        b.branch(cond_block)
        b.position_at_end(cond_block)
        b.cbranch(self.expr.emit(node.condition, state), body_block, end_block)

        b.position_at_end(body_block)
        if is_for:
            assert node.body.stmts and is_instance_compat(
                node.body.stmts[-1], AstAssign
            ), node.body.stmts
            self._emit_stmts(node.body.stmts[:-1], state)
            if not b.block.is_terminated:
                b.branch(inc_block)
            b.position_at_end(inc_block)
            self._emit_stmts(node.body.stmts[-1:], state)
        else:
            self.emit(node.body, state)
        if not b.block.is_terminated:
            b.branch(cond_block)
        b.position_at_end(end_block)

    def emit_AstBreak(self, node: AstBreak, state: CompileState) -> None:
        target = self.loop_break_blocks[state.enclosing_loops[node]]
        self.builder.branch(target)

    def emit_AstContinue(self, node: AstContinue, state: CompileState) -> None:
        target = self.loop_continue_blocks[state.enclosing_loops[node]]
        self.builder.branch(target)

    def emit_AstAssign(self, node: AstAssign, state: CompileState) -> None:
        sym = state.resolved_symbols[node.lhs]
        # The rhs is coerced to the target's type, so its emitted value
        # already matches the slot's element type. Emit it before the lhs
        # pointer so a failing bounds check in the lhs still evaluates the
        # rhs first
        value = self.expr.emit(node.rhs, state)
        if is_instance_compat(sym, VariableSymbol):
            self.builder.store(value, state.backend.variable_ptrs[sym])
            return
        # A field/element target (x.f = ..., a[i] = ...): store through a
        # pointer into the base variable's storage, in place. Semantics only
        # lets through targets rooted at a variable, so _emit_ptr cannot copy
        # the target to a temp slot (which would lose the store).
        assert is_instance_compat(sym, FieldAccess), sym
        assert is_instance_compat(sym.base_sym, VariableSymbol), sym
        self.builder.store(value, self.expr._emit_ptr(node.lhs, state))

    def emit_AstIf(self, node: AstIf, state: CompileState) -> None:
        builder = self.builder
        func = builder.function
        end_block = func.append_basic_block("if_end")
        cases = [(node.condition, node.body)]
        cases += [(case.condition, case.body) for case in node.elifs]

        for condition, case_body in cases:
            cond = self.expr.emit(condition, state)
            then_block = func.append_basic_block("if_then")
            next_block = func.append_basic_block("if_next")
            builder.cbranch(cond, then_block, next_block)

            builder.position_at_end(then_block)
            self.emit(case_body, state)
            # block might already be terminated from a return
            if not builder.block.is_terminated:
                builder.branch(end_block)

            # Subsequent cases (and the else) are tested/run when this condition
            # was false, i.e. in next_block.
            builder.position_at_end(next_block)

        if node.els is not None:
            self.emit(node.els, state)
        if not builder.block.is_terminated:
            builder.branch(end_block)

        builder.position_at_end(end_block)

    def emit_AstAssert(self, node: AstAssert, state: CompileState) -> None:
        builder = self.builder
        func = builder.function
        condition = self.expr.emit(node.condition, state)

        fail_block = func.append_basic_block(name="assert_fail")
        ok_block = func.append_basic_block(name="assert_ok")
        builder.cbranch(condition, ok_block, fail_block)

        builder.position_at_end(fail_block)
        if node.exit_code is None:
            code = ir.Constant(
                ERROR_CODE_TYPE, DirectiveErrorCode.EXIT_WITH_ERROR.value
            )
        else:
            code = self.expr.emit(node.exit_code, state)
        builder.call(builder.module.globals[HOST_EXIT_FUNC_NAME], [code])
        builder.unreachable()

        # Success path: continue lowering subsequent statements here.
        builder.position_at_end(ok_block)


class DeclareFunctions(Visitor):
    """Declares an ir.Function for every used script function before any body
    is lowered, so every call site -- including recursive, mutually recursive,
    and forward calls -- resolves to a declared function."""

    def visit_AstDef(self, node: AstDef, state: CompileState):
        if node not in state.used_funcs:
            return
        func = state.resolved_symbols[node.name]
        fn_type = ir.FunctionType(
            func.return_type.llvm_type,
            [arg_type.llvm_type for _name, arg_type, _default in func.args],
        )
        module = state.backend.module
        fn = ir.Function(module, fn_type, name=module.get_unique_name(func.name))
        fn.linkage = "internal"
        state.backend.funcs[node] = fn


class GenerateLlvmModule:
    """Builds the LLVM module for a sequence: declares an LLVM function for
    the entry point and for every used script function, then defines each one
    by assigning its storage and lowering its body."""

    def emit(self, root_block: AstBlock, state: CompileState) -> ir.Module:
        assert (
            root_block is state.root_block
        ), "module generator must be run on the root block"
        if state.this_seq_arg_specs:
            # A parameter's value comes from whoever runs the sequence, and the
            # host interface has no way to deliver one.
            raise BackendError("LLVM backend can't lower sequences with parameters yet")
        module = ir.Module(name="seq")
        module.triple = LLVM_TRIPLE
        declare_host_imports(module)
        state.backend = LlvmBackendState(module)
        self._create_flags_slot(state)

        # The user entrypoint (see FPY_ENTRY_POINT): void main(), where
        # returning at all means success. Created before the script functions
        # so it owns the exported name; a script function named "main" is
        # uniqued away from it.
        main = ir.Function(
            module, ir.FunctionType(ir.VoidType(), []), name=FPY_ENTRY_POINT
        )
        DeclareFunctions().run(root_block, state)

        # Main first: its storage walk creates the module-global slots (the
        # top-level variables) that the function bodies load and store.
        self._define_function(main, state.main_block, [], state)
        for def_node, fn in state.backend.funcs.items():
            self._define_function(
                fn, def_node.body, self._param_vars(def_node, state), state
            )
        return module

    def _define_function(
        self,
        fn: ir.Function,
        body: AstBlock,
        params: list[VariableSymbol],
        state: CompileState,
    ) -> None:
        """Emit *fn*'s definition: storage for everything *body* addresses,
        the incoming arguments stored into their parameter slots, then the
        lowered body, closed off with the function's terminator."""
        builder = ir.IRBuilder(fn.append_basic_block(name="entry"))
        AssignAddresses(builder).run(body, state)
        assert len(params) == len(fn.args), (params, fn.args)
        for var, arg in zip(params, fn.args):
            arg.name = var.name
            builder.store(arg, state.backend.variable_ptrs[var])
        EmitLlvmStmt(builder).emit(body, state)
        if not builder.block.is_terminated:
            if isinstance(fn.function_type.return_type, ir.VoidType):
                # Fell off the end without failing: an implicit empty return
                # (and, for main, sequence success).
                builder.ret_void()
            else:
                # CheckFunctionReturns proved every path returns, so this
                # point is unreachable -- but the block needs a terminator.
                builder.unreachable()

    def _param_vars(self, node: AstDef, state: CompileState) -> list[VariableSymbol]:
        """The parameter variables of *node*'s body scope, in signature order."""
        func = state.resolved_symbols[node.name]
        values = state.enclosing_scope[node.body].group(NameGroup.VALUE)
        return [values[name] for name, _type, _default in func.args]

    def _create_flags_slot(self, state: CompileState) -> None:
        """Create the built-in `flags` struct, initialized to its defaults.
        It lives in the base scope, outside every block, so no storage walk
        reaches it."""
        flags = state.flags_var
        g = ir.GlobalVariable(
            state.backend.module, flags.type.llvm_type, name=flags.name
        )
        g.linkage = "internal"
        g.initializer = FpyValue(
            flags.type, dict(flags.type.member_defaults)
        ).llvm_value
        state.backend.variable_ptrs[flags] = g


_llvm_targets_initialized = False


def _ensure_llvm_targets() -> None:
    global _llvm_targets_initialized
    if not _llvm_targets_initialized:
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
        _llvm_targets_initialized = True


def llvm_module_to_wasm(module: ir.Module) -> bytes:
    """Compile an llvmlite module targeting wasm32 into a runnable wasm binary."""
    _ensure_llvm_targets()
    parsed = llvm.parse_assembly(str(module))
    parsed.verify()
    target = llvm.Target.from_triple(LLVM_TRIPLE)
    machine = target.create_target_machine(cpu=LLVM_CPU)
    obj = machine.emit_object(parsed)
    return _link_wasm_object(obj)


def llvm_module_to_wasm_text(module: ir.Module) -> str:
    """Lower an llvmlite module to WebAssembly text (the LLVM `.s` textual
    assembly: a human-readable listing of the wasm instructions).

    This is the textual form of the same code emit_object produces, taken
    before linking -- analogous to dumping textual LLVM IR. It is meant for
    inspection/debugging, not for feeding back to wat2wasm (it's the `.s`
    assembly format, not the s-expression `(module ...)` form)."""
    _ensure_llvm_targets()
    parsed = llvm.parse_assembly(str(module))
    parsed.verify()
    target = llvm.Target.from_triple(LLVM_TRIPLE)
    machine = target.create_target_machine(cpu=LLVM_CPU)
    return machine.emit_assembly(parsed)


def _wasm_ld_command() -> list[str]:
    """The argv prefix that runs ziglang's bundled wasm-ld."""
    try:
        import ziglang  # noqa: F401
    except ImportError:
        raise BackendError(
            "the 'ziglang' package is required to link wasm output (it provides "
            "wasm-ld); install it with 'pip install ziglang'"
        )
    return [
        sys.executable,
        "-m",
        "ziglang",
        "wasm-ld",
        "--page-size=1",
        # Reserve a real shadow stack for allocas, placed at the very start of
        # linear memory so an overflow traps (the stack pointer wraps below 0
        # and the store lands out of bounds) instead of silently corrupting
        # the globals laid out after it.
        "-zstack-size=4096",
        "--stack-first",
    ]


def _llvm_version_str() -> str:
    """The version of LLVM that llvmlite is bound to (e.g. "20.1.8")."""
    return ".".join(str(n) for n in llvm.llvm_version_info)


def _wasm_ld_version_str() -> str:
    """The version line reported by the bundled wasm-ld (e.g. "LLD 21.1.0").

    wasm-ld is shipped separately (via the 'ziglang' package), so its version
    is independent of the LLVM that compiled the IR. Returns "unavailable" if
    wasm-ld can't be run rather than failing -- this is only for --version."""
    try:
        result = subprocess.run(_wasm_ld_command() + ["--version"], capture_output=True)
    except (BackendError, OSError):
        return "unavailable"
    if result.returncode != 0:
        return "unavailable"
    # wasm-ld prints a single line like "LLD 21.1.0 (compatible with GNU linkers)";
    # keep just the "LLD <version>" part.
    line = result.stdout.decode(errors="replace").strip()
    return line if line else "unavailable"


def backend_version_str() -> str:
    """Human-readable summary of the LLVM/wasm toolchain the backend uses, for
    the compiler's --version output: the LLVM version that lowers the IR, the
    WebAssembly spec version we target, and the wasm-ld that links the module."""
    return (
        f"LLVM {_llvm_version_str()}, "
        f"WASM {WASM_VERSION}, "
        f"wasm-ld {_wasm_ld_version_str()}"
    )


def _link_wasm_object(obj: bytes) -> bytes:
    """Link a relocatable wasm object into a runnable module with wasm-ld."""
    flags = [
        # No C-style _start crt wrapper; the host invokes the exported
        # entrypoint directly.
        "--no-entry",
        # Undefined symbols become host imports (for future host calls like
        # commands/telemetry); harmless while there are none.
        "--allow-undefined",
        f"--export={FPY_ENTRY_POINT}",
    ]

    if os.name != "nt":
        # POSIX: pipe the object in via /dev/stdin and read the linked module
        # back from stdout (-o -), so no temp files are needed. wasm-ld reads
        # the object sequentially, which works fine over a pipe, and its
        # diagnostics go to stderr so stdout stays pure binary.
        return _run_wasm_ld(flags + ["/dev/stdin", "-o", "-"], stdin=obj)

    # Windows has no /dev/stdin (nor any general stdin path wasm-ld can open),
    # so round-trip the object and result through temp files there instead.
    # no guarantee windows stuff actually works. I haven't tested it.
    with tempfile.TemporaryDirectory() as tmp:
        obj_path = Path(tmp) / "seq.o"
        out_path = Path(tmp) / "seq.wasm"
        obj_path.write_bytes(obj)
        _run_wasm_ld(flags + [str(obj_path), "-o", str(out_path)])
        return out_path.read_bytes()


def _run_wasm_ld(args: list[str], stdin: bytes | None = None) -> bytes:
    """Run wasm-ld with *args*; return its stdout. Raises on link failure."""
    result = subprocess.run(_wasm_ld_command() + args, input=stdin, capture_output=True)
    if result.returncode != 0:
        raise BackendError(
            f"wasm-ld failed to link the sequence:\n{result.stderr.decode()}"
        )
    return result.stdout
