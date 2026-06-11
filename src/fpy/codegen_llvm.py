from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from llvmlite import ir
import llvmlite.binding as llvm

from fpy.error import BackendError
from fpy.model import DirectiveErrorCode
from fpy.state import CompileState
from fpy.symbols import BuiltinFuncSymbol, CastSymbol, VariableSymbol
from fpy.syntax import (
    AstAssert,
    AstAssign,
    AstBinaryOp,
    AstBlock,
    AstDef,
    AstFuncCall,
    AstIdent,
    AstIf,
    AstUnaryOp,
    BinaryStackOp,
    COMPARISON_OPS,
    UnaryStackOp,
)
from fpy.types import FpyValue
from fpy.visitors import STOP_DESCENT, Emitter, TopDownVisitor


LLVM_TRIPLE = "wasm32-unknown-unknown"

# Target the original WebAssembly 1.0 MVP (the W3C Core Spec 1.0)
LLVM_CPU = "mvp"
WASM_VERSION = "1.0 (MVP)"

# TODO enable custom page sizes

# TODO strip custom sections from WASM--wasm opt crate, optimize for size?

# TODO with wasm mvp llvm will provide pow/fmod?? 

# TODO will have to make exit() macro a host function

# TODO could just start with a .a and a header??

# exit
# command (opcode i32, ptr i32, len i32)
# telemetry ()
# param ()

# The sequence entry point returns an error code. 0 means success
ERROR_CODE_TYPE = ir.IntType(32)


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
            # Const values are stored at the node's contextual type already.
            return value.llvm_value
        result = super().emit(node, state)
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
            return b.fcmp_ordered(op, lhs, rhs)
        # Enums and bools lower to integers too, so any integer-typed value
        # (not just numeric types) compares with icmp; aggregates don't.
        if isinstance(lhs.type, ir.IntType):
            return b.icmp_signed(op, lhs, rhs) if is_signed \
                else b.icmp_unsigned(op, lhs, rhs)
        raise BackendError(
            f"LLVM backend can't compare values of type "
            f"'{intermediate_type.display_name}' yet"
        )

    def _emit_floor_divide(
        self, lhs: ir.Value, rhs: ir.Value, is_float: bool, is_signed: bool
    ) -> ir.Value:
        """Emit `lhs // rhs`, flooring toward -inf (Python `//`, matching the VM).

        Floats floor the quotient directly via llvm.floor (a native f64.floor on
        wasm). Integer sdiv/udiv truncate toward zero, which differs from floor
        only when the operands have opposite signs and the division is inexact;
        in that case we subtract one.
        """
        b = self.builder
        if is_float:
            # Floats have a real floor: divide, then floor the quotient. The
            # llvm.floor intrinsic lowers to a native f64.floor on wasm (no
            # libcall), so e.g. -5.5 / 2.0 = -2.75 floors to -3.0.
            quotient = b.fdiv(lhs, rhs)
            floor_fn = b.module.declare_intrinsic(
                "llvm.floor", [quotient.type]
            )
            return b.call(floor_fn, [quotient])
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
        """Emit `lhs % rhs` with the VM's *floored* semantics (Python `%`): the
        result takes the sign of the divisor.

        The IR remainder ops (srem/frem) are *truncated* -- the result takes the
        sign of the dividend -- so we correct it by adding the divisor back when
        the remainder is nonzero and its sign differs from the divisor's. (frem
        lowers to an fmod libcall on wasm, hence the imported env.fmod.)
        """
        b = self.builder
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

    def _emit_short_circuit(
        self, node: AstBinaryOp, state: CompileState
    ) -> ir.Value:
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
        # integers (matching the VM, which multiplies by -1; sub is the simpler
        # equivalent here).
        assert node.op == UnaryStackOp.NEGATE, node.op
        if intermediate_type.is_float:
            return b.fneg(val)
        return b.neg(val)

    def emit_AstIdent(self, node: AstIdent, state: CompileState) -> ir.Value:
        sym = state.resolved_symbols[node]
        assert isinstance(sym, VariableSymbol), sym
        # Load the variable at its stored (declared) type, which is the ident's
        # synthesized type; emit() handles any widening to the contextual type.
        return self.builder.load(sym.llvm_ptr, name=str(node.name))

    def emit_AstFuncCall(self, node: AstFuncCall, state: CompileState) -> ir.Value | None:
        func = state.resolved_symbols[node.func]
        if isinstance(func, CastSymbol):
            # the actual conversion happens already as part of the 
            # synthesized -> contextual conversion
            return self.emit(node.args[0], state)
        if not isinstance(func, BuiltinFuncSymbol):
            raise BackendError(
                f"LLVM backend can't lower a call to {type(func).__name__} yet"
            )
        # Pass each argument as (emitted ir.Value, its constant FpyValue or None
        # if it isn't a compile-time constant). The builtin's generate_llvm picks
        # whichever it needs.
        args: list[tuple[ir.Value, FpyValue | None]] = []
        for arg in node.args or []:
            if isinstance(arg, FpyValue):  # a filled-in default argument
                args.append((arg.llvm_value, arg))
            else:
                args.append((self.emit(arg, state), state.const_expr_values.get(arg)))
        return func.generate_llvm(self.builder, args)

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
            to_float = self.builder.sitofp if from_type.is_signed else self.builder.uitofp
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
        fn = self.builder.module.declare_intrinsic(base, (result_type, value.type), ir.FunctionType(result_type, [value.type]))
        return self.builder.call(fn, [value])


FPY_ENTRY_POINT = "fpy_main"


class CollectFrameVariables(TopDownVisitor):
    """Collects every variable declared in a frame"""

    def __init__(self):
        super().__init__()
        self.symbols: list[VariableSymbol] = []

    def visit_AstAssign(self, node: AstAssign, state: CompileState):
        sym = state.resolved_symbols.get(node.lhs)
        # Only variable declarations/reassignments need storage; a field or
        # element target (x.f = ..., a[i] = ...) resolves to something else.
        if isinstance(sym, VariableSymbol):
            self.symbols.append(sym)

    def visit_AstDef(self, node: AstDef, state: CompileState):
        return STOP_DESCENT  # a def's locals belong to its own frame


class GenerateLlvmModule:
    """Lowers a sequence's top-level statements into an LLVM module.
    """

    def emit(self, body: AstBlock, state: CompileState) -> ir.Module:
        assert body is state.root, "module generator must be run on the root block"
        module = ir.Module(name="seq")
        module.triple = LLVM_TRIPLE

        func_type = ir.FunctionType(ERROR_CODE_TYPE, [])
        func = ir.Function(module, func_type, name=FPY_ENTRY_POINT)
        builder = ir.IRBuilder(func.append_basic_block(name="entry"))

        # The built-in flags struct (a global with no declaring statement).
        self._declare_flags(module, state)
        # Declare storage for every variable in this frame up front.
        collector = CollectFrameVariables()
        collector.run(body, state)
        for sym in collector.symbols:
            self._declare_variable(module, builder, sym)

        self._emit_block(func, builder, body, state)

        # Fell off the end of the sequence without failing: success.
        if not builder.block.is_terminated:
            builder.ret(ir.Constant(ERROR_CODE_TYPE, DirectiveErrorCode.NO_ERROR.value))
        return module

    def _emit_block(
        self,
        func: ir.Function,
        builder: ir.IRBuilder,
        block: AstBlock,
        state: CompileState,
    ) -> None:
        """Lower the statements of *block* into the current basic block(s)."""
        for stmt in block.stmts:
            if isinstance(stmt, AstAssign):
                self._emit_assign(builder, stmt, state)
            elif isinstance(stmt, AstAssert):
                self._emit_assert(func, builder, stmt, state)
            elif isinstance(stmt, AstIf):
                self._emit_if(func, builder, stmt, state)
            elif isinstance(stmt, AstFuncCall):
                # A call statement (e.g. exit(...)); its result, if any, is
                # discarded. Unsupported calls raise inside emit_AstFuncCall.
                EmitLlvmExpr(builder).emit(stmt, state)
            elif isinstance(stmt, AstDef):
                # Function definitions (incl. the prepended builtin library)
                # aren't lowered here; a call to one is handled at the call site.
                continue
            else:
                assert False, (
                    f"LLVM backend doesn't handle statement {type(stmt).__name__}"
                )

    def _emit_if(
        self,
        func: ir.Function,
        builder: ir.IRBuilder,
        node: AstIf,
        state: CompileState,
    ) -> None:
        """Lower an if / elif* / else chain.

        Each case tests its condition; on true it runs its body and jumps to a
        shared end block, on false it falls through to test the next case. A
        block that already ends in a terminator (e.g. its body called exit())
        is not given a redundant branch to the end.
        """
        end_block = func.append_basic_block("if_end")
        cases = [(node.condition, node.body)]
        cases += [(case.condition, case.body) for case in node.elifs]

        for condition, case_body in cases:
            cond = EmitLlvmExpr(builder).emit(condition, state)
            then_block = func.append_basic_block("if_then")
            next_block = func.append_basic_block("if_next")
            builder.cbranch(cond, then_block, next_block)

            builder.position_at_end(then_block)
            self._emit_block(func, builder, case_body, state)
            if not builder.block.is_terminated:
                builder.branch(end_block)

            # Subsequent cases (and the else) are tested/run when this condition
            # was false, i.e. in next_block.
            builder.position_at_end(next_block)

        if node.els is not None:
            self._emit_block(func, builder, node.els, state)
        if not builder.block.is_terminated:
            builder.branch(end_block)

        builder.position_at_end(end_block)

    def _declare_flags(self, module: ir.Module, state: CompileState) -> None:
        """Create the built-in ``flags`` struct as a global, seeded with its
        defaults (e.g. assert_cmd_success = True). It has no declaring statement,
        so it isn't reached by the variable walk."""
        flags = state.flags_var
        g = ir.GlobalVariable(module, flags.type.llvm_type, name=flags.name)
        g.linkage = "internal"
        g.initializer = FpyValue(
            flags.type, dict(flags.type.member_defaults)
        ).llvm_value
        flags.llvm_ptr = g

    def _declare_variable(
        self, module: ir.Module, builder: ir.IRBuilder, sym: VariableSymbol
    ) -> None:
        """Declare storage for *sym*, once.

        is_global variables become module-level globals in linear memory, so a
        function can read/write the same slot main does. Locals become an
        entry-block alloca. Any type works for either; promoting slots to
        registers (sroa/mem2reg/globalopt) is left to optimization passes.
        """
        if sym.llvm_ptr is not None:
            return  # already declared (a reassignment to the same symbol)
        if sym.is_global:
            gvar = ir.GlobalVariable(module, sym.type.llvm_type, name=sym.name)
            gvar.linkage = "internal"
            # Zero-initialized; the declaring assignment writes the real value.
            gvar.initializer = ir.Constant(sym.type.llvm_type, None)
            sym.llvm_ptr = gvar
        else:
            sym.llvm_ptr = builder.alloca(sym.type.llvm_type, name=sym.name)

    def _emit_assign(
        self, builder: ir.IRBuilder, node: AstAssign, state: CompileState
    ) -> None:
        sym = state.resolved_symbols[node.lhs]
        # The rhs is coerced to the variable's type, so its emitted value
        # already matches the slot's element type.
        value = EmitLlvmExpr(builder).emit(node.rhs, state)
        builder.store(value, sym.llvm_ptr)

    def _emit_assert(
        self,
        func: ir.Function,
        builder: ir.IRBuilder,
        node: AstAssert,
        state: CompileState,
    ) -> None:
        """Emit ``assert cond``: if cond is false, return the exit code.

        On success, control continues in a fresh block so subsequent statements
        keep lowering after the check.
        """
        condition = EmitLlvmExpr(builder).emit(node.condition, state)

        fail_block = func.append_basic_block(name="assert_fail")
        ok_block = func.append_basic_block(name="assert_ok")
        builder.cbranch(condition, ok_block, fail_block)

        # Failure path: return the exit code the user wrote (verbatim), or
        # EXIT_WITH_ERROR by default. A written code is coerced to U8 (i8), so
        # widen it to the i32 return type.
        builder.position_at_end(fail_block)
        if node.exit_code is None:
            code = ir.Constant(
                ERROR_CODE_TYPE, DirectiveErrorCode.EXIT_WITH_ERROR.value
            )
        else:
            code = EmitLlvmExpr(builder).emit(node.exit_code, state)
            if code.type != ERROR_CODE_TYPE:
                code = builder.zext(code, ERROR_CODE_TYPE)
        builder.ret(code)

        # Success path: continue lowering subsequent statements here.
        builder.position_at_end(ok_block)



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
    return [sys.executable, "-m", "ziglang", "wasm-ld"]


def _llvm_version_str() -> str:
    """The version of LLVM that llvmlite is bound to (e.g. "20.1.8")."""
    return ".".join(str(n) for n in llvm.llvm_version_info)


def _wasm_ld_version_str() -> str:
    """The version line reported by the bundled wasm-ld (e.g. "LLD 21.1.0").

    wasm-ld is shipped separately (via the 'ziglang' package), so its version
    is independent of the LLVM that compiled the IR. Returns "unavailable" if
    wasm-ld can't be run rather than failing -- this is only for --version."""
    try:
        result = subprocess.run(
            _wasm_ld_command() + ["--version"], capture_output=True
        )
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
        # No C-style _start entry point; we call fpy_main directly.
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
    result = subprocess.run(
        _wasm_ld_command() + args, input=stdin, capture_output=True
    )
    if result.returncode != 0:
        raise BackendError(
            f"wasm-ld failed to link the sequence:\n{result.stderr.decode()}"
        )
    return result.stdout
