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
from fpy.symbols import VariableSymbol
from fpy.syntax import (
    AstAssert,
    AstAssign,
    AstBinaryOp,
    AstBlock,
    AstIdent,
    BinaryStackOp,
)
from fpy.types import FLOAT, INTEGER, INTERNAL_STRING, FpyValue, TypeKind
from fpy.visitors import Emitter


LLVM_TRIPLE = "wasm32-unknown-unknown"

# The sequence entry point returns an error code. 0 means success; a failed
# assert returns its exit code verbatim (or EXIT_WITH_ERROR if none was given).
# Note this intentionally does NOT collapse codes the way the bytecode VM's
# handle_exit does -- the code the user writes is the code returned.
# The codes fit in a byte, but we return an i32 to match the wasm ABI boundary:
# wasm has no i8 value type, so the export is `() -> i32` regardless.
ERROR_CODE_TYPE = ir.IntType(32)


class EmitLlvmExpr(Emitter):
    """Lowers a single Fpy arithmetic/comparison expression into LLVM IR.

    Each ``emit_*`` returns the ``ir.Value`` holding the computed result at the
    node's *synthesized* type; emit() then converts it to the node's contextual
    (coerced) type. (Constants are stored already at their contextual type, so
    they skip that conversion.)
    """

    def __init__(
        self,
        builder: ir.IRBuilder,
        variables: dict[int, ir.AllocaInstr],
    ):
        super().__init__()
        self.builder = builder
        # Keyed by id(VariableSymbol): the symbol is an unhashable dataclass,
        # and each variable has exactly one symbol instance, so identity is the
        # right key.
        self.variables = variables

    def emit(self, node, state: CompileState) -> ir.Value:
        value = state.const_expr_values.get(node)
        if value is not None:
            # Const values are stored at the node's contextual type already.
            return self._emit_const_value(value)
        result = super().emit(node, state)
        # Runtime emitters produce the node's synthesized type; coerce to the
        # contextual type the surrounding expression expects (e.g. a U32 var
        # read in a U64 operator context gets widened here).
        synthesized = state.synthesized_types[node]
        contextual = state.contextual_types[node]
        if synthesized != contextual:
            result = self.convert_numeric_type(result, synthesized, contextual)
        return result

    def _emit_const_value(self, value) -> ir.Value:
        """Emit an FpyValue as an LLVM constant"""
        fpy_type = value.type
        kind = fpy_type.kind

        # Internal types have no LLVM representation and should never be emitted.
        assert fpy_type not in (INTEGER, FLOAT, INTERNAL_STRING), value

        # ir.Constant wants a native Python number matching the LLVM type's family:
        if fpy_type.is_float:
            # float types store a Decimal; float() gives the double/float value.
            return ir.Constant(fpy_type.llvm_type, float(value.val))
        if fpy_type.is_integer or kind == TypeKind.BOOL:
            # ints store a Python int; BOOL stores a bool (int(True) == 1).
            return ir.Constant(fpy_type.llvm_type, int(value.val))
        if kind == TypeKind.ENUM:
            # an enum const stores its member name; map it to the integer rep.
            return ir.Constant(fpy_type.llvm_type, fpy_type.enum_dict[value.val])
        if kind == TypeKind.STRUCT:
            members = [
                self._emit_const_value(value.val[m.name]) for m in fpy_type.members
            ]
            return ir.Constant(fpy_type.llvm_type, members)
        if kind == TypeKind.ARRAY:
            elements = [self._emit_const_value(elem) for elem in value.val]
            return ir.Constant(fpy_type.llvm_type, elements)

        raise BackendError(
            f"LLVM backend cannot emit a constant of type {fpy_type.display_name}"
        )

    def emit_AstBinaryOp(self, node: AstBinaryOp, state: CompileState) -> ir.Value:
        intermediate_type = state.op_intermediate_types[node]
        is_float = intermediate_type.is_float

        lhs = self.emit(node.lhs, state)
        rhs = self.emit(node.rhs, state)

        if node.op == BinaryStackOp.ADD:
            return self.builder.fadd(lhs, rhs) if is_float \
                else self.builder.add(lhs, rhs)
        if node.op == BinaryStackOp.EQUAL:
            return self.builder.fcmp_ordered("==", lhs, rhs) if is_float \
                else self.builder.icmp_signed("==", lhs, rhs)
        if node.op == BinaryStackOp.NOT_EQUAL:
            return self.builder.fcmp_ordered("!=", lhs, rhs) if is_float \
                else self.builder.icmp_signed("!=", lhs, rhs)

        raise BackendError(
            f"LLVM backend only supports '+', '==' and '!=' for now, got '{node.op}'"
        )

    def emit_AstIdent(self, node: AstIdent, state: CompileState) -> ir.Value:
        sym = state.resolved_symbols[node]
        assert isinstance(sym, VariableSymbol), sym
        # Load the variable at its stored (declared) type, which is the ident's
        # synthesized type; emit() handles any widening to the contextual type.
        return self.builder.load(self.variables[id(sym)], name=str(node.name))

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
            to_int = self.builder.fptosi if to_type.is_signed else self.builder.fptoui
            return to_int(value, target)
        # float -> float
        if to_type.bits > from_type.bits:
            return self.builder.fpext(value, target)
        if to_type.bits < from_type.bits:
            return self.builder.fptrunc(value, target)
        return value


FPY_ENTRY_POINT = "fpy_main"


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

        # Give every variable a stack slot up front, in the entry block. We
        # always use alloca and leave it to optimization passes to promote slots
        # to registers where worthwhile.
        self.variables: dict[int, ir.AllocaInstr] = {}
        self._declare_flags(builder, state)
        for stmt in body.stmts:
            if isinstance(stmt, AstAssign):
                self._declare_variable(builder, stmt, state)

        for stmt in body.stmts:
            if isinstance(stmt, AstAssign):
                self._emit_assign(builder, stmt, state)
            elif isinstance(stmt, AstAssert):
                self._emit_assert(func, builder, stmt, state)
            # Anything else (the prepended builtin library defs, bare
            # expressions, commands, ...) is not lowered yet and is silently
            # skipped until the backend grows to cover it.

        # Fell off the end of the sequence without failing: success.
        builder.ret(ir.Constant(ERROR_CODE_TYPE, DirectiveErrorCode.NO_ERROR.value))
        return module

    def _declare_flags(self, builder: ir.IRBuilder, state: CompileState) -> None:
        """Allocate and initialize the built-in ``flags`` struct."""
        flags = state.flags_var
        slot = builder.alloca(flags.type.llvm_type, name=flags.name)
        self.variables[id(flags)] = slot
        default = FpyValue(flags.type, dict(flags.type.member_defaults))
        builder.store(EmitLlvmExpr(builder, self.variables)._emit_const_value(default), slot)

    def _declare_variable(
        self, builder: ir.IRBuilder, node: AstAssign, state: CompileState
    ) -> None:
        """Allocate a stack slot for the variable assigned by *node*, once.

        Any type gets an alloca -- aggregates (structs/arrays) included, since
        alloca handles them fine. We leave promoting these slots to registers
        (sroa/mem2reg) to later optimization passes.
        """
        sym = state.resolved_symbols[node.lhs]
        assert isinstance(sym, VariableSymbol), sym
        if id(sym) in self.variables:
            return  # already declared (this is a reassignment)
        self.variables[id(sym)] = builder.alloca(sym.type.llvm_type, name=sym.name)

    def _emit_assign(
        self, builder: ir.IRBuilder, node: AstAssign, state: CompileState
    ) -> None:
        sym = state.resolved_symbols[node.lhs]
        # The rhs is coerced to the variable's type, so its emitted value
        # already matches the slot's element type.
        value = EmitLlvmExpr(builder, self.variables).emit(node.rhs, state)
        builder.store(value, self.variables[id(sym)])

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
        condition = EmitLlvmExpr(builder, self.variables).emit(node.condition, state)

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
            code = EmitLlvmExpr(builder, self.variables).emit(node.exit_code, state)
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
    machine = target.create_target_machine()
    obj = machine.emit_object(parsed)
    return _link_wasm_object(obj)


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
