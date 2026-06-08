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
from fpy.syntax import (
    AstAssert,
    AstBinaryOp,
    AstBlock,
    BinaryStackOp,
)
from fpy.types import FLOAT, INTEGER, INTERNAL_STRING
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

    Each ``emit_*`` returns the ``ir.Value`` holding the computed result, built
    into ``self.builder``'s current basic block.
    """

    def __init__(self, builder: ir.IRBuilder):
        super().__init__()
        self.builder = builder

    def emit(self, node, state: CompileState) -> ir.Value:
        value = state.const_expr_values.get(node)
        if value is not None:
            return self._emit_const_value(value)
        return super().emit(node, state)

    def _emit_const_value(self, value) -> ir.Value:
        """Emit an FpyValue as an LLVM constant."""
        fpy_type = value.type

        assert fpy_type not in (INTEGER, FLOAT, INTERNAL_STRING), value

        # ir.Constant wants a native Python number matching the LLVM type's
        # family. The two branches map FpyValue's stored .val accordingly:
        #   - float types store a Decimal -> float() gives the double/float LLVM
        #     types the Python float they expect.
        #   - integer-family types (U*/I*, plus BOOL and ENUM, which are integers
        #     in LLVM) store a Python int (BOOL stores a bool; int(True) == 1) ->
        #     int() feeds the iN constant.
        if fpy_type.is_float:
            return ir.Constant(fpy_type.llvm_type, float(value.val))
        return ir.Constant(fpy_type.llvm_type, int(value.val))

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

        for stmt in body.stmts:
            if isinstance(stmt, AstAssert):
                self._emit_assert(func, builder, stmt, state)
            # Anything else (the prepended builtin library defs, bare
            # expressions, assignments, commands, ...) is not lowered yet and is
            # silently skipped until the backend grows to cover it.

        # Fell off the end of the sequence without failing: success.
        builder.ret(ir.Constant(ERROR_CODE_TYPE, DirectiveErrorCode.NO_ERROR.value))
        return module

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
