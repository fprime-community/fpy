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
from fpy.symbols import BuiltinFuncSymbol, VariableSymbol
from fpy.syntax import (
    AstAssert,
    AstAssign,
    AstBinaryOp,
    AstBlock,
    AstDef,
    AstFuncCall,
    AstIdent,
    AstIf,
    BinaryStackOp,
)
from fpy.types import FpyValue
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

    def emit_AstFuncCall(self, node: AstFuncCall, state: CompileState) -> ir.Value | None:
        func = state.resolved_symbols[node.func]
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

        # self.variables maps id(VariableSymbol) -> the pointer to its storage:
        # an ir.GlobalVariable for is_global variables, or an alloca for
        # function-locals. Both are load/store pointers, so reads/writes are
        # uniform; only how the storage is created differs.
        self.variables: dict[int, ir.Value] = {}
        # The built-in flags struct (a global with no declaring statement).
        self._declare_flags(module, state)
        # Walk the whole top-level region -- including nested if/elif/else blocks
        # -- declaring each variable's storage. Fpy is block-scoped and a var is
        # is_global when it's outside any function, so a variable declared in a
        # top-level if block is still a global; the walk must reach it (declaring
        # only the outermost scope would drop it). Created up front so function
        # bodies can reference globals declared later in the source.
        self._declare_variables(module, builder, body, state)

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
                EmitLlvmExpr(builder, self.variables).emit(stmt, state)
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
            cond = EmitLlvmExpr(builder, self.variables).emit(condition, state)
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
        self.variables[id(flags)] = g

    def _declare_variables(
        self,
        module: ir.Module,
        builder: ir.IRBuilder,
        block: AstBlock,
        state: CompileState,
    ) -> None:
        """Recursively declare storage for every variable assigned in *block* and
        its nested blocks (if/elif/else bodies). Does not descend into function
        defs -- their locals belong to their own frame (and nested defs can't
        exist anyway)."""
        for stmt in block.stmts:
            if isinstance(stmt, AstAssign):
                self._declare_variable(module, builder, stmt, state)
            elif isinstance(stmt, AstIf):
                self._declare_variables(module, builder, stmt.body, state)
                for case in stmt.elifs:
                    self._declare_variables(module, builder, case.body, state)
                if stmt.els is not None:
                    self._declare_variables(module, builder, stmt.els, state)

    def _declare_variable(
        self,
        module: ir.Module,
        builder: ir.IRBuilder,
        node: AstAssign,
        state: CompileState,
    ) -> None:
        """Declare storage for the variable assigned by *node*, once."""
        sym = state.resolved_symbols[node.lhs]
        assert isinstance(sym, VariableSymbol), sym
        if id(sym) in self.variables:
            return  # already declared (a reassignment to the same symbol)
        if sym.is_global:
            g = ir.GlobalVariable(module, sym.type.llvm_type, name=sym.name)
            g.linkage = "internal"
            # Zero-initialized; the declaring assignment writes the real value.
            g.initializer = ir.Constant(sym.type.llvm_type, None)
            self.variables[id(sym)] = g
        else:
            self.variables[id(sym)] = builder.alloca(
                sym.type.llvm_type, name=sym.name
            )

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
