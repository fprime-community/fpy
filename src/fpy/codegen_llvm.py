from __future__ import annotations

from llvmlite import ir

from fpy.state import CompileState
from fpy.syntax import AstBlock
from fpy.visitors import Visitor


# Generic 64-bit Linux target. This is a placeholder; the real target triple /
# data layout for the flight target will be decided as the backend is built out.
LLVM_TRIPLE = "x86_64-unknown-linux-gnu"


class GenerateLlvmModule:

    def emit(self, body: AstBlock, state: CompileState) -> ir.Module:
        assert body is state.root, "module generator must be run on the root block"
        module = ir.Module(name="seq")
        module.triple = LLVM_TRIPLE
        # TODO: lower the sequence (and its functions) into LLVM IR here.
        return module

