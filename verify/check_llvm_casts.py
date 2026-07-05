#!/usr/bin/env python3
"""Prove that fpy's LLVM cast codegen implements the MATH_CASTS_DRAFT.md spec.

For every ordered pair (S, T) of the 10 numeric types, this script:

  1. calls the *production* codegen (EmitLlvmExpr.convert_numeric_type in
     src/fpy/codegen_llvm.py) to emit a one-cast LLVM function S -> T,
  2. round-trips the module text through LLVM's own parser/verifier and
     denotes the function as a Z3 term using verify/llvm_semantics.py
     (instruction semantics transcribed from Alive2 -- an encoding of LLVM
     independent of the spec's),
  3. proves the denotation equal to the spec function cast() from
     verify/cast_properties.py for all inputs.

A PASS on a pair is a machine-checked theorem: "the IR fpy emits for this
cast computes exactly the function the spec defines" -- modulo the shared
FP-theory blind spot for NaN payloads (see llvm_semantics.py docstring).

Run:
    uv run --with z3-solver python verify/check_llvm_casts.py
"""

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llvmlite import ir

from fpy import types as fpy_types
from fpy.codegen_llvm import EmitLlvmExpr

import cast_properties as spec
from llvm_semantics import denote_module


def emit_cast_module() -> str:
    """One LLVM function per (S, T) pair, bodies produced by fpy's codegen."""
    module = ir.Module(name="fpy_casts")
    names = list(spec.TYPES)
    for from_name, to_name in product(names, names):
        from_fpy = getattr(fpy_types, from_name)
        to_fpy = getattr(fpy_types, to_name)
        fnty = ir.FunctionType(to_fpy.llvm_type, [from_fpy.llvm_type])
        fn = ir.Function(module, fnty, name=f"cast_{from_name}_{to_name}")
        fn.args[0].name = "x"
        builder = ir.IRBuilder(fn.append_basic_block("entry"))
        result = EmitLlvmExpr(builder).convert_numeric_type(
            fn.args[0], from_fpy, to_fpy
        )
        builder.ret(result)
    return str(module)


def main():
    print("fpy LLVM cast codegen vs MATH_CASTS_DRAFT.md spec\n")

    denotations = denote_module(emit_cast_module())

    names = list(spec.TYPES)
    for from_name, to_name in product(names, names):
        frm, to = spec.TYPES[from_name], spec.TYPES[to_name]
        args, impl = denotations[f"cast_{from_name}_{to_name}"]
        assert len(args) == 1
        x = args[0]
        assert x.sort() == frm.sort, (x.sort(), frm.sort)
        assert impl.sort() == to.sort, (impl.sort(), to.sort)
        # SMT `=`: on bitvectors exact, on floats it distinguishes +0/-0 and
        # is true on NaN==NaN, i.e. equality of spec values.
        spec.prove(f"codegen == spec {from_name}->{to_name}", impl == spec.cast(x, frm, to))

    n_pass = sum(1 for s, _, _ in spec.results if s == spec.PASS)
    n_fail = sum(1 for s, _, _ in spec.results if s == spec.FAIL)
    n_warn = sum(1 for s, _, _ in spec.results if s == spec.WARN)
    print(f"\n{n_pass} passed, {n_fail} failed, {n_warn} unknown/timeout")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
