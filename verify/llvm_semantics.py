"""Z3 denotations of the LLVM IR instructions emitted by fpy's cast codegen.

Each encoder maps one LLVM instruction to a Z3 term over the SMT bitvector
and IEEE-754 floating-point theories. The encodings are transcribed from
Alive2 (https://github.com/AliveToolkit/alive2, commit 1d1bc4f), which is the
de facto SMT semantics of LLVM IR:

  * int conversions:   ir/instr.cpp   ConversionOp::toSMT
  * float conversions: ir/instr.cpp   FpConversionOp::toSMT
  * Z3-level ops:      smt/expr.cpp   fp2sint / fp2uint / sint2fp / uint2fp /
                                      float2Float / round

The transcription is deliberately *independent* of the spec functions in
cast_properties.py: notably, Alive2 detects float->int overflow with an
RTZ round-trip check (sint2fp(fp2sint(x)) == round(x)) instead of the spec's
bound comparisons, so proving the two equal is a real theorem, not an
identity.

Denotational scope (asserted, not silently ignored): straight-line functions
of one basic block, all instructions total and non-poison -- exactly the
fragment fpy's convert_numeric_type emits. Extending to branching/trapping
code (Tier 3: arithmetic overflow checks) will require path conditions and a
(value, halted) result pair.

Known blind spot, shared with cast_properties.py: Z3's FP sort has a single
NaN value, so NaN *payloads* are outside this model. LLVM leaves the payload
of NaN results loosely specified, so the spec's canonical-NaN clause cannot
be verified here (Alive2 models floats as bitvectors precisely for this
reason). See MATH_CASTS_DRAFT.md.
"""

from z3 import (
    RNE,
    RTZ,
    BitVecSort,
    BitVecVal,
    Const,
    Extract,
    FPSort,
    FPVal,
    If,
    Or,
    SignExt,
    ZeroExt,
    fpIsNaN,
    fpIsNegative,
    fpIsZero,
    fpRoundToIntegral,
    fpSignedToFP,
    fpToFP,
    fpToSBV,
    fpToUBV,
    fpUnsignedToFP,
    is_bv_sort,
)

import llvmlite.binding as llvm_binding

F32S = FPSort(8, 24)
F64S = FPSort(11, 53)

_SORTS = {
    "i1": BitVecSort(1),
    "i8": BitVecSort(8),
    "i16": BitVecSort(16),
    "i32": BitVecSort(32),
    "i64": BitVecSort(64),
    "float": F32S,
    "double": F64S,
}


def sort_of(llvm_type_str: str):
    assert llvm_type_str in _SORTS, f"unsupported LLVM type: {llvm_type_str}"
    return _SORTS[llvm_type_str]


# --- instruction encoders ------------------------------------------------------
# One function per opcode. Signature: (operand z3 exprs..., result sort) -> expr.


def encode_trunc(v, to_sort):
    # alive2 ConversionOp::toSMT, Trunc: val.trunc(bits)
    return Extract(to_sort.size() - 1, 0, v)


def encode_zext(v, to_sort):
    # alive2 ConversionOp::toSMT, ZExt: val.zext(delta)
    return ZeroExt(to_sort.size() - v.size(), v)


def encode_sext(v, to_sort):
    # alive2 ConversionOp::toSMT, SExt: val.sext(delta)
    return SignExt(to_sort.size() - v.size(), v)


def encode_sitofp(v, to_sort):
    # alive2 FpConversionOp::toSMT, SIntToFP: val.sint2fp(dummy, rm), rm = RNE
    return fpSignedToFP(RNE(), v, to_sort)


def encode_uitofp(v, to_sort):
    # alive2 FpConversionOp::toSMT, UIntToFP: val.uint2fp(dummy, rm), rm = RNE
    # (fpy never sets the nneg flag, so no poison condition)
    return fpUnsignedToFP(RNE(), v, to_sort)


def encode_fpext(v, to_sort):
    # alive2 FpConversionOp::toSMT, FPExt/FPTrunc: val.float2Float(dummy, rm)
    # = Z3_mk_fpa_to_fp_float with rm = RNE. fpy emits no fast-math flags, so
    # alive2's fm_poison wrapper contributes nothing.
    return fpToFP(RNE(), v, to_sort)


encode_fptrunc = encode_fpext


def encode_fptosi_sat(v, to_sort):
    """alive2 FpConversionOp::toSMT, FPToSInt_Sat.

    Overflow is detected by an RTZ round-trip instead of bound comparisons:
    bv is trusted iff sint2fp(bv) reproduces round(v, RTZ). fp.to_sbv is
    underspecified out of range, but then no in-range bv round-trips onto the
    (out-of-range) rounded value, so every interpretation falls through to
    the saturation arms.
    """
    bits = to_sort.size()
    rm = RTZ()
    bv = fpToSBV(rm, v, to_sort)                    # val.fp2sint(bits, rm)
    fp2 = fpSignedToFP(rm, bv, v.sort())            # bv.sint2fp(val, rm)
    val_rounded = fpRoundToIntegral(rm, v)          # val.round(rm)
    # "-0.xx is converted to 0 and then to 0.0, though -0.xx is ok to convert"
    # (fp2 is +0 while val_rounded is -0, and SMT `=` distinguishes them)
    no_overflow = Or(fpIsZero(val_rounded), fp2 == val_rounded)
    return If(
        fpIsNaN(v),
        BitVecVal(0, bits),
        If(
            no_overflow,
            bv,
            If(
                fpIsNegative(v),
                BitVecVal(-(1 << (bits - 1)), bits),  # expr::IntSMin
                BitVecVal((1 << (bits - 1)) - 1, bits),  # expr::IntSMax
            ),
        ),
    )


def encode_fptoui_sat(v, to_sort):
    """alive2 FpConversionOp::toSMT, FPToUInt_Sat."""
    bits = to_sort.size()
    rm = RTZ()
    bv = fpToUBV(rm, v, to_sort)                    # val.fp2uint(bits, rm)
    fp2 = fpUnsignedToFP(rm, bv, v.sort())          # bv.uint2fp(val, rm)
    val_rounded = fpRoundToIntegral(rm, v)          # val.round(rm)
    # "-0.xx must be converted to 0, not poison."
    no_overflow = Or(fpIsZero(val_rounded), fp2 == val_rounded)
    return If(
        Or(fpIsNaN(v), fpIsNegative(v)),
        BitVecVal(0, bits),
        If(no_overflow, bv, BitVecVal((1 << bits) - 1, bits)),  # expr::IntUMax
    )


_SAT_INTRINSICS = {
    "llvm.fptosi.sat": encode_fptosi_sat,
    "llvm.fptoui.sat": encode_fptoui_sat,
}

_OPCODE_ENCODERS = {
    "trunc": encode_trunc,
    "zext": encode_zext,
    "sext": encode_sext,
    "sitofp": encode_sitofp,
    "uitofp": encode_uitofp,
    "fpext": encode_fpext,
    "fptrunc": encode_fptrunc,
}


# --- the symbolic executor -----------------------------------------------------


def _constant_expr(operand):
    sort = sort_of(str(operand.type))
    val = operand.get_constant_value()
    if is_bv_sort(sort):
        return BitVecVal(val, sort.size())
    return FPVal(val, sort)


def denote_function(fn):
    """Denote a parsed straight-line LLVM function as a Z3 term.

    `fn` is an llvmlite.binding ValueRef of a defined function. Returns
    (args, result): `args` are fresh Z3 constants standing for the function
    parameters, `result` is the Z3 expression of the returned value.
    """
    env = {}
    args = []
    for a in fn.arguments:
        assert a.name, f"unnamed argument in @{fn.name}"
        c = Const(f"{fn.name}!{a.name}", sort_of(str(a.type)))
        env[a.name] = c
        args.append(c)

    def resolve(operand):
        if operand.name:
            assert operand.name in env, (
                f"@{fn.name}: operand %{operand.name} used before definition"
            )
            return env[operand.name]
        return _constant_expr(operand)

    blocks = list(fn.blocks)
    assert len(blocks) == 1, (
        f"@{fn.name}: denotation only covers straight-line code, "
        f"got {len(blocks)} basic blocks"
    )

    result = None
    for inst in blocks[0].instructions:
        opcode = inst.opcode
        operands = list(inst.operands)

        if opcode == "ret":
            assert len(operands) == 1, f"@{fn.name}: void or multi-value ret"
            assert result is None
            result = resolve(operands[0])
        elif opcode == "call":
            callee = operands[-1].name
            base = ".".join(callee.split(".")[:3])  # llvm.fptosi.sat.i8.f64 ->
            assert base in _SAT_INTRINSICS, f"@{fn.name}: unsupported call to @{callee}"
            assert len(operands) == 2, f"@{fn.name}: sat intrinsics take one arg"
            assert inst.name, f"@{fn.name}: unnamed instruction result"
            env[inst.name] = _SAT_INTRINSICS[base](
                resolve(operands[0]), sort_of(str(inst.type))
            )
        else:
            assert opcode in _OPCODE_ENCODERS, f"@{fn.name}: unsupported opcode {opcode}"
            assert len(operands) == 1
            assert inst.name, f"@{fn.name}: unnamed instruction result"
            env[inst.name] = _OPCODE_ENCODERS[opcode](
                resolve(operands[0]), sort_of(str(inst.type))
            )

    assert result is not None, f"@{fn.name}: no ret instruction"
    return args, result


def denote_module(ll_text: str):
    """Parse LLVM assembly (via LLVM itself, which also verifies it) and
    denote every defined function. Returns {name: (args, result)}."""
    mod = llvm_binding.parse_assembly(ll_text)
    out = {}
    for fn in mod.functions:
        if fn.is_declaration:
            continue
        out[fn.name] = denote_function(fn)
    return out
