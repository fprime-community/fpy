from __future__ import annotations

import struct
from dataclasses import dataclass, fields
from enum import Enum
from typing import ClassVar

from fpy.types import (
    FpyType,
    FpyValue,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    BOOL,
)
from fpy.syntax import (
    BinaryStackOp,
    UnaryStackOp,
    COMPARISON_OPS,
    NUMERIC_OPERATORS,
    BOOLEAN_OPERATORS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Bytecode-level type aliases (FpyType singletons)
# ─────────────────────────────────────────────────────────────────────────────

FwSizeType = U64
FwChanIdType = U32
FwPrmIdType = U32
FwOpcodeType = U32
ArrayIndexType = I64
StackSizeType = U32
SignedStackSizeType = I32
LoopVarType = I64  # same as ArrayIndexType


# ─────────────────────────────────────────────────────────────────────────────
# DirectiveId enum
# ─────────────────────────────────────────────────────────────────────────────


class DirectiveId(Enum):
    INVALID = 0
    WAIT_REL = 1
    WAIT_ABS = 2
    GOTO = 3
    IF = 4
    NO_OP = 5
    PUSH_TLM_VAL = 6
    PUSH_PRM = 7
    CONST_CMD = 8
    # stack op directives
    # all of these are handled at the CPP level by one StackOpDirective to save boilerplate
    # you MUST keep them all in between OR and ITRUNC_64_32 inclusive
    # boolean ops
    OR = 9
    AND = 10
    # integer equalities
    IEQ = 11
    INE = 12
    # unsigned integer inequalities
    ULT = 13
    ULE = 14
    UGT = 15
    UGE = 16
    # signed integer inequalities
    SLT = 17
    SLE = 18
    SGT = 19
    SGE = 20
    # floating point equalities
    FEQ = 21
    FNE = 22
    # floating point inequalities
    FLT = 23
    FLE = 24
    FGT = 25
    FGE = 26
    NOT = 27
    # floating point conversion to signed/unsigned integer,
    # and vice versa
    FPTOSI = 28
    FPTOUI = 29
    SITOFP = 30
    UITOFP = 31
    # integer arithmetic
    ADD = 32
    SUB = 33
    MUL = 34
    UDIV = 35
    SDIV = 36
    UMOD = 37
    SMOD = 38
    # float arithmetic
    FADD = 39
    FSUB = 40
    FMUL = 41
    FDIV = 42
    FPOW = 43
    FLOG = 44
    FMOD = 45
    # floating point bitwidth conversions
    FPEXT = 46
    FPTRUNC = 47
    # integer bitwidth conversions
    # signed integer extend
    SIEXT_8_64 = 48
    SIEXT_16_64 = 49
    SIEXT_32_64 = 50
    # zero (unsigned) integer extend
    ZIEXT_8_64 = 51
    ZIEXT_16_64 = 52
    ZIEXT_32_64 = 53
    # integer truncate
    ITRUNC_64_8 = 54
    ITRUNC_64_16 = 55
    ITRUNC_64_32 = 56
    # end stack op dirs

    EXIT = 57
    ALLOCATE = 58
    STORE_REL_CONST_OFFSET = 59
    LOAD_REL = 60
    PUSH_VAL = 61
    DISCARD = 62
    MEMCMP = 63
    STACK_CMD = 64
    PUSH_TLM_VAL_AND_TIME = 65
    PUSH_TIME = 66
    SET_FLAG = 67
    GET_FLAG = 68
    GET_FIELD = 69
    PEEK = 70
    STORE_REL = 71
    CALL = 72
    RETURN = 73
    LOAD_ABS = 74
    STORE_ABS = 75
    STORE_ABS_CONST_OFFSET = 76


# ─────────────────────────────────────────────────────────────────────────────
# Directive base class
# ─────────────────────────────────────────────────────────────────────────────


class Directive:
    opcode: ClassVar[DirectiveId] = DirectiveId.INVALID
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {}

    def serialize(self) -> bytes:
        arg_bytes = self.serialize_args()
        output = FpyValue(U8, self.opcode.value).serialize()
        output += FpyValue(U16, len(arg_bytes)).serialize()
        output += arg_bytes
        return output

    def serialize_args(self) -> bytes:
        output = bytes()
        for f in fields(self):
            value = getattr(self, f.name)

            if isinstance(value, FpyValue):
                output += value.serialize()
                continue

            if isinstance(value, bytes):
                output += value
                continue

            # Look up the FpyType for this field and serialize
            fpy_type = self._FIELD_TYPES.get(f.name)
            assert fpy_type is not None, (
                f"No type mapping for field {f.name} in {type(self).__name__}"
            )
            output += FpyValue(fpy_type, value).serialize()

        return output

    def __repr__(self):
        r = self.__class__.__old_repr__(self)
        name = self.__class__.__name__.replace("Directive", "").upper()
        value = "".join(r.split("(")[1:])
        return name + "(" + value

    @classmethod
    def deserialize(cls, data: bytes, offset: int) -> tuple[int, Directive] | None:
        if len(data) - offset < 3:
            return None
        opcode = struct.unpack_from(">B", data, offset)[0]
        arg_size = struct.unpack_from(">H", data, offset + 1)[0]
        offset += 3
        if len(data) - offset < arg_size:
            return None
        args = data[offset : (offset + arg_size)]
        offset += arg_size

        dir_type_list = [
            c
            for c in (Directive.__subclasses__() + StackOpDirective.__subclasses__())
            if c.opcode.value == opcode
        ]
        if len(dir_type_list) != 1:
            return None

        dir_type = dir_type_list[0]
        arg_offset = 0
        arg_values = []

        for f in fields(dir_type):
            fpy_type = dir_type._FIELD_TYPES.get(f.name)
            if fpy_type is not None:
                val, arg_offset = FpyValue.deserialize(fpy_type, args, arg_offset)
                arg_values.append(val.val)
            else:
                # Must be raw bytes — consume the remainder
                arg_values.append(args[arg_offset:])
                arg_offset = len(args)

        return offset, dir_type(*arg_values)


# ─────────────────────────────────────────────────────────────────────────────
# StackOpDirective base
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StackOpDirective(Directive):
    """Base for directives that operate on the expression stack."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Concrete directives — with fields
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StackCmdDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.STACK_CMD
    args_size: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {"args_size": StackSizeType}


@dataclass
class MemCompareDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.MEMCMP
    size: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {"size": StackSizeType}


@dataclass
class LoadRelDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.LOAD_REL
    lvar_offset: int
    size: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {
        "lvar_offset": SignedStackSizeType,
        "size": StackSizeType,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Integer width-conversion directives (no fields)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class IntegerSignedExtend8To64Directive(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.SIEXT_8_64


@dataclass
class IntegerSignedExtend16To64Directive(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.SIEXT_16_64


@dataclass
class IntegerSignedExtend32To64Directive(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.SIEXT_32_64


@dataclass
class IntegerZeroExtend8To64Directive(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.ZIEXT_8_64


@dataclass
class IntegerZeroExtend16To64Directive(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.ZIEXT_16_64


@dataclass
class IntegerZeroExtend32To64Directive(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.ZIEXT_32_64


@dataclass
class IntegerTruncate64To8Directive(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.ITRUNC_64_8


@dataclass
class IntegerTruncate64To16Directive(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.ITRUNC_64_16


@dataclass
class IntegerTruncate64To32Directive(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.ITRUNC_64_32


# ─────────────────────────────────────────────────────────────────────────────
# Memory / stack management directives
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AllocateDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.ALLOCATE
    size: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {"size": StackSizeType}


@dataclass
class StoreRelDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.STORE_REL
    size: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {"size": StackSizeType}


@dataclass
class StoreRelConstOffsetDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.STORE_REL_CONST_OFFSET
    lvar_offset: int
    size: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {
        "lvar_offset": SignedStackSizeType,
        "size": StackSizeType,
    }


@dataclass
class DiscardDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.DISCARD
    size: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {"size": StackSizeType}


@dataclass
class PushValDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.PUSH_VAL
    val: bytes


@dataclass
class ConstCmdDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.CONST_CMD
    cmd_opcode: int
    args: bytes
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {"cmd_opcode": FwOpcodeType}


# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic stack op directives (no fields)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FloatModuloDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FMOD


@dataclass
class SignedModuloDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.SMOD


@dataclass
class UnsignedModuloDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.UMOD


@dataclass
class IntAddDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.ADD


@dataclass
class IntSubtractDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.SUB


@dataclass
class IntMultiplyDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.MUL


@dataclass
class UnsignedIntDivideDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.UDIV


@dataclass
class SignedIntDivideDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.SDIV


@dataclass
class FloatAddDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FADD


@dataclass
class FloatSubtractDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FSUB


@dataclass
class FloatMultiplyDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FMUL


@dataclass
class FloatExponentDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FPOW


@dataclass
class FloatDivideDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FDIV


@dataclass
class FloatLogDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FLOG


# ─────────────────────────────────────────────────────────────────────────────
# Control flow directives
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WaitRelDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.WAIT_REL


@dataclass
class WaitAbsDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.WAIT_ABS


@dataclass
class GotoDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.GOTO
    dir_idx: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {"dir_idx": U32}


@dataclass
class IfDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.IF
    false_goto_dir_index: int
    """U32: The dir index to go to if the top of stack is false."""
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {"false_goto_dir_index": U32}


@dataclass
class NoOpDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.NO_OP


@dataclass
class PushTlmValDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.PUSH_TLM_VAL
    chan_id: int
    """FwChanIdType: The telemetry channel ID to get."""
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {"chan_id": FwChanIdType}


@dataclass
class PushPrmDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.PUSH_PRM
    prm_id: int
    """FwPrmIdType: The parameter ID to get the value of."""
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {"prm_id": FwPrmIdType}


# ─────────────────────────────────────────────────────────────────────────────
# Comparison / boolean stack ops (no fields)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OrDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.OR


@dataclass
class AndDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.AND


@dataclass
class IntEqualDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.IEQ


@dataclass
class IntNotEqualDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.INE


@dataclass
class UnsignedLessThanDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.ULT


@dataclass
class UnsignedLessThanOrEqualDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.ULE


@dataclass
class UnsignedGreaterThanDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.UGT


@dataclass
class UnsignedGreaterThanOrEqualDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.UGE


@dataclass
class SignedLessThanDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.SLT


@dataclass
class SignedLessThanOrEqualDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.SLE


@dataclass
class SignedGreaterThanDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.SGT


@dataclass
class SignedGreaterThanOrEqualDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.SGE


@dataclass
class FloatGreaterThanOrEqualDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FGE


@dataclass
class FloatLessThanOrEqualDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FLE


@dataclass
class FloatLessThanDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FLT


@dataclass
class FloatGreaterThanDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FGT


@dataclass
class FloatEqualDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FEQ


@dataclass
class FloatNotEqualDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FNE


@dataclass
class NotDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.NOT


# ─────────────────────────────────────────────────────────────────────────────
# Type-conversion stack ops (no fields)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FloatTruncateDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FPTRUNC


@dataclass
class FloatExtendDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FPEXT


@dataclass
class FloatToSignedIntDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FPTOSI


@dataclass
class SignedIntToFloatDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.SITOFP


@dataclass
class FloatToUnsignedIntDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.FPTOUI


@dataclass
class UnsignedIntToFloatDirective(StackOpDirective):
    opcode: ClassVar[DirectiveId] = DirectiveId.UITOFP


# ─────────────────────────────────────────────────────────────────────────────
# Miscellaneous directives
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExitDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.EXIT


@dataclass
class GetFieldDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.GET_FIELD
    parent_size: int
    member_size: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {
        "parent_size": StackSizeType,
        "member_size": StackSizeType,
    }


@dataclass
class PeekDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.PEEK


@dataclass
class PushTimeDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.PUSH_TIME


@dataclass
class SetFlagDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.SET_FLAG
    flag_idx: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {"flag_idx": U8}


@dataclass
class GetFlagDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.GET_FLAG
    flag_idx: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {"flag_idx": U8}


@dataclass
class CallDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.CALL


@dataclass
class ReturnDirective(Directive):
    opcode: ClassVar[DirectiveId] = DirectiveId.RETURN
    return_val_size: int
    call_args_size: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {
        "return_val_size": StackSizeType,
        "call_args_size": StackSizeType,
    }


@dataclass
class LoadAbsDirective(Directive):
    """Load a value from a global variable (absolute offset from start of stack)"""

    opcode: ClassVar[DirectiveId] = DirectiveId.LOAD_ABS
    global_offset: int
    size: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {
        "global_offset": SignedStackSizeType,
        "size": StackSizeType,
    }


@dataclass
class StoreAbsDirective(Directive):
    """Store a value to a global variable (absolute offset popped from stack)"""

    opcode: ClassVar[DirectiveId] = DirectiveId.STORE_ABS
    size: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {"size": StackSizeType}


@dataclass
class StoreAbsConstOffsetDirective(Directive):
    """Store a value to a global variable at a constant absolute offset"""

    opcode: ClassVar[DirectiveId] = DirectiveId.STORE_ABS_CONST_OFFSET
    global_offset: int
    size: int
    _FIELD_TYPES: ClassVar[dict[str, FpyType]] = {
        "global_offset": SignedStackSizeType,
        "size": StackSizeType,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fix __repr__ for all directive subclasses
# ─────────────────────────────────────────────────────────────────────────────

for cls in Directive.__subclasses__():
    cls.__old_repr__ = cls.__repr__
    cls.__repr__ = Directive.__repr__

for cls in StackOpDirective.__subclasses__():
    cls.__old_repr__ = cls.__repr__
    cls.__repr__ = StackOpDirective.__repr__


# ─────────────────────────────────────────────────────────────────────────────
# Operator dispatch tables  (keys are FpyType singletons)
# ─────────────────────────────────────────────────────────────────────────────

UNARY_STACK_OPS: dict[str, dict[FpyType, type[StackOpDirective]]] = {
    UnaryStackOp.NOT: {BOOL: NotDirective},
    UnaryStackOp.IDENTITY: {
        I64: NoOpDirective,
        U64: NoOpDirective,
        F64: NoOpDirective,
    },
    UnaryStackOp.NEGATE: {
        I64: IntMultiplyDirective,
        U64: IntMultiplyDirective,  # TODO disallow uint negation
        F64: FloatMultiplyDirective,
    },
}

BINARY_STACK_OPS: dict[str, dict[FpyType, type[StackOpDirective]]] = {
    BinaryStackOp.EXPONENT: {F64: FloatExponentDirective},
    BinaryStackOp.MODULUS: {
        I64: SignedModuloDirective,
        U64: UnsignedModuloDirective,
        F64: FloatModuloDirective,
    },
    BinaryStackOp.ADD: {
        I64: IntAddDirective,
        U64: IntAddDirective,
        F64: FloatAddDirective,
    },
    BinaryStackOp.SUBTRACT: {
        I64: IntSubtractDirective,
        U64: IntSubtractDirective,
        F64: FloatSubtractDirective,
    },
    BinaryStackOp.MULTIPLY: {
        I64: IntMultiplyDirective,
        U64: IntMultiplyDirective,
        F64: FloatMultiplyDirective,
    },
    BinaryStackOp.DIVIDE: {
        I64: SignedIntDivideDirective,
        U64: UnsignedIntDivideDirective,
        F64: FloatDivideDirective,
    },
    BinaryStackOp.FLOOR_DIVIDE: {
        I64: SignedIntDivideDirective,
        U64: UnsignedIntDivideDirective,
        # special case for float floor div
    },
    BinaryStackOp.GREATER_THAN: {
        I64: SignedGreaterThanDirective,
        U64: UnsignedGreaterThanDirective,
        F64: FloatGreaterThanDirective,
    },
    BinaryStackOp.GREATER_THAN_OR_EQUAL: {
        I64: SignedGreaterThanOrEqualDirective,
        U64: UnsignedGreaterThanOrEqualDirective,
        F64: FloatGreaterThanOrEqualDirective,
    },
    BinaryStackOp.LESS_THAN_OR_EQUAL: {
        I64: SignedLessThanOrEqualDirective,
        U64: UnsignedLessThanOrEqualDirective,
        F64: FloatLessThanOrEqualDirective,
    },
    BinaryStackOp.LESS_THAN: {
        I64: SignedLessThanDirective,
        U64: UnsignedLessThanDirective,
        F64: FloatLessThanDirective,
    },
    BinaryStackOp.EQUAL: {
        I64: IntEqualDirective,
        U64: IntEqualDirective,
        F64: FloatEqualDirective,
    },
    BinaryStackOp.NOT_EQUAL: {
        I64: IntNotEqualDirective,
        U64: IntNotEqualDirective,
        F64: FloatNotEqualDirective,
    },
    BinaryStackOp.OR: {BOOL: OrDirective},
    BinaryStackOp.AND: {BOOL: AndDirective},
}
