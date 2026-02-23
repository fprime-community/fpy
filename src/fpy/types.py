from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Union, get_args, get_origin
from fpy.syntax import (
    BinaryStackOp,
    COMPARISON_OPS,
)

# In Python 3.10+, the `|` operator creates a `types.UnionType`.
# We need to handle this for forward compatibility, but it won't exist in 3.9.
try:
    from types import UnionType

    UNION_TYPES = (Union, UnionType)
except ImportError:
    UNION_TYPES = (Union,)

# Default values for sequence limits - may be overridden by dictionary constants
DEFAULT_MAX_DIRECTIVES_COUNT = 1024
DEFAULT_MAX_DIRECTIVE_SIZE = 2048

# Keep old names as aliases for backward compatibility
MAX_DIRECTIVES_COUNT = DEFAULT_MAX_DIRECTIVES_COUNT
MAX_DIRECTIVE_SIZE = DEFAULT_MAX_DIRECTIVE_SIZE

COMPILER_MAX_STRING_SIZE = 128

# FPP wire-format constants for boolean serialization
FW_SERIALIZE_TRUE_VALUE = 0xFF
FW_SERIALIZE_FALSE_VALUE = 0x00


class TypeKind(str, Enum):
    # Concrete primitive types
    U8 = "U8"
    U16 = "U16"
    U32 = "U32"
    U64 = "U64"
    I8 = "I8"
    I16 = "I16"
    I32 = "I32"
    I64 = "I64"
    F32 = "F32"
    F64 = "F64"
    BOOL = "bool"
    STRING = "string"
    # Concrete compound types
    ENUM = "enum"
    STRUCT = "struct"
    ARRAY = "array"
    # Compiler-internal types (never serialized to bytecode as stack values)
    INTEGER = "Integer"  # arbitrary-precision integer literal
    FLOAT = "Float"  # arbitrary-precision float literal
    INTERNAL_STRING = "InternalString"  # arbitrary-length string
    RANGE = "Range"  # range expression
    NOTHING = "Nothing"  # void / no-value


# struct format for each primitive kind
_PRIMITIVE_FORMATS: dict[TypeKind, str] = {
    TypeKind.U8: ">B",
    TypeKind.U16: ">H",
    TypeKind.U32: ">I",
    TypeKind.U64: ">Q",
    TypeKind.I8: ">b",
    TypeKind.I16: ">h",
    TypeKind.I32: ">i",
    TypeKind.I64: ">q",
    TypeKind.F32: ">f",
    TypeKind.F64: ">d",
    TypeKind.BOOL: ">B",
}

# Size in bytes for each primitive kind
_PRIMITIVE_SIZES: dict[TypeKind, int] = {
    TypeKind.U8: 1,
    TypeKind.U16: 2,
    TypeKind.U32: 4,
    TypeKind.U64: 8,
    TypeKind.I8: 1,
    TypeKind.I16: 2,
    TypeKind.I32: 4,
    TypeKind.I64: 8,
    TypeKind.F32: 4,
    TypeKind.F64: 8,
    TypeKind.BOOL: 1,
}

# Bit widths
_PRIMITIVE_BITS: dict[TypeKind, int] = {
    TypeKind.U8: 8,
    TypeKind.U16: 16,
    TypeKind.U32: 32,
    TypeKind.U64: 64,
    TypeKind.I8: 8,
    TypeKind.I16: 16,
    TypeKind.I32: 32,
    TypeKind.I64: 64,
    TypeKind.F32: 32,
    TypeKind.F64: 64,
    TypeKind.BOOL: 8,
}

# Inclusive integer ranges
_INTEGER_RANGES: dict[TypeKind, tuple[int, int]] = {
    TypeKind.U8: (0, 255),
    TypeKind.U16: (0, 65535),
    TypeKind.U32: (0, 2**32 - 1),
    TypeKind.U64: (0, 2**64 - 1),
    TypeKind.I8: (-128, 127),
    TypeKind.I16: (-32768, 32767),
    TypeKind.I32: (-(2**31), 2**31 - 1),
    TypeKind.I64: (-(2**63), 2**63 - 1),
}

# Kind sets for fast membership tests
_SIGNED_INTEGER_KINDS = frozenset(
    {TypeKind.I8, TypeKind.I16, TypeKind.I32, TypeKind.I64}
)
_UNSIGNED_INTEGER_KINDS = frozenset(
    {TypeKind.U8, TypeKind.U16, TypeKind.U32, TypeKind.U64}
)
_CONCRETE_INTEGER_KINDS = _SIGNED_INTEGER_KINDS | _UNSIGNED_INTEGER_KINDS
_ALL_INTEGER_KINDS = _CONCRETE_INTEGER_KINDS | {TypeKind.INTEGER}
_CONCRETE_FLOAT_KINDS = frozenset({TypeKind.F32, TypeKind.F64})
_ALL_FLOAT_KINDS = _CONCRETE_FLOAT_KINDS | {TypeKind.FLOAT}
_ALL_NUMERICAL_KINDS = _ALL_INTEGER_KINDS | _ALL_FLOAT_KINDS
_INTERNAL_KINDS = frozenset(
    {
        TypeKind.INTEGER,
        TypeKind.FLOAT,
        TypeKind.INTERNAL_STRING,
        TypeKind.RANGE,
        TypeKind.NOTHING,
    }
)


@dataclass
class StructMember:
    name: str
    type: FpyType


class FpyType:
    """Describes an FPP type.  Singletons for primitives, constructed instances
    for compound types (enums, structs, arrays, strings with length)."""

    __slots__ = (
        "kind",
        "name",
        "max_length",
        "enum_dict",
        "rep_type",
        "members",
        "elem_type",
        "length",
        "default",
    )

    def __init__(
        self,
        kind: TypeKind,
        name: str,
        *,
        max_length: int | None = None,
        enum_dict: dict[str, int] | None = None,
        rep_type: FpyType | None = None,
        members: tuple[StructMember, ...] | None = None,
        elem_type: FpyType | None = None,
        length: int | None = None,
        default: object | None = None,
    ):
        self.kind = kind
        self.name = name
        self.max_length = max_length
        self.enum_dict = enum_dict
        self.rep_type = rep_type
        self.members = members
        self.elem_type = elem_type
        self.length = length
        self.default = default

    # -- identity ----------------------------------------------------------

    def __eq__(self, other):
        if not isinstance(other, FpyType):
            return NotImplemented
        return self.kind == other.kind and self.name == other.name

    def __hash__(self):
        return hash((self.kind, self.name))

    def __repr__(self):
        if self.kind == TypeKind.STRING:
            return f"FpyType(String[{self.max_length}])"
        return f"FpyType({self.name})"

    # -- classification properties -----------------------------------------

    @property
    def is_integer(self) -> bool:
        """True for U8..I64 and the internal INTEGER type."""
        return self.kind in _ALL_INTEGER_KINDS

    @property
    def is_float(self) -> bool:
        """True for F32, F64, and the internal FLOAT type."""
        return self.kind in _ALL_FLOAT_KINDS

    @property
    def is_numerical(self) -> bool:
        return self.kind in _ALL_NUMERICAL_KINDS

    @property
    def is_signed(self) -> bool:
        return self.kind in _SIGNED_INTEGER_KINDS

    @property
    def is_unsigned(self) -> bool:
        return self.kind in _UNSIGNED_INTEGER_KINDS

    @property
    def is_concrete_integer(self) -> bool:
        return self.kind in _CONCRETE_INTEGER_KINDS

    @property
    def is_concrete_float(self) -> bool:
        return self.kind in _CONCRETE_FLOAT_KINDS

    @property
    def is_concrete(self) -> bool:
        """True if this type can appear at runtime (not a compiler-internal type)."""
        return self.kind not in _INTERNAL_KINDS

    @property
    def is_primitive(self) -> bool:
        """True for U8..F64 and BOOL."""
        return self.kind in _PRIMITIVE_FORMATS

    @property
    def is_string(self) -> bool:
        """True for both concrete STRING and internal INTERNAL_STRING."""
        return self.kind in (TypeKind.STRING, TypeKind.INTERNAL_STRING)

    @property
    def display_name(self) -> str:
        """Human-readable type name for error messages."""
        if self.kind == TypeKind.INTEGER:
            return "Integer"
        if self.kind == TypeKind.FLOAT:
            return "Float"
        if self.kind == TypeKind.INTERNAL_STRING:
            return "String"
        return self.name

    # -- size / range properties -------------------------------------------

    @property
    def max_size(self) -> int:
        """Maximum serialized size in bytes."""
        if self.kind in _PRIMITIVE_SIZES:
            return _PRIMITIVE_SIZES[self.kind]
        if self.kind in (TypeKind.STRING, TypeKind.INTERNAL_STRING):
            assert (
                self.max_length is not None
            ), "Cannot compute size of arbitrary-length string"
            return 2 + self.max_length
        if self.kind == TypeKind.ENUM:
            return self.rep_type.max_size
        if self.kind == TypeKind.STRUCT:
            return sum(m.type.max_size for m in self.members)
        if self.kind == TypeKind.ARRAY:
            return self.elem_type.max_size * self.length
        if self.kind == TypeKind.NOTHING:
            return 0
        assert False, f"Cannot compute max_size for {self}"

    @property
    def bits(self) -> int | float:
        """Bit width of the type, inf for arbitrary-precision."""
        if self.kind in _PRIMITIVE_BITS:
            return _PRIMITIVE_BITS[self.kind]
        if self.kind in (TypeKind.INTEGER, TypeKind.FLOAT):
            return math.inf
        assert False, f"Cannot compute bits for {self}"

    def value_range(self) -> tuple[int | float, int | float]:
        """(min, max) inclusive range for integer types."""
        if self.kind in _INTEGER_RANGES:
            return _INTEGER_RANGES[self.kind]
        if self.kind == TypeKind.INTEGER:
            return (-math.inf, math.inf)
        assert False, f"Cannot compute range for {self}"

    def validate_value(self, val) -> None:
        """Raise ValueError if *val* is invalid for this type."""
        if self.kind == TypeKind.INTEGER:
            if not isinstance(val, int):
                raise ValueError(f"Expected int, got {type(val)}")
        elif self.kind == TypeKind.FLOAT:
            if not isinstance(val, Decimal):
                raise ValueError(f"Expected Decimal, got {type(val)}")
        elif self.kind in _CONCRETE_INTEGER_KINDS:
            lo, hi = self.value_range()
            if not (lo <= val <= hi):
                raise ValueError(
                    f"Value {val} out of range [{lo}, {hi}] for {self.name}"
                )


U8 = FpyType(TypeKind.U8, "U8")
U16 = FpyType(TypeKind.U16, "U16")
U32 = FpyType(TypeKind.U32, "U32")
U64 = FpyType(TypeKind.U64, "U64")
I8 = FpyType(TypeKind.I8, "I8")
I16 = FpyType(TypeKind.I16, "I16")
I32 = FpyType(TypeKind.I32, "I32")
I64 = FpyType(TypeKind.I64, "I64")
F32 = FpyType(TypeKind.F32, "F32")
F64 = FpyType(TypeKind.F64, "F64")
BOOL = FpyType(TypeKind.BOOL, "bool")
TIME = FpyType(
    TypeKind.STRUCT,
    "Fw.Time",
    members=(
        StructMember("time_base", U16),
        StructMember("time_context", U8),
        StructMember("seconds", U32),
        StructMember("useconds", U32),
    ),
)
INTEGER = FpyType(TypeKind.INTEGER, "Integer")
FLOAT = FpyType(TypeKind.FLOAT, "Float")
INTERNAL_STRING = FpyType(TypeKind.INTERNAL_STRING, "InternalString")
RANGE = FpyType(TypeKind.RANGE, "Range")
NOTHING = FpyType(TypeKind.NOTHING, "Nothing")

# Tuples of concrete types for iteration / membership tests
SPECIFIC_NUMERIC_TYPES = (U32, U16, U64, U8, I16, I32, I64, I8, F32, F64)
SPECIFIC_INTEGER_TYPES = (U32, U16, U64, U8, I16, I32, I64, I8)
SIGNED_INTEGER_TYPES = (I16, I32, I64, I8)
UNSIGNED_INTEGER_TYPES = (U32, U16, U64, U8)
SPECIFIC_FLOAT_TYPES = (F32, F64)
ARBITRARY_PRECISION_TYPES = (FLOAT, INTEGER)

# Map from canonical name to FpyType (primitives only)
PRIMITIVE_TYPE_MAP: dict[str, FpyType] = {
    "U8": U8,
    "U16": U16,
    "U32": U32,
    "U64": U64,
    "I8": I8,
    "I16": I16,
    "I32": I32,
    "I64": I64,
    "F32": F32,
    "F64": F64,
    "bool": BOOL,
}


class FpyValue:
    """A concrete value with an associated FPP type."""

    __slots__ = ("type", "val")

    def __init__(self, type: FpyType, val: Any):
        self.type = type
        self.val = val

    def __repr__(self):
        return f"FpyValue({self.type.name}, {self.val!r})"

    def __eq__(self, other):
        if not isinstance(other, FpyValue):
            return NotImplemented
        return self.type == other.type and self.val == other.val

    def __hash__(self):
        try:
            return hash((self.type, self.val))
        except TypeError:
            return hash(self.type)

    # -- serialization -----------------------------------------------------

    def serialize(self) -> bytes:
        """Serialize this value to bytes (big-endian, FPP wire format)."""
        kind = self.type.kind

        if kind in _PRIMITIVE_FORMATS:
            val = self.val
            if kind == TypeKind.BOOL:
                val = FW_SERIALIZE_TRUE_VALUE if val else FW_SERIALIZE_FALSE_VALUE
            return struct.pack(_PRIMITIVE_FORMATS[kind], val)

        if kind in (TypeKind.STRING, TypeKind.INTERNAL_STRING):
            encoded = (
                self.val.encode("utf-8") if isinstance(self.val, str) else self.val
            )
            if self.type.max_length is not None:
                if len(encoded) > self.type.max_length:
                    raise ValueError(
                        f"String too long: {len(encoded)} > {self.type.max_length}"
                    )
                padding = b"\x00" * (self.type.max_length - len(encoded))
                return struct.pack(">H", len(encoded)) + encoded + padding
            else:
                return struct.pack(">H", len(encoded)) + encoded

        if kind == TypeKind.ENUM:
            val = self.val
            if isinstance(val, str):
                assert val in self.type.enum_dict, f"Unknown enum constant: {val}"
                val = self.type.enum_dict[val]
            return FpyValue(self.type.rep_type, val).serialize()

        if kind == TypeKind.STRUCT:
            output = b""
            for m in self.type.members:
                member_val = self.val[m.name]
                if not isinstance(member_val, FpyValue):
                    member_val = FpyValue(m.type, member_val)
                output += member_val.serialize()
            return output

        if kind == TypeKind.ARRAY:
            output = b""
            for elem in self.val:
                if isinstance(elem, FpyValue):
                    output += elem.serialize()
                else:
                    output += FpyValue(self.type.elem_type, elem).serialize()
            return output

        assert False, f"Cannot serialize {self.type}"

    @staticmethod
    def deserialize(typ: FpyType, data: bytes, offset: int = 0) -> tuple[FpyValue, int]:
        """Deserialize a value of *typ* from *data* at *offset*.
        Returns ``(value, new_offset)``."""
        kind = typ.kind

        if kind in _PRIMITIVE_FORMATS:
            fmt = _PRIMITIVE_FORMATS[kind]
            size = _PRIMITIVE_SIZES[kind]
            raw = struct.unpack_from(fmt, data, offset)[0]
            if kind == TypeKind.BOOL:
                raw = bool(raw)
            return FpyValue(typ, raw), offset + size

        if kind in (TypeKind.STRING, TypeKind.INTERNAL_STRING):
            str_len = struct.unpack_from(">H", data, offset)[0]
            offset += 2
            s = data[offset : offset + str_len].decode("utf-8")
            if typ.max_length is not None:
                offset += typ.max_length  # skip padding
            else:
                offset += str_len
            return FpyValue(typ, s), offset

        if kind == TypeKind.ENUM:
            rep_val, new_offset = FpyValue.deserialize(typ.rep_type, data, offset)
            for name, val in typ.enum_dict.items():
                if val == rep_val.val:
                    return FpyValue(typ, name), new_offset
            return FpyValue(typ, rep_val.val), new_offset

        if kind == TypeKind.STRUCT:
            members_dict: dict[str, FpyValue] = {}
            for m in typ.members:
                member_val, offset = FpyValue.deserialize(m.type, data, offset)
                members_dict[m.name] = member_val
            return FpyValue(typ, members_dict), offset

        if kind == TypeKind.ARRAY:
            elements: list[FpyValue] = []
            for _ in range(typ.length):
                elem, offset = FpyValue.deserialize(typ.elem_type, data, offset)
                elements.append(elem)
            return FpyValue(typ, elements), offset

        assert False, f"Cannot deserialize {typ}"


# Sentinel value for void (no-value) expressions
NOTHING_VALUE = FpyValue(NOTHING, None)


@dataclass
class CmdDef:
    """Command definition (replaces CmdTemplate)."""

    name: str
    opcode: int
    args: list[tuple[str, str, FpyType]]  # (name, description, type)
    description: str = ""

    @property
    def component(self) -> str:
        return self.name.rsplit(".", 1)[0]

    @property
    def mnemonic(self) -> str:
        return self.name.rsplit(".", 1)[1]

    @property
    def arguments(self) -> list[tuple[str, str, FpyType]]:
        return self.args


@dataclass
class ChDef:
    """Telemetry channel definition (replaces ChTemplate)."""

    name: str
    ch_id: int
    ch_type: FpyType
    description: str = ""


@dataclass
class PrmDef:
    """Parameter definition (replaces PrmTemplate)."""

    name: str
    prm_id: int
    prm_type: FpyType
    default: Any = None
    description: str = ""


# The canonical Svc.Fpy.FlagId enum type
FLAG_ID = FpyType(
    TypeKind.ENUM,
    "Svc.Fpy.FlagId",
    enum_dict={"EXIT_ON_CMD_FAIL": 0},
    rep_type=U8,
)

# The canonical Fw.CmdResponse enum type
CMD_RESPONSE = FpyType(
    TypeKind.ENUM,
    "Fw.CmdResponse",
    enum_dict={
        "OK": 0,
        "INVALID_OPCODE": 1,
        "VALIDATION_ERROR": 2,
        "FORMAT_ERROR": 3,
        "EXECUTION_ERROR": 4,
        "BUSY": 5,
    },
    rep_type=U8,
)

# The canonical Fw.TimeComparison enum type
TIME_COMPARISON = FpyType(
    TypeKind.ENUM,
    "Fw.TimeComparison",
    enum_dict={"LT": -1, "EQ": 0, "GT": 1, "INCOMPARABLE": 2},
    rep_type=I32,
)

# The canonical Fw.TimeIntervalValue struct type
TIME_INTERVAL = FpyType(
    TypeKind.STRUCT,
    "Fw.TimeIntervalValue",
    members=(
        StructMember("seconds", U32),
        StructMember("useconds", U32),
    ),
)

# Internal type (prefixed with $) not directly accessible to users,
# used for desugaring check statements.
CHECK_STATE = FpyType(
    TypeKind.STRUCT,
    "$CheckState",
    members=(
        StructMember("persist", TIME_INTERVAL),
        StructMember("timeout", TIME),
        StructMember("freq", TIME_INTERVAL),
        StructMember("result", BOOL),
        StructMember("last_was_true", BOOL),
        StructMember("last_time_true", TIME),
        StructMember("time_started", TIME),
    ),
)


def is_instance_compat(obj, cls):
    """
    A wrapper for isinstance() that correctly handles Union types in Python 3.9+.
    """
    origin = get_origin(cls)
    if origin in UNION_TYPES:
        return isinstance(obj, get_args(cls))
    return isinstance(obj, cls)


# Time operator overloads:
# maps (lhs_type, rhs_type, op) -> (intermediate_type, result_type, func_name, is_comparison)
TIME_OPS: dict[
    tuple[FpyType, FpyType, BinaryStackOp], tuple[FpyType, FpyType, str, bool]
] = {
    # Time - Time -> TimeInterval
    (TIME, TIME, BinaryStackOp.SUBTRACT): (
        TIME,
        TIME_INTERVAL,
        "time_sub",
        False,
    ),
    # Time + TimeInterval -> Time
    (TIME, TIME_INTERVAL, BinaryStackOp.ADD): (TIME, TIME, "time_add", False),
    # TimeInterval +/- TimeInterval -> TimeInterval
    (TIME_INTERVAL, TIME_INTERVAL, BinaryStackOp.ADD): (
        TIME_INTERVAL,
        TIME_INTERVAL,
        "time_interval_add",
        False,
    ),
    (TIME_INTERVAL, TIME_INTERVAL, BinaryStackOp.SUBTRACT): (
        TIME_INTERVAL,
        TIME_INTERVAL,
        "time_interval_sub",
        False,
    ),
    # Time comparisons -> Bool
    **{
        (TIME, TIME, op): (TIME, BOOL, "time_cmp_assert_comparable", True)
        for op in COMPARISON_OPS
    },
    # TimeInterval comparisons -> Bool
    **{
        (TIME_INTERVAL, TIME_INTERVAL, op): (
            TIME_INTERVAL,
            BOOL,
            "time_interval_cmp",
            True,
        )
        for op in COMPARISON_OPS
    },
}
