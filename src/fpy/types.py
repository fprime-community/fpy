from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, auto
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable, Union, get_args, get_origin

if TYPE_CHECKING:
    from llvmlite import ir
from fpy.syntax import (
    BinaryStackOp,
    BOOLEAN_OPERATORS,
    COMPARISON_OPS,
    UnaryStackOp,
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
DEFAULT_MAX_SEQ_ARG_COUNT = 16
DEFAULT_MAX_STACK_SIZE = 65535

# The TimeBase constant used as the default timeBase of time() and of the
# Fw.TimeValue constructor - may be overridden via get_base_compile_state.
DEFAULT_TIME_BASE = "TB_WORKSTATION_TIME"

# Keep old names as aliases for backward compatibility
MAX_DIRECTIVES_COUNT = DEFAULT_MAX_DIRECTIVES_COUNT
MAX_DIRECTIVE_SIZE = DEFAULT_MAX_DIRECTIVE_SIZE

COMPILER_MAX_STRING_SIZE = 128

# FPP wire-format constants for boolean serialization.
# The live FW_SERIALIZE_* values may be overridden from the dictionary at
# compile time (see get_base_compile_state); the DEFAULT_* values are the
# framework fallbacks used when the dictionary does not define them.
DEFAULT_FW_SERIALIZE_TRUE_VALUE = 0xFF
DEFAULT_FW_SERIALIZE_FALSE_VALUE = 0x00

FW_SERIALIZE_TRUE_VALUE = DEFAULT_FW_SERIALIZE_TRUE_VALUE
FW_SERIALIZE_FALSE_VALUE = DEFAULT_FW_SERIALIZE_FALSE_VALUE


class DeserializeError(ValueError):
    """Bytes that cannot be deserialized as a value of the requested type
    (fprime's FW_DESERIALIZE_* error statuses)."""


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
    ENUM = "enum"
    STRUCT = "struct"
    ARRAY = "array"
    SINGLETON = "singleton"
    RANGE = "range"  # range expression
    UNIT = "unit"  # the type of an expression that produces the Unit value
    ANON_STRUCT = "AnonStruct"  # anonymous struct literal
    ANON_ARRAY = "AnonArray"  # anonymous array literal
    SIZED = "sized"  # internal: matches any serializable, statically-sized argument
    UNION = "union"
    ANY = "any"  # a type containing all possible values


_PRIMITIVE_KINDS = frozenset(
    {
        TypeKind.U8,
        TypeKind.U16,
        TypeKind.U32,
        TypeKind.U64,
        TypeKind.I8,
        TypeKind.I16,
        TypeKind.I32,
        TypeKind.I64,
        TypeKind.F32,
        TypeKind.F64,
        TypeKind.BOOL,
    }
)
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
    # FIXME should any of the literal types be included?
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
_FLOAT_RANGES: dict[TypeKind, tuple[Decimal, Decimal]] = {
    TypeKind.F32: (Decimal(-3.4028234663852886e38), Decimal(3.4028234663852886e38)),
    TypeKind.F64: (Decimal(-1.7976931348623157e308), Decimal(1.7976931348623157e308)),
}
# Kind sets for fast membership tests
_SIGNED_INTEGER_KINDS = frozenset(
    {TypeKind.I8, TypeKind.I16, TypeKind.I32, TypeKind.I64}
)
_UNSIGNED_INTEGER_KINDS = frozenset(
    {TypeKind.U8, TypeKind.U16, TypeKind.U32, TypeKind.U64}
)
_ALL_INTEGER_KINDS = _SIGNED_INTEGER_KINDS | _UNSIGNED_INTEGER_KINDS
_ALL_FLOAT_KINDS = frozenset({TypeKind.F32, TypeKind.F64})
_ALL_NUMERICAL_KINDS = _ALL_INTEGER_KINDS | _ALL_FLOAT_KINDS
_SERIALIZABLE_KINDS = _PRIMITIVE_KINDS | frozenset(
    {
        TypeKind.STRING,
        TypeKind.ENUM,
        TypeKind.STRUCT,
        TypeKind.ARRAY,
        TypeKind.ANON_STRUCT,
        TypeKind.ANON_ARRAY,
        TypeKind.SIZED,
        TypeKind.UNIT,  # FIXME should unit be included?
    }
)


@lru_cache(maxsize=1)
def _scalar_llvm_types() -> dict[TypeKind, "ir.Type"]:
    """LLVM types for the scalar Fpy kinds.

    Built lazily (and cached) so that importing this module does not pull in
    llvmlite / the LLVM native library on the bytecode-only path.
    """
    from llvmlite import ir

    return {
        TypeKind.U8: ir.IntType(8),
        TypeKind.U16: ir.IntType(16),
        TypeKind.U32: ir.IntType(32),
        TypeKind.U64: ir.IntType(64),
        TypeKind.I8: ir.IntType(8),
        TypeKind.I16: ir.IntType(16),
        TypeKind.I32: ir.IntType(32),
        TypeKind.I64: ir.IntType(64),
        TypeKind.F32: ir.FloatType(),
        TypeKind.F64: ir.DoubleType(),
        TypeKind.BOOL: ir.IntType(1),
    }


@dataclass
class StructMember:
    name: str
    type: FpyType


class FpyType:

    __slots__ = (
        "kind",
        "name",
        "max_length",
        "enum_dict",
        "rep_type",
        "members",
        "elem_type",
        "length",
        "json_default",
        "member_defaults",
        "elem_defaults",
        "literal_val",
        "union_types",
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
        json_default: object | None = None,
        member_defaults: dict[str, FpyValue] | None = None,
        elem_defaults: tuple[FpyValue, ...] | None = None,
        literal_val: int | Decimal | str | bool | None = None,
        union_types: frozenset[FpyType] | None = None,
    ):
        self.kind = kind
        self.name = name
        self.max_length = max_length
        """max number of characters in a string type. inf for INTERNAL_STRING"""
        self.enum_dict = enum_dict
        self.rep_type = rep_type
        self.members = members
        self.elem_type = elem_type
        self.length = length
        """max number of elements in an array type"""
        self.json_default = json_default
        self.member_defaults = member_defaults
        self.elem_defaults = elem_defaults
        self.literal_val = literal_val
        """the sole value of this literal type"""
        self.union_types = union_types
        """the types which this union type contains"""

    # -- identity ----------------------------------------------------------

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, FpyType):
            return NotImplemented
        return all(
            getattr(self, field) == getattr(other, field) for field in FpyType.__slots__
        )

    def __hash__(self):
        return hash((self.kind, self.name))

    def __repr__(self):
        if self.kind == TypeKind.STRING:
            return f"FpyType(String[{self.max_length}])"
        return f"FpyType({self.name})"

    # -- classification properties -----------------------------------------

    @property
    def is_integer(self) -> bool:
        """True if all values of this type are integers"""
        if self.kind == TypeKind.UNION:
            return all(t.is_integer for t in self.union_types)
        return self.kind in _ALL_INTEGER_KINDS

    @property
    def is_float(self) -> bool:
        """True if all values of this type are floats"""
        if self.kind == TypeKind.UNION:
            return all(t.is_float for t in self.union_types)
        return self.kind in _ALL_FLOAT_KINDS

    @property
    def is_numerical(self) -> bool:
        """True if all values of this type are integers or floats"""
        if self.kind == TypeKind.UNION:
            return all(t.is_numerical for t in self.union_types)
        return self.kind in _ALL_NUMERICAL_KINDS

    @property
    def is_signed_integer(self) -> bool:
        """True if all values of this type are signed integers.
        Literals are neither signed nor unsigned"""
        if self.kind == TypeKind.UNION:
            return all(t.is_signed_integer for t in self.union_types)
        return self.kind in _SIGNED_INTEGER_KINDS

    @property
    def is_unsigned_integer(self) -> bool:
        """True if all values of this type are unsigned integers.
        Literals are neither signed nor unsigned"""
        assert self.kind != TypeKind.LITERAL_INT
        if self.kind == TypeKind.UNION:
            return all(t.is_unsigned_integer for t in self.union_types)
        return self.kind in _UNSIGNED_INTEGER_KINDS

    @property
    def is_primitive(self) -> bool:
        """True for U8..F64 and BOOL."""
        if self.kind == TypeKind.UNION:
            return all(t.is_primitive for t in self.union_types)
        return self.kind in _PRIMITIVE_FORMATS

    @property
    def is_string(self) -> bool:
        """True if all values of this type are strings"""
        if self.kind == TypeKind.UNION:
            return all(t.is_string for t in self.union_types)
        return self.kind in (TypeKind.STRING, TypeKind.LITERAL_STRING)

    @property
    def is_array(self) -> bool:
        """True if all values of this type are arrays"""
        if self.kind == TypeKind.UNION:
            return all(t.is_array for t in self.union_types)
        return self.kind == TypeKind.ARRAY or self.kind == TypeKind.ANON_ARRAY

    @property
    def is_struct(self) -> bool:
        """True if all values of this type are structs"""
        if self.kind == TypeKind.UNION:
            return all(t.is_struct for t in self.union_types)
        return self.kind == TypeKind.STRUCT or self.kind == TypeKind.ANON_STRUCT

    @property
    def is_literal(self) -> bool:
        """True if all values of this type are literals"""
        if self.kind == TypeKind.UNION:
            return all(t.is_literal for t in self.union_types)
        return self.kind in _ALL_LITERAL_KINDS

    @property
    def display_name(self) -> str:
        """Human-readable type name for error messages."""
        if self.kind == TypeKind.LITERAL_INT:
            return "int literal"
        if self.kind == TypeKind.LITERAL_FLOAT:
            return "float literal"
        if self.kind == TypeKind.LITERAL_STRING:
            return "string literal"
        if self.kind == TypeKind.ANON_STRUCT:
            return "struct literal"
        if self.kind == TypeKind.ANON_ARRAY:
            return "array literal"
        if self.kind == TypeKind.SIZED:
            return "serializable, statically-sized value"
        if self.kind == TypeKind.UNION:
            return " | ".join(sorted(t.name for t in self.union_types))
        return self.name

    # -- size / range properties -------------------------------------------

    @property
    def is_serializable(self) -> bool:
        """True if all values of this type are serializable"""
        if self.kind == TypeKind.UNION:
            return all(t.is_serializable for t in self.union_types)
        return self.kind in _SERIALIZABLE_KINDS

    @property
    def has_static_serialized_size(self) -> bool:
        """True if all values of this serializable type have the same serialized size"""
        assert self.is_serializable
        if self.kind == TypeKind.UNION:
            if not all(t.has_static_serialized_size for t in self.union_types):
                # at least one member type does not have a static serialized size
                return False
            member_sizes = set(t.actual_size for t in self.union_types)
            # if the members take up diff sizes, the union doesn't have a static serialized size
            return len(member_sizes) == 1

        # all other serializable types except non-literal strings have a static serialized size
        return self.kind != TypeKind.STRING

    @property
    def max_size(self) -> int:
        """Maximum serialized size in bytes. Raises an error if the type is not serializable"""
        assert self.is_serializable, self

        if self.kind == TypeKind.UNION:
            # the maximum size of a union is the max of the max sizes of its member types
            return max(t.max_size for t in self.union_types)

        if self.kind == TypeKind.STRING:
            return FwSizeStoreType.max_size + self.max_length
        if self.is_struct:
            return sum(m.type.max_size for m in self.members)
        if self.is_array:
            return self.elem_type.max_size * self.length

        # all other serializable types have a known size
        return self.actual_size

    @property
    def actual_size(self) -> int:
        """Actual serialized size in bytes. Raises an error if the type is not serializable, or not all values have the same serialized size"""
        assert self.has_static_serialized_size, self

        if self.kind == TypeKind.UNION:
            # the actual size of a union of statically sized types is the size of any one of those statically sized types
            return next(t.actual_size for t in self.union_types)

        if self.is_primitive:
            return _PRIMITIVE_SIZES[self.kind]
        if self.kind == TypeKind.ENUM:
            return self.rep_type.actual_size
        if self.is_struct:
            return sum(m.type.actual_size for m in self.members)
        if self.is_array:
            return self.elem_type.actual_size * self.length
        if self.kind == TypeKind.UNIT:
            return 0
        assert False, f"Cannot compute max_size for {self}"

    @property
    def bits(self) -> int | float:
        """Bit width of the primitive type"""
        if self.kind in _PRIMITIVE_BITS:
            return _PRIMITIVE_BITS[self.kind]
        assert False, f"Cannot compute bits for {self}"

    @property
    def llvm_type(self) -> "ir.Type":
        """The LLVM IR type used to represent this type in the wasm backend."""
        from llvmlite import ir

        scalars = _scalar_llvm_types()
        if self.kind in scalars:
            return scalars[self.kind]
        if self.kind == TypeKind.ENUM:
            # An enum is represented by its underlying integer type.
            return self.rep_type.llvm_type
        if self.is_struct:
            return ir.LiteralStructType([m.type.llvm_type for m in self.members])
        if self.is_array:
            return ir.ArrayType(self.elem_type.llvm_type, self.length)
        if self.is_string:
            # Fprime string: 2-byte length prefix + fixed-capacity byte buffer.
            assert self.max_length is not None, "string type needs a max_length"
            return ir.LiteralStructType(
                [ir.IntType(16), ir.ArrayType(ir.IntType(8), self.max_length)]
            )
        if self.kind == TypeKind.UNIT:
            return ir.VoidType()
        raise NotImplementedError(f"No LLVM type mapping for {self.display_name}")

    @property
    def value_range(self) -> tuple[int | Decimal, int | Decimal]:
        """(min, max) inclusive range for non-literal numeric types."""
        if self.kind in _INTEGER_RANGES:
            return _INTEGER_RANGES[self.kind]
        if self.kind in _FLOAT_RANGES:
            return _FLOAT_RANGES[self.kind]
        if self.kind == TypeKind.LITERAL_FLOAT or self.kind == TypeKind.LITERAL_INT:
            return (self.literal_val, self.literal_val)

        assert False, f"Cannot compute range for {self}"

    def is_integer_exact_in_type(self, value: int) -> bool:
        """return True if the integer value is representable exactly in this non-literal float type"""
        assert self.is_float and not self.is_literal, self.kind
        if value == 0:
            return True
        value = abs(value)
        value >>= (value & -value).bit_length() - 1
        mantissa_bits = 53 if self.kind == TypeKind.F64 else 24
        return value.bit_length() <= mantissa_bits

    def is_numeric_value_in_type(self, value: Decimal | int) -> bool:
        assert self.is_numerical
        assert not self.is_literal, self

        # inf and nan lie outside every value range, but every float type can
        # represent them. this must come before the range check, because
        # ordering a nan against anything raises InvalidOperation
        if isinstance(value, Decimal) and not value.is_finite():
            return self.is_float

        min_val, max_val = self.value_range
        # if value is out of range of this type
        if value < min_val or value > max_val:
            return False

        # does value have a fractional part?
        if isinstance(value, Decimal) and value.to_integral_value() != value:
            # must be a float type
            return self.is_float

        # otherwise, the value is either an int, or a Decimal which is an exact int.
        if self.is_float:
            return self.is_integer_exact_in_type(int(value))

        return True

    def is_subtype_of(self, super_type: FpyType) -> bool:
        """True if all values of self are values of super_type, with some minor exceptions. See comments"""
        if self == super_type:
            return True

        if self.kind == TypeKind.UNION:
            # all of these member types must be subtypes of the super type for
            # the whole union type to be a subtype of the super type
            return all(t.is_subtype_of(super_type) for t in self.union_types)

        if self.kind == TypeKind.ANY:
            # any type is only a subtype of itself
            return False

        if super_type.kind == TypeKind.ANY:
            # any type is a supertype of all types
            return True

        if super_type.kind == TypeKind.UNION:
            # if we are a subtype of any one of the member types of the super union type,
            # then we are a subtype of the whole super union type

            # note this misses the case in which we are a subtype of various combinations
            # of the member types of the super union type. e.g. bool would not be considered
            # a subtype of Literal[True] | Literal[False] even though it absolutely is.
            # but in order to calculate that case, we'd have to work with the sets of values
            # in each type. it gets complicated and we don't really need to worry about this
            # as it's a relatively rare case
            return any(self.is_subtype_of(t) for t in super_type.union_types)

        if super_type.kind == TypeKind.SIZED:
            # sized is a super type of any type which is serializable, and has a statically-known
            # serialized size
            return self.is_serializable and self.has_static_serialized_size

        if super_type.is_literal:
            # super type is a literal, so it has one value
            # the only way this could be a subtype of it is if
            # it is equal to it, which was already checked
            return False

        if super_type.kind == TypeKind.ENUM:
            # only way an enum type can be a subtype of another enum type
            # is if they are the same type. i.e. no enum constant is shared
            # between two enum types
            return False

        if super_type.kind == TypeKind.RANGE:
            return False

        if super_type.kind == TypeKind.UNIT:
            return False

        if super_type.kind == TypeKind.BOOL:
            # only the bool type is a subtype of the bool type
            return self.kind == TypeKind.BOOL

        if super_type.is_float:
            if self.is_numerical and self.is_literal:
                # a literal numeric type is a subtype of any type containing the literal type's value
                return super_type.is_numeric_value_in_type(self.literal_val)

            if self.is_float:
                return self.bits <= super_type.bits

            if self.is_integer:
                # okay, this one is the integer, super_type is a float
                if self.bits >= 64:
                    # if I64, U64, cannot fit in any float
                    return False

                if self.bits == 32:
                    # a 32 bit u/i int can only fit into an f64
                    return super_type.kind == TypeKind.F64

                return True

            return False

        if super_type.is_integer:
            if self.is_numerical and self.is_literal:
                # a literal numeric type is a subtype of any type containing the literal type's value
                return super_type.is_numeric_value_in_type(self.literal_val)

            if self.is_float:
                # no non-literal float type is a subtype of an integer type
                return False

            if self.is_integer:
                # an int type B is a subtype of another int type A if all B's values are contained within
                # the value range of A
                this_min, this_max = self.value_range
                super_min, super_max = super_type.value_range

                return this_max <= super_max and this_min >= super_min

            return False

        if super_type.is_array:
            if self.kind == TypeKind.ANON_ARRAY:
                # an anonymous array may only be a subtype of an array (anonymous or not)
                # with the same length
                if self.length != super_type.length:
                    return False

                # the element type of this array type must be a subtype of the super type's element type
                return self.elem_type.is_subtype_of(super_type.elem_type)

            return False

        if super_type.is_struct:
            if self.kind == TypeKind.ANON_STRUCT:
                # if there's a member in the super type which isn't present in this
                # type, this type cannot be a subtype
                for super_member in super_type.members:
                    sub_member = next(
                        (m for m in self.members if m.name == super_member.name), None
                    )
                    if sub_member is None:
                        return False
                    # the member of this type is not a subtype of the member of the potential super type
                    if not sub_member.type.is_subtype_of(super_member.type):
                        return False
                return True

            return False

        if super_type.is_string:
            if self.is_string:
                # this handles string literals too
                return self.max_length <= super_type.max_length
            return False

        assert False, super_type.kind


def union_of(*types: FpyType, name: str | None = None) -> FpyType:
    """The type whose values are the values of every type in *types*.

    Not itself a union type when one of the given types already contains all
    the others."""
    members: list[FpyType] = []
    # sorted so the result does not depend on the iteration order of a frozenset
    for member in sorted(_flatten_unions(types), key=lambda member: member.name):
        if any(member.is_subtype_of(kept) for kept in members):
            # a type we already kept contains every value of this one
            continue
        # this type contains every value of some of the types we kept
        members = [kept for kept in members if not kept.is_subtype_of(member)]
        members.append(member)

    if len(members) == 1:
        return members[0]
    if len(members) == 0:
        return NEVER

    if name is None:
        name = " | ".join(member.name for member in members)
    return FpyType(TypeKind.UNION, name, union_types=frozenset(members))


def _flatten_unions(types: Iterable[FpyType]) -> list[FpyType]:
    """Each of *types*, with the member types of any union type in its place."""
    flat: list[FpyType] = []
    for type in types:
        if type.kind == TypeKind.UNION:
            flat.extend(_flatten_unions(type.union_types))
        else:
            flat.append(type)
    return flat


NEVER = FpyType(TypeKind.UNION, "Never", union_types=frozenset())

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
ANY = FpyType(TypeKind.ANY, "Any")

# distinct singleton so that the in-place update
# is visible everywhere the object is referenced.
FwSizeStoreType = FpyType(TypeKind.U16, "U16")

# the problem is, that the integer type sounds like it should contain all integers
# but it doesn't, it only contains those which can be represented by a u64 or i64
INTEGER = union_of(U64, I64, name="Integer")
NUMBER = union_of(INTEGER, F64, name="Number")

# The canonical TimeBase enum type — default placeholder.
# The full set of enum constants and representation type are loaded from the
# dictionary at compile time.  Only TB_NONE is required to exist.
TIME_BASE = FpyType(
    TypeKind.ENUM,
    "TimeBase",
    enum_dict={"TB_NONE": 0},
    rep_type=U16,
)

LOG_SEVERITY = FpyType(
    TypeKind.ENUM,
    "Fw.LogSeverity",
    enum_dict={
        "FATAL": 1,
        "WARNING_HI": 2,
        "WARNING_LO": 3,
        "COMMAND": 4,
        "ACTIVITY_HI": 5,
        "ACTIVITY_LO": 6,
        "DIAGNOSTIC": 7,
    },
    rep_type=U8,
)

TIME = FpyType(
    TypeKind.STRUCT,
    "Fw.TimeValue",
    members=(
        StructMember("timeBase", TIME_BASE),
        StructMember("timeContext", U8),
        StructMember("seconds", U32),
        StructMember("useconds", U32),
    ),
)
RANGE = FpyType(TypeKind.RANGE, "Range")
UNIT = FpyType(TypeKind.UNIT, "Unit")
SIZED = FpyType(TypeKind.SIZED, "Sized")

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
    # TODO do we need fwsizestoretype in here?
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

    # -- lowering ----------------------------------------------------------

    @property
    def llvm_value(self) -> "ir.Constant":
        """The LLVM constant representing this value. Raises an error if
        not representable"""
        from llvmlite import ir

        kind = self.type.kind
        # Internal/abstract types have no LLVM representation.
        assert kind not in (
            TypeKind.INTEGER,
            TypeKind.INTERNAL_STRING,
        ), self

        llvm_type = self.type.llvm_type
        if self.type.is_float:
            # float types store a Decimal; float() gives the double/float value.
            return ir.Constant(llvm_type, float(self.val))
        if self.type.is_integer or kind == TypeKind.BOOL:
            # ints store a Python int; BOOL stores a bool (int(True) == 1).
            return ir.Constant(llvm_type, int(self.val))
        if kind == TypeKind.ENUM:
            # an enum const stores its member name; map it to the integer rep.
            return ir.Constant(llvm_type, self.type.enum_dict[self.val])
        if self.type.is_struct:
            return ir.Constant(
                llvm_type, [self.val[m.name].llvm_value for m in self.type.members]
            )
        if self.type.is_array:
            return ir.Constant(llvm_type, [elem.llvm_value for elem in self.val])
        # FIXME how do i do a unit type here? can i have an "i0"?
        raise NotImplementedError(
            f"No LLVM constant for a value of type {self.type.display_name}"
        )

    # -- serialization -----------------------------------------------------

    def serialize(self) -> bytes:
        """Serialize this value to bytes (big-endian, FPP wire format)."""
        kind = self.type.kind

        if kind in _PRIMITIVE_FORMATS:
            val = self.val
            if kind == TypeKind.BOOL:
                val = FW_SERIALIZE_TRUE_VALUE if val else FW_SERIALIZE_FALSE_VALUE
            return struct.pack(_PRIMITIVE_FORMATS[kind], val)

        if self.type.is_string:
            encoded = (
                self.val.encode("utf-8") if isinstance(self.val, str) else self.val
            )
            if self.type.max_length is not None:
                if len(encoded) > self.type.max_length:
                    raise ValueError(
                        f"String too long: {len(encoded)} > {self.type.max_length}"
                    )
            return FpyValue(FwSizeStoreType, len(encoded)).serialize() + encoded

        if kind == TypeKind.ENUM:
            val = self.val
            if isinstance(val, str):
                assert val in self.type.enum_dict, f"Unknown enum constant: {val}"
                val = self.type.enum_dict[val]
            return FpyValue(self.type.rep_type, val).serialize()

        if self.type.is_struct:
            output = b""
            for m in self.type.members:
                member_val = self.val[m.name]
                if not isinstance(member_val, FpyValue):
                    member_val = FpyValue(m.type, member_val)
                output += member_val.serialize()
            return output

        if self.type.is_array:
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
        Returns ``(value, new_offset)``. Raises DeserializeError on bytes the
        fprime C++ deserializer would reject"""
        kind = typ.kind

        if kind in _PRIMITIVE_FORMATS:
            fmt = _PRIMITIVE_FORMATS[kind]
            size = _PRIMITIVE_SIZES[kind]
            if offset + size > len(data):
                raise DeserializeError(
                    f"Buffer too short for {typ.display_name}: need {size} bytes "
                    f"at offset {offset}, have {len(data) - offset}"
                )
            raw = struct.unpack_from(fmt, data, offset)[0]
            if kind == TypeKind.BOOL:
                if raw == FW_SERIALIZE_TRUE_VALUE:
                    raw = True
                elif raw == FW_SERIALIZE_FALSE_VALUE:
                    raw = False
                else:
                    raise DeserializeError(f"Invalid bool byte 0x{raw:02x}")
            return FpyValue(typ, raw), offset + size

        if typ.is_string:
            size_val, offset = FpyValue.deserialize(FwSizeStoreType, data, offset)
            str_len = size_val.val
            if typ.max_length is not None and str_len > typ.max_length:
                raise DeserializeError(
                    f"String length {str_len} exceeds max length "
                    f"{typ.max_length} of {typ.display_name}"
                )
            if offset + str_len > len(data):
                raise DeserializeError(
                    f"Buffer too short for {typ.display_name}: need {str_len} "
                    f"bytes at offset {offset}, have {len(data) - offset}"
                )
            s = data[offset : offset + str_len].decode("utf-8")
            offset += str_len
            return FpyValue(typ, s), offset

        if kind == TypeKind.ENUM:
            rep_val, new_offset = FpyValue.deserialize(typ.rep_type, data, offset)
            for name, val in typ.enum_dict.items():
                if val == rep_val.val:
                    return FpyValue(typ, name), new_offset
            return FpyValue(typ, rep_val.val), new_offset

        if typ.is_struct:
            members_dict: dict[str, FpyValue] = {}
            for m in typ.members:
                member_val, offset = FpyValue.deserialize(m.type, data, offset)
                members_dict[m.name] = member_val
            return FpyValue(typ, members_dict), offset

        if typ.is_array:
            elements: list[FpyValue] = []
            for _ in range(typ.length):
                elem, offset = FpyValue.deserialize(typ.elem_type, data, offset)
                elements.append(elem)
            return FpyValue(typ, elements), offset

        # TODO deser/ser unit type?

        assert False, f"Cannot deserialize {typ}"


# The sole value of UNIT
UNIT_VALUE = FpyValue(UNIT, None)


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


# The built-in flags struct that controls sequencer behavior.
# Allocated as a magic global variable at the start of the stack.
FLAGS_TYPE = FpyType(
    TypeKind.STRUCT,
    "$Flags",
    members=(StructMember("assert_cmd_success", BOOL),),
    member_defaults={"assert_cmd_success": FpyValue(BOOL, True)},
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

# The canonical Fw.TlmValid enum type: the validity a telemetry-channel read
# reports.
TLM_VALID = FpyType(
    TypeKind.ENUM,
    "Fw.TlmValid",
    enum_dict={
        "VALID": 0,
        "INVALID": 1,
    },
    rep_type=U8,
)

# The canonical Fw.ParamValid enum type: the validity a parameter read
# reports.
PARAM_VALID = FpyType(
    TypeKind.ENUM,
    "Fw.ParamValid",
    enum_dict={
        "UNINIT": 0,
        "VALID": 1,
        "INVALID": 2,
        "DEFAULT": 3,
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

# The canonical Svc.BlockState enum type. Both the seq dispatcher's RUN_ARGS and
# the fpy sequencer's RUN command take their blocking arg as this type, so the
# compiler can match sequence-run commands by this exact type.
BLOCK_STATE = FpyType(
    TypeKind.ENUM,
    "Svc.BlockState",
    enum_dict={"BLOCK": 0, "NO_BLOCK": 1},
    rep_type=U8,
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

# Placeholder buffer size for Svc.SeqArgs; replaced from the dictionary at
# compile time (see _update_seq_args_from_dict in compiler.py).
DEFAULT_SEQ_ARGS_BUFFER_SIZE = 255

# The canonical Svc.SeqArgs struct type used for passing arguments to subsequences.
# The buffer's length and name are updated from the dictionary at compile time,
# and member_defaults is populated by _populate_type_defaults after the load.
# FPP struct: { $size: FwSizeType, buffer: [N] U8 }
_SEQ_ARGS_BUFFER_TYPE = FpyType(
    TypeKind.ARRAY,
    "Array_U8_255",
    elem_type=U8,
    length=DEFAULT_SEQ_ARGS_BUFFER_SIZE,
)
SEQ_ARGS = FpyType(
    TypeKind.STRUCT,
    "Svc.SeqArgs",
    members=(
        StructMember("size", U64),
        StructMember("buffer", _SEQ_ARGS_BUFFER_TYPE),
    ),
)


_TIME_INTERVAL_DEFAULT = {"seconds": 0, "useconds": 0}
_TIME_DEFAULT = {
    "timeBase": "TimeBase.TB_NONE",
    "timeContext": 0,
    "seconds": 0,
    "useconds": 0,
}

# Internal type not directly accessible to users,
# used for desugaring check statements.
CHECK_STATE = FpyType(
    TypeKind.STRUCT,
    "$CheckState",
    members=(
        StructMember("persist", TIME_INTERVAL),
        StructMember("timeout", TIME),
        StructMember("period", TIME_INTERVAL),
        StructMember("result", BOOL),
        StructMember("last_was_true", BOOL),
        StructMember("last_time_true", TIME),
        StructMember("time_started", TIME),
    ),
    json_default={
        "persist": _TIME_INTERVAL_DEFAULT,
        "timeout": _TIME_DEFAULT,
        "period": _TIME_INTERVAL_DEFAULT,
        "result": False,
        "last_was_true": False,
        "last_time_true": _TIME_DEFAULT,
        "time_started": _TIME_DEFAULT,
    },
)


# FIXME let's remove isinstancecompat
def is_instance_compat(obj, cls):
    """
    A wrapper for isinstance() that correctly handles Union types in Python 3.9+.
    """
    origin = get_origin(cls)
    if origin in UNION_TYPES:
        return isinstance(obj, get_args(cls))
    return isinstance(obj, cls)


class OpCase(Enum):
    """How an operator expression is evaluated: the operator specialized to the
    category of its intermediate type. Suffixes: INT is any integer, SINT/UINT
    a signed/unsigned integer, FLOAT a float, BYTES a non-numeric value
    compared by its serialized bytes."""

    NOT = auto()
    AND = auto()
    OR = auto()
    IDENTITY = auto()
    NEGATE_INT = auto()
    NEGATE_FLOAT = auto()
    ADD_INT = auto()
    ADD_FLOAT = auto()
    SUBTRACT_INT = auto()
    SUBTRACT_FLOAT = auto()
    MULTIPLY_INT = auto()
    MULTIPLY_FLOAT = auto()
    DIVIDE_FLOAT = auto()
    EXPONENT_FLOAT = auto()
    MODULUS_SINT = auto()
    MODULUS_UINT = auto()
    MODULUS_FLOAT = auto()
    FLOOR_DIVIDE_SINT = auto()
    FLOOR_DIVIDE_UINT = auto()
    FLOOR_DIVIDE_FLOAT = auto()
    LESS_THAN_SINT = auto()
    LESS_THAN_UINT = auto()
    LESS_THAN_FLOAT = auto()
    GREATER_THAN_SINT = auto()
    GREATER_THAN_UINT = auto()
    GREATER_THAN_FLOAT = auto()
    LESS_THAN_OR_EQUAL_SINT = auto()
    LESS_THAN_OR_EQUAL_UINT = auto()
    LESS_THAN_OR_EQUAL_FLOAT = auto()
    GREATER_THAN_OR_EQUAL_SINT = auto()
    GREATER_THAN_OR_EQUAL_UINT = auto()
    GREATER_THAN_OR_EQUAL_FLOAT = auto()
    EQUAL_INT = auto()
    EQUAL_FLOAT = auto()
    EQUAL_BYTES = auto()
    NOT_EQUAL_INT = auto()
    NOT_EQUAL_FLOAT = auto()
    NOT_EQUAL_BYTES = auto()


@dataclass
class OperatorFunc:
    arg_types: set[FpyType]


NOT = OperatorFunc([BOOL])
IDENTITY = OperatorFunc([ANY])
NEGATE_INT = OperatorFunc([INTEGER])

# challenge: what is the type of 1 + 1?
# is it IntLiteral[2]? well, it's not a literal... so how could it be?
# well, here's how i'm going to think about it. The IntLiteral[X] type, where X is a metavariable of an integer, represents the type whose sole value is [X]


# op -> its case over a (signed int, unsigned int, float) intermediate type.
# Unary and binary ops are tabled apart because `+` and `-` spell one of each,
# and as str enums those members compare (and hash) equal.
_NumericOpCases = dict[str, tuple[OpCase | None, OpCase | None, OpCase | None]]
_UNARY_NUMERIC_OP_CASES: _NumericOpCases = {
    UnaryStackOp.IDENTITY: (OpCase.IDENTITY, OpCase.IDENTITY, OpCase.IDENTITY),
    UnaryStackOp.NEGATE: (OpCase.NEGATE_INT, OpCase.NEGATE_INT, OpCase.NEGATE_FLOAT),
}
_BINARY_NUMERIC_OP_CASES: _NumericOpCases = {
    BinaryStackOp.ADD: (OpCase.ADD_INT, OpCase.ADD_INT, OpCase.ADD_FLOAT),
    BinaryStackOp.SUBTRACT: (
        OpCase.SUBTRACT_INT,
        OpCase.SUBTRACT_INT,
        OpCase.SUBTRACT_FLOAT,
    ),
    BinaryStackOp.MULTIPLY: (
        OpCase.MULTIPLY_INT,
        OpCase.MULTIPLY_INT,
        OpCase.MULTIPLY_FLOAT,
    ),
    BinaryStackOp.DIVIDE: (None, None, OpCase.DIVIDE_FLOAT),
    BinaryStackOp.EXPONENT: (None, None, OpCase.EXPONENT_FLOAT),
    BinaryStackOp.MODULUS: (
        OpCase.MODULUS_SINT,
        OpCase.MODULUS_UINT,
        OpCase.MODULUS_FLOAT,
    ),
    BinaryStackOp.FLOOR_DIVIDE: (
        OpCase.FLOOR_DIVIDE_SINT,
        OpCase.FLOOR_DIVIDE_UINT,
        OpCase.FLOOR_DIVIDE_FLOAT,
    ),
    BinaryStackOp.LESS_THAN: (
        OpCase.LESS_THAN_SINT,
        OpCase.LESS_THAN_UINT,
        OpCase.LESS_THAN_FLOAT,
    ),
    BinaryStackOp.GREATER_THAN: (
        OpCase.GREATER_THAN_SINT,
        OpCase.GREATER_THAN_UINT,
        OpCase.GREATER_THAN_FLOAT,
    ),
    BinaryStackOp.LESS_THAN_OR_EQUAL: (
        OpCase.LESS_THAN_OR_EQUAL_SINT,
        OpCase.LESS_THAN_OR_EQUAL_UINT,
        OpCase.LESS_THAN_OR_EQUAL_FLOAT,
    ),
    BinaryStackOp.GREATER_THAN_OR_EQUAL: (
        OpCase.GREATER_THAN_OR_EQUAL_SINT,
        OpCase.GREATER_THAN_OR_EQUAL_UINT,
        OpCase.GREATER_THAN_OR_EQUAL_FLOAT,
    ),
    BinaryStackOp.EQUAL: (OpCase.EQUAL_INT, OpCase.EQUAL_INT, OpCase.EQUAL_FLOAT),
    BinaryStackOp.NOT_EQUAL: (
        OpCase.NOT_EQUAL_INT,
        OpCase.NOT_EQUAL_INT,
        OpCase.NOT_EQUAL_FLOAT,
    ),
}
_BOOLEAN_OP_CASES = {
    UnaryStackOp.NOT: OpCase.NOT,
    BinaryStackOp.AND: OpCase.AND,
    BinaryStackOp.OR: OpCase.OR,
}
_BYTES_OP_CASES = {
    BinaryStackOp.EQUAL: OpCase.EQUAL_BYTES,
    BinaryStackOp.NOT_EQUAL: OpCase.NOT_EQUAL_BYTES,
}


def pick_unary_op_case(op: UnaryStackOp, intermediate_type: FpyType) -> OpCase:
    """The case evaluating unary *op* over an operand coerced to
    *intermediate_type*."""
    return _pick_op_case(_UNARY_NUMERIC_OP_CASES, op, intermediate_type)


def pick_binary_op_case(op: BinaryStackOp, intermediate_type: FpyType) -> OpCase:
    """The case evaluating binary *op* over operands coerced to
    *intermediate_type*."""
    return _pick_op_case(_BINARY_NUMERIC_OP_CASES, op, intermediate_type)


def _pick_op_case(
    numeric_op_cases: _NumericOpCases, op: str, intermediate_type: FpyType
) -> OpCase:
    if op in BOOLEAN_OPERATORS:
        assert intermediate_type == BOOL, intermediate_type
        return _BOOLEAN_OP_CASES[op]
    if not intermediate_type.is_numerical:
        return _BYTES_OP_CASES[op]
    signed_case, unsigned_case, float_case = numeric_op_cases[op]
    if intermediate_type.is_float:
        case = float_case
    elif intermediate_type.is_unsigned_integer:
        case = unsigned_case
    else:
        case = signed_case
    assert case is not None, (op, intermediate_type)
    return case


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
