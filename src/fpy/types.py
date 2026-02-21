from __future__ import annotations
from abc import ABC
from decimal import Decimal
from dataclasses import dataclass
import math
from typing import Union, get_args, get_origin

from fpy.bytecode.directives import (
    BinaryStackOp,
    COMPARISON_OPS,
)
from fprime_gds.common.models.serialize.serializable_type import (
    SerializableType as StructValue,
)
from fprime_gds.common.models.serialize.numerical_types import (
    U8Type as U8Value,
    U16Type as U16Value,
    U32Type as U32Value,
    U64Type as U64Value,
    I8Type as I8Value,
    I16Type as I16Value,
    I32Type as I32Value,
    I64Type as I64Value,
    F32Type as F32Value,
    F64Type as F64Value,
    IntegerType as IntegerValue,
    FloatType as FloatValue,
    NumericalType as NumericalValue,
)
from fprime_gds.common.models.serialize.string_type import StringType as StringValue
from fprime_gds.common.models.serialize.time_type import TimeType as TimeValue
from fprime_gds.common.models.serialize.bool_type import BoolType as BoolValue
from fprime_gds.common.models.serialize.type_base import BaseType as FppValue
from fprime_gds.common.models.serialize.enum_type import EnumType as EnumValue

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


def typename(typ: FppType) -> str:
    if typ == FpyIntegerValue:
        return "Integer"
    if typ == FpyFloatValue:
        return "Float"
    if issubclass(typ, NumericalValue):
        return typ.get_canonical_name()
    if typ == FpyStringValue:
        return "String"
    if typ == RangeValue:
        return "Range"
    return typ.__name__


# this is the "internal" integer type that integer literals have by
# default. it is arbitrary precision. it is also only used in places where
# we know the value is constant
class FpyIntegerValue(IntegerValue):
    @classmethod
    def range(cls):
        return (-math.inf, math.inf)

    @staticmethod
    def get_serialize_format():
        raise NotImplementedError()

    @classmethod
    def get_bits(cls):
        return math.inf

    @classmethod
    def validate(cls, val):
        if not isinstance(val, int):
            raise RuntimeError()


# this is the "internal" float type that float literals have by
# default. it is arbitrary precision. it is also only used in places where
# we know the value is constant
class FpyFloatValue(FloatValue):
    @staticmethod
    def get_serialize_format():
        raise NotImplementedError()

    @classmethod
    def get_bits(cls):
        return math.inf

    @classmethod
    def validate(cls, val):
        if not isinstance(val, Decimal):
            raise RuntimeError()


class RangeValue(FppValue):
    """the type produced by range expressions `X .. Y`"""

    def serialize(self):
        raise NotImplementedError()

    def deserialize(self, data, offset):
        raise NotImplementedError()

    def getSize(self):
        raise NotImplementedError()

    @classmethod
    def getMaxSize(cls):
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__

    def to_jsonable(self):
        raise NotImplementedError()


# this is the "internal" string type that string literals have by
# default. it is arbitrary length. it is also only used in places where
# we know the value is constant
FpyStringValue = StringValue.construct_type("FpyStringValue", None)

SPECIFIC_NUMERIC_TYPES = (
    U32Value,
    U16Value,
    U64Value,
    U8Value,
    I16Value,
    I32Value,
    I64Value,
    I8Value,
    F32Value,
    F64Value,
)
SPECIFIC_INTEGER_TYPES = (
    U32Value,
    U16Value,
    U64Value,
    U8Value,
    I16Value,
    I32Value,
    I64Value,
    I8Value,
)
SIGNED_INTEGER_TYPES = (
    I16Value,
    I32Value,
    I64Value,
    I8Value,
)
UNSIGNED_INTEGER_TYPES = (
    U32Value,
    U16Value,
    U64Value,
    U8Value,
)
SPECIFIC_FLOAT_TYPES = (
    F32Value,
    F64Value,
)
ARBITRARY_PRECISION_TYPES = (FpyFloatValue, FpyIntegerValue)

# The canonical Svc.Fpy.FlagId enum type
# This must match the dictionary's Svc.Fpy.FlagId definition
FlagIdValue = EnumValue.construct_type(
    "Svc.Fpy.FlagId",
    {"EXIT_ON_CMD_FAIL": 0},
    "U8",
)

# The canonical Fw.TimeIntervalValue struct type
# This must match the dictionary's Fw.TimeIntervalValue definition
# The format string '{}' and empty description match what the GDS JSON loader produces
TimeIntervalValue = StructValue.construct_type(
    "Fw.TimeIntervalValue",
    [
        ("seconds", U32Value, "{}", ""),
        ("useconds", U32Value, "{}", ""),
    ],
)

# Time operator overloads: maps (lhs_type, rhs_type, op) -> (intermediate_type, result_type, func_name, is_comparison)
TIME_OPS: dict[tuple[type, type, BinaryStackOp], tuple[type, type, str, bool]] = {
    # Time - Time -> TimeInterval
    (TimeValue, TimeValue, BinaryStackOp.SUBTRACT): (TimeValue, TimeIntervalValue, "time_sub", False),
    # Time + TimeInterval -> Time  
    (TimeValue, TimeIntervalValue, BinaryStackOp.ADD): (TimeValue, TimeValue, "time_add", False),
    # TimeInterval +/- TimeInterval -> TimeInterval
    (TimeIntervalValue, TimeIntervalValue, BinaryStackOp.ADD): (TimeIntervalValue, TimeIntervalValue, "time_interval_add", False),
    (TimeIntervalValue, TimeIntervalValue, BinaryStackOp.SUBTRACT): (TimeIntervalValue, TimeIntervalValue, "time_interval_sub", False),
    # Time comparisons -> Bool
    **{(TimeValue, TimeValue, op): (TimeValue, BoolValue, "time_cmp_assert_comparable", True) for op in COMPARISON_OPS},
    # TimeInterval comparisons -> Bool
    **{(TimeIntervalValue, TimeIntervalValue, op): (TimeIntervalValue, BoolValue, "time_interval_cmp", True) for op in COMPARISON_OPS},
}


def is_instance_compat(obj, cls):
    """
    A wrapper for isinstance() that correctly handles Union types in Python 3.9+.

    Args:
        obj: The object to check.
        cls: The class, tuple of classes, or Union type to check against.

    Returns:
        True if the object is an instance of the class or any type in the Union.
    """
    origin = get_origin(cls)
    if origin in UNION_TYPES:
        # It's a Union type, so get its arguments.
        # e.g., get_args(Union[int, str]) returns (int, str)
        return isinstance(obj, get_args(cls))

    # It's not a Union, so it's a regular type (like int) or a
    # tuple of types ((int, str)), which isinstance handles natively.
    return isinstance(obj, cls)


# a value of type FppType is a Python `type` object representing
# the type of an Fprime value
FppType = type[FppValue]


class NothingValue(ABC):
    """a type which has no valid values in fprime. used to denote
    a function which doesn't return a value"""

    @classmethod
    def __subclasscheck__(cls, subclass):
        return False


# the `type` object representing the NothingType class
NothingType = type[NothingValue]
