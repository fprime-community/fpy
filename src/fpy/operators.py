from dataclasses import dataclass
from enum import Enum, auto

from fpy.syntax import BinaryOp, UnaryOp
from fpy.types import (
    ANY,
    BOOL,
    F64,
    FLOAT_TYPES,
    I64,
    NUMERIC_TYPES,
    SIGNED_INTEGER_TYPES,
    U64,
    FpyType,
    _ALL_NUMERICAL_KINDS,
)


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


@dataclass
class OperatorFunc:
    name: str
    arg_types: list[FpyType]
    return_type: FpyType


def unary_op_with_types_from(name, types: frozenset[FpyType]) -> list[OperatorFunc]:
    return [OperatorFunc(f"{name}_{typ.name}_{typ.name}", [typ], typ) for typ in types]


UNARY_OPS: dict[UnaryOp, list[OperatorFunc]] = {
    UnaryOp.IDENTITY: unary_op_with_types_from("identity", NUMERIC_TYPES),
    UnaryOp.NOT: unary_op_with_types_from("not", frozenset({BOOL})),
    UnaryOp.NEGATE: unary_op_with_types_from(
        "negate", SIGNED_INTEGER_TYPES.union(FLOAT_TYPES)
    ),
}


def binary_op_with_types_from(name, types) -> list[OperatorFunc]:
    return [
        OperatorFunc(f"{name}_{typ.name}_{typ.name}_{typ.name}", [typ, typ], typ)
        for typ in types
    ]


BINARY_OPS: dict[BinaryOp, list[OperatorFunc]] = {
    BinaryOp.AND: binary_op_with_types_from("and", frozenset({BOOL})),
    BinaryOp.OR: binary_op_with_types_from("or", frozenset({BOOL})),
    # you can add, sub, mul any two of the same numeric types together
    BinaryOp.ADD: binary_op_with_types_from("add", NUMERIC_TYPES),
    BinaryOp.SUBTRACT: binary_op_with_types_from("sub", NUMERIC_TYPES),
    BinaryOp.MULTIPLY: binary_op_with_types_from("mul", NUMERIC_TYPES),
    BinaryOp.MODULUS: binary_op_with_types_from("mod", NUMERIC_TYPES),
    BinaryOp.FLOOR_DIVIDE: binary_op_with_types_from("floor_div", NUMERIC_TYPES),
    # exponent is only supported at F64
    BinaryOp.EXPONENT: binary_op_with_types_from("exp", frozenset({F64})),
    BinaryOp.DIVIDE: binary_op_with_types_from("div", frozenset({F64})),
    # comparison ops are supported for all numeric types
    BinaryOp.LESS_THAN: binary_op_with_types_from("lt", NUMERIC_TYPES),
    BinaryOp.LESS_THAN_OR_EQUAL: binary_op_with_types_from("le", NUMERIC_TYPES),
    BinaryOp.GREATER_THAN: binary_op_with_types_from("gt", NUMERIC_TYPES),
    BinaryOp.GREATER_THAN_OR_EQUAL: binary_op_with_types_from("ge", NUMERIC_TYPES),
    # so are equality ops
    BinaryOp.EQUAL: binary_op_with_types_from("eq", NUMERIC_TYPES),
    BinaryOp.NOT_EQUAL: binary_op_with_types_from("ne", NUMERIC_TYPES),
}

# this leaves the special case of eq/neq generic over any Sized type T

# DIVIDE_FLOAT = auto()
# EQUAL_BYTES = auto()
# NOT_EQUAL_BYTES = auto()


NOT = OperatorFunc([BOOL])
IDENTITY_I64 = OperatorFunc([I64], I64)
IDENTITY_F64 = OperatorFunc([F64], F64)
IDENTITY_U64 = OperatorFunc([U64], U64)
NEGATE_I64 = OperatorFunc([I64], I64)

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
