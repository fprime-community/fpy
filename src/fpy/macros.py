from __future__ import annotations
from fpy.bytecode.directives import (
    ExitDirective,
    FloatLogDirective,
    GetFlagDirective,
    PushTimeDirective,
    SetFlagDirective,
    SignedIntToFloatDirective,
    WaitAbsDirective,
    WaitRelDirective,
)
from fpy.ir import Ir, IrIf, IrLabel
from fpy.syntax import Ast
from fpy.types import BuiltinFuncSymbol, FlagIdValue, FpyStringValue, NothingValue
from fprime_gds.common.models.serialize.time_type import TimeType as TimeValue
from fprime_gds.common.models.serialize.numerical_types import (
    U8Type as U8Value,
    U16Type as U16Value,
    U32Type as U32Value,
    I64Type as I64Value,
    F64Type as F64Value,
)
from fprime_gds.common.models.serialize.bool_type import BoolType as BoolValue
from fprime_gds.common.models.serialize.type_base import BaseType as FppValue
from fpy.bytecode.directives import (
    FloatLessThanDirective,
    FloatMultiplyDirective,
    FloatSubtractDirective,
    FloatToUnsignedIntDirective,
    IntMultiplyDirective,
    IntegerTruncate64To32Directive,
    IntegerZeroExtend32To64Directive,
    PeekDirective,
    PushTimeDirective,
    PushValDirective,
    FloatLogDirective,
    Directive,
    ExitDirective,
    SignedLessThanDirective,
    StackSizeType,
    UnsignedIntToFloatDirective,
    WaitAbsDirective,
    WaitRelDirective,
)


def generate_abs_float(node: Ast, const_args: dict[int, FppValue]) -> list[Directive | Ir]:
    # if input is < 0 multiply by -1
    leave_unmodified = IrLabel(node, "else")
    dirs = [
        # copy the f64
        PushValDirective(StackSizeType(8).serialize()),
        PushValDirective(StackSizeType(0).serialize()),
        PeekDirective(),
        # push 0
        PushValDirective(F64Value(0.0).serialize()),
        # check <
        FloatLessThanDirective(),
        IrIf(leave_unmodified),
        # push -1
        PushValDirective(F64Value(-1.0).serialize()),
        # and multiply
        FloatMultiplyDirective(),
        # otherwise do nothing
        leave_unmodified,
    ]
    return dirs


MACRO_ABS_FLOAT = BuiltinFuncSymbol("abs", F64Value, [("value", F64Value, None)], generate_abs_float)


def generate_abs_signed_int(node: Ast, const_args: dict[int, FppValue]) -> list[Directive | Ir]:
    # if input is < 0 multiply by -1
    leave_unmodified = IrLabel(node, "else")
    dirs = [
        # copy the I64
        PushValDirective(StackSizeType(8).serialize()),
        PushValDirective(StackSizeType(0).serialize()),
        PeekDirective(),
        # push 0
        PushValDirective(I64Value(0).serialize()),
        # check <
        SignedLessThanDirective(),
        IrIf(leave_unmodified),
        # push -1
        PushValDirective(I64Value(-1).serialize()),
        # and multiply
        IntMultiplyDirective(),
        # otherwise do nothing
        leave_unmodified,
    ]
    return dirs


MACRO_ABS_SIGNED_INT = BuiltinFuncSymbol(
    "abs", I64Value, [("value", I64Value, None)], generate_abs_signed_int
)

MACRO_SLEEP_SECONDS_USECONDS = BuiltinFuncSymbol(
    "sleep",
    NothingValue,
    [
        (
            "seconds",
            U32Value,
            U32Value(0),
        ),
        ("useconds", U32Value, U32Value(0)),
    ],
    lambda n, c: [WaitRelDirective()],
)


def generate_sleep_float(node: Ast, const_args: dict[int, FppValue]) -> list[Directive | Ir]:
    # convert F64 to seconds and microseconds
    dirs = [
        # first do seconds
        # copy the f64
        PushValDirective(StackSizeType(8).serialize()),
        PushValDirective(StackSizeType(0).serialize()),
        PeekDirective(),
        # convert to U64
        FloatToUnsignedIntDirective(),
        # and then U32
        IntegerTruncate64To32Directive(),
        # now we have f64, u32 (seconds) on stack
        # now do microseconds
        # copy the f64 and u32
        PushValDirective(StackSizeType(12).serialize()),
        PushValDirective(StackSizeType(0).serialize()),
        PeekDirective(),
        # turn the u32 into a float
        IntegerZeroExtend32To64Directive(),
        UnsignedIntToFloatDirective(),
        # subtract, this should give us the frac
        FloatSubtractDirective(),
        # okay now multiply by 1000000
        PushValDirective(F64Value(1_000_000.0).serialize()),
        # now convert to u32
        FloatToUnsignedIntDirective(),
        IntegerTruncate64To32Directive(),
    ]

    return dirs


MACRO_SLEEP_FLOAT = BuiltinFuncSymbol(
    "sleep", NothingValue, [("seconds", F64Value, None)], generate_sleep_float
)


def generate_log_signed_int(node: Ast, const_args: dict[int, FppValue]) -> list[Directive | Ir]:
    return [
        # convert int to float
        SignedIntToFloatDirective(),
        FloatLogDirective(),
    ]

TIME_MACRO = BuiltinFuncSymbol(
        "time",
        TimeValue,
        [
            ("timestamp", FpyStringValue, None),
            ("time_base", U16Value, U16Value(0)),
            ("time_context", U8Value, U8Value(0)),
        ],
        lambda n, c: [],  # placeholder - const eval handles this
    )

MACROS: dict[str, BuiltinFuncSymbol] = {
    "sleep": MACRO_SLEEP_SECONDS_USECONDS,
    "sleep_until": BuiltinFuncSymbol(
        "sleep_until",
        NothingValue,
        [("wakeup_time", TimeValue, None)],
        lambda n, c: [WaitAbsDirective()],
    ),
    "exit": BuiltinFuncSymbol(
        "exit", NothingValue, [("exit_code", U8Value, None)], lambda n, c: [ExitDirective()]
    ),
    "log": BuiltinFuncSymbol(
        "log", F64Value, [("operand", F64Value, None)], lambda n, c: [FloatLogDirective()]
    ),
    "now": BuiltinFuncSymbol("now", TimeValue, [], lambda n, c: [PushTimeDirective()]),
    "iabs": MACRO_ABS_SIGNED_INT,
    "fabs": MACRO_ABS_FLOAT,
    # time() parses ISO 8601 timestamps at compile time
    # The generate function should never be called since this is always const-evaluated
    "time": TIME_MACRO,
    "set_flag": BuiltinFuncSymbol(
        "set_flag",
        NothingValue,
        [("flag_idx", FlagIdValue, None), ("value", BoolValue, None)],
        lambda n, c: [SetFlagDirective(FlagIdValue.ENUM_DICT[c[0].val])],
        const_arg_indices=frozenset({0}),
    ),
    "get_flag": BuiltinFuncSymbol(
        "get_flag",
        BoolValue,
        [("flag_idx", FlagIdValue, None)],
        lambda n, c: [GetFlagDirective(FlagIdValue.ENUM_DICT[c[0].val])],
        const_arg_indices=frozenset({0}),
    ),
}
