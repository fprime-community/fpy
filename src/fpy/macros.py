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
from fpy.types import FLAG_ID, INTERNAL_STRING, NOTHING, TIME, BOOL, U8, U16, U32, I64, F64, FpyValue, FpyType
from fpy.state import BuiltinFuncSymbol
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


def generate_abs_float(node: Ast, const_args: dict[int, FpyValue]) -> list[Directive | Ir]:
    # if input is < 0 multiply by -1
    leave_unmodified = IrLabel(node, "else")
    dirs = [
        # copy the f64
        PushValDirective(FpyValue(StackSizeType, 8).serialize()),
        PushValDirective(FpyValue(StackSizeType, 0).serialize()),
        PeekDirective(),
        # push 0
        PushValDirective(FpyValue(F64, 0.0).serialize()),
        # check <
        FloatLessThanDirective(),
        IrIf(leave_unmodified),
        # push -1
        PushValDirective(FpyValue(F64, -1.0).serialize()),
        # and multiply
        FloatMultiplyDirective(),
        # otherwise do nothing
        leave_unmodified,
    ]
    return dirs


MACRO_ABS_FLOAT = BuiltinFuncSymbol("abs", F64, [("value", F64, None)], generate_abs_float)


def generate_abs_signed_int(node: Ast, const_args: dict[int, FpyValue]) -> list[Directive | Ir]:
    # if input is < 0 multiply by -1
    leave_unmodified = IrLabel(node, "else")
    dirs = [
        # copy the I64
        PushValDirective(FpyValue(StackSizeType, 8).serialize()),
        PushValDirective(FpyValue(StackSizeType, 0).serialize()),
        PeekDirective(),
        # push 0
        PushValDirective(FpyValue(I64, 0).serialize()),
        # check <
        SignedLessThanDirective(),
        IrIf(leave_unmodified),
        # push -1
        PushValDirective(FpyValue(I64, -1).serialize()),
        # and multiply
        IntMultiplyDirective(),
        # otherwise do nothing
        leave_unmodified,
    ]
    return dirs


MACRO_ABS_SIGNED_INT = BuiltinFuncSymbol(
    "abs", I64, [("value", I64, None)], generate_abs_signed_int
)

MACRO_SLEEP_SECONDS_USECONDS = BuiltinFuncSymbol(
    "sleep",
    NOTHING,
    [
        (
            "seconds",
            U32,
            FpyValue(U32, 0),
        ),
        ("useconds", U32, FpyValue(U32, 0)),
    ],
    lambda n, c: [WaitRelDirective()],
)


def generate_sleep_float(node: Ast, const_args: dict[int, FpyValue]) -> list[Directive | Ir]:
    # convert F64 to seconds and microseconds
    dirs = [
        # first do seconds
        # copy the f64
        PushValDirective(FpyValue(StackSizeType, 8).serialize()),
        PushValDirective(FpyValue(StackSizeType, 0).serialize()),
        PeekDirective(),
        # convert to U64
        FloatToUnsignedIntDirective(),
        # and then U32
        IntegerTruncate64To32Directive(),
        # now we have f64, u32 (seconds) on stack
        # now do microseconds
        # copy the f64 and u32
        PushValDirective(FpyValue(StackSizeType, 12).serialize()),
        PushValDirective(FpyValue(StackSizeType, 0).serialize()),
        PeekDirective(),
        # turn the u32 into a float
        IntegerZeroExtend32To64Directive(),
        UnsignedIntToFloatDirective(),
        # subtract, this should give us the frac
        FloatSubtractDirective(),
        # okay now multiply by 1000000
        PushValDirective(FpyValue(F64, 1_000_000.0).serialize()),
        # now convert to u32
        FloatToUnsignedIntDirective(),
        IntegerTruncate64To32Directive(),
    ]

    return dirs


MACRO_SLEEP_FLOAT = BuiltinFuncSymbol(
    "sleep", NOTHING, [("seconds", F64, None)], generate_sleep_float
)


def generate_log_signed_int(node: Ast, const_args: dict[int, FpyValue]) -> list[Directive | Ir]:
    return [
        # convert int to float
        SignedIntToFloatDirective(),
        FloatLogDirective(),
    ]

TIME_MACRO = BuiltinFuncSymbol(
        "time",
        TIME,
        [
            ("timestamp", INTERNAL_STRING, None),
            ("time_base", U16, FpyValue(U16, 0)),
            ("time_context", U8, FpyValue(U8, 0)),
        ],
        lambda n, c: [],  # placeholder - const eval handles this
    )

MACROS: dict[str, BuiltinFuncSymbol] = {
    "sleep": MACRO_SLEEP_SECONDS_USECONDS,
    "sleep_until": BuiltinFuncSymbol(
        "sleep_until",
        NOTHING,
        [("wakeup_time", TIME, None)],
        lambda n, c: [WaitAbsDirective()],
    ),
    "exit": BuiltinFuncSymbol(
        "exit", NOTHING, [("exit_code", U8, None)], lambda n, c: [ExitDirective()]
    ),
    "log": BuiltinFuncSymbol(
        "log", F64, [("operand", F64, None)], lambda n, c: [FloatLogDirective()]
    ),
    "now": BuiltinFuncSymbol("now", TIME, [], lambda n, c: [PushTimeDirective()]),
    "iabs": MACRO_ABS_SIGNED_INT,
    "fabs": MACRO_ABS_FLOAT,
    # time() parses ISO 8601 timestamps at compile time
    # The generate function should never be called since this is always const-evaluated
    "time": TIME_MACRO,
    "set_flag": BuiltinFuncSymbol(
        "set_flag",
        NOTHING,
        [("flag_idx", FLAG_ID, None), ("value", BOOL, None)],
        lambda n, c: [SetFlagDirective(FLAG_ID.enum_dict[c[0].val])],
        const_arg_indices=frozenset({0}),
    ),
    "get_flag": BuiltinFuncSymbol(
        "get_flag",
        BOOL,
        [("flag_idx", FLAG_ID, None)],
        lambda n, c: [GetFlagDirective(FLAG_ID.enum_dict[c[0].val])],
        const_arg_indices=frozenset({0}),
    ),
}
