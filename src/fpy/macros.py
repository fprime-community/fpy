from __future__ import annotations

from llvmlite import ir
from fpy.bytecode.directives import (
    PushRandDirective,
    ExitDirective,
    FloatLogDirective,
    PopEventDirective,
    PushTimeDirective,
    SetSeedDirective,
    PushValDirective,
    SignedIntToFloatDirective,
    WaitAbsDirective,
    WaitRelDirective,
)
from fpy.ir import Ir
from fpy.symbols import BuiltinFuncSymbol
from fpy.syntax import Ast
from fpy.types import (
    INTERNAL_STRING,
    LOG_SEVERITY,
    NOTHING,
    TIME,
    TIME_BASE,
    BOOL,
    U8,
    U16,
    U32,
    I64,
    F64,
    FpyValue,
    FpyType,
)
from fpy.bytecode.directives import (
    FloatDivideDirective,
    FloatSubtractDirective,
    FloatToUnsignedIntDirective,
    FloatAbsDirective,
    IntAbsDirective,
    IntegerTruncate64To32Directive,
    IntegerZeroExtend32To64Directive,
    PeekDirective,
    PushTimeDirective,
    PushValDirective,
    FloatLogDirective,
    Directive,
    ExitDirective,
    ErrorCodeType,
    StackSizeType,
    UnsignedIntToFloatDirective,
    WaitAbsDirective,
    WaitRelDirective,
)


def generate_abs_float(
    node: Ast, const_args: dict[int, FpyValue]
) -> list[Directive | Ir]:
    return [FloatAbsDirective()]


def generate_abs_signed_int(
    node: Ast, const_args: dict[int, FpyValue]
) -> list[Directive | Ir]:
    return [IntAbsDirective()]


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


def generate_sleep_float(
    node: Ast, const_args: dict[int, FpyValue]
) -> list[Directive | Ir]:
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


def generate_log_signed_int(
    node: Ast, const_args: dict[int, FpyValue]
) -> list[Directive | Ir]:
    return [
        # convert int to float
        SignedIntToFloatDirective(),
        FloatLogDirective(),
    ]


def generate_exit_llvm(builder, args):
    """LLVM/wasm lowering of exit(code): call the host fpy_exit function, which
    ends the whole sequence from any call depth (code 0 is a normal exit,
    nonzero a fault).
    """
    from fpy.codegen_llvm import emit_host_exit

    [(code, _const)] = args
    emit_host_exit(builder, code)
    builder.position_at_end(builder.function.append_basic_block("after_exit"))
    return None


def generate_abs_float_llvm(builder, args):
    [(value, _)] = args
    fn = builder.module.declare_intrinsic("llvm.fabs", [value.type])
    return builder.call(fn, [value])


def generate_abs_signed_int_llvm(builder, args):
    [(value, _)] = args
    fn = builder.module.declare_intrinsic(
        "llvm.abs",
        [value.type, ir.IntType(1)],
        ir.FunctionType(ir.IntType(64), [value.type, ir.IntType(1)]),
    )
    return builder.call(fn, [value, ir.Constant(ir.IntType(1), 0)])


def generate_log_llvm(builder, args):
    [(value, _)] = args
    fn = builder.module.declare_intrinsic(
        "llvm.log",
        [value.type],
        ir.FunctionType(value.type, [value.type]),
    )
    return builder.call(fn, [value])


MACRO_ABS_FLOAT = BuiltinFuncSymbol(
    "abs", F64, [("value", F64, None)], generate_abs_float, generate_abs_float_llvm
)

MACRO_ABS_SIGNED_INT = BuiltinFuncSymbol(
    "abs",
    I64,
    [("value", I64, None)],
    generate_abs_signed_int,
    generate_abs_signed_int_llvm,
)


def generate_randf(node: Ast, const_args: dict[int, FpyValue]) -> list[Directive | Ir]:
    return [
        PushRandDirective(),
        IntegerZeroExtend32To64Directive(),
        UnsignedIntToFloatDirective(),
        PushValDirective(FpyValue(F64, 2**32).serialize()),
        FloatDivideDirective(),
    ]


TIME_MACRO = BuiltinFuncSymbol(
    "time",
    TIME,
    [
        ("timestamp", INTERNAL_STRING, None),
        ("timeBase", TIME_BASE, FpyValue(TIME_BASE, "TB_NONE")),
        ("timeContext", U8, FpyValue(U8, 0)),
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
        "exit",
        NOTHING,
        [("exit_code", ErrorCodeType, None)],
        lambda n, c: [ExitDirective()],
        generate_llvm=generate_exit_llvm,
    ),
    "ln": BuiltinFuncSymbol(
        "ln", F64, [("operand", F64, None)], lambda n, c: [FloatLogDirective()], generate_log_llvm
    ),
    "now": BuiltinFuncSymbol("now", TIME, [], lambda n, c: [PushTimeDirective()]),
    "rand": BuiltinFuncSymbol("rand", U32, [], lambda n, c: [PushRandDirective()]),
    "randf": BuiltinFuncSymbol("randf", F64, [], generate_randf),
    "set_seed": BuiltinFuncSymbol(
        "set_seed", NOTHING, [("seed", U32, None)], lambda n, c: [SetSeedDirective()]
    ),
    "iabs": MACRO_ABS_SIGNED_INT,
    "fabs": MACRO_ABS_FLOAT,
    # time() parses ISO 8601 timestamps at compile time
    # The generate function should never be called since this is always const-evaluated
    "time": TIME_MACRO,
    # Event logging builtin — compile-time string + severity, defaults to ACTIVITY_HI
    "log": BuiltinFuncSymbol(
        "log",
        NOTHING,
        [
            ("message", INTERNAL_STRING, None),
            ("severity", LOG_SEVERITY, FpyValue(LOG_SEVERITY, "ACTIVITY_HI")),
        ],
        lambda n, c: [
            PushValDirective(c[1].serialize()),
            PushValDirective(c[0].val.encode("utf-8")),
            PushValDirective(
                FpyValue(StackSizeType, len(c[0].val.encode("utf-8"))).serialize()
            ),
            PopEventDirective(),
        ],
        const_arg_indices=frozenset({0, 1}),
    ),
}
