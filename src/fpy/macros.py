from __future__ import annotations

from fpy.bytecode.directives import (
    PushRandDirective,
    ExitDirective,
    FloatLogDirective,
    PopEventDirective,
    PopSerializableDirective,
    PushTimeDirective,
    SetSeedDirective,
    PushValDirective,
    SignedIntToFloatDirective,
    WaitAbsDirective,
    WaitRelDirective,
)
from fpy.ir import Ir
from fpy.symbols import BuiltinFuncSymbol
from fpy.wasm_host import (
    HOST_ASLEEP_FUNC_NAME,
    HOST_EVENT_FUNC_NAME,
    HOST_EXIT_FUNC_NAME,
    HOST_RSLEEP_FUNC_NAME,
    HOST_SERIAL_SEND_FUNC_NAME,
    HOST_TIME_FUNC_NAME,
)
from fpy.syntax import Ast
from fpy.bytecode.directives import SerialPortIndex
from fpy.types import (
    INTERNAL_STRING,
    LOG_SEVERITY,
    NOTHING,
    SIZED,
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
    node: Ast, const_args: dict[int, FpyValue], arg_types: list[FpyType]
) -> list[Directive | Ir]:
    return [FloatAbsDirective()]


def generate_abs_signed_int(
    node: Ast, const_args: dict[int, FpyValue], arg_types: list[FpyType]
) -> list[Directive | Ir]:
    return [IntAbsDirective()]


def _emit_micros_u64(builder, seconds, useconds):
    """Combine a (seconds, useconds) pair of i32 values into one i64
    microsecond count, the unit the host sleep imports take."""
    from llvmlite import ir

    i64 = ir.IntType(64)
    return builder.add(
        builder.mul(builder.zext(seconds, i64), ir.Constant(i64, 1_000_000)),
        builder.zext(useconds, i64),
    )


def generate_now_llvm(builder, args):
    """LLVM/wasm lowering of now(): the host time import writes the serialized
    Fw.Time into a buffer in linear memory, which is then deserialized into
    the Fw.Time value."""
    from llvmlite import ir

    # The load helper lives with the rest of the wire-format emission in the
    # LLVM backend; import it here so this module stays importable without
    # llvmlite.
    from fpy.codegen_llvm import EmitLlvmExpr, create_byte_buffer

    assert not args
    module = builder.module
    buf = create_byte_buffer(module, "time_buf", bytearray(TIME.max_size))
    builder.call(
        module.globals[HOST_TIME_FUNC_NAME],
        [
            builder.bitcast(buf, ir.IntType(8).as_pointer()),
            ir.Constant(ir.IntType(32), TIME.max_size),
        ],
    )
    return EmitLlvmExpr(builder)._emit_load_big_endian(TIME, buf, 0)


def generate_sleep_llvm(builder, args):
    """LLVM/wasm lowering of sleep(seconds, useconds): the host rsleep import
    takes the duration as one microsecond count."""
    [(seconds, _, _), (useconds, _, _)] = args
    builder.call(
        builder.module.globals[HOST_RSLEEP_FUNC_NAME],
        [_emit_micros_u64(builder, seconds, useconds)],
    )
    return None


def generate_sleep_until_llvm(builder, args):
    """LLVM/wasm lowering of sleep_until(wakeup_time): the host asleep import
    takes the wake-up time as microseconds since the epoch of the host's time
    base (the wakeup time's own base and context do not travel)."""
    [(wakeup, _, _)] = args
    member_idx = {m.name: i for i, m in enumerate(TIME.members)}
    seconds = builder.extract_value(wakeup, member_idx["seconds"])
    useconds = builder.extract_value(wakeup, member_idx["useconds"])
    builder.call(
        builder.module.globals[HOST_ASLEEP_FUNC_NAME],
        [_emit_micros_u64(builder, seconds, useconds)],
    )
    return None


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
    lambda n, c, t: [WaitRelDirective()],
    generate_sleep_llvm,
)


def generate_sleep_float(
    node: Ast, const_args: dict[int, FpyValue], arg_types: list[FpyType]
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
    node: Ast, const_args: dict[int, FpyValue], arg_types: list[FpyType]
) -> list[Directive | Ir]:
    return [
        # convert int to float
        SignedIntToFloatDirective(),
        FloatLogDirective(),
    ]


def generate_exit_llvm(builder, args):
    """LLVM/wasm lowering of exit(code): call the host exit function, which
    ends the whole sequence from any call depth (code 0 is a normal exit,
    nonzero an error).
    """
    [(code, _const, _)] = args
    builder.call(builder.module.globals[HOST_EXIT_FUNC_NAME], [code])
    builder.unreachable()
    builder.position_at_end(builder.function.append_basic_block("after_exit"))
    return None


def generate_abs_float_llvm(builder, args):
    [(value, _, _)] = args
    fn = builder.module.declare_intrinsic("llvm.fabs", [value.type])
    return builder.call(fn, [value])


def generate_abs_signed_int_llvm(builder, args):
    from llvmlite import ir

    [(value, _, _)] = args
    fn = builder.module.declare_intrinsic(
        "llvm.abs",
        [value.type, ir.IntType(1)],
        ir.FunctionType(ir.IntType(64), [value.type, ir.IntType(1)]),
    )
    return builder.call(fn, [value, ir.Constant(ir.IntType(1), 0)])


def generate_log_llvm(builder, args):
    from llvmlite import ir

    [(value, _, _)] = args
    fn = builder.module.declare_intrinsic(
        "llvm.log",
        [value.type],
        ir.FunctionType(value.type, [value.type]),
    )
    return builder.call(fn, [value])


def generate_log_event_llvm(builder, args):
    """LLVM/wasm lowering of log(message, severity): place the utf-8 message
    bytes in a constant in linear memory and call the host
    event(severity, ptr, len)."""
    from llvmlite import ir

    [(_, message, _), (_, severity, _)] = args
    data = message.val.encode("utf-8")
    module = builder.module
    msg_type = ir.ArrayType(ir.IntType(8), len(data))
    msg = ir.GlobalVariable(module, msg_type, name=module.get_unique_name("log_msg"))
    msg.linkage = "private"
    msg.global_constant = True
    msg.initializer = ir.Constant(msg_type, bytearray(data))

    i32 = ir.IntType(32)
    builder.call(
        builder.module.globals[HOST_EVENT_FUNC_NAME],
        [
            ir.Constant(i32, severity.type.enum_dict[severity.val]),
            builder.bitcast(msg, ir.IntType(8).as_pointer()),
            ir.Constant(i32, len(data)),
        ],
    )
    return None


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


def generate_randf(
    node: Ast, const_args: dict[int, FpyValue], arg_types: list[FpyType]
) -> list[Directive | Ir]:
    return [
        PushRandDirective(),
        IntegerZeroExtend32To64Directive(),
        UnsignedIntToFloatDirective(),
        PushValDirective(FpyValue(F64, 2**32).serialize()),
        FloatDivideDirective(),
    ]


def generate_write_to_port(
    node: Ast, const_args: dict[int, FpyValue], arg_types: list[FpyType]
) -> list[Directive | Ir]:
    # The value is already on the stack; pop it out the serial port. It is
    # coerced to a concrete sized type, so max_size is the exact size to pop.
    # The port is a const dictionary SerialPortIndex enum; .val is the
    # constant name, resolve to its int index.
    port_val = const_args[0]
    assert isinstance(port_val.val, str), port_val
    return [
        PopSerializableDirective(
            portIndex=port_val.type.enum_dict[port_val.val],
            size=arg_types[1].max_size,
        )
    ]


def generate_write_to_port_llvm(builder, args):
    """LLVM/wasm lowering of write_to_port(port, value): serialize the value
    into a buffer in linear memory in fprime wire format and call the host
    serial_send(port, ptr, len) import."""
    from llvmlite import ir

    # The store helper lives with the rest of the wire-format emission in the
    # LLVM backend; import it here so this module stays importable without
    # llvmlite.
    from fpy.codegen_llvm import EmitLlvmExpr, create_byte_buffer

    [(_, port_val, _), (value, const_val, value_type)] = args
    assert isinstance(port_val.val, str), port_val
    # The value is coerced to a concrete sized type, so max_size is the exact
    # serialized size.
    size = value_type.max_size
    module = builder.module
    if const_val is not None:
        # A constant serializes at compile time, straight into the buffer's
        # initializer. This is also the only way a string value travels: a
        # runtime string can't exist.
        data = const_val.serialize()
        assert len(data) == size, (const_val, size)
        buf = create_byte_buffer(module, "serial_buf", bytearray(data))
        buf.global_constant = True
    else:
        buf = create_byte_buffer(module, "serial_buf", bytearray(size))
        written = EmitLlvmExpr(builder)._emit_store_big_endian(
            value, value_type, buf, 0
        )
        assert written == size, (value_type, written)

    i32 = ir.IntType(32)
    builder.call(
        module.globals[HOST_SERIAL_SEND_FUNC_NAME],
        [
            ir.Constant(i32, port_val.type.enum_dict[port_val.val]),
            builder.bitcast(buf, ir.IntType(8).as_pointer()),
            ir.Constant(i32, size),
        ],
    )
    return None


TIME_MACRO = BuiltinFuncSymbol(
    "time",
    TIME,
    [
        ("timestamp", INTERNAL_STRING, None),
        ("timeBase", TIME_BASE, FpyValue(TIME_BASE, "TB_NONE")),
        ("timeContext", U8, FpyValue(U8, 0)),
    ],
    lambda n, c, t: [],  # placeholder - const eval handles this
)

MACROS: dict[str, BuiltinFuncSymbol] = {
    "sleep": MACRO_SLEEP_SECONDS_USECONDS,
    "sleep_until": BuiltinFuncSymbol(
        "sleep_until",
        NOTHING,
        [("wakeup_time", TIME, None)],
        lambda n, c, t: [WaitAbsDirective()],
        generate_sleep_until_llvm,
    ),
    "exit": BuiltinFuncSymbol(
        "exit",
        NOTHING,
        [("exit_code", ErrorCodeType, None)],
        lambda n, c, t: [ExitDirective()],
        generate_llvm=generate_exit_llvm,
    ),
    "ln": BuiltinFuncSymbol(
        "ln",
        F64,
        [("operand", F64, None)],
        lambda n, c, t: [FloatLogDirective()],
        generate_log_llvm,
    ),
    "now": BuiltinFuncSymbol(
        "now", TIME, [], lambda n, c, t: [PushTimeDirective()], generate_now_llvm
    ),
    "rand": BuiltinFuncSymbol("rand", U32, [], lambda n, c, t: [PushRandDirective()]),
    "randf": BuiltinFuncSymbol("randf", F64, [], generate_randf),
    "set_seed": BuiltinFuncSymbol(
        "set_seed", NOTHING, [("seed", U32, None)], lambda n, c, t: [SetSeedDirective()]
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
        lambda n, c, t: [
            PushValDirective(c[1].serialize()),
            PushValDirective(c[0].val.encode("utf-8")),
            PushValDirective(
                FpyValue(StackSizeType, len(c[0].val.encode("utf-8"))).serialize()
            ),
            PopEventDirective(),
        ],
        generate_log_event_llvm,
        const_arg_indices=frozenset({0, 1}),
    ),
    # Serial write: port typed by the dictionary-backed Svc.Fpy.SerialPortIndex enum; value typed SIZED
    "write_to_port": BuiltinFuncSymbol(
        "write_to_port",
        NOTHING,
        [
            ("port", SerialPortIndex, None),
            ("value", SIZED, None),
        ],
        generate_write_to_port,
        generate_write_to_port_llvm,
        const_arg_indices=frozenset({0}),  # port must be compile-time constant
    ),
}
