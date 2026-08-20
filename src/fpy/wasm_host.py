"""The host interface of a compiled wasm sequence: the functions the flight
sequencer provides to the sequence. All of them are imported under the wasm
import module named by HOST_MODULE_NAME."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fpy.bytecode.directives import ErrorCodeType

if TYPE_CHECKING:
    from llvmlite import ir

HOST_MODULE_NAME = "fprime_v1"

HOST_EXIT_FUNC_NAME = "exit"
HOST_PANIC_FUNC_NAME = "panic"
HOST_EVENT_FUNC_NAME = "event"
HOST_CMD_FUNC_NAME = "cmd"
HOST_TLM_FUNC_NAME = "tlm"
HOST_PRM_FUNC_NAME = "prm"
HOST_TIME_FUNC_NAME = "time"
HOST_RSLEEP_FUNC_NAME = "rsleep"
HOST_ASLEEP_FUNC_NAME = "asleep"


def __getattr__(name: str):
    # ERROR_CODE_TYPE is an llvmlite ir.Type, so building it needs llvmlite --
    # an optional dependency (the `wasm` extra). Serving it lazily keeps this
    # module importable (for the HOST_*_NAME constants) without llvmlite.
    if name == "ERROR_CODE_TYPE":
        return ErrorCodeType.llvm_type
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _declare_host_func(
    module: ir.Module, name: str, fn_type: ir.FunctionType
) -> ir.Function:
    """Declare the host function *name* on *module* as an import from the wasm
    module HOST_MODULE_NAME.

    Wasm imports are two-level (a module name plus a field name), and a
    function symbol's module name defaults to "env" unless the IR function
    carries the string attribute ``"wasm-import-module"``, which is what
    clang's ``__attribute__((import_module(...)))`` lowers to; see
    "import_module" in https://clang.llvm.org/docs/AttributeReference.html.
    llvmlite's attribute set only knows enum attributes, so bypass its
    allowlist to attach the string-valued one."""
    from llvmlite import ir

    fn = ir.Function(module, fn_type, name=name)
    set.add(fn.attributes, f'"wasm-import-module"="{HOST_MODULE_NAME}"')
    return fn


def declare_host_imports(module: ir.Module) -> None:
    """Declare the full expected host interface on *module*; emit sites look
    the functions up in ``module.globals``. The float libcalls
    (env.pow/fmod/log) are deliberately absent: LLVM materializes those itself
    when lowering llvm.pow/llvm.log/frem, and they stay under wasm-ld's
    default import module "env"."""
    from llvmlite import ir

    error_code_type = ErrorCodeType.llvm_type

    # exit(code) ends the whole sequence. It never returns to wasm (the host
    # unwinds the interpreter); noreturn lets LLVM drop the dead code after it.
    exit_fn = _declare_host_func(
        module, HOST_EXIT_FUNC_NAME, ir.FunctionType(ir.VoidType(), [error_code_type])
    )
    exit_fn.attributes.add("noreturn")

    # panic(code) reports a runtime error; like exit, it never returns.
    panic_fn = _declare_host_func(
        module, HOST_PANIC_FUNC_NAME, ir.FunctionType(ir.VoidType(), [error_code_type])
    )
    panic_fn.attributes.add("noreturn")

    # event(severity, msg_ptr, msg_len) emits a log message.
    _declare_host_func(
        module,
        HOST_EVENT_FUNC_NAME,
        ir.FunctionType(
            ir.VoidType(),
            [ir.IntType(32), ir.IntType(8).as_pointer(), ir.IntType(32)],
        ),
    )

    # cmd(buf_ptr, buf_len) dispatches a command buffer and returns its
    # Fw.CmdResponse, widened to i32.
    _declare_host_func(
        module,
        HOST_CMD_FUNC_NAME,
        ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer(), ir.IntType(32)]),
    )

    # tlm(chan_id, time_ptr, time_len, value_ptr, value_len) reads the latest
    # value of a telemetry channel: the host writes the serialized Fw.Time
    # into the time buffer (whose length must be exactly the serialized time
    # size) and the fprime-serialized value into the value buffer, returning
    # the read's Fw.TlmValid (the canonical TLM_VALID enum in fpy.types)
    # widened to i32. The channel id travels as i64 whatever width
    # FwChanIdType has.
    _declare_host_func(
        module,
        HOST_TLM_FUNC_NAME,
        ir.FunctionType(
            ir.IntType(32),
            [
                ir.IntType(64),
                ir.IntType(8).as_pointer(),
                ir.IntType(32),
                ir.IntType(8).as_pointer(),
                ir.IntType(32),
            ],
        ),
    )

    # time(time_ptr, time_len) reads the current time: the host writes the
    # serialized Fw.Time into the buffer, whose length must be exactly the
    # serialized time size.
    _declare_host_func(
        module,
        HOST_TIME_FUNC_NAME,
        ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer(), ir.IntType(32)]),
    )

    # rsleep(useconds) suspends the sequence for the given duration, relative
    # to the current time.
    _declare_host_func(
        module, HOST_RSLEEP_FUNC_NAME, ir.FunctionType(ir.VoidType(), [ir.IntType(64)])
    )

    # asleep(useconds) suspends the sequence until the given absolute time,
    # expressed in microseconds since the epoch of the host's time base.
    _declare_host_func(
        module, HOST_ASLEEP_FUNC_NAME, ir.FunctionType(ir.VoidType(), [ir.IntType(64)])
    )

    # prm(prm_id, buf_ptr, buf_len) reads a parameter: the host writes the
    # fprime-serialized value into the buffer, returning the read's
    # Fw.ParamValid (the canonical PARAM_VALID enum in fpy.types) widened to
    # i32. The buffer contents are meaningful only for a VALID read. The
    # parameter id travels as i64 whatever width FwPrmIdType has.
    _declare_host_func(
        module,
        HOST_PRM_FUNC_NAME,
        ir.FunctionType(
            ir.IntType(32),
            [ir.IntType(64), ir.IntType(8).as_pointer(), ir.IntType(32)],
        ),
    )
