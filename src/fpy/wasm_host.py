"""The host interface of a compiled wasm sequence: the functions the flight
sequencer provides to the sequence. All of them are imported under the wasm
import module named by HOST_MODULE_NAME."""

from llvmlite import ir

from fpy.bytecode.directives import ErrorCodeType

HOST_MODULE_NAME = "fprime"

HOST_EXIT_FUNC_NAME = "exit"
HOST_FAULT_FUNC_NAME = "fault"
HOST_EVENT_FUNC_NAME = "event"
HOST_CMD_FUNC_NAME = "cmd"

ERROR_CODE_TYPE = ErrorCodeType.llvm_type


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
    fn = ir.Function(module, fn_type, name=name)
    set.add(fn.attributes, f'"wasm-import-module"="{HOST_MODULE_NAME}"')
    return fn


def declare_host_imports(module: ir.Module) -> None:
    """Declare the full expected host interface on *module*; emit sites look
    the functions up in ``module.globals``. The float libcalls
    (env.pow/fmod/log) are deliberately absent: LLVM materializes those itself
    when lowering llvm.pow/llvm.log/frem, and they stay under wasm-ld's
    default import module "env"."""
    # exit(code) ends the whole sequence. It never returns to wasm (the host
    # unwinds the interpreter); noreturn lets LLVM drop the dead code after it.
    exit_fn = _declare_host_func(
        module, HOST_EXIT_FUNC_NAME, ir.FunctionType(ir.VoidType(), [ERROR_CODE_TYPE])
    )
    exit_fn.attributes.add("noreturn")

    # fault(code) reports a runtime error; like exit, it never returns.
    fault_fn = _declare_host_func(
        module, HOST_FAULT_FUNC_NAME, ir.FunctionType(ir.VoidType(), [ERROR_CODE_TYPE])
    )
    fault_fn.attributes.add("noreturn")

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
