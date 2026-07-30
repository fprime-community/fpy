//! Runs a single compiled fpy `.wasm` sequence through the NASA spacewasm
//! interpreter and reports the result of its `main` export.
//!
//! Usage: `fpy-spacewasm-runner <path-to.wasm> [entry-name]
//!             [--fail-opcode <u32>]... [--cmd-response <i32>]`
//!
//! On success it prints the sequence's i32 error code as a decimal on the
//! final line of stdout and exits 0. The code comes from `fprime.exit` when
//! the sequence calls exit() or fails an assert, or from `fprime.fault` when
//! it hits an implicit runtime trap (array index out of bounds, zero
//! divisor). The entry itself is a `void main()` -- no arguments, no return
//! value -- since every failure (and explicit exit) leaves through exit/fault
//! and never comes back, so returning at all means success (reported as code
//! 0). The caller reads the printed code; the process exit status only
//! distinguishes "ran cleanly" (0) from "harness/runtime fault" (2), so a
//! nonzero error code is not conflated with a trap.
//!
//! Each `fprime.event` call the sequence makes (the log() builtin) is
//! reported before the code line as `event <severity> <message>`, one line
//! per event, with the message Rust-escaped so it stays on one line.
//!
//! Each `fprime.cmd` call is reported the same way as a `cmd <hex>` line
//! holding the received command buffer (the big-endian FwOpcodeType followed
//! by the serialized arguments) as lowercase hex. The call returns
//! `--cmd-response` (an Fw.CmdResponse value, default 0 = OK), except that a
//! command whose opcode is listed via `--fail-opcode` returns
//! EXECUTION_ERROR (4) -- mirroring the always-failing commands of the
//! bytecode reference model.
//!
//! Two host modules are provided: `env` holds the float libcalls
//! (`pow`/`fmod`/`log`) LLVM materializes itself, backed by libm so they
//! match the C/IEEE semantics the LLVM intrinsics lower to; `fprime` mirrors
//! the flight sequencer's host interface (`exit`, `fault`, `event`, `cmd`).

use std::alloc::Layout;
use std::cell::Cell;
use std::ops::ControlFlow;
use std::process::ExitCode;
use std::ptr::NonNull;
use std::rc::Rc;

use spacewasm::{
    AllocError, Allocator, CodeBuilder, CompilerOptions, Engine, ExportDesc, HostFunction,
    HostFunctionBreak, HostModule, InnerVec, Interpreter, InterpreterResult, InterpreterRunner,
    Module, ModuleRef, Ref, StartInvocation, Value, WasmMemoryAllocator, WasmRef, WasmStream,
    global_allocator,
};

/// Exit status used for any harness/runtime failure (read error, decode error,
/// failed instantiation, trap, missing export). Distinct from a clean run so
/// the Python side can tell a sequencer error code from a runtime fault.
const FAULT: u8 = 2;

/// Validation limits for decoding, matching spacewasm's own test harness.
const MAX_CONTROL_FRAMES: usize = 128;
const MAX_STACK_DEPTH: usize = 256;
const MAX_CODE_PAGES: u32 = 256;

/// Instruction budget per interpreter run, to catch infinite loops.
const FUEL: usize = 10_000_000;

// ---------------------------------------------------------------------------
// Allocator plumbing (mirrors spacewasm's own test harness). spacewasm is
// no_std and asks the embedder to supply the global alloc hooks plus an
// allocator for wasm linear memory.
// ---------------------------------------------------------------------------

struct RustSystemAllocator;

unsafe impl Allocator for RustSystemAllocator {
    unsafe fn alloc(&self, layout: Layout) -> Result<*mut u8, AllocError> {
        unsafe { Ok(std::alloc::alloc(layout)) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { std::alloc::dealloc(ptr, layout) }
    }

    fn memory_statistics(&self) -> spacewasm::MemoryStatistics {
        spacewasm::MemoryStatistics {
            total_bytes: 0,
            pad_bytes: 0,
        }
    }
}

impl WasmMemoryAllocator for RustSystemAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        unsafe { NonNull::new(std::alloc::alloc(layout)).ok_or(AllocError::AllocationFailed) }
    }

    fn reallocate(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        layout: Layout,
    ) -> Result<NonNull<u8>, AllocError> {
        unsafe {
            NonNull::new(std::alloc::realloc(ptr.as_ptr(), old_layout, layout.size()))
                .ok_or(AllocError::AllocationFailed)
        }
    }

    fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) }
    }
}

global_allocator!(RustSystemAllocator, RustSystemAllocator);

/// A one-shot WasmStream over an in-memory byte buffer.
struct ByteStream {
    buffer: Option<Vec<u8>>,
    consumed: bool,
}

impl ByteStream {
    fn new(data: &[u8]) -> Self {
        Self {
            buffer: Some(data.to_vec()),
            consumed: false,
        }
    }
}

impl WasmStream for ByteStream {
    fn read(&mut self) -> Result<Option<InnerVec<u8>>, u8> {
        if self.consumed {
            return Ok(None);
        }
        if let Some(ref mut vec) = self.buffer {
            self.consumed = true;
            Ok(Some(InnerVec {
                ptr: vec.as_mut_ptr(),
                capacity: vec.len() as u32,
                len: vec.len() as u32,
            }))
        } else {
            Ok(None)
        }
    }

    fn return_(&mut self, _chunk: InnerVec<u8>) {}
}

fn wasm_alloc() -> spacewasm::Rc<dyn WasmMemoryAllocator> {
    spacewasm::Rc::new(RustSystemAllocator)
        .unwrap()
        .into_wasm_memory_allocator()
}

/// The `env` host module: the float libcalls (`pow`/`fmod`/`log`) LLVM
/// materializes under wasm-ld's default import module. Backed by libm so edge
/// cases (e.g. `pow(0, -1)` -> +inf, domain errors -> NaN) match what the
/// LLVM intrinsics produce.
fn env_host_module() -> HostModule {
    fn arg_f64(args: &[Value], i: usize) -> f64 {
        match args[i] {
            Value::F64(v) => v,
            other => panic!("expected f64 host arg, got {other:?}"),
        }
    }
    HostModule {
        name: "env".into(),
        globals: spacewasm::vec![],
        functions: spacewasm::vec![
            HostFunction::new("pow", "dd".into(), "d".into(), |_, args| {
                ControlFlow::Continue(Some(Value::F64(libm::pow(
                    arg_f64(args, 0),
                    arg_f64(args, 1),
                ))))
            }),
            HostFunction::new("fmod", "dd".into(), "d".into(), |_, args| {
                ControlFlow::Continue(Some(Value::F64(libm::fmod(
                    arg_f64(args, 0),
                    arg_f64(args, 1),
                ))))
            }),
            HostFunction::new("log", "d".into(), "d".into(), |_, args| {
                ControlFlow::Continue(Some(Value::F64(libm::log(arg_f64(args, 0)))))
            }),
        ],
        memory: spacewasm::vec![],
        table: spacewasm::vec![],
    }
}

/// The `fprime` host module, mirroring the host interface the flight
/// sequencer (Svc/WasmSequencer) registers under that wasm module name.
///
/// `exit(code)` and `fault(code)` end the whole sequence: they record the code
/// into *exit_code* and unwind the interpreter with a host trap, so they
/// terminate the program from any call depth (a `ret` would only unwind the
/// current function). The recorded code -- not the trap -- carries the result.
/// `exit` is a termination the sequence asked for (exit(), a failing assert;
/// code 0 is a normal exit, nonzero an error), `fault` an implicit runtime
/// trap (array index out of bounds, zero divisor; always an error). This
/// harness reports both the same way, as the sequence's resulting error code.
///
/// `event(severity, ptr, len)` is the log() builtin's event report: the
/// pointer/length name a utf-8 message in guest linear memory. This harness
/// prints each one to stdout as an `event <severity> <message>` line.
///
/// `cmd(ptr, len)` dispatches a command: the pointer/length name the encoded
/// command buffer (big-endian FwOpcodeType + serialized arguments) in guest
/// linear memory. This harness prints each buffer as a `cmd <hex>` line and
/// returns *cmd_response*, or EXECUTION_ERROR (4) when the buffer's opcode is
/// in *fail_opcodes*.
fn fprime_host_module(
    exit_code: Rc<Cell<Option<i32>>>,
    fail_opcodes: Vec<u32>,
    cmd_response: i32,
) -> HostModule {
    fn arg_i32(args: &[Value], i: usize) -> i32 {
        match args[i] {
            Value::I32(v) => v,
            other => panic!("expected i32 host arg, got {other:?}"),
        }
    }
    fn record_code_and_unwind(
        exit_code: &Rc<Cell<Option<i32>>>,
        args: &[Value],
    ) -> ControlFlow<HostFunctionBreak, Option<Value>> {
        let code = match args[0] {
            Value::I32(v) => v,
            other => panic!("expected i32 exit code, got {other:?}"),
        };
        exit_code.set(Some(code));
        // Unwind the whole interpreter; run() reads the code back.
        ControlFlow::Break(HostFunctionBreak::Trap)
    }
    let fault_code = exit_code.clone();
    HostModule {
        name: "fprime".into(),
        globals: spacewasm::vec![],
        functions: spacewasm::vec![
            HostFunction::new("exit", "i".into(), "".into(), move |_, args| {
                record_code_and_unwind(&exit_code, args)
            }),
            HostFunction::new("fault", "i".into(), "".into(), move |_, args| {
                record_code_and_unwind(&fault_code, args)
            }),
            HostFunction::new("event", "iii".into(), "".into(), |engine, args| {
                let severity = arg_i32(args, 0);
                let ptr = arg_i32(args, 1) as u32 as usize;
                let len = arg_i32(args, 2) as u32 as usize;
                match engine.memory.load(ptr, len) {
                    Ok(bytes) => {
                        let message = String::from_utf8_lossy(bytes);
                        println!("event {severity} {}", message.escape_default());
                        ControlFlow::Continue(None)
                    }
                    Err(e) => {
                        eprintln!("event: bad message pointer/length: {e:?}");
                        ControlFlow::Break(HostFunctionBreak::Trap)
                    }
                }
            }),
            HostFunction::new("cmd", "ii".into(), "i".into(), move |engine, args| {
                let ptr = arg_i32(args, 0) as u32 as usize;
                let len = arg_i32(args, 1) as u32 as usize;
                match engine.memory.load(ptr, len) {
                    Ok(bytes) => {
                        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
                        println!("cmd {hex}");
                        // FwOpcodeType is a U32 in the test dictionary: the
                        // buffer leads with the opcode's 4 big-endian bytes.
                        let Some(opcode_bytes) = bytes.get(0..4) else {
                            eprintln!("cmd: buffer too short for an opcode: {len} bytes");
                            return ControlFlow::Break(HostFunctionBreak::Trap);
                        };
                        let opcode = u32::from_be_bytes(opcode_bytes.try_into().unwrap());
                        let response = if fail_opcodes.contains(&opcode) {
                            4 // Fw.CmdResponse.EXECUTION_ERROR
                        } else {
                            cmd_response
                        };
                        ControlFlow::Continue(Some(Value::I32(response)))
                    }
                    Err(e) => {
                        eprintln!("cmd: bad buffer pointer/length: {e:?}");
                        ControlFlow::Break(HostFunctionBreak::Trap)
                    }
                }
            }),
        ],
        memory: spacewasm::vec![],
        table: spacewasm::vec![],
    }
}

fn run(
    wasm_path: &str,
    entry: &str,
    fail_opcodes: Vec<u32>,
    cmd_response: i32,
) -> Result<i32, String> {
    let wasm = std::fs::read(wasm_path).map_err(|e| format!("read {wasm_path}: {e}"))?;

    // exit/fault write the sequence's error code here and unwind the interpreter.
    let exit_code: Rc<Cell<Option<i32>>> = Rc::new(Cell::new(None));

    let mut engine = Engine::new(
        1024,
        256,
        spacewasm::vec![
            env_host_module(),
            fprime_host_module(exit_code.clone(), fail_opcodes, cmd_response)
        ],
    )
    .map_err(|e| format!("engine: {e:?}"))?;
    let mut code_builder = CodeBuilder::new(CompilerOptions {
        allow_memory_grow: true,
        max_backpatch_iterations: 0,
        max_code_pages: MAX_CODE_PAGES,
    })
    .map_err(|e| format!("code builder: {e:?}"))?;

    let mut stream = ByteStream::new(&wasm);
    let module = Module::new::<MAX_CONTROL_FRAMES, MAX_STACK_DEPTH>(
        "seq",
        &mut stream,
        &mut engine.store,
        &mut code_builder,
        wasm_alloc(),
    )
    .map_err(|e| format!("decode: {e:?}"))?;

    // Instantiate: push the module onto the store and drive its start function
    // (if any) to completion before touching the exports.
    let module_ref = engine
        .push_module(module)
        .map_err(|e| format!("push module: {e:?}"))?;
    let start_result = match engine.invoke_start(module_ref) {
        StartInvocation::Finished => InterpreterResult::Finished,
        StartInvocation::Trap(t) => InterpreterResult::Trap(t),
        StartInvocation::Pause => InterpreterResult::Pause,
        StartInvocation::Running => Interpreter.run(code_builder.pages(), &mut engine, FUEL),
    };
    // exit/fault during the start function is still the sequence's verdict.
    if let Some(code) = exit_code.get() {
        return Ok(code);
    }
    if start_result != InterpreterResult::Finished {
        return Err(format!("initialize: {start_result:?}"));
    }

    // Resolve the entry export to a WasmRef.
    let module_index = module_ref.0 as usize;
    let f_ref = {
        let module = &engine.store.modules()[module_index];
        let export = module
            .exports
            .iter()
            .find(|e| e.name == entry)
            .ok_or_else(|| format!("export {entry:?} not found"))?;
        let func_idx = match &export.desc {
            ExportDesc::Func(idx) => *idx,
            other => return Err(format!("export {entry:?} is not a function: {other:?}")),
        };
        match module
            .get_func_ref(func_idx)
            .ok_or_else(|| format!("no func ref for {entry:?}"))?
        {
            Ref::Module(index) => WasmRef {
                module: ModuleRef(module_index as u8),
                index,
            },
            Ref::Extern { module, index } => WasmRef { module, index },
            other => return Err(format!("{entry:?} resolved to non-function: {other:?}")),
        }
    };

    engine
        .invoke(f_ref, &[])
        .map_err(|e| format!("invoke: {e:?}"))?;
    let result = Interpreter.run(code_builder.pages(), &mut engine, FUEL);

    // If exit/fault ran, the recorded code is authoritative -- they unwind via
    // a host trap, so the interpreter result is a trap we must not treat as a
    // harness fault. exit()/assert/implicit traps take this path; falling off
    // the end of main does not (it returns normally below).
    if let Some(code) = exit_code.get() {
        return Ok(code);
    }

    match result {
        // main returned without calling exit/fault. The entry is void, so
        // reaching its end is the only way here and it always means success.
        InterpreterResult::Finished => Ok(0),
        InterpreterResult::Trap(t) => Err(format!("trap: {t:?}")),
        InterpreterResult::Pause => Err("interpreter paused unexpectedly".into()),
        InterpreterResult::OutOfFuel => Err("out of fuel (infinite loop?)".into()),
    }
}

fn main() -> ExitCode {
    let usage = "usage: fpy-spacewasm-runner <path-to.wasm> [entry-name] \
                 [--fail-opcode <u32>]... [--cmd-response <i32>]";
    let mut positional: Vec<String> = Vec::new();
    let mut fail_opcodes: Vec<u32> = Vec::new();
    let mut cmd_response: i32 = 0;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let flag_value = |args: &mut dyn Iterator<Item = String>| {
            args.next().ok_or_else(|| format!("{arg} needs a value"))
        };
        let parsed = match arg.as_str() {
            "--fail-opcode" => flag_value(&mut args)
                .and_then(|v| v.parse().map_err(|e| format!("{arg} {v}: {e}")))
                .map(|v| fail_opcodes.push(v)),
            "--cmd-response" => flag_value(&mut args)
                .and_then(|v| v.parse().map_err(|e| format!("{arg} {v}: {e}")))
                .map(|v| cmd_response = v),
            _ => Ok(positional.push(arg.clone())),
        };
        if let Err(msg) = parsed {
            eprintln!("{msg}\n{usage}");
            return ExitCode::from(FAULT);
        }
    }
    let mut positional = positional.into_iter();
    let Some(wasm_path) = positional.next() else {
        eprintln!("{usage}");
        return ExitCode::from(FAULT);
    };
    let entry = positional.next().unwrap_or_else(|| "main".to_string());

    match run(&wasm_path, &entry, fail_opcodes, cmd_response) {
        Ok(code) => {
            println!("{code}");
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("fpy-spacewasm-runner: {msg}");
            ExitCode::from(FAULT)
        }
    }
}
