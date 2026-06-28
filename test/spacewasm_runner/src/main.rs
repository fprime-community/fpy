//! Runs a single compiled fpy `.wasm` sequence through the NASA spacewasm
//! interpreter and reports the result of its `fpy_main` export.
//!
//! Usage: `fpy-spacewasm-runner <path-to.wasm> [entry-name]`
//!
//! On success it prints the i32 `fpy_main` return value (an fpy
//! `DirectiveErrorCode`) as a single decimal line to stdout and exits 0. The
//! caller reads the printed code; the process exit status only distinguishes
//! "ran cleanly" (0) from "harness/runtime fault" (2), so a nonzero
//! DirectiveErrorCode is not conflated with a trap.
//!
//! The `env.{pow,fmod,log}` host imports the fpy LLVM backend may emit are
//! provided here, backed by libm so they match the C/IEEE semantics the LLVM
//! intrinsics lower to.

use std::alloc::Layout;
use std::ops::ControlFlow;
use std::process::ExitCode;
use std::ptr::NonNull;

use spacewasm::{
    AllocError, Allocator, CodeBuilder, CompilerOptions, ExportDesc, HostFunction, HostModule,
    InitializeResult, InnerVec, InterpreterBreak, InterpreterResult, InterpreterRunner, Module,
    ModuleRef, ReaderError, Ref, Value, WasmMemoryAllocator, WasmRef, WasmStream, global_allocator,
};

/// Exit status used for any harness/runtime failure (read error, decode error,
/// failed instantiation, trap, missing export). Distinct from a clean run so
/// the Python side can tell a sequencer error code from a runtime fault.
const FAULT: u8 = 2;

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
    fn read(&mut self) -> Result<Option<InnerVec<u8>>, ReaderError> {
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

/// The host imports the fpy LLVM/wasm backend may emit, all under module `env`.
/// Backed by libm so edge cases (e.g. `pow(0, -1)` -> +inf, domain errors ->
/// NaN) match what the LLVM intrinsics produce.
fn fpy_host_module() -> HostModule {
    fn arg_f64(args: &[Value], i: usize) -> f64 {
        match args[i] {
            Value::F64(v) => v,
            other => panic!("expected f64 host arg, got {other:?}"),
        }
    }
    HostModule {
        name: "env",
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

fn run(wasm_path: &str, entry: &str) -> Result<i32, String> {
    let wasm = std::fs::read(wasm_path).map_err(|e| format!("read {wasm_path}: {e}"))?;

    let mut store =
        spacewasm::Store::new(256, [fpy_host_module()]).map_err(|e| format!("store: {e:?}"))?;
    let mut code_builder = CodeBuilder::<256>::default();

    let mut stream = ByteStream::new(&wasm);
    let module = Module::new::<256>(
        "seq",
        &mut stream,
        &mut store,
        &mut code_builder,
        wasm_alloc(),
        CompilerOptions {
            allow_memory_grow: true,
        },
    )
    .map_err(|e| format!("decode: {e:?}"))?;

    let (text, _) = code_builder.clone().finish().unwrap();

    // Instantiate the module (runs any start function); it is pushed onto the
    // store so we can look up its exports afterwards.
    {
        let mut state = store.allocate(1024).map_err(|e| format!("allocate: {e:?}"))?;
        match state.initialize_module(spacewasm::Box::new(module).unwrap(), &text, usize::MAX) {
            InitializeResult::Ok => {}
            other => return Err(format!("initialize: {other:?}")),
        }
    }

    // Resolve the entry export to a WasmRef (immutable borrows finish before we
    // re-borrow the store mutably to allocate the interpreter state).
    let module_index = store.modules().len() - 1;
    let f_ref = {
        let module = &store.modules()[module_index];
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

    let mut state = store.allocate(1024).map_err(|e| format!("allocate: {e:?}"))?;
    state.invoke(f_ref, &[]).map_err(|e| format!("invoke: {e:?}"))?;

    let interpreter = spacewasm::Interpreter::default();
    match interpreter.run(&text, &mut state, 10_000_000) {
        InterpreterResult::Instruction(InterpreterBreak::Finished) => {
            let raw = state.result.ok_or("entry returned no value")?;
            match raw.to_value(spacewasm::ValType::I32) {
                Value::I32(code) => Ok(code),
                other => Err(format!("entry returned non-i32: {other:?}")),
            }
        }
        InterpreterResult::Instruction(brk) => Err(format!("trap/break: {brk:?}")),
        InterpreterResult::OutOfFuel => Err("out of fuel (infinite loop?)".into()),
        InterpreterResult::ReaderError(e) => Err(format!("reader: {e:?}")),
    }
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(wasm_path) = args.next() else {
        eprintln!("usage: fpy-spacewasm-runner <path-to.wasm> [entry-name]");
        return ExitCode::from(FAULT);
    };
    let entry = args.next().unwrap_or_else(|| "fpy_main".to_string());

    match run(&wasm_path, &entry) {
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
