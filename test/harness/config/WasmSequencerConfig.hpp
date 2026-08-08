// ======================================================================
// \title  WasmSequencerConfig.hpp
// \brief  Svc::WasmSequencer limits for the harness
//
// Overrides the component's own defaults, which are sized for flight rather
// than for what fpy emits. A deployment running fpy sequences has to make the
// same adjustment: the compiler links each module with a 4 KiB guest stack
// (-zstack-size=4096 in codegen_llvm.py), so the component's default
// GUEST_MEMORY_SIZE of 2048 cannot hold one and the module fails to load with
// ERR_GUEST_MEMORY_ALLOC_FAILED.
//
// This file shadows the copy in the WasmSequencer checkout by sitting earlier
// on the include path; keep it in step when that one changes.
// ======================================================================

#ifndef WASMSEQUENCERCONFIG_HPP
#define WASMSEQUENCERCONFIG_HPP

#include <Fw/FPrimeBasicTypes.hpp>

namespace Svc {
namespace WasmSequencerConfig {

//! Page size of the interpreter's own heap, and how many pages back it.
constexpr FwSizeType SPACEWASM_PAGE_SIZE = 8192;
constexpr FwSizeType SPACEWASM_MAX_PAGES = 64;

//! Total static pool backing the interpreter heap.
constexpr FwSizeType DYNAMIC_MEMORY_SIZE = SPACEWASM_PAGE_SIZE * SPACEWASM_MAX_PAGES;

//! Interpreter value-stack depth available to a guest module.
constexpr FwSizeType GUEST_STACK_SIZE = 1024;

//! Upper bound on a module's code size, in pages.
constexpr U32 MAX_CODE_PAGES = 256;

//! Linear memory a guest module may claim. Must hold the stack the compiler
//! links in, plus the module's own data.
constexpr FwSizeType GUEST_MEMORY_SIZE = 65536;

//! How many modules the store holds at once.
constexpr U8 MAX_GUEST_MODULES = 8;

}  // namespace WasmSequencerConfig
}  // namespace Svc

#endif
