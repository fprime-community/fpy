// The harness process pytest starts once per test session: it runs compiled
// wasm sequences on a real Svc::WasmSequencer. See test/harness/Harness.hpp
// for the stdin/stdout protocol.

#include "test/harness/Harness.hpp"
#include "test/harness/wasm/tester/WasmSequencerTester.hpp"

int main() {
    return harness::harnessMain<Svc::WasmSequencerTester>();
}
