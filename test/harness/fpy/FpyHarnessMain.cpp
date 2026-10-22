// The harness process pytest starts once per test session: it runs compiled
// fpy sequences on a real Svc::FpySequencer. See Harness.hpp for the
// stdin/stdout protocol.

#include "test/harness/FpySequencerTester.hpp"
#include "test/harness/Harness.hpp"

int main() {
    return harness::harnessMain<Svc::FpySequencerTester>();
}
