"""Tests of the sequencer harnesses themselves: a run that did not succeed
must never read as a success, whatever shape the failure takes."""

import pytest

from fpy.bytecode.directives import PushValDirective
from fpy.test_helpers import compile_seq, expected_final_stack, run_seq, run_wasm

# A minimal module whose exported entrypoint immediately hits `unreachable`,
# assembled by hand so the canary cannot drift with the compiler backend:
#   (module (func (export "main") unreachable))
UNREACHABLE_MODULE = (
    b"\x00asm\x01\x00\x00\x00"  # magic + version
    b"\x01\x04\x01\x60\x00\x00"  # type section: () -> ()
    b"\x03\x02\x01\x00"  # function section: one func of type 0
    b"\x07\x08\x01\x04main\x00\x00"  # export section: "main" = func 0
    b"\x0a\x05\x01\x03\x00\x00\x0b"  # code section: unreachable; end
)

# The same module with an empty body: the smallest module that succeeds.
# It pins the loader and entrypoint working, so the trap canary above cannot
# pass vacuously.
EMPTY_MODULE = (
    b"\x00asm\x01\x00\x00\x00"
    b"\x01\x04\x01\x60\x00\x00"
    b"\x03\x02\x01\x00"
    b"\x07\x08\x01\x04main\x00\x00"
    b"\x0a\x04\x01\x02\x00\x0b"  # code section: end
)


@pytest.mark.wasm
class TestWasmOutcome:
    def test_empty_module_succeeds(self):
        code, _, _ = run_wasm(EMPTY_MODULE)
        assert code == 0

    def test_raw_trap_is_not_success(self):
        """A wasm-level trap ends the run without an exit or panic code; it
        must surface as an error, never as a clean zero."""
        with pytest.raises(RuntimeError, match="without reporting a code"):
            run_wasm(UNREACHABLE_MODULE)


def test_stack_leak_is_detected():
    """The final-stack oracle must fire when a sequence leaves an extra byte."""
    state, directives, _ = compile_seq("x: U8 = 1\n")
    directives.append(PushValDirective(b"\xaa"))
    with pytest.raises(RuntimeError, match="leaked 1 bytes"):
        run_seq(directives, final_stack=expected_final_stack(state))
