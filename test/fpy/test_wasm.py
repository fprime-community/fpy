"""End-to-end tests for the LLVM/wasm backend.

These compile a sequence all the way to a runnable wasm module, run it in
wasmtime, and assert on the error code that ``fpy_main`` returns.

The backend currently only supports ``assert`` over compile-time-constant
conditions (all-literal expressions fold at compile time). Testing *runtime*
arithmetic needs a runtime operand -- i.e. variables -- which the backend
doesn't have yet, so that's deferred. What's meaningfully exercised here is the
wasm round-trip and the assert/exit-code semantics: a failed assert returns its
exit code verbatim, or EXIT_WITH_ERROR by default.
"""

import pytest

from fpy.model import DirectiveErrorCode
from fpy.test_helpers import run_seq_wasm


NO_ERROR = DirectiveErrorCode.NO_ERROR.value
EXIT_WITH_ERROR = DirectiveErrorCode.EXIT_WITH_ERROR.value


class TestWasmAssert:
    def test_passing_assert_succeeds(self):
        assert run_seq_wasm("assert 1 == 1\n") == NO_ERROR

    def test_empty_sequence_succeeds(self):
        assert run_seq_wasm("") == NO_ERROR

    @pytest.mark.parametrize(
        "exit_code, expected",
        [
            (None, EXIT_WITH_ERROR),  # no code written -> default
            (42, 42),                 # written code returned verbatim
            (123, 123),
        ],
    )
    def test_failing_assert_returns_written_code(self, exit_code, expected):
        # A false assert returns its exit code. The condition is constant-false
        # so the failure branch is taken.
        suffix = "" if exit_code is None else f", {exit_code}"
        assert run_seq_wasm(f"assert 1 == 2{suffix}\n") == expected
