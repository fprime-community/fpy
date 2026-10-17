"""Every main sequence validates its argument bytes, even without parameters."""

import pytest

from fpy import compiler, test_helpers as th
from fpy.bytecode.directives import DirectiveErrorCode
from fpy.harness import wasm_harness
from fpy.types import FpyValue, U8, U32


@pytest.mark.parametrize("declaration", ["", "sequence()\n"])
@pytest.mark.parametrize(
    "args", [[FpyValue(U8, 1)], [FpyValue(U8, 1), FpyValue(U8, 2)]]
)
def test_parameterless_sequence_rejects_args(
    single_backend, tmp_path, declaration, args
):
    source = declaration + 'log("ran")\n'
    if single_backend == th.FPYBC:
        th.assert_run_failure(None, source, validation_error=True, args=args)
    else:
        assert_wasm_rejects_oversized_args(tmp_path, source, args)


def assert_wasm_rejects_oversized_args(tmp_path, source, args):
    # The real host rejects an undersized destination before returning to the
    # guest; this is a host failure, not a guest INVALID_ARG exit code.
    (tmp_path / "seq.wasm").write_bytes(th.compile_seq_wasm(source))
    result = wasm_harness().run(
        {
            "cwd": str(tmp_path),
            "seqFile": "seq.wasm",
            "args": b"".join(value.serialize() for value in args).hex(),
        }
    )
    assert result["cmdResponse"] == th.CMD_RESPONSE_EXECUTION_ERROR
    assert not any(event.get("guest") for event in result["events"])
    assert not result["cmds"]
    assert any(
        "BufferTooSmall" in event["text"] and "ARGS" in event["text"]
        for event in result["events"]
    )


@pytest.mark.parametrize("declaration", ["", "sequence()\n"])
@pytest.mark.parametrize("args", [None, []])
def test_parameterless_sequence_accepts_no_args(fprime_test_api, declaration, args):
    th.assert_run_success(fprime_test_api, declaration + "assert True\n", args=args)


@pytest.mark.parametrize(
    "args", [None, [FpyValue(U8, 42)], [FpyValue(U32, 42), FpyValue(U8, 1)]]
)
def test_parameterized_sequence_rejects_wrong_size(single_backend, tmp_path, args):
    source = "sequence(value: U32)\nassert value == 42\n"
    if single_backend == th.FPYBC:
        th.assert_run_failure(None, source, validation_error=True, args=args)
    elif args is not None and len(args) == 2:
        assert_wasm_rejects_oversized_args(tmp_path, source, args)
    else:
        th.assert_run_failure(
            None, source, error_code=DirectiveErrorCode.INVALID_ARG, args=args
        )


def test_parameterized_sequence_preserves_value(fprime_test_api):
    th.assert_run_success(
        fprime_test_api,
        "sequence(value: U32)\nassert value == 42\n",
        args=[FpyValue(U32, 42)],
    )


def test_normal_function_has_no_sequence_arg_prologue(fprime_test_api):
    th.assert_run_success(
        fprime_test_api,
        "sequence(value: U32)\n"
        "def increment(x: U32) -> U32:\n"
        "    return U32(x + 1)\n"
        "assert increment(value) == 43\n",
        args=[FpyValue(U32, 42)],
    )


@pytest.mark.wasm
@pytest.mark.parametrize("declaration", ["", "sequence()\n"])
def test_parameterless_main_imports_and_checks_zero_args(declaration):
    state = th.analyze_seq(declaration + "assert True\n")
    module, _ = compiler.analysis_to_llvm_module(state)
    main = str(module.globals["main"])
    assert 'call i32 @"args"' in main
    assert "i32 0)" in main
    assert "icmp ne i32" in main
