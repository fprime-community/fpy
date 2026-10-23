"""Excessive nesting must produce a diagnostic, not a Python traceback."""

import sys

import pytest

from fpy import compiler, test_helpers as th
from fpy.error import BackendError
from fpy.main import compile_main


def nested_statements(depth, statement="if"):
    lines = ["x: I64 = 0"]
    lines += ["    " * i + f"{statement} x == 0:" for i in range(depth)]
    lines += ["    " * depth + "x = 1", "assert x == 1"]
    return "\n".join(lines) + "\n"


def nested_expression(terms):
    return "x: I64 = 1\ny: I64 = " + " + ".join(["x"] * terms) + "\n"


GENERATORS = {
    "fpybc": compiler.analysis_to_fpybc_directives,
    "llvm-ir": compiler.analysis_to_llvm_module,
    "wasm": compiler.analysis_to_wasm,
    "wat": compiler.analysis_to_wat,
}
BACKENDS = [
    pytest.param(name, marks=pytest.mark.wasm) if name != "fpybc" else name
    for name in GENERATORS
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("statement", ["if", "while"])
def test_deep_statements_finish_or_report_nesting(backend, statement):
    # Analysis must succeed: this specifically exercises the codegen boundary.
    state = th.analyze_seq(nested_statements(200, statement))
    original_limit = sys.getrecursionlimit()
    try:
        GENERATORS[backend](state)
    except BackendError as error:
        assert "too deeply nested" in str(error)
    assert sys.getrecursionlimit() == original_limit


@pytest.mark.parametrize("backend", BACKENDS)
def test_deep_expression_reports_backend_error(backend):
    state = th.analyze_seq(nested_expression(300))
    # Some Python/backend versions can compile this expression. Do not reject
    # supported input just to force the same depth limit on both backends.
    try:
        GENERATORS[backend](state)
    except BackendError as error:
        assert "too deeply nested" in str(error)


@pytest.mark.parametrize(
    "emit",
    ["fpybin", "fpyasm"]
    + [
        pytest.param(name, marks=pytest.mark.wasm)
        for name in ("llvm-ir", "wasm", "wat")
    ],
)
def test_cli_reports_nesting_without_traceback(tmp_path, capsys, emit):
    source, output = tmp_path / "nested.fpy", tmp_path / "output"
    source.write_text(nested_statements(200), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        compile_main(
            [
                str(source),
                "-d",
                th.default_dictionary,
                "--emit",
                emit,
                "-o",
                str(output),
            ]
        )
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "too deeply nested" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


@pytest.mark.parametrize("statement", ["if", "while"])
def test_supported_statements_still_run(fprime_test_api, statement):
    th.assert_run_success(fprime_test_api, nested_statements(8, statement))


def test_supported_expression_still_runs(fprime_test_api):
    th.assert_run_success(fprime_test_api, nested_expression(20) + "assert y == 20\n")


@pytest.mark.wasm
def test_supported_long_expression_still_runs(fprime_test_api):
    th.assert_run_success(fprime_test_api, nested_expression(100) + "assert y == 100\n")


@pytest.mark.parametrize("backend", BACKENDS)
def test_codegen_recursion_is_reported_at_each_entry_point(monkeypatch, backend):
    state = th.analyze_seq("assert True\n")

    def recurse(*args, **kwargs):
        raise RecursionError("test recursion")

    if backend == "fpybc":
        monkeypatch.setattr(compiler.GenerateSequence, "emit", recurse)
    else:
        from fpy import codegen_llvm

        target = {
            "llvm-ir": "GenerateLlvmModule",
            "wasm": "llvm_module_to_wasm",
            "wat": "llvm_module_to_wasm_text",
        }[backend]
        if backend == "llvm-ir":
            monkeypatch.setattr(codegen_llvm.GenerateLlvmModule, "emit", recurse)
        else:
            monkeypatch.setattr(codegen_llvm, target, recurse)
    with pytest.raises(BackendError, match="too deeply nested") as error:
        GENERATORS[backend](state)
    assert error.value.__suppress_context__


@pytest.mark.parametrize(
    "error", [BackendError("original error"), AssertionError("bug")]
)
def test_other_codegen_errors_are_preserved(monkeypatch, error):
    state = th.analyze_seq("assert True\n")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(compiler.GenerateSequence, "emit", fail)
    with pytest.raises(type(error)) as caught:
        compiler.analysis_to_fpybc_directives(state)
    assert caught.value is error
