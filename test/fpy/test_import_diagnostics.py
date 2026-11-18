"""Imported diagnostics retain their actual source after parsing returns."""

from dataclasses import FrozenInstanceError

import pytest

import fpy.error
from fpy.compiler import analysis_to_fpybc_directives, analyze_ast, text_to_ast
from fpy.error import BackendError, CompileError, WarningType
from fpy.state import get_base_compile_state
from fpy.test_helpers import assert_run_success, default_dictionary

BAD = "def ok() -> U8:\n    return 1\ndef bad() -> U8:\n    return ok() + 10\n"
MAIN = "import bad\nx: U8 = 1\ny: U8 = 2\nz: U8 = bad.ok()\n"


def analyze(source, directory, *, name="main.fpy", promoted=()):
    fpy.error.file_name = str(directory / name)
    state = get_base_compile_state(
        default_dictionary,
        import_directories=[str(directory)],
        main_file_dir=str(directory),
        main_file_path=str(directory / name),
        error_warnings=set(promoted),
    )
    return analyze_ast(text_to_ast(source), state)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("delayed", [False, True])
def test_imported_semantic_error_uses_its_own_filename_and_lines(
    tmp_path, nested, delayed
):
    (tmp_path / "bad.fpy").write_text(BAD)
    if nested:
        (tmp_path / "outer.fpy").write_text(
            "import bad\ndef ok() -> U8:\n    return bad.ok()\n"
        )
    source = MAIN.replace("bad", "outer") if nested else MAIN
    with pytest.raises(CompileError) as result:
        analyze(source, tmp_path)
    if delayed:
        analyze("assert True\n", tmp_path, name="later.fpy")
    rendered = str(result.value)
    assert f"{tmp_path / 'bad.fpy'}:4" in rendered.splitlines()[0]
    assert "return ok() + 10" in rendered
    assert "z: U8 =" not in rendered
    assert "^" in rendered


@pytest.mark.parametrize("promoted", [False, True])
def test_imported_warning_has_its_own_context(tmp_path, promoted):
    (tmp_path / "warn.fpy").write_text(
        "def ok() -> U8:\n    for i in 5 .. 3:\n        pass\n    return 1\n"
    )
    source = "import warn\nassert warn.ok() == 1\n"
    if promoted:
        with pytest.raises(CompileError) as result:
            analyze(source, tmp_path, promoted=[WarningType.EMPTY_RANGE])
        diagnostic = result.value
    else:
        state = analyze(source, tmp_path)
        diagnostic = next(
            w for w in state.warnings if w.type == WarningType.EMPTY_RANGE
        )
    analyze("assert True\n", tmp_path, name="later.fpy")
    rendered = str(diagnostic)
    assert f"{tmp_path / 'warn.fpy'}:2" in rendered.splitlines()[0]
    assert "for i in 5 .. 3" in rendered


def test_imported_backend_error_has_its_own_context(tmp_path):
    (tmp_path / "large.fpy").write_text('def run():\n    log("' + "x" * 3000 + '")\n')
    state = analyze("import large\nlarge.run()\n", tmp_path)
    with pytest.raises(BackendError) as result:
        analysis_to_fpybc_directives(state)
    rendered = str(result.value)
    assert f"{tmp_path / 'large.fpy'}:2" in rendered.splitlines()[0]
    assert "log(" in rendered


def test_imported_desugared_error_keeps_source(tmp_path):
    (tmp_path / "aug.fpy").write_text("def run():\n    x: U8 = 1\n    x += 300\n")
    with pytest.raises(CompileError) as result:
        analyze("import aug\naug.run()\n", tmp_path)
    rendered = str(result.value)
    assert f"{tmp_path / 'aug.fpy'}:3" in rendered.splitlines()[0]
    assert "x += 300" in rendered


def test_main_error_after_valid_import_keeps_main_context(tmp_path):
    (tmp_path / "valid.fpy").write_text("def value() -> U8:\n    return 7\n")
    with pytest.raises(CompileError) as result:
        analyze("import valid\nunknown_main_name\n", tmp_path)
    assert f"{tmp_path / 'main.fpy'}:2" in str(result.value).splitlines()[0]
    assert "unknown_main_name" in str(result.value)


def test_imported_parse_error_still_has_imported_context(tmp_path):
    (tmp_path / "syntax.fpy").write_text("def broken( ->\n")
    with pytest.raises(CompileError) as result:
        analyze("import syntax\n", tmp_path)
    assert f"{tmp_path / 'syntax.fpy'}:1" in str(result.value)


def test_valid_import_still_runs_on_both_backends(fprime_test_api, tmp_path):
    (tmp_path / "valid.fpy").write_text("def value() -> U8:\n    return 7\n")
    assert_run_success(
        fprime_test_api,
        "import valid\nassert valid.value() == 7\n",
        import_directories=[str(tmp_path)],
    )


def test_one_immutable_snapshot_is_shared_by_file_nodes(tmp_path):
    fpy.error.file_name = str(tmp_path / "source.fpy")
    body = text_to_ast("x: U8 = 1\ny: U8 = 2\n")
    source = body.meta.diagnostic_source
    assert body.stmts[0].meta.diagnostic_source is source
    assert body.stmts[1].meta.diagnostic_source is source
    assert source.lines == ("x: U8 = 1", "y: U8 = 2")
    with pytest.raises(FrozenInstanceError):
        source.lines = ()


def test_multiline_diagnostic_retains_its_snapshot_after_next_parse(tmp_path):
    path = str(tmp_path / "source.fpy")
    fpy.error.file_name = path
    body = text_to_ast("def value() -> U8:\n    x: U8 = 1\n    return x\n")
    error = CompileError("multiline marker", body.stmts[0])
    fpy.error.file_name = str(tmp_path / "later.fpy")
    text_to_ast("assert True\n")
    rendered = str(error)
    # Lark includes the trailing newline in the definition's end position.
    assert f"{path}:1-4" in rendered.splitlines()[0]
    assert "return x" in rendered
    assert "assert True" not in rendered
