"""Tests for sequence argument declarations and sequence imports.

Tests cover:
- Parsing of `sequence(...)` declarations
- Sequence parameter semantics (type resolution, defaults, ordering)
- Import statement parsing and resolution
- Calling imported sequences
- Header argument_count in compiled output
- Frame offset assignment for sequence args
"""

import pytest
import tempfile
import os
from pathlib import Path

import fpy.error
from fpy.compiler import text_to_ast, ast_to_directives, CompileResult
from fpy.error import CompileError, BackendError
from fpy.test_helpers import (
    assert_compile_failure,
    assert_compile_success,
    compile_seq,
    default_dictionary,
    CompilationFailed,
)
from fpy.bytecode.assembler import serialize_directives
from fpy.bytecode.directives import AllocateDirective


# When --use-gds is NOT passed (the default), override fprime_test_api with None
# so tests run against the Python model instead of a live GDS.
@pytest.fixture(name="fprime_test_api", scope="module")
def fprime_test_api_override(request):
    if request.config.getoption("--use-gds"):
        return request.getfixturevalue("fprime_test_api_session")
    return None


def compile_source(source: str, compile_args: dict | None = None) -> CompileResult | CompileError | BackendError:
    """Compile source text and return the raw result (not unwrapped)."""
    fpy.error.file_name = "<test>"
    fpy.error.input_text = source
    fpy.error.input_lines = source.splitlines()
    body = text_to_ast(source)
    if body is None:
        raise CompilationFailed("Parsing failed")
    return ast_to_directives(body, default_dictionary, compile_args)


# ──────────────────────── Parsing ─────────────────────────

class TestSequenceParsing:
    """Tests for parsing the `sequence(...)` declaration."""

    def test_sequence_no_args(self, fprime_test_api):
        """sequence() with no parameters should parse and compile."""
        assert_compile_success(fprime_test_api, "sequence()\n")

    def test_sequence_one_arg(self, fprime_test_api):
        """sequence with one typed parameter."""
        assert_compile_success(fprime_test_api, "sequence(x: U32)\n")

    def test_sequence_multiple_args(self, fprime_test_api):
        """sequence with multiple typed parameters."""
        assert_compile_success(fprime_test_api, "sequence(x: U32, y: F64, z: bool)\n")

    def test_sequence_with_defaults(self, fprime_test_api):
        """sequence parameters with default values."""
        seq = "sequence(x: U32, y: U32 = 42)\n"
        assert_compile_success(fprime_test_api, seq)

    def test_sequence_all_defaults(self, fprime_test_api):
        """sequence where all parameters have defaults."""
        seq = "sequence(x: U32 = 1, y: U32 = 2)\n"
        assert_compile_success(fprime_test_api, seq)

    def test_sequence_with_body(self, fprime_test_api):
        """sequence declaration followed by regular statements."""
        seq = """\
sequence(x: U32)
var: U32 = x
"""
        assert_compile_success(fprime_test_api, seq)

    def test_empty_sequence_with_body(self, fprime_test_api):
        """Empty sequence() followed by statements."""
        seq = """\
sequence()
var: U32 = 42
"""
        assert_compile_success(fprime_test_api, seq)


class TestSequenceParsingErrors:
    """Tests for parse/transform errors in sequence declarations."""

    def test_duplicate_sequence_decl(self, fprime_test_api):
        """Only one sequence() declaration is allowed."""
        seq = """\
sequence(x: U32)
sequence(y: U32)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_sequence_after_statement(self, fprime_test_api):
        """sequence() must appear before other statements."""
        seq = """\
var: U32 = 1
sequence(x: U32)
"""
        assert_compile_failure(fprime_test_api, seq)


# ──────────────────────── Semantics ──────────────────────

class TestSequenceSemantics:
    """Tests for semantic analysis of sequence declarations."""

    def test_args_are_usable_as_variables(self, fprime_test_api):
        """Sequence args should be usable as normal variables in the body."""
        seq = """\
sequence(x: U32, y: F64)
z: U32 = x
"""
        assert_compile_success(fprime_test_api, seq)

    def test_default_after_required_ok(self, fprime_test_api):
        """Default args after required args is allowed."""
        seq = "sequence(a: U32, b: U32 = 5)\n"
        assert_compile_success(fprime_test_api, seq)

    def test_required_after_default_error(self, fprime_test_api):
        """Required arg after default arg is an error."""
        seq = "sequence(a: U32 = 5, b: U32)\n"
        assert_compile_failure(fprime_test_api, seq)

    def test_duplicate_param_names(self, fprime_test_api):
        """Duplicate parameter names are rejected."""
        seq = "sequence(x: U32, x: U32)\n"
        assert_compile_failure(fprime_test_api, seq)

    def test_arg_type_resolution(self, fprime_test_api):
        """Sequence args should have their types resolved."""
        seq = """\
sequence(x: U32)
y: U32 = x
"""
        assert_compile_success(fprime_test_api, seq)

    def test_args_visible_in_functions(self, fprime_test_api):
        """Sequence args (globals) should be visible inside function defs."""
        seq = """\
sequence(x: U32)
def foo() -> U32:
    return x
y: U32 = foo()
"""
        assert_compile_success(fprime_test_api, seq)

    def test_arg_name_conflicts_with_variable(self, fprime_test_api):
        """A variable with the same name as a sequence arg should shadow it."""
        # CLAUDE this doesn't actually test what it says it tests
        seq = """\
sequence(x: U32)
x = 42
"""
        assert_compile_success(fprime_test_api, seq)

    def test_function_arg_same_name_as_seq_arg(self, fprime_test_api):
        """A function parameter can shadow a sequence arg name."""
        seq = """\
sequence(x: U32)
def foo(x: U32) -> U32:
    return x
y: U32 = foo(1)
"""
        assert_compile_success(fprime_test_api, seq)


# ──────────────────── Argument Count ─────────────────────

class TestArgumentCount:
    """Tests that argument_count is correctly reported."""

    def test_no_sequence_decl(self):
        """No sequence() → argument_count = 0."""
        result = compile_source("var: U32 = 1\n")
        assert isinstance(result, CompileResult)
        assert result.argument_count == 0

    def test_empty_sequence(self):
        """sequence() with no params → argument_count = 0."""
        result = compile_source("sequence()\n")
        assert isinstance(result, CompileResult)
        assert result.argument_count == 0

    def test_one_arg(self):
        """sequence(x: U32) → argument_count = 1."""
        result = compile_source("sequence(x: U32)\n")
        assert isinstance(result, CompileResult)
        assert result.argument_count == 1

    def test_multiple_args(self):
        """sequence(x: U32, y: F64) → argument_count = 2."""
        result = compile_source("sequence(x: U32, y: F64)\n")
        assert isinstance(result, CompileResult)
        assert result.argument_count == 2

    def test_args_with_defaults(self):
        """Default values don't change argument count."""
        result = compile_source("sequence(a: U32, b: U32 = 5, c: U32 = 10)\n")
        assert isinstance(result, CompileResult)
        assert result.argument_count == 3


# ──────────────────── Frame Offsets ──────────────────────

class TestFrameOffsets:
    """Tests that sequence args occupy the first stack slots."""
# CLAUDE let's not test this, too in the weeds. use golden tests for this

    def test_arg_takes_first_offset(self):
        """Sequence args should be at offset 0 (first stack slot)."""
        source = """\
sequence(x: U32)
y: U32 = 1
"""
        result = compile_source(source)
        assert isinstance(result, CompileResult)
        # The ALLOCATE directive should only allocate for y, not x
        alloc = [d for d in result.directives if isinstance(d, AllocateDirective)]
        assert len(alloc) == 1
        # x is U32 = 4 bytes at offset 0, y is U32 = 4 bytes at offset 4
        # ALLOCATE should be for 4 bytes (just y), not 8 (x + y)
        assert alloc[0].size == 4

    def test_no_extra_alloc_for_args_only(self):
        """If only sequence args exist, no ALLOCATE is needed."""
        source = "sequence(x: U32)\n"
        result = compile_source(source)
        assert isinstance(result, CompileResult)
        alloc = [d for d in result.directives if isinstance(d, AllocateDirective)]
        assert len(alloc) == 0


# ──────────────────── Serialization ──────────────────────

class TestSerialization:
    """Tests that argument_count makes it into the serialized binary."""

    def test_argument_count_in_header(self):
        """The header in the binary should have the correct argument count."""
        result = compile_source("sequence(x: U32, y: F64)\n")
        assert isinstance(result, CompileResult)
        binary, _ = serialize_directives(
            result.directives, argument_count=result.argument_count
        )
        # Header layout: major(1) + minor(1) + patch(1) + schema(1) + argCount(1) + stmtCount(2) + bodySize(4)
        # argumentCount is at byte 4, a single U8
        arg_count = binary[4]
        assert arg_count == 2


# ──────────────────── Import Parsing ─────────────────────

class TestImportParsing:
    """Tests for parsing import statements."""

    def test_simple_import_parses(self, fprime_test_api):
        """A simple import statement should parse."""
        # Note: this will fail at ResolveImports because the file doesn't exist,
        # but it should parse successfully
        body = text_to_ast("import foo\n")
        assert body is not None
        assert len(body.imports) == 1
        assert body.imports[0].path == ["foo"]
        assert body.imports[0].alias is None

    def test_dotted_import_parses(self, fprime_test_api):
        """A dotted import statement should parse."""
        body = text_to_ast("import foo.bar.baz\n")
        assert body is not None
        assert len(body.imports) == 1
        assert body.imports[0].path == ["foo", "bar", "baz"]
        assert body.imports[0].alias is None

    def test_import_with_alias_parses(self, fprime_test_api):
        """Import with 'as' alias should parse."""
        body = text_to_ast("import foo.bar as fb\n")
        assert body is not None
        assert len(body.imports) == 1
        assert body.imports[0].path == ["foo", "bar"]
        assert body.imports[0].alias == "fb"

    def test_multiple_imports_parse(self, fprime_test_api):
        """Multiple import statements should parse."""
        body = text_to_ast("import foo\nimport bar\n")
        assert body is not None
        assert len(body.imports) == 2

    def test_import_after_statement_fails(self, fprime_test_api):
        """Import after a regular statement should fail."""
        assert_compile_failure(fprime_test_api, "var: U32 = 1\nimport foo\n")

    def test_import_after_sequence_ok(self, fprime_test_api):
        """Import after sequence() is allowed (sequence comes first)."""
        body = text_to_ast("sequence(x: U32)\nimport foo\n")
        assert body is not None
        assert body.sequence_decl is not None
        assert len(body.imports) == 1


# ──────────────────── Import Resolution ──────────────────

class TestImportResolution:
    """Tests for ResolveImports — finding and parsing imported files."""

    @pytest.fixture
    def import_dir(self, tmp_path):
        """Create a temporary directory with importable .fpy files."""
        return tmp_path

    def _write_seq(self, directory, name, content):
        # CLAUDE use the temp file python api for this
        """Write a .fpy file into the directory."""
        parts = name.split(".")
        file_path = directory
        for part in parts[:-1]:
            file_path = file_path / part
            file_path.mkdir(exist_ok=True)
        file_path = file_path / (parts[-1] + ".fpy")
        file_path.write_text(content)
        return file_path

    def _compile_with_imports(self, source, search_paths):
        """Compile source with search_paths set."""
        fpy.error.file_name = "<test>"
        fpy.error.input_text = source
        fpy.error.input_lines = source.splitlines()
        body = text_to_ast(source)
        if body is None:
            raise CompilationFailed("Parsing failed")
        compile_args = {"search_paths": [str(p) for p in search_paths]}
        return ast_to_directives(body, default_dictionary, compile_args)

    def test_import_simple_sequence(self, import_dir):
        """Import a simple sequence with no args."""
        self._write_seq(import_dir, "helper", "sequence()\n")
        result = self._compile_with_imports(
            "import helper\n",
            [import_dir],
        )
        assert isinstance(result, CompileResult)

    def test_import_sequence_with_args(self, import_dir):
        """Import a sequence that declares parameters."""
        self._write_seq(import_dir, "helper", "sequence(x: U32, y: F64)\n")
        result = self._compile_with_imports(
            "import helper\n",
            [import_dir],
        )
        assert isinstance(result, CompileResult)

    def test_import_dotted_path(self, import_dir):
        """Import via dotted path creates subdirectories."""
        sub = import_dir / "pkg"
        sub.mkdir()
        self._write_seq(import_dir, "pkg.helper", "sequence(x: U32)\n")
        result = self._compile_with_imports(
            "import pkg.helper as helper\n",
            [import_dir],
        )
        assert isinstance(result, CompileResult)

    def test_import_nonexistent_file_fails(self, import_dir):
        """Importing a file that doesn't exist should fail."""
        result = self._compile_with_imports(
            "import nonexistent\n",
            [import_dir],
        )
        assert isinstance(result, CompileError)

    def test_import_file_without_sequence_decl(self, import_dir):
        """Importing a file with no sequence() declaration should still work.
        The imported sequence just has no args."""
        self._write_seq(import_dir, "helper", "var: U32 = 1\n")
        result = self._compile_with_imports(
            "import helper\n",
            [import_dir],
        )
        assert isinstance(result, CompileResult)

    def test_import_alias_conflict(self, import_dir):
        """Importing with an alias that conflicts with an existing name should error."""
        self._write_seq(import_dir, "foo", "sequence()\n")
        self._write_seq(import_dir, "bar", "sequence()\n")
        result = self._compile_with_imports(
            "import foo\nimport bar as foo\n",
            [import_dir],
        )
        assert isinstance(result, CompileError)


# ──────────────────── Sequence Call Codegen ───────────────

class TestSequenceCallCodegen:
    """Tests for calling imported sequences and the generated directives."""

    @pytest.fixture
    def import_dir(self, tmp_path):
        return tmp_path

    def _write_seq(self, directory, name, content):
        parts = name.split(".")
        file_path = directory
        for part in parts[:-1]:
            file_path = file_path / part
            file_path.mkdir(exist_ok=True)
        file_path = file_path / (parts[-1] + ".fpy")
        file_path.write_text(content)
        return file_path

    def _compile_with_imports(self, source, search_paths, compile_args=None):
        fpy.error.file_name = "<test>"
        fpy.error.input_text = source
        fpy.error.input_lines = source.splitlines()
        body = text_to_ast(source)
        if body is None:
            raise CompilationFailed("Parsing failed")
        args = {"search_paths": [str(p) for p in search_paths]}
        if compile_args:
            args.update(compile_args)
        return ast_to_directives(body, default_dictionary, args)

    def test_call_imported_sequence_no_args(self, import_dir):
        """Calling an imported sequence with no args should produce directives."""
        self._write_seq(import_dir, "helper", "sequence()\n")
        result = self._compile_with_imports(
            "import helper\nhelper()\n",
            [import_dir],
        )
        assert isinstance(result, CompileResult)
        assert len(result.directives) > 0

    def test_call_imported_sequence_with_args(self, import_dir):
        """Calling an imported sequence with args should compile."""
        self._write_seq(import_dir, "helper", "sequence(x: U32)\n")
        result = self._compile_with_imports(
            "import helper\nhelper(42)\n",
            [import_dir],
        )
        assert isinstance(result, CompileResult)

    def test_call_wrong_arg_count(self, import_dir):
        """Calling with wrong number of args should fail."""
        self._write_seq(import_dir, "helper", "sequence(x: U32, y: U32)\n")
        result = self._compile_with_imports(
            "import helper\nhelper(42)\n",
            [import_dir],
        )
        assert isinstance(result, CompileError)

    def test_call_wrong_arg_type(self, import_dir):
        """Calling with incompatible arg type should fail."""
        # CLAUDE "bool" is the right type here. I think you'll need to make sure that all
        # cases of corrupted/otherwise bad imported files are gracefully handled
        self._write_seq(import_dir, "helper", "sequence(x: Bool)\n")
        # String can't coerce to Bool
        result = self._compile_with_imports(
            'import helper\nhelper("hello")\n',
            [import_dir],
        )
        assert isinstance(result, CompileError)

    def test_call_with_alias(self, import_dir):
        """Calling via alias should work."""
        self._write_seq(import_dir, "helper", "sequence(x: U32)\n")
        result = self._compile_with_imports(
            "import helper as h\nh(42)\n",
            [import_dir],
        )
        assert isinstance(result, CompileResult)

    def test_call_uncalled_import(self, import_dir):
        """Importing without calling should be fine (no-op)."""
        self._write_seq(import_dir, "helper", "sequence(x: U32)\n")
        result = self._compile_with_imports(
            "import helper\nvar: U32 = 1\n",
            [import_dir],
        )
        assert isinstance(result, CompileResult)

    def test_call_with_named_args(self, import_dir):
        """Calling with named args should work."""
        self._write_seq(import_dir, "helper", "sequence(x: U32, y: U32)\n")
        result = self._compile_with_imports(
            "import helper\nhelper(y=2, x=1)\n",
            [import_dir],
        )
        assert isinstance(result, CompileResult)
