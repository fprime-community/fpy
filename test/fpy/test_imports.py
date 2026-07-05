"""Tests for the `import` statement (spec / TDD).

`import foo` inlines the definitions of a sibling `foo.fpy` source file into
the importing sequence and sections its symbols off under the name `foo`, so a
function `bar` defined in `foo.fpy` is called as `foo.bar()`.

Design decisions encoded here:
  * Import paths resolve against `state.import_search_dirs`, an ordered list of
    directories searched first-match-wins.  In the CLI this is the importing
    file's own directory followed by each `-i/--include` directory.  This is
    DISTINCT from `ground_binary_dir`, which roots runtime
    sequence-binary (.bin) paths -- imports are a compile-time-only source
    inlining and never survive into the emitted bytecode.
  * A bare name resolves to a sibling file: `import foo` -> `foo.fpy`.  Dotted
    module paths resolve through package directories, Pythonically: `import
    a.b.c` -> `a/b/c.fpy`, searched against each `import_search_dirs` entry
    first-match-wins.  Package directories are plain namespace packages -- no
    `__init__.fpy` marker is required.  The name bound is the top segment, and
    members are reached by the full path: `import a.b.c` makes `a.b.c.bar()`
    callable (and `import foo` makes `foo.bar()` callable).
  * File/package precedence follows Python's namespace-package rules.  Because
    no `__init__.fpy` marker is required, directories are namespace portions,
    which rank BELOW modules: at an import's leaf segment a module file
    `foo.fpy` outranks a same-named `foo/` directory.  Non-leaf segments must be
    package directories -- `import a.b` descends into `a/` to reach `a/b.fpy`
    regardless of any sibling `a.fpy` module.  Importing a leaf that resolves
    only to a package directory (no module file to inline) is an error: a
    namespace package has nothing to inline.
  * `import` is only valid as a top-level statement (not nested in a block),
    but an imported module MAY itself import other modules (transitive imports
    are supported, with cycle detection).
  * Inlining an imported file that does more than define functions emits the
    `import-side-effects` warning (its top-level code runs as part of the
    importing sequence).
  * Importing a file that declares sequence arguments (`sequence(x: U32)`) is a
    hard error.
  * The module namespace obeys name groups: it exists only in the name groups
    of the symbols the module defines.  So `import lib` of a functions-only
    module collides with a local function `lib` (callable name group) but
    coexists with a local variable `lib` (value name group).
  * Importing the same module path more than once in one file is a hard
    error.  The rule is per-file: a file and a module it imports may each
    import the same module.

Every sequence that is expected to compile is also *run* (via
`assert_run_success`), so the asserts embedded in the sequences actually
execute.  Every sequence that is expected to fail compilation uses
`assert_compile_failure`.  These tests are `xfail` until the import passes are
implemented; they define the target behavior.  Remove the `pytestmark` once
`import` is implemented.
"""

from pathlib import Path

import pytest

from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_success,
    compile_seq,
)
from fpy.error import WarningType

# The whole module targets not-yet-implemented behavior.
pytestmark = pytest.mark.xfail(
    reason="import statement not yet implemented", strict=False
)


def _write_module(search_dir: Path, dotted_name: str, src: str) -> None:
    """Write an importable module for `import <dotted_name>` into *search_dir*.

    A bare name `foo` becomes `<search_dir>/foo.fpy`; a dotted name `a.b.c`
    becomes `<search_dir>/a/b/c.fpy`, creating the intervening package
    directories."""
    rel = Path(*dotted_name.split(".")).with_suffix(".fpy")
    path = search_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src)


class TestImportInlining:
    """An imported function is inlined and callable under the module name."""

    def test_call_imported_function(self, fprime_test_api, tmp_path):
        _write_module(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return x + 1
""",
        )
        main = """\
import lib

result: U32 = lib.add_one(41)
assert result == 42
"""
        # Funcs-only module: compiles cleanly with no side-effect warning...
        state, _, _ = compile_seq(main, import_search_dirs=[str(tmp_path)])
        assert state.warnings == []
        # ...and the embedded assert holds at run time.
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_imported_function_runs(self, fprime_test_api, tmp_path):
        _write_module(
            tmp_path,
            "lib",
            """\
def double(x: U32) -> U32:
    return x * 2
""",
        )
        main = """\
import lib

v: U32 = lib.double(21)
assert v == 42
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_local_and_imported_names_coexist(self, fprime_test_api, tmp_path):
        """A local `helper` and an imported `lib.helper` must not collide --
        the imported symbols are sectioned off under `lib`."""
        _write_module(
            tmp_path,
            "lib",
            """\
def helper() -> U32:
    return 1
""",
        )
        main = """\
import lib

def helper() -> U32:
    return 2

a: U32 = helper()
b: U32 = lib.helper()
assert a == 2
assert b == 1
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])


class TestImportSideEffects:
    """Importing a file with top-level, non-def code warns."""

    SIDE_EFFECT_MODULE = """\
CdhCore.cmdDisp.CMD_NO_OP()

def noop_wrapper():
    CdhCore.cmdDisp.CMD_NO_OP()
"""

    def test_side_effecting_import_warns(self, fprime_test_api, tmp_path):
        _write_module(tmp_path, "side_effects", self.SIDE_EFFECT_MODULE)
        main = """\
import side_effects

side_effects.noop_wrapper()
"""
        state, _, _ = compile_seq(main, import_search_dirs=[str(tmp_path)])
        assert any(
            w.type == WarningType.IMPORT_SIDE_EFFECTS for w in state.warnings
        ), f"expected an import-side-effects warning, got {state.warnings}"
        # The warning is non-fatal: the sequence still compiles and runs.
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_side_effect_warning_can_be_ignored(self, fprime_test_api, tmp_path):
        _write_module(tmp_path, "side_effects", self.SIDE_EFFECT_MODULE)
        main = """\
import side_effects

side_effects.noop_wrapper()
"""
        state, _, _ = compile_seq(
            main,
            import_search_dirs=[str(tmp_path)],
            ignored_warnings={WarningType.IMPORT_SIDE_EFFECTS},
        )
        assert state.warnings == []
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_side_effect_warning_can_be_escalated(self, fprime_test_api, tmp_path):
        _write_module(tmp_path, "side_effects", self.SIDE_EFFECT_MODULE)
        main = """\
import side_effects

side_effects.noop_wrapper()
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="import-side-effects",
            import_search_dirs=[str(tmp_path)],
            error_warnings={WarningType.IMPORT_SIDE_EFFECTS},
        )

    def test_functions_only_module_does_not_warn(self, fprime_test_api, tmp_path):
        _write_module(
            tmp_path,
            "clean",
            """\
def a() -> U32:
    return 1

def b() -> U32:
    return 2
""",
        )
        main = """\
import clean

x: U32 = clean.a() + clean.b()
assert x == 3
"""
        state, _, _ = compile_seq(main, import_search_dirs=[str(tmp_path)])
        assert state.warnings == []
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])


class TestImportErrors:
    """Error cases for import."""

    def test_cannot_import_sequence_with_arguments(self, fprime_test_api, tmp_path):
        _write_module(
            tmp_path,
            "with_arguments",
            """\
sequence(x: U32)

def f() -> U32:
    return x
""",
        )
        main = """\
import with_arguments
"""
        assert_compile_failure(
            fprime_test_api, main, match="argument", import_search_dirs=[str(tmp_path)]
        )

    def test_missing_module_is_an_error(self, fprime_test_api, tmp_path):
        main = """\
import does_not_exist
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_no_arg_sequence_is_importable(self, fprime_test_api, tmp_path):
        """A bare `sequence()` with no arguments is importable -- only
        sequences *with arguments* are rejected (per the feature's wording)."""
        _write_module(
            tmp_path,
            "no_argument_sequence",
            """\
sequence()

def f() -> U32:
    return 1
""",
        )
        main = """\
import no_argument_sequence

x: U32 = no_argument_sequence.f()
assert x == 1
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])


class TestImportFileErrors:
    """Failure modes rooted in the imported file itself."""

    def test_parse_error_in_imported_file_fails(self, fprime_test_api, tmp_path):
        """A syntax error inside the imported file is a hard compile error.
        Ideally the diagnostic points into the imported file, not the importer."""
        _write_module(
            tmp_path,
            "broken",
            """\
def f( ->
    return 1
""",
        )
        main = """\
import broken
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_import_path_is_a_directory_fails(self, fprime_test_api, tmp_path):
        """If the resolved `<name>.fpy` is a directory, importing fails cleanly
        rather than crashing with an IO error."""
        (tmp_path / "directory.fpy").mkdir()
        main = """\
import directory
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_empty_module_compiles_without_warning(self, fprime_test_api, tmp_path):
        """An empty module has no definitions and no side effects."""
        _write_module(tmp_path, "empty", "")
        main = """\
import empty

CdhCore.cmdDisp.CMD_NO_OP()
"""
        state, _, _ = compile_seq(main, import_search_dirs=[str(tmp_path)])
        assert state.warnings == []
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])


class TestImportNamespaceIsolation:
    """Imported symbols are sectioned under the module name and isolated."""

    def test_imported_symbol_requires_module_prefix(self, fprime_test_api, tmp_path):
        """`add_one` is only reachable as `lib.add_one`, never bare."""
        _write_module(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return x + 1
""",
        )
        main = """\
import lib

y: U32 = add_one(1)
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_module_name_not_usable_as_value(self, fprime_test_api, tmp_path):
        """The module name is a namespace, not an expression."""
        _write_module(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return x + 1
""",
        )
        main = """\
import lib

y: U32 = lib
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_same_function_name_in_two_modules_no_collision(
        self, fprime_test_api, tmp_path
    ):
        _write_module(
            tmp_path,
            "lib_a",
            """\
def helper() -> U32:
    return 1
""",
        )
        _write_module(
            tmp_path,
            "lib_b",
            """\
def helper() -> U32:
    return 2
""",
        )
        main = """\
import lib_a
import lib_b

a: U32 = lib_a.helper()
b: U32 = lib_b.helper()
assert a == 1
assert b == 2
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_imported_function_cannot_see_importer_globals(
        self, fprime_test_api, tmp_path
    ):
        """An imported function is analyzed in its own module scope: a name
        defined only in the importing sequence must NOT resolve inside it."""
        _write_module(
            tmp_path,
            "iso",
            """\
def uses_outside() -> U32:
    return main_global
""",
        )
        main = """\
import iso

main_global: U32 = 5
x: U32 = iso.uses_outside()
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )


class TestImportNameCollisions:
    """The module namespace collides with existing top-level names per name
    group: the namespace exists only in the name groups of the symbols the
    module defines."""

    def test_import_collides_with_local_function(self, fprime_test_api, tmp_path):
        """A functions-only module occupies the callable name group, so a
        local function with the module's name collides."""
        _write_module(
            tmp_path,
            "dup",
            """\
def f() -> U32:
    return 1
""",
        )
        main = """\
import dup

def dup() -> U32:
    return 2
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_import_collides_with_local_variable(self, fprime_test_api, tmp_path):
        """A module with a top-level variable occupies the value name group,
        so a local variable with the module's name collides."""
        _write_module(
            tmp_path,
            "dup",
            """\
v: U32 = 1
""",
        )
        main = """\
import dup

dup: U32 = 3
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_import_coexists_with_local_variable(self, fprime_test_api, tmp_path):
        """A functions-only module does NOT occupy the value name group, so a
        local variable with the module's name is legal, and both remain
        usable in their respective name groups."""
        _write_module(
            tmp_path,
            "dup",
            """\
def f() -> U32:
    return 1
""",
        )
        main = """\
import dup

dup: U32 = 3
x: U32 = dup.f()
assert x == 1
assert dup == 3
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])


class TestImportDuplicates:
    """Importing the same module path twice in one file is an error; the rule
    is per-file, so a file and a module it imports may each import the same
    module."""

    def test_duplicate_import_is_error(self, fprime_test_api, tmp_path):
        _write_module(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return x + 1
""",
        )
        main = """\
import lib
import lib
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_duplicate_across_files_is_allowed(self, fprime_test_api, tmp_path):
        """main imports both `a` and `c`; `a` also imports `c` internally.
        Each file imports `c` only once, so no duplicate error."""
        _write_module(tmp_path, "c", "def g() -> U32:\n    return 7\n")
        _write_module(
            tmp_path,
            "a",
            """\
import c

def f() -> U32:
    return c.g()
""",
        )
        main = """\
import a
import c

x: U32 = a.f() + c.g()
assert x == 14
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])


class TestImportOnlyAtTopLevel:
    """`import` is only valid as a top-level statement (never nested in a
    block).  Note: an imported *module* may still contain its own top-level
    imports -- see TestImportTransitive."""

    def test_import_inside_if_block_fails(self, fprime_test_api, tmp_path):
        _write_module(
            tmp_path,
            "lib",
            """\
def f() -> U32:
    return 1
""",
        )
        main = """\
if 1 == 1:
    import lib
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_import_inside_function_fails(self, fprime_test_api, tmp_path):
        _write_module(
            tmp_path,
            "lib",
            """\
def f() -> U32:
    return 1
""",
        )
        main = """\
def wrapper() -> U32:
    import lib
    return 1
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )


class TestImportTransitive:
    """An imported module may itself import other modules."""

    def test_transitive_import_works(self, fprime_test_api, tmp_path):
        """main -> a -> b: `a` uses `b` internally, and main runs `a.f()`."""
        _write_module(
            tmp_path,
            "b",
            """\
def g() -> U32:
    return 7
""",
        )
        _write_module(
            tmp_path,
            "a",
            """\
import b

def f() -> U32:
    return b.g()
""",
        )
        main = """\
import a

x: U32 = a.f()
assert x == 7
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_transitive_dependency_is_private(self, fprime_test_api, tmp_path):

        _write_module(
            tmp_path,
            "b",
            """\
def g() -> U32:
    return 7
""",
        )
        _write_module(
            tmp_path,
            "a",
            """\
import b

def f() -> U32:
    return b.g()
""",
        )
        main = """\
import a

x: U32 = b.g()
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )


class TestImportCycles:
    """Import cycles are detected and rejected."""

    def test_self_import_is_cycle_error(self, fprime_test_api, tmp_path):
        _write_module(
            tmp_path,
            "self_import",
            """\
import self_import

def f() -> U32:
    return 1
""",
        )
        main = """\
import self_import
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="(?i)(circular|cycle)",
            import_search_dirs=[str(tmp_path)],
        )

    def test_mutual_import_is_cycle_error(self, fprime_test_api, tmp_path):
        _write_module(
            tmp_path,
            "mod_a",
            """\
import mod_b

def a() -> U32:
    return mod_b.b()
""",
        )
        _write_module(
            tmp_path,
            "mod_b",
            """\
import mod_a

def b() -> U32:
    return 1
""",
        )
        main = """\
import mod_a
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="(?i)(circular|cycle)",
            import_search_dirs=[str(tmp_path)],
        )

    def test_three_way_cycle_error(self, fprime_test_api, tmp_path):
        _write_module(tmp_path, "c1", "import c2\n\ndef f() -> U32:\n    return 1\n")
        _write_module(tmp_path, "c2", "import c3\n\ndef f() -> U32:\n    return 1\n")
        _write_module(tmp_path, "c3", "import c1\n\ndef f() -> U32:\n    return 1\n")
        main = """\
import c1
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="(?i)(circular|cycle)",
            import_search_dirs=[str(tmp_path)],
        )


class TestImportDottedPaths:
    """Dotted module paths resolve through package directories, Pythonically."""

    def test_single_dotted_import(self, fprime_test_api, tmp_path):
        """`import pkg.mod` resolves `pkg/mod.fpy` and binds `pkg.mod`."""
        _write_module(
            tmp_path,
            "pkg.mod",
            """\
def add_one(x: U32) -> U32:
    return x + 1
""",
        )
        main = """\
import pkg.mod

result: U32 = pkg.mod.add_one(41)
assert result == 42
"""
        state, _, _ = compile_seq(main, import_search_dirs=[str(tmp_path)])
        assert state.warnings == []
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_deeply_nested_dotted_import(self, fprime_test_api, tmp_path):
        """`import a.b.c` resolves `a/b/c.fpy` (arbitrary nesting depth)."""
        _write_module(
            tmp_path,
            "a.b.c",
            """\
def val() -> U32:
    return 7
""",
        )
        main = """\
import a.b.c

x: U32 = a.b.c.val()
assert x == 7
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_dotted_symbol_requires_full_path(self, fprime_test_api, tmp_path):
        """A member of `pkg.mod` is only reachable as `pkg.mod.f`, never as a
        bare `f` nor via a truncated `mod.f`."""
        _write_module(
            tmp_path,
            "pkg.mod",
            """\
def f() -> U32:
    return 1
""",
        )
        main = """\
import pkg.mod

y: U32 = mod.f()
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_missing_leaf_in_existing_package_is_error(self, fprime_test_api, tmp_path):
        """The package dir exists but the leaf module file does not."""
        _write_module(
            tmp_path,
            "pkg.other",
            """\
def f() -> U32:
    return 1
""",
        )
        main = """\
import pkg.missing
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_two_modules_in_same_package_no_collision(self, fprime_test_api, tmp_path):
        """Sibling modules under one package are independently namespaced."""
        _write_module(tmp_path, "pkg.a", "def f() -> U32:\n    return 1\n")
        _write_module(tmp_path, "pkg.b", "def f() -> U32:\n    return 2\n")
        main = """\
import pkg.a
import pkg.b

x: U32 = pkg.a.f()
y: U32 = pkg.b.f()
assert x == 1
assert y == 2
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])


class TestImportPackagePrecedence:
    """File-vs-directory precedence follows Python's namespace-package rules:
    a module file outranks a same-named (init-less) package directory at a
    leaf, but non-leaf segments always descend into the directory."""

    def test_module_file_beats_namespace_directory(self, fprime_test_api, tmp_path):
        """`foo.fpy` and a `foo/` directory both exist; `import foo` resolves
        the module file (namespace dirs rank below modules)."""
        _write_module(tmp_path, "foo", "def f() -> U32:\n    return 1\n")
        # This also creates the sibling `foo/` directory:
        _write_module(tmp_path, "foo.inner", "def g() -> U32:\n    return 2\n")
        main = """\
import foo

x: U32 = foo.f()
assert x == 1
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_package_dir_used_for_dotted_descent(self, fprime_test_api, tmp_path):
        """A `pkg.fpy` module does not block `import pkg.mod` from descending
        into the `pkg/` directory to reach `pkg/mod.fpy`."""
        _write_module(tmp_path, "pkg", "def top() -> U32:\n    return 1\n")
        _write_module(tmp_path, "pkg.mod", "def f() -> U32:\n    return 5\n")
        main = """\
import pkg.mod

x: U32 = pkg.mod.f()
assert x == 5
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_bare_package_import_is_error(self, fprime_test_api, tmp_path):
        """`import pkg` where only a `pkg/` directory exists (no `pkg.fpy`) is
        an error -- a namespace package has nothing to inline."""
        # Creates `pkg/mod.fpy`, so `pkg/` exists as a directory but `pkg.fpy`
        # does not.
        _write_module(tmp_path, "pkg.mod", "def f() -> U32:\n    return 1\n")
        main = """\
import pkg
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_dotted_leaf_package_import_is_error(self, fprime_test_api, tmp_path):
        """`import a.b` where `a/b/` is a directory but `a/b.fpy` does not exist
        is likewise an error at the dotted leaf."""
        _write_module(tmp_path, "a.b.c", "def f() -> U32:\n    return 1\n")
        main = """\
import a.b
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )


class TestImportSearchDirs:
    """`import_search_dirs` is an ordered search path: first match wins."""

    def test_module_found_in_later_search_dir(self, fprime_test_api, tmp_path):
        """A module present only in the second search dir is still found."""
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        _write_module(d2, "lib", "def f() -> U32:\n    return 9\n")
        main = """\
import lib

x: U32 = lib.f()
assert x == 9
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(d1), str(d2)])

    def test_first_search_dir_shadows_later(self, fprime_test_api, tmp_path):
        """When a module name exists in two search dirs, the earlier dir wins."""
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        _write_module(d1, "lib", "def f() -> U32:\n    return 1\n")
        _write_module(d2, "lib", "def f() -> U32:\n    return 2\n")
        main = """\
import lib

x: U32 = lib.f()
assert x == 1
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(d1), str(d2)])

    def test_search_order_respects_dir_order(self, fprime_test_api, tmp_path):
        """Reversing the search-dir order flips which module wins."""
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        _write_module(d1, "lib", "def f() -> U32:\n    return 1\n")
        _write_module(d2, "lib", "def f() -> U32:\n    return 2\n")
        main = """\
import lib

x: U32 = lib.f()
assert x == 2
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(d2), str(d1)])

    def test_dotted_module_resolved_across_search_dirs(self, fprime_test_api, tmp_path):
        """Dotted resolution honors the search path: `pkg/mod.fpy` lives only in
        the second dir."""
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        _write_module(d2, "pkg.mod", "def f() -> U32:\n    return 5\n")
        main = """\
import pkg.mod

x: U32 = pkg.mod.f()
assert x == 5
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(d1), str(d2)])

    def test_no_search_dirs_cannot_resolve(self, fprime_test_api, tmp_path):
        """With an empty search path, no import can resolve."""
        _write_module(tmp_path, "lib", "def f() -> U32:\n    return 1\n")
        main = """\
import lib
"""
        assert_compile_failure(fprime_test_api, main, import_search_dirs=[])


class TestImportVariables:
    """A module's top-level variable is both a side effect and a namespaced
    symbol."""

    def test_top_level_variable_is_side_effect_and_namespaced(
        self, fprime_test_api, tmp_path
    ):
        _write_module(
            tmp_path,
            "with_variable",
            """\
counter: U32 = 5

def get() -> U32:
    return counter
""",
        )
        main = """\
import with_variable

x: U32 = with_variable.counter
assert x == 5
assert with_variable.counter == 5
assert with_variable.get() == 5
"""
        # The top-level assignment runs at sequence start -> side effect warning,
        # but `with_variable.counter` still resolves as a namespaced symbol.
        state, _, _ = compile_seq(main, import_search_dirs=[str(tmp_path)])
        assert any(
            w.type == WarningType.IMPORT_SIDE_EFFECTS for w in state.warnings
        ), f"expected an import-side-effects warning, got {state.warnings}"
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])
