"""Tests for the `import` statement (spec / TDD).

`import foo` inlines the definitions of a sibling `foo.fpy` sequence into the
importing sequence and sections its symbols off under a module named `foo`, so
a function `bar` defined in `foo.fpy` is called as `foo.bar()`.

Terminology: a *sequence* is an importable `.fpy` file; a *module* is the
name-to-symbol mapping an import introduces to hold a sequence's symbols.

Design decisions encoded here:
  * Imports come in two disjoint resolution styles (PEP-328-like).  An
    ABSOLUTE import (`import foo`, `import a.b.c`) resolves only against the
    shared base search path `state.import_search_dirs` (the `-i/--include`
    directories); the importing file's own location plays no role, so an
    absolute path names the same file in every sequence of a compilation.  A
    RELATIVE import (leading dots: `import .foo`, `import ..util`,
    `from .foo import f`) resolves only against its anchor -- one dot is the
    importing sequence's own directory, each extra dot one parent up -- and
    never consults the base search path.  So a library directory's sequences
    reference one another relatively and work unmodified wherever the library
    is mounted, while consumers name the library absolutely from anywhere.
    The search dirs are DISTINCT from `ground_binary_dir`, which roots
    runtime sequence-binary (.bin) paths -- imports are a compile-time-only
    source inlining and never survive into the emitted bytecode.
  * An absolute import that resolves in more than one base search dir is
    AMBIGUOUS -- a hard error, even if the candidates agree -- so adding a
    file to one search dir can never silently rebind another sequence's
    import.  There is no shadowing order among the search dirs.
  * Dotted sequence paths resolve through package directories, Pythonically:
    `import a.b.c` -> `a/b/c.fpy`.  Package directories need no
    `__init__.fpy` marker.  The imported symbols live under a module chain,
    reached by the full path: `import a.b.c` makes `a.b.c.bar()` callable
    (and `import foo` makes `foo.bar()` callable).  A relative import binds
    the path after its dots: `import .sub.mod` -> `sub.mod.f()`, and
    `import ..util` -> `util.f()`.
  * Unlike Python, `import .foo` is legal (binding module `foo`) and
    `from . import foo` is not: `from` members are always definitions, never
    sequences.
  * File/directory precedence: at an import's leaf segment a sequence file
    `foo.fpy` outranks a same-named `foo/` directory.  Non-leaf segments must
    be directories -- `import a.b` descends into `a/` to reach `a/b.fpy`
    regardless of any sibling `a.fpy`.  Importing a leaf that resolves only to
    a directory (no sequence file to inline) is an error.
  * `import` is only valid as a top-level statement (not nested in a block),
    but an imported sequence MAY itself import other sequences (transitive
    imports are supported, with cycle detection).
  * Inlining an imported sequence that does more than define functions emits
    the `import-side-effects` warning (its top-level code runs as part of the
    importing sequence).
  * A leading underscore marks a definition as library-internal: an importer
    naming it (`lib._helper()`, `import lib._helper`, `from lib import
    _helper`) emits the `import-underscore` warning.  The library's own
    references to it never warn, and an alias silences later uses.
  * Importing a sequence that declares sequence arguments (`sequence(x: U32)`)
    is a hard error.
  * A module obeys name groups: it exists only in the name groups of the
    symbols the sequence defines.  So `import lib` of a functions-only sequence
    collides with a local function `lib` (callable name group) but coexists
    with a local variable `lib` (value name group).
  * Package modules (non-leaf chain segments) merge when paths overlap
    (`import pkg.a` + `import pkg.b` share `pkg`), but two SEQUENCE modules
    (a module holding a file's definitions, whether chain leaf or alias)
    never merge: binding two different files' modules to one name is a
    collision.
  * Importing the same sequence more than once in one file is a hard error,
    across every form (`import seq`, `import seq.foo`, `from seq import bar`).
    The rule is per-file: a file and a sequence it imports may each import the
    same sequence.

Beyond the plain `import seq` form, this file also covers member imports
(`import seq.func`), aliases (`import seq as x`, `... as y`), and `from` imports
(`from seq import a, b`, `from seq import *`).

"""

from pathlib import Path

from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_success,
    compile_seq,
)
from fpy.error import WarningType
from fpy.types import U32, FpyValue


def _write_sequence(search_dir: Path, dotted_name: str, src: str) -> None:
    """Write an importable sequence for `import <dotted_name>` into *search_dir*.

    A bare name `foo` becomes `<search_dir>/foo.fpy`; a dotted name `a.b.c`
    becomes `<search_dir>/a/b/c.fpy`, creating the intervening package
    directories."""
    rel = Path(*dotted_name.split(".")).with_suffix(".fpy")
    path = search_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src)


class TestImportInlining:
    """An imported function is inlined and callable under the sequence's
    module name."""

    def test_call_imported_function(self, fprime_test_api, tmp_path):
        _write_sequence(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return U32(x + 1)
""",
        )
        main = """\
import lib

result: U32 = lib.add_one(41)
assert result == 42
"""
        # Funcs-only sequence: compiles cleanly with no side-effect warning...
        state, _, _ = compile_seq(main, import_search_dirs=[str(tmp_path)])
        assert state.warnings == []
        # ...and the embedded assert holds at run time.
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_local_and_imported_names_coexist(self, fprime_test_api, tmp_path):
        """A local `helper` and an imported `lib.helper` must not collide --
        the imported symbols are sectioned off under `lib`."""
        _write_sequence(
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
    """An imported sequence may contain only function definitions and imports."""

    SIDE_EFFECT_SEQUENCE = """\
CdhCore.cmdDisp.CMD_NO_OP()

def noop_wrapper():
    CdhCore.cmdDisp.CMD_NO_OP()
"""

    def test_side_effecting_import_is_error(self, fprime_test_api, tmp_path):
        _write_sequence(tmp_path, "side_effects", self.SIDE_EFFECT_SEQUENCE)
        main = """\
import side_effects

side_effects.noop_wrapper()
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="only function definitions and imports",
            import_search_dirs=[str(tmp_path)],
        )

    def test_functions_only_sequence_compiles(self, fprime_test_api, tmp_path):
        _write_sequence(
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

x: U32 = U32(clean.a() + clean.b())
assert x == 3
"""
        state, _, _ = compile_seq(main, import_search_dirs=[str(tmp_path)])
        assert state.warnings == []
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])


class TestImportUnderscore:
    """A leading underscore marks a definition as internal to its sequence:
    the importer naming it emits the `import-underscore` warning; the
    library's own references to it do not."""

    LIB = """\
def _helper() -> U32:
    return 7

def public() -> U32:
    return _helper()
"""

    def test_underscore_module_access_warns(self, fprime_test_api, tmp_path):
        """`lib._helper()` names an imported underscore definition: warn."""
        _write_sequence(tmp_path, "lib", self.LIB)
        main = """\
import lib

x: U32 = lib._helper()
assert x == 7
"""
        expected = {WarningType.IMPORT_UNDERSCORE}
        state, _, _ = compile_seq(
            main, import_search_dirs=[str(tmp_path)], expected_warnings=expected
        )
        assert any(
            w.type == WarningType.IMPORT_UNDERSCORE for w in state.warnings
        ), f"expected an import-underscore warning, got {state.warnings}"
        # The warning is non-fatal: the sequence still compiles and runs.
        assert_run_success(
            fprime_test_api,
            main,
            import_search_dirs=[str(tmp_path)],
            expected_warnings=expected,
        )

    def test_underscore_member_import_warns(self, fprime_test_api, tmp_path):
        """`import lib._helper` names the underscore definition as a member."""
        _write_sequence(tmp_path, "lib", self.LIB)
        main = """\
import lib._helper

x: U32 = lib._helper()
assert x == 7
"""
        expected = {WarningType.IMPORT_UNDERSCORE}
        state, _, _ = compile_seq(
            main, import_search_dirs=[str(tmp_path)], expected_warnings=expected
        )
        # FIXME redundant
        assert any(
            w.type == WarningType.IMPORT_UNDERSCORE for w in state.warnings
        ), f"expected an import-underscore warning, got {state.warnings}"
        assert_run_success(
            fprime_test_api,
            main,
            import_search_dirs=[str(tmp_path)],
            expected_warnings=expected,
        )

    def test_underscore_from_import_warns(self, fprime_test_api, tmp_path):
        """`from lib import _helper` names the underscore definition."""
        _write_sequence(tmp_path, "lib", self.LIB)
        main = """\
from lib import _helper

x: U32 = _helper()
assert x == 7
"""
        expected = {WarningType.IMPORT_UNDERSCORE}
        state, _, _ = compile_seq(
            main, import_search_dirs=[str(tmp_path)], expected_warnings=expected
        )
        assert any(
            w.type == WarningType.IMPORT_UNDERSCORE for w in state.warnings
        ), f"expected an import-underscore warning, got {state.warnings}"
        assert_run_success(
            fprime_test_api,
            main,
            import_search_dirs=[str(tmp_path)],
            expected_warnings=expected,
        )

    def test_star_import_underscore_use_warns(self, fprime_test_api, tmp_path):
        """`from lib import *` binds `_helper` without naming it; the later
        bare `_helper()` use is what warns."""
        _write_sequence(tmp_path, "lib", self.LIB)
        main = """\
from lib import *

x: U32 = _helper()
assert x == 7
"""
        expected = {WarningType.IMPORT_UNDERSCORE}
        state, _, _ = compile_seq(
            main, import_search_dirs=[str(tmp_path)], expected_warnings=expected
        )
        assert any(
            w.type == WarningType.IMPORT_UNDERSCORE for w in state.warnings
        ), f"expected an import-underscore warning, got {state.warnings}"
        assert_run_success(
            fprime_test_api,
            main,
            import_search_dirs=[str(tmp_path)],
            expected_warnings=expected,
        )

    def test_library_internal_use_does_not_warn(self, fprime_test_api, tmp_path):
        """`lib.public()` internally calls `_helper`; the importer never names
        an underscore definition, so nothing warns."""
        _write_sequence(tmp_path, "lib", self.LIB)
        main = """\
import lib

x: U32 = lib.public()
assert x == 7
"""
        state, _, _ = compile_seq(main, import_search_dirs=[str(tmp_path)])
        assert state.warnings == []
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_underscore_alias_statement_warns_but_uses_do_not(
        self, fprime_test_api, tmp_path
    ):
        """`from lib import _helper as helper` warns once, at the statement;
        uses of the alias `helper` add nothing."""
        _write_sequence(tmp_path, "lib", self.LIB)
        main = """\
from lib import _helper as helper

x: U32 = helper()
y: U32 = helper()
assert x == 7
assert y == 7
"""
        expected = {WarningType.IMPORT_UNDERSCORE}
        state, _, _ = compile_seq(
            main, import_search_dirs=[str(tmp_path)], expected_warnings=expected
        )
        underscore_warnings = [
            w for w in state.warnings if w.type == WarningType.IMPORT_UNDERSCORE
        ]
        assert (
            len(underscore_warnings) == 1
        ), f"expected exactly one import-underscore warning, got {state.warnings}"
        assert_run_success(
            fprime_test_api,
            main,
            import_search_dirs=[str(tmp_path)],
            expected_warnings=expected,
        )


class TestImportErrors:
    """Error cases for import."""

    def test_cannot_import_sequence_with_arguments(self, fprime_test_api, tmp_path):
        _write_sequence(
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

    def test_missing_sequence_is_an_error(self, fprime_test_api, tmp_path):
        main = """\
import does_not_exist
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_no_arg_sequence_is_importable(self, fprime_test_api, tmp_path):
        """A bare `sequence()` with no arguments is importable -- only
        sequences *with arguments* are rejected (per the feature's wording)."""
        _write_sequence(
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


class TestImporterWithArguments:
    """Importing a sequence WITH arguments is rejected, but the *importer*
    declaring its own sequence arguments is fine: it may both take arguments and
    import another (argument-less) sequence, and use both together."""

    def test_importer_declares_args_and_imports(self, fprime_test_api, tmp_path):
        _write_sequence(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return U32(x + 1)
""",
        )
        main = """\
sequence(n: U32)

import lib

result: U32 = lib.add_one(n)
assert result == 42
"""
        state, _, _ = compile_seq(main, import_search_dirs=[str(tmp_path)])
        assert state.warnings == []
        assert_run_success(
            fprime_test_api,
            main,
            args=[FpyValue(U32, 41)],
            import_search_dirs=[str(tmp_path)],
        )


class TestImportBuiltinLibrary:
    """The builtin library functions (`time_add`, `time_sub`,
    `time_interval_cmp`, ...) register into the shared base callable scope that
    every sequence's scope descends from, so an imported sequence may reference
    them too -- whether written explicitly or produced by desugaring a
    `check`/timeout statement.

    Found by a one-time import-inlining sweep."""

    def test_imported_sequence_can_use_time_builtin(self, fprime_test_api, tmp_path):
        """A `check ... timeout` in an imported sequence desugars into a
        `time_add`/`time_interval_cmp` call that must resolve."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def wait_ok() -> bool:
    check True timeout Fw.TimeIntervalValue(1, 0) persist Fw.TimeIntervalValue(0, 0) period Fw.TimeIntervalValue(0, 100000):
        return True
    timeout:
        return False
    return False
""",
        )
        main = """\
import lib

ok: bool = lib.wait_ok()
assert ok
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_imported_sequence_can_use_time_operators(self, fprime_test_api, tmp_path):
        """Time operators inside an imported sequence desugar to builtin calls
        (`time_sub`, `time_interval_cmp`) that must resolve there too."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def is_past(deadline: Fw.Time) -> bool:
    return (now() - deadline) > Fw.TimeIntervalValue(0, 0)
""",
        )
        main = """\
import lib

past: bool = lib.is_past(Fw.Time(TimeBase.TB_NONE, 0, 0, 0))
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])


class TestImportFileErrors:
    """Failure modes rooted in the imported file itself."""

    def test_parse_error_in_imported_file_fails(self, fprime_test_api, tmp_path):
        """A syntax error inside the imported file is a hard compile error.
        Ideally the diagnostic points into the imported file, not the importer."""
        _write_sequence(
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

    def test_empty_sequence_compiles_without_warning(self, fprime_test_api, tmp_path):
        """An empty module has no definitions and no side effects."""
        _write_sequence(tmp_path, "empty", "")
        main = """\
import empty

CdhCore.cmdDisp.CMD_NO_OP()
"""
        state, _, _ = compile_seq(main, import_search_dirs=[str(tmp_path)])
        assert state.warnings == []
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])


class TestImportModuleIsolation:
    """Imported symbols are sectioned under the module name and isolated."""

    def test_imported_symbol_requires_module_prefix(self, fprime_test_api, tmp_path):
        """`add_one` is only reachable as `lib.add_one`, never bare."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return U32(x + 1)
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
        """A module name resolves to a module, not a value, so it is not usable as an expression."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return U32(x + 1)
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
        _write_sequence(
            tmp_path,
            "lib_a",
            """\
def helper() -> U32:
    return 1
""",
        )
        _write_sequence(
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
        _write_sequence(
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
    """An imported module collides with existing top-level names per name
    group: the module exists only in the name groups of the symbols the
    sequence defines."""

    def test_import_collides_with_local_function(self, fprime_test_api, tmp_path):
        """A functions-only module occupies the callable name group, so a
        local function with the module's name collides."""
        _write_sequence(
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
        _write_sequence(
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
        _write_sequence(
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


class TestImportedSequenceShadowingBuiltins:
    """An imported sequence's own scope layers on top of the shared dictionary /
    builtin base scope. Binding a name already taken by that base (a cast like
    `U32`, a type constructor, a dictionary namespace) does not collide -- the
    base name lives in an ENCLOSING scope, so it is shadowed, with a warning,
    like any other shadow. A same-scope collision or a side effect stays a hard
    error. These also guard the child-of-base scope shape: a child scope resolves
    base names up its parent chain, so the shadow must be detected there (not
    only in the importer's own dict)."""

    def test_nested_import_alias_shadowing_builtin_cast_warns(
        self, fprime_test_api, tmp_path
    ):
        """An IMPORTED sequence that binds an alias matching a builtin base name
        (`U32`) shadows it. The importer's scope is a child of base, so the
        shadow is surfaced up the parent chain, not just in the importer's own
        dict -- and it warns rather than erroring."""
        _write_sequence(
            tmp_path,
            "inner",
            """\
def helper() -> U32:
    return U32(5)
""",
        )
        _write_sequence(
            tmp_path,
            "outer",
            """\
from inner import helper as U32
""",
        )
        main = "import outer\nexit(0)\n"
        assert_run_success(
            fprime_test_api,
            main,
            import_search_dirs=[str(tmp_path)],
            expected_warnings={WarningType.SHADOW_CALLABLE},
        )

    def test_imported_sequence_cannot_declare_top_level_flags(
        self, fprime_test_api, tmp_path
    ):
        """An imported sequence can not declare a top-level `flags` (or any
        other top-level variable): a top-level assignment is a side effect, so it
        is rejected outright -- a side effect stays an error, it is not a shadow
        warning."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
flags: U32 = 5
""",
        )
        main = "import lib\n"
        assert_compile_failure(
            fprime_test_api,
            main,
            match="only function definitions and imports",
            import_search_dirs=[str(tmp_path)],
        )

    def test_nested_import_of_module_named_after_dictionary_namespace_warns(
        self, fprime_test_api, tmp_path
    ):
        """An imported sequence that itself imports a module whose name matches
        a dictionary namespace (`Ref`) shadows that base namespace. Bound through
        `_bind_module`, so the shadow is detected up the parent chain. Allowed
        with a warning (it does change what `Ref.*` names in that sequence)."""
        _write_sequence(
            tmp_path,
            "Ref",
            """\
def f() -> U32:
    return U32(1)
""",
        )
        _write_sequence(
            tmp_path,
            "outer",
            """\
import Ref
""",
        )
        main = "import outer\nexit(0)\n"
        assert_run_success(
            fprime_test_api,
            main,
            import_search_dirs=[str(tmp_path)],
            expected_warnings={WarningType.SHADOW_CALLABLE},
        )


class TestImportShadowsBuiltins:
    """A main-sequence import that binds a name taken only by the builtin /
    dictionary base scope shadows it -- a warning categorized by the bound name's
    name group. Imports of function-only sequences occupy the callable group, so
    they warn as `shadow-callable`, and still compile and run. A same-scope
    collision stays an error."""

    def test_import_module_shadowing_builtin_warns(self, fprime_test_api, tmp_path):
        # A sequence file named after the builtin library function `time_add`;
        # `import time_add` binds module `time_add` over the builtin callable.
        _write_sequence(
            tmp_path,
            "time_add",
            """\
def f() -> U32:
    return U32(1)
""",
        )
        # expected_warnings both requires shadow-callable and (by promoting
        # anything else) asserts it is not miscategorized as shadow-value.
        main = "import time_add\nexit(0)\n"
        assert_run_success(
            fprime_test_api,
            main,
            import_search_dirs=[str(tmp_path)],
            expected_warnings={WarningType.SHADOW_CALLABLE},
        )

    def test_from_import_alias_shadowing_builtin_warns(self, fprime_test_api, tmp_path):
        # `... as time_cmp` binds the imported function under a builtin's name.
        _write_sequence(
            tmp_path,
            "lib",
            """\
def public() -> U32:
    return U32(7)
""",
        )
        main = "from lib import public as time_cmp\nexit(0)\n"
        assert_run_success(
            fprime_test_api,
            main,
            import_search_dirs=[str(tmp_path)],
            expected_warnings={WarningType.SHADOW_CALLABLE},
        )

    def test_imported_sequence_function_shadowing_builtin_warns(
        self, fprime_test_api, tmp_path
    ):
        # A function defined INSIDE an imported sequence that shadows a builtin
        # cast (`U32`) warns rather than erroring.
        _write_sequence(
            tmp_path,
            "lib",
            """\
def U32() -> U32:
    return 0
""",
        )
        main = "import lib\nexit(0)\n"
        assert_run_success(
            fprime_test_api,
            main,
            import_search_dirs=[str(tmp_path)],
            expected_warnings={WarningType.SHADOW_CALLABLE},
        )

    def test_import_over_local_definition_is_error(self, fprime_test_api, tmp_path):
        # Binding an import over a name defined in the SAME (importer's) scope is
        # a same-scope collision -- it stays a hard error, not a shadow warning.
        _write_sequence(
            tmp_path,
            "lib",
            """\
def public() -> U32:
    return U32(7)
""",
        )
        main = """\
def public() -> U32:
    return U32(1)
from lib import public
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="collides with an existing definition",
            import_search_dirs=[str(tmp_path)],
        )


class TestImportDuplicates:
    """A sequence is compiled once however many import statements name it, and
    importing it more than once in one file is governed by the ordinary
    collision rule: two `import lib` statements bind the module `lib` twice and
    collide."""

    def test_import_same_sequence_twice_collides(self, fprime_test_api, tmp_path):
        _write_sequence(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return U32(x + 1)
""",
        )
        main = """\
import lib
import lib
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="collides with an existing imported sequence",
            import_search_dirs=[str(tmp_path)],
        )

    def test_duplicate_across_files_is_allowed(self, fprime_test_api, tmp_path):
        """main imports both `a` and `c`; `a` also imports `c` internally. `c`
        is imported from two places but compiled once and shared, not
        duplicated."""
        _write_sequence(tmp_path, "c", "def g() -> U32:\n    return 7\n")
        _write_sequence(
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

x: U32 = U32(a.f() + c.g())
assert x == 14
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])


class TestImportOnlyAtTopLevel:
    """`import` is only valid as a top-level statement (never nested in a
    block).  Note: an imported *sequence* may still contain its own top-level
    imports -- see TestImportTransitive."""

    def test_import_inside_if_block_fails(self, fprime_test_api, tmp_path):
        _write_sequence(
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
        _write_sequence(
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
    """An imported sequence may itself import other sequences."""

    def test_transitive_import_works(self, fprime_test_api, tmp_path):
        """main -> a -> b: `a` uses `b` internally, and main runs `a.f()`."""
        _write_sequence(
            tmp_path,
            "b",
            """\
def g() -> U32:
    return 7
""",
        )
        _write_sequence(
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

        _write_sequence(
            tmp_path,
            "b",
            """\
def g() -> U32:
    return 7
""",
        )
        _write_sequence(
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
        _write_sequence(
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
        _write_sequence(
            tmp_path,
            "mod_a",
            """\
import mod_b

def a() -> U32:
    return mod_b.b()
""",
        )
        _write_sequence(
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
        _write_sequence(tmp_path, "c1", "import c2\n\ndef f() -> U32:\n    return 1\n")
        _write_sequence(tmp_path, "c2", "import c3\n\ndef f() -> U32:\n    return 1\n")
        _write_sequence(tmp_path, "c3", "import c1\n\ndef f() -> U32:\n    return 1\n")
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
    """Dotted sequence paths resolve through package directories, Pythonically."""

    def test_single_dotted_import(self, fprime_test_api, tmp_path):
        """`import pkg.mod` resolves `pkg/mod.fpy` and binds `pkg.mod`."""
        _write_sequence(
            tmp_path,
            "pkg.mod",
            """\
def add_one(x: U32) -> U32:
    return U32(x + 1)
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
        _write_sequence(
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
        _write_sequence(
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
        """The package dir exists but the leaf sequence file does not."""
        _write_sequence(
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

    def test_two_sequences_in_same_package_no_collision(
        self, fprime_test_api, tmp_path
    ):
        """Sibling sequences under one package get independent modules."""
        _write_sequence(tmp_path, "pkg.a", "def f() -> U32:\n    return 1\n")
        _write_sequence(tmp_path, "pkg.b", "def f() -> U32:\n    return 2\n")
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
    """File-vs-directory precedence: a sequence file outranks a same-named
    (init-less) directory at a leaf, but non-leaf segments always descend into
    the directory."""

    def test_sequence_file_beats_directory(self, fprime_test_api, tmp_path):
        """`foo.fpy` and a `foo/` directory both exist; `import foo` resolves
        the sequence file (a directory ranks below a sequence file)."""
        _write_sequence(tmp_path, "foo", "def f() -> U32:\n    return 1\n")
        # This also creates the sibling `foo/` directory:
        _write_sequence(tmp_path, "foo.inner", "def g() -> U32:\n    return 2\n")
        main = """\
import foo

x: U32 = foo.f()
assert x == 1
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_package_dir_used_for_dotted_descent(self, fprime_test_api, tmp_path):
        """A `pkg.fpy` module does not block `import pkg.mod` from descending
        into the `pkg/` directory to reach `pkg/mod.fpy`."""
        _write_sequence(tmp_path, "pkg", "def top() -> U32:\n    return 1\n")
        _write_sequence(tmp_path, "pkg.mod", "def f() -> U32:\n    return 5\n")
        main = """\
import pkg.mod

x: U32 = pkg.mod.f()
assert x == 5
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_bare_package_import_is_error(self, fprime_test_api, tmp_path):
        """`import pkg` where only a `pkg/` directory exists (no `pkg.fpy`) is
        an error -- a directory has no sequence file to inline."""
        # Creates `pkg/mod.fpy`, so `pkg/` exists as a directory but `pkg.fpy`
        # does not.
        _write_sequence(tmp_path, "pkg.mod", "def f() -> U32:\n    return 1\n")
        main = """\
import pkg
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_dotted_leaf_package_import_is_error(self, fprime_test_api, tmp_path):
        """`import a.b` where `a/b/` is a directory but `a/b.fpy` does not exist
        is likewise an error at the dotted leaf."""
        _write_sequence(tmp_path, "a.b.c", "def f() -> U32:\n    return 1\n")
        main = """\
import a.b
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )


class TestImportSearchDirs:
    """`import_search_dirs` holds the candidate directories for absolute
    imports: an import must resolve in exactly one of them.  Resolving in
    none is a missing-sequence error; resolving in more than one is an
    ambiguity error.  There is no shadowing order."""

    def test_sequence_found_in_later_search_dir(self, fprime_test_api, tmp_path):
        """A module present in only one search dir is found, wherever that
        dir sits in the list."""
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        _write_sequence(d2, "lib", "def f() -> U32:\n    return 9\n")
        main = """\
import lib

x: U32 = lib.f()
assert x == 9
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(d1), str(d2)])

    def test_same_name_in_two_dirs_is_ambiguous(self, fprime_test_api, tmp_path):
        """A module name that resolves in two search dirs is ambiguous and
        fails, even though either candidate alone would compile."""
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        _write_sequence(d1, "lib", "def f() -> U32:\n    return 1\n")
        _write_sequence(d2, "lib", "def f() -> U32:\n    return 2\n")
        main = """\
import lib

x: U32 = lib.f()
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="(?i)ambig",
            import_search_dirs=[str(d1), str(d2)],
        )

    def test_whole_vs_member_across_dirs_is_ambiguous(self, fprime_test_api, tmp_path):
        """Whole-path preference applies only within a single directory.  A
        member split in one dir and a whole-path split in another are two
        candidate directories with splits: ambiguous."""
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        # d1 can split `import a.b` as sequence `a` plus member `b`...
        _write_sequence(d1, "a", "def b() -> U32:\n    return 1\n")
        # ...while d2 resolves it whole as the sequence `a.b`.
        _write_sequence(d2, "a.b", "def f() -> U32:\n    return 2\n")
        main = """\
import a.b
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="(?i)ambig",
            import_search_dirs=[str(d1), str(d2)],
        )

    def test_dotted_sequence_resolved_across_search_dirs(
        self, fprime_test_api, tmp_path
    ):
        """Dotted resolution tries every search dir: `pkg/mod.fpy` lives only
        in the second dir."""
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        _write_sequence(d2, "pkg.mod", "def f() -> U32:\n    return 5\n")
        main = """\
import pkg.mod

x: U32 = pkg.mod.f()
assert x == 5
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(d1), str(d2)])

    def test_no_search_dirs_cannot_resolve(self, fprime_test_api, tmp_path):
        """With an empty search path, no import can resolve."""
        _write_sequence(tmp_path, "lib", "def f() -> U32:\n    return 1\n")
        main = """\
import lib
"""
        assert_compile_failure(fprime_test_api, main, import_search_dirs=[])


class TestImportVariables:
    """A sequence's top-level variable is a side effect (its assignment runs at
    sequence start), so an imported sequence may not declare one -- there is no
    module-level variable, only functions."""

    def test_top_level_variable_is_error(self, fprime_test_api, tmp_path):
        _write_sequence(
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

x: U32 = with_variable.get()
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="only function definitions and imports",
            import_search_dirs=[str(tmp_path)],
        )


class TestImportMember:
    """`import seq.member` imports a single symbol of sequence `seq`, reachable
    under the full dotted name `seq.member` and no shorter name.  The sequence
    path is the longest prefix that resolves to a file; the remaining suffix is
    the member."""

    def test_import_member_function(self, fprime_test_api, tmp_path):
        """`import lib.add_one` exposes only `lib.add_one`, callable under its
        full path."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return U32(x + 1)
""",
        )
        main = """\
import lib.add_one

result: U32 = lib.add_one(41)
assert result == 42
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_import_member_of_dotted_sequence(self, fprime_test_api, tmp_path):
        """The sequence path may itself be dotted: `import pkg.mod.f` splits
        into sequence `pkg.mod` and member `f`."""
        _write_sequence(
            tmp_path,
            "pkg.mod",
            """\
def f() -> U32:
    return 5
""",
        )
        main = """\
import pkg.mod.f

x: U32 = pkg.mod.f()
assert x == 5
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_member_requires_full_path(self, fprime_test_api, tmp_path):
        """A member import is reachable only under its full path, never as a
        bare name."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return U32(x + 1)
""",
        )
        main = """\
import lib.add_one

result: U32 = add_one(41)
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_member_import_hides_other_symbols(self, fprime_test_api, tmp_path):
        """`import lib.a` imports only `a`; the sibling `b` is not bound, even
        though the whole sequence is inlined at execution."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1

def b() -> U32:
    return 2
""",
        )
        main = """\
import lib.a

y: U32 = lib.b()
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_missing_member_is_error(self, fprime_test_api, tmp_path):
        """`import lib.nope` where `lib` has no symbol `nope` is an error."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1
""",
        )
        main = """\
import lib.nope
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )


class TestImportAlias:
    """An `as` clause introduces no module chain: the alias is bound to the
    leaf of the import -- the imported sequence's module (empty member path) or
    the imported symbol (non-empty member path)."""

    def test_import_sequence_as_alias(self, fprime_test_api, tmp_path):
        """`import lib as L` binds `L` to `lib`'s module; members are reached as
        `L.member`."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return U32(x + 1)
""",
        )
        main = """\
import lib as L

result: U32 = L.add_one(41)
assert result == 42
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_dotted_import_as_alias(self, fprime_test_api, tmp_path):
        """`import pkg.mod as m` binds `m` to the leaf module `mod`."""
        _write_sequence(
            tmp_path,
            "pkg.mod",
            """\
def f() -> U32:
    return 5
""",
        )
        main = """\
import pkg.mod as m

x: U32 = m.f()
assert x == 5
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_import_member_as_alias(self, fprime_test_api, tmp_path):
        """`import lib.add_one as inc` binds `inc` directly to the symbol."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return U32(x + 1)
""",
        )
        main = """\
import lib.add_one as inc

result: U32 = inc(41)
assert result == 42
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_alias_hides_chain(self, fprime_test_api, tmp_path):
        """With `import pkg.mod as m`, the chain `pkg.mod` is not introduced;
        only `m` is bound."""
        _write_sequence(
            tmp_path,
            "pkg.mod",
            """\
def f() -> U32:
    return 5
""",
        )
        main = """\
import pkg.mod as m

x: U32 = pkg.mod.f()
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_alias_collides_with_local(self, fprime_test_api, tmp_path):
        """An alias is an ordinary symbol: it collides with a local symbol in
        the same name group."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def f() -> U32:
    return 1
""",
        )
        main = """\
import lib as dup

def dup() -> U32:
    return 2
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )


class TestImportFrom:
    """A `from` statement binds its imported symbols directly in the importing
    sequence's global scope, introducing no module chain."""

    def test_from_import_function(self, fprime_test_api, tmp_path):
        """`from lib import add_one` binds the bare name `add_one`."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return U32(x + 1)
""",
        )
        main = """\
from lib import add_one

result: U32 = add_one(41)
assert result == 42
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_from_dotted_sequence(self, fprime_test_api, tmp_path):
        """The sequence path of a `from` may be dotted."""
        _write_sequence(
            tmp_path,
            "pkg.mod",
            """\
def f() -> U32:
    return 5
""",
        )
        main = """\
from pkg.mod import f

x: U32 = f()
assert x == 5
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_from_import_as_alias(self, fprime_test_api, tmp_path):
        """`from lib import add_one as inc` binds `inc`."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return U32(x + 1)
""",
        )
        main = """\
from lib import add_one as inc

result: U32 = inc(41)
assert result == 42
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_from_import_star(self, fprime_test_api, tmp_path):
        """`from lib import *` binds every top-level symbol of `lib` under its
        own name."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1

def b() -> U32:
    return 2
""",
        )
        main = """\
from lib import *

x: U32 = U32(a() + b())
assert x == 3
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_from_import_multiple_members(self, fprime_test_api, tmp_path):
        """`from lib import a, b` binds each member under its own name."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1

def b() -> U32:
    return 2
""",
        )
        main = """\
from lib import a, b

x: U32 = U32(a() + b())
assert x == 3
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_from_import_multiple_with_aliases(self, fprime_test_api, tmp_path):
        """Each member in a list may carry its own `as` alias."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1

def b() -> U32:
    return 2
""",
        )
        main = """\
from lib import a as first, b as second

x: U32 = U32(first() + second())
assert x == 3
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_from_import_parenthesized_single_line(self, fprime_test_api, tmp_path):
        """The member list may be parenthesized."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1

def b() -> U32:
    return 2
""",
        )
        main = """\
from lib import (a, b)

x: U32 = U32(a() + b())
assert x == 3
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_from_import_parenthesized_multiline(self, fprime_test_api, tmp_path):
        """The parenthesized member list may span multiple lines and end with a
        trailing comma."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1

def b() -> U32:
    return 2

def c() -> U32:
    return 3
""",
        )
        main = """\
from lib import (
    a,
    b as bb,
    c,
)

x: U32 = U32(a() + bb() + c())
assert x == 6
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_from_import_duplicate_member_is_error(self, fprime_test_api, tmp_path):
        """Binding the same name twice in one member list is a collision."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1
""",
        )
        main = """\
from lib import a, a
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_from_import_does_not_introduce_sequence_name(
        self, fprime_test_api, tmp_path
    ):
        """`from lib import add_one` introduces no module chain, so `lib.add_one`
        is not reachable."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def add_one(x: U32) -> U32:
    return U32(x + 1)
""",
        )
        main = """\
from lib import add_one

result: U32 = lib.add_one(41)
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_from_import_missing_member_is_error(self, fprime_test_api, tmp_path):
        """`from lib import nope` where `lib` has no `nope` is an error."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1
""",
        )
        main = """\
from lib import nope
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_from_import_collides_with_local(self, fprime_test_api, tmp_path):
        """A `from`-imported name is ordinary and collides with a local symbol
        in the same name group."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def f() -> U32:
    return 1
""",
        )
        main = """\
from lib import f

def f() -> U32:
    return 2
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_from_star_collides_across_sequences(self, fprime_test_api, tmp_path):
        """Two `from ... import *` statements that both define `f` collide;
        unlike module chains, these names do not merge."""
        _write_sequence(tmp_path, "a", "def f() -> U32:\n    return 1\n")
        _write_sequence(tmp_path, "b", "def f() -> U32:\n    return 2\n")
        main = """\
from a import *
from b import *
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_from_import_side_effects_is_error(self, fprime_test_api, tmp_path):
        """A `from` import brings in the whole sequence, so a side-effecting
        imported sequence is an error just as a plain import of it would be."""
        _write_sequence(
            tmp_path,
            "side_effects",
            """\
CdhCore.cmdDisp.CMD_NO_OP()

def noop_wrapper():
    CdhCore.cmdDisp.CMD_NO_OP()
""",
        )
        main = """\
from side_effects import noop_wrapper

noop_wrapper()
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="only function definitions and imports",
            import_search_dirs=[str(tmp_path)],
        )


class TestImportDuplicateSequence:
    """A sequence may be imported more than once in one file. Whether that is
    allowed follows from the collision rule on the names each import binds, and
    the sequence is compiled just once regardless."""

    def test_import_and_from_same_sequence_coexist(self, fprime_test_api, tmp_path):
        """`import lib` binds the module `lib`; `from lib import b` binds `b`.
        The names differ, so they coexist."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1

def b() -> U32:
    return 2
""",
        )
        main = """\
import lib
from lib import b

x: U32 = U32(lib.a() + b())
assert x == 3
"""
        state, _, _ = compile_seq(main, import_search_dirs=[str(tmp_path)])
        assert state.warnings == []
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_two_members_same_sequence_collides(self, fprime_test_api, tmp_path):
        """`import lib.a` and `import lib.b` both bind the sequence module `lib`,
        which collides."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1

def b() -> U32:
    return 2
""",
        )
        main = """\
import lib.a
import lib.b
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="collides with an existing imported sequence",
            import_search_dirs=[str(tmp_path)],
        )

    def test_whole_and_member_same_sequence_collides(self, fprime_test_api, tmp_path):
        """`import lib` and `import lib.a` both bind the sequence module `lib`,
        which collides."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1
""",
        )
        main = """\
import lib
import lib.a
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="collides with an existing imported sequence",
            import_search_dirs=[str(tmp_path)],
        )

    def test_two_from_same_sequence_coexist(self, fprime_test_api, tmp_path):
        """Two `from lib import ...` statements that name distinct members bind
        distinct names, so they coexist (and `lib` is compiled once)."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1

def b() -> U32:
    return 2
""",
        )
        main = """\
from lib import a
from lib import b

x: U32 = U32(a() + b())
assert x == 3
"""
        state, _, _ = compile_seq(main, import_search_dirs=[str(tmp_path)])
        assert state.warnings == []
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])


class TestImportRelative:
    """Leading dots make an import relative: one dot anchors at the importing
    sequence's own directory, each extra dot one parent up, and the base
    search path is never consulted.  Absolute imports, conversely, never see
    the importing sequence's own directory.  The dots affect resolution only;
    the bound module chain is the path after them."""

    def test_relative_import_of_sibling(self, fprime_test_api, tmp_path):
        """`pkg/helper.fpy` imports its sibling `pkg/sibling.fpy` with
        `import .sibling`; `pkg/` is not on the base search path, only the
        anchor makes it reachable."""
        _write_sequence(tmp_path, "pkg.sibling", "def thing() -> U32:\n    return 9\n")
        _write_sequence(
            tmp_path,
            "pkg.helper",
            """\
import .sibling

def f() -> U32:
    return sibling.thing()
""",
        )
        main = """\
import pkg.helper

x: U32 = pkg.helper.f()
assert x == 9
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_absolute_import_ignores_importer_directory(
        self, fprime_test_api, tmp_path
    ):
        """An absolute `import util` sees only the base search path: a
        same-named file sitting next to the importing sequence must not
        shadow it (no Python-2-style implicit relative imports)."""
        _write_sequence(tmp_path, "util", "def v() -> U32:\n    return 1\n")
        _write_sequence(tmp_path, "pkg.util", "def v() -> U32:\n    return 2\n")
        _write_sequence(
            tmp_path,
            "pkg.helper",
            """\
import util

def f() -> U32:
    return util.v()
""",
        )
        main = """\
import pkg.helper

x: U32 = pkg.helper.f()
assert x == 1
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_relative_import_does_not_search_base_path(self, fprime_test_api, tmp_path):
        """`import .nope` must not fall back to the base search path, even
        when a `nope.fpy` exists there."""
        _write_sequence(tmp_path, "nope", "def f() -> U32:\n    return 1\n")
        _write_sequence(
            tmp_path,
            "pkg.helper",
            """\
import .nope

def f() -> U32:
    return nope.f()
""",
        )
        main = """\
import pkg.helper
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_relative_binds_path_after_dots(self, fprime_test_api, tmp_path):
        """`import .sub.mod` resolves `<anchor>/sub/mod.fpy` and binds the
        chain `sub.mod`: the dots are not part of the name."""
        _write_sequence(tmp_path, "lib.sub.mod", "def f() -> U32:\n    return 3\n")
        _write_sequence(
            tmp_path,
            "lib.helper",
            """\
import .sub.mod

def f() -> U32:
    return sub.mod.f()
""",
        )
        main = """\
import lib.helper

x: U32 = lib.helper.f()
assert x == 3
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_parent_relative_import(self, fprime_test_api, tmp_path):
        """`import ..util` in `lib/sub/inner.fpy` anchors one parent up,
        resolving `lib/util.fpy` and binding module `util`."""
        _write_sequence(tmp_path, "lib.util", "def v() -> U32:\n    return 4\n")
        _write_sequence(
            tmp_path,
            "lib.sub.inner",
            """\
import ..util

def f() -> U32:
    return util.v()
""",
        )
        main = """\
import lib.sub.inner

x: U32 = lib.sub.inner.f()
assert x == 4
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_relative_member_import(self, fprime_test_api, tmp_path):
        """Member splitting applies to relative imports too: `import
        .sibling.thing` splits into sequence `.sibling` and member `thing`."""
        _write_sequence(tmp_path, "pkg.sibling", "def thing() -> U32:\n    return 9\n")
        _write_sequence(
            tmp_path,
            "pkg.helper",
            """\
import .sibling.thing

def f() -> U32:
    return sibling.thing()
""",
        )
        main = """\
import pkg.helper

x: U32 = pkg.helper.f()
assert x == 9
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_relative_alias(self, fprime_test_api, tmp_path):
        """`import .sibling as s` binds only `s`, as with absolute aliases."""
        _write_sequence(tmp_path, "pkg.sibling", "def thing() -> U32:\n    return 9\n")
        _write_sequence(
            tmp_path,
            "pkg.helper",
            """\
import .sibling as s

def f() -> U32:
    return s.thing()
""",
        )
        main = """\
import pkg.helper

x: U32 = pkg.helper.f()
assert x == 9
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_relative_from_import(self, fprime_test_api, tmp_path):
        """`from .sibling import thing` binds the bare member name."""
        _write_sequence(tmp_path, "pkg.sibling", "def thing() -> U32:\n    return 9\n")
        _write_sequence(
            tmp_path,
            "pkg.helper",
            """\
from .sibling import thing

def f() -> U32:
    return thing()
""",
        )
        main = """\
import pkg.helper

x: U32 = pkg.helper.f()
assert x == 9
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_bare_dot_from_is_error(self, fprime_test_api, tmp_path):
        """`from . import sibling` is invalid: the leading dots must be
        followed by at least one name (`from` members are definitions, not
        sequences).  Write `import .sibling` instead."""
        _write_sequence(tmp_path, "sibling", "def f() -> U32:\n    return 1\n")
        main = """\
from . import sibling
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            import_search_dirs=[str(tmp_path)],
            main_file_dir=str(tmp_path),
        )

    def test_relative_import_in_main(self, fprime_test_api, tmp_path):
        """The main sequence's own relative imports anchor at its directory
        (`main_file_dir` here; the input file's directory in the CLI).  An
        empty base search path proves it plays no role."""
        _write_sequence(tmp_path, "util", "def v() -> U32:\n    return 6\n")
        main = """\
import .util

x: U32 = util.v()
assert x == 6
"""
        assert_run_success(
            fprime_test_api, main, import_search_dirs=[], main_file_dir=str(tmp_path)
        )

    def test_relative_import_without_location_is_error(self, fprime_test_api, tmp_path):
        """A relative import in a sequence with no containing directory (a
        stream; `main_file_dir=None` here) has no anchor and is an error."""
        _write_sequence(tmp_path, "util", "def v() -> U32:\n    return 6\n")
        main = """\
import .util
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_relative_and_absolute_same_file_collides(self, fprime_test_api, tmp_path):
        """An absolute and a relative import that resolve to the same file import
        the same sequence; both bind the sequence module `util`, which collides
        (the shared file is still compiled only once)."""
        _write_sequence(tmp_path, "util", "def v() -> U32:\n    return 6\n")
        main = """\
import util
import .util
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="collides with an existing imported sequence",
            import_search_dirs=[str(tmp_path)],
            main_file_dir=str(tmp_path),
        )


class TestImportModuleMerging:
    # FIXME i think these might be duplicates?
    """Package modules (non-leaf chain segments) merge freely; sequence
    modules (a module holding a file's definitions -- chain leaf or alias)
    never merge with each other."""

    def test_sequence_and_package_module_merge(self, fprime_test_api, tmp_path):
        """`import pkg` (the file `pkg.fpy`) and `import pkg.mod` (the file
        `pkg/mod.fpy`) share the name `pkg`: the sequence module and the
        package module merge, holding `pkg.fpy`'s definitions alongside
        module `mod`."""
        _write_sequence(tmp_path, "pkg", "def top() -> U32:\n    return 1\n")
        _write_sequence(tmp_path, "pkg.mod", "def f() -> U32:\n    return 5\n")
        main = """\
import pkg
import pkg.mod

x: U32 = U32(pkg.top() + pkg.mod.f())
assert x == 6
"""
        assert_run_success(fprime_test_api, main, import_search_dirs=[str(tmp_path)])

    def test_two_aliases_same_name_collide(self, fprime_test_api, tmp_path):
        """Two different sequences aliased to one name are two sequence
        modules on the same name: a collision, not a merge."""
        _write_sequence(tmp_path, "a", "def f() -> U32:\n    return 1\n")
        _write_sequence(tmp_path, "b", "def g() -> U32:\n    return 2\n")
        main = """\
import a as m
import b as m
"""
        assert_compile_failure(
            fprime_test_api, main, import_search_dirs=[str(tmp_path)]
        )

    def test_relative_and_absolute_leaf_collision(self, fprime_test_api, tmp_path):
        """An absolute `import util` and a relative `import .util` naming two
        DIFFERENT files both bind a sequence module `util`: a collision, even
        though the two files' definitions do not overlap."""
        d_base = tmp_path / "base"
        d_main = tmp_path / "main"
        d_base.mkdir()
        d_main.mkdir()
        _write_sequence(d_base, "util", "def f() -> U32:\n    return 1\n")
        _write_sequence(d_main, "util", "def g() -> U32:\n    return 2\n")
        main = """\
import util
import .util
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            import_search_dirs=[str(d_base)],
            main_file_dir=str(d_main),
        )


# FIXME what happens if an import has relative imports? do we test this? the relative imports from the imported lib should
# resolve relative to the imported lib, not the original importing file
