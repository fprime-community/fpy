"""Tests for the `import` statement (spec / TDD).

`import foo` inlines the definitions of a sibling `foo.fpy` sequence into the
importing sequence and sections its symbols off under the name `foo`, so
a function `bar` defined in `foo.fpy` is called as `foo.bar()`.

Terminology (per SPEC.md's Imports section): files and directories are *definitions* -- each
`.fpy` file a *sequence definition* holding the definitions of its sequence,
each directory a *directory definition*. One file or directory is one
definition, and an import associates names with definitions.

Design decisions encoded here (per SPEC.md's Imports section):
  * Imports come in two disjoint resolution styles (PEP-328-like).  An
    ABSOLUTE import (`import foo`, `import a.b.c`) resolves only against the
    import directories (`state.import_directories`, the `-i/--imports`
    directories); the importing file's own location plays no role, so an
    absolute path names the same file in every sequence of a compilation.  A
    RELATIVE import (leading dots: `import .foo`, `import ..util`,
    `from .foo import f`) resolves only against its anchor directory -- one
    dot is the importing sequence's own directory, each extra dot one parent
    up -- and never consults the import directories.  So a library
    directory's sequences reference one another relatively and work
    unmodified wherever the library is mounted, while consumers name the
    library absolutely from anywhere.  The import directories are DISTINCT
    from the seq maps (`--seq-map`), which locate the sources of separately
    compiled called sequences -- imports are a compile-time-only source
    inlining and never survive into the emitted bytecode.
  * The import directories are an ORDERED list: the first identifier of an
    absolute import path is resolved in each of them in order until it
    succeeds.  A name present in several import directories resolves in the
    earliest, like PATH lookup; there is no ambiguity error.
  * Import paths resolve identifier by identifier, left to right:
    `import a.b.c` -> `a/b/c.fpy`.  Directories need no
    `__init__.fpy` marker.  The imported symbols are reached by the full
    path: `import a.b.c` makes `a.b.c.bar()` callable
    (and `import foo` makes `foo.bar()` callable).  A relative import binds
    the path after its dots: `import .sub.mod` -> `sub.mod.f()`, and
    `import ..util` -> `util.f()`.
  * Unlike Python, `import .foo` is legal (binding `foo`) and
    `from . import foo` is not: `from` members are always definitions, never
    sequences.
  * In a directory, an identifier refers to the definition of that name
    (a sequence file or a subdirectory); a directory that contains BOTH
    `foo.fpy` and `foo/` makes `foo` an error, not a precedence choice.
    An import path must refer to a sequence definition: one that refers to a
    directory (no sequence file to inline) is an error, and one that names a
    definition in a sequence (`import lib.func`) is an error -- those come
    only from `from` imports (`from lib import func`).
  * `import` is only valid as a top-level statement (not nested in a block),
    but an imported sequence MAY itself import other sequences.  Transitive
    imports are supported, cycles included: a sequence is compiled once and an
    import executes nothing, so a cycle needs no special handling.
  * An imported sequence that does more than define functions and import is a
    hard error: its top-level code would have to run at the importer's position,
    and an imported sequence is a sibling block that never runs inline.
  * A leading underscore marks a definition as library-internal: an importer
    naming it (`lib._helper()`, `from lib import _helper`) emits the
    `import-underscore` warning.  The library's own references to it never
    warn, and an alias silences later uses.
  * Importing a sequence that declares sequence arguments (`sequence(x: U32)`)
    is a hard error.
  * A sequence symbol obeys name groups: it exists only in the name groups
    of the definitions it holds.  So `import lib` of a functions-only sequence
    collides with a local function `lib` (callable name group) but coexists
    with a local variable `lib` (value name group).
  * One file or directory is one definition, and a name may be
    associated with at most one definition.  So `import pkg.a` +
    `import pkg.b` share the directory definition `pkg`, while two DIFFERENT
    directories or files on one name collide, and a directory and a sequence
    file are never the same definition.
  * Associations fold by identity rather than colliding blindly: the SAME
    file imported more than one way is one sequence definition, and the SAME
    definition aliased more than once under one name (`from seq import bar`
    twice, or a facade re-export plus a direct import -- a diamond) binds
    once.  Only two different definitions meeting on one name collide.  A
    single member list that binds one name to one definition twice
    additionally warns (`import-duplicate`): legal, but a near-certain typo.

Beyond the plain `import seq` form, this file also covers aliases
(`import seq as x`, `... as y`) and `from` imports (`from seq import a, b`,
`from seq import *`).

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
    becomes `<search_dir>/a/b/c.fpy`, creating the intervening directories."""
    rel = Path(*dotted_name.split(".")).with_suffix(".fpy")
    path = search_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src)


class TestImportInlining:
    """An imported function is inlined and callable under the sequence's
    name."""

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
        state, _, _ = compile_seq(main, import_directories=[str(tmp_path)])
        assert state.warnings == []
        # ...and the embedded assert holds at run time.
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])


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
            import_directories=[str(tmp_path)],
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
        # No expected_warnings: any warning at all would fail the test.
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])


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
        assert_run_success(
            fprime_test_api,
            main,
            import_directories=[str(tmp_path)],
            expected_warnings={WarningType.IMPORT_UNDERSCORE},
        )

    def test_underscore_from_import_warns(self, fprime_test_api, tmp_path):
        """`from lib import _helper` names the underscore definition."""
        _write_sequence(tmp_path, "lib", self.LIB)
        main = """\
from lib import _helper

x: U32 = _helper()
assert x == 7
"""
        assert_run_success(
            fprime_test_api,
            main,
            import_directories=[str(tmp_path)],
            expected_warnings={WarningType.IMPORT_UNDERSCORE},
        )

    def test_star_import_does_not_bind_underscore_names(
        self, fprime_test_api, tmp_path
    ):
        """`from lib import *` names nothing, so it binds only the public
        surface: `_helper` stays unbound and using it is an error, not a
        warning."""
        _write_sequence(tmp_path, "lib", self.LIB)
        main = """\
from lib import *

x: U32 = _helper()
assert x == 7
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="Unknown callable '_helper'",
            import_directories=[str(tmp_path)],
        )

    def test_star_import_binds_public_names(self, fprime_test_api, tmp_path):
        """The public half of the same star import still binds, and `public`
        internally calling `_helper` does not warn the importer."""
        _write_sequence(tmp_path, "lib", self.LIB)
        main = """\
from lib import *

x: U32 = public()
assert x == 7
"""
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

    def test_library_internal_use_does_not_warn(self, fprime_test_api, tmp_path):
        """`lib.public()` internally calls `_helper`; the importer never names
        an underscore definition, so nothing warns."""
        _write_sequence(tmp_path, "lib", self.LIB)
        main = """\
import lib

x: U32 = lib.public()
assert x == 7
"""
        # No expected_warnings: any warning at all would fail the test.
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
            main, import_directories=[str(tmp_path)], expected_warnings=expected
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
            import_directories=[str(tmp_path)],
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
            fprime_test_api,
            main,
            match="sequence arguments",
            import_directories=[str(tmp_path)],
        )

    def test_missing_sequence_is_an_error(self, fprime_test_api, tmp_path):
        main = """\
import does_not_exist
"""
        assert_compile_failure(
            fprime_test_api, main, import_directories=[str(tmp_path)]
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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

    def test_imported_sequence_metadata_must_be_first(self, fprime_test_api, tmp_path):
        """An imported sequence keeps its `sequence()` statement, so the
        position rule is enforced on its block by the same pass that checks the
        main sequence's."""
        _write_sequence(
            tmp_path,
            "late_metadata",
            """\
def f() -> U32:
    return 1

sequence()
""",
        )
        main = """\
import late_metadata
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="must be the first statement",
            import_directories=[str(tmp_path)],
        )


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
        state, _, _ = compile_seq(main, import_directories=[str(tmp_path)])
        assert state.warnings == []
        assert_run_success(
            fprime_test_api,
            main,
            args=[FpyValue(U32, 41)],
            import_directories=[str(tmp_path)],
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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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

past: bool = lib.is_past(Fw.Time(TimeBase.TB_WORKSTATION_TIME, 0, 0, 0))
"""
        assert_run_success(
            fprime_test_api, main, import_directories=[str(tmp_path)], time_base=2
        )


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
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )

    def test_import_path_is_a_directory_fails(self, fprime_test_api, tmp_path):
        """If the resolved `<name>.fpy` is a directory, importing fails cleanly
        rather than crashing with an IO error."""
        (tmp_path / "directory.fpy").mkdir()
        main = """\
import directory
"""
        assert_compile_failure(
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )

    def test_empty_sequence_compiles_without_warning(self, fprime_test_api, tmp_path):
        """An empty sequence has no definitions and no side effects."""
        _write_sequence(tmp_path, "empty", "")
        main = """\
import empty

CdhCore.cmdDisp.CMD_NO_OP()
"""
        # No expected_warnings: any warning at all would fail the test.
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])


class TestImportIsolation:
    """Imported symbols are sectioned under the sequence's name and isolated."""

    def test_imported_symbol_requires_qualification(self, fprime_test_api, tmp_path):
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
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )

    def test_sequence_name_not_usable_as_value(self, fprime_test_api, tmp_path):
        """A sequence's name resolves to its sequence symbol, not a value, so it is not usable as an expression."""
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
        # A functions-only sequence symbol binds only in the callable group, so a value-context
        # use fails at name resolution ("Unknown value") before the more general
        # module-as-value check (see test_return_nothing_expr_in_void_func).
        assert_compile_failure(
            fprime_test_api,
            main,
            match="Unknown value 'lib'",
            import_directories=[str(tmp_path)],
        )

    def test_same_function_name_in_two_sequences_no_collision(
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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

    def test_imported_function_cannot_see_importer_globals(
        self, fprime_test_api, tmp_path
    ):
        """An imported function is analyzed in its own sequence's scope: a name
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
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )


class TestImportNameCollisions:
    """An imported sequence symbol collides with existing top-level names
    per name group: it exists only in the name groups of the definitions it
    holds."""

    def test_import_collides_with_local_function(self, fprime_test_api, tmp_path):
        """A functions-only sequence symbol occupies the callable name
        group, so a local function with the sequence's name collides."""
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
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )

    def test_import_collides_with_local_variable(self, fprime_test_api, tmp_path):
        """A sequence symbol with a top-level variable occupies the value
        name group, so a local variable with the sequence's name collides."""
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
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )

    def test_import_coexists_with_local_variable(self, fprime_test_api, tmp_path):
        """A functions-only sequence symbol does NOT occupy the value name
        group, so a local variable with the sequence's name is legal, and both
        remain usable in their respective name groups."""
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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])


class TestImportedSequenceShadowingBuiltins:
    """An imported sequence's own scope layers on top of the shared dictionary /
    builtin base scope. Binding a name already taken by that base (a cast like
    `U32`, a type constructor, a dictionary module) does not collide -- the
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
            import_directories=[str(tmp_path)],
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
            import_directories=[str(tmp_path)],
        )

    def test_nested_import_shadowing_dictionary_module_warns(
        self, fprime_test_api, tmp_path
    ):
        """An imported sequence that itself imports a sequence whose name
        matches a dictionary module (`Ref`) shadows that base module. Bound
        through `_define_name`, so the shadow is detected up the parent chain. Allowed
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
            import_directories=[str(tmp_path)],
            expected_warnings={WarningType.SHADOW_CALLABLE},
        )


class TestImportShadowsBuiltins:
    """A main-sequence import that binds a name taken only by the builtin /
    dictionary base scope shadows it -- a warning categorized by the bound name's
    name group. Imports of function-only sequences occupy the callable group, so
    they warn as `shadow-callable`, and still compile and run. A same-scope
    collision stays an error."""

    def test_import_shadowing_builtin_warns(self, fprime_test_api, tmp_path):
        # A sequence file named after the builtin library function `time_add`;
        # `import time_add` binds `time_add` over the builtin callable.
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
            import_directories=[str(tmp_path)],
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
            import_directories=[str(tmp_path)],
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
            import_directories=[str(tmp_path)],
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
            import_directories=[str(tmp_path)],
        )


class TestImportDuplicates:
    """A sequence is compiled once however many import statements name it;
    two `import lib` statements define the same sequence symbol and fold
    idempotently."""

    def test_import_same_sequence_twice_is_idempotent(self, fprime_test_api, tmp_path):
        """Importing the same file twice binds the same definitions both times,
        so the second import merges idempotently rather than colliding."""
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
x: U32 = lib.add_one(4)
assert x == 5
"""
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])


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
            fprime_test_api, main, import_directories=[str(tmp_path)]
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
            fprime_test_api, main, import_directories=[str(tmp_path)]
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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )


class TestImportCycles:
    """A sequence may import a sequence that transitively imports it. A cycle
    needs no special handling: a sequence is compiled once, an import binds names
    and executes nothing, and no sequence's definitions are needed until every
    sequence has been compiled."""

    def test_self_import(self, fprime_test_api, tmp_path):
        """The tightest cycle: a sequence importing itself, and reaching its own
        definitions back through the symbol it binds."""
        _write_sequence(
            tmp_path,
            "self_import",
            """\
import self_import

def f() -> U32:
    return 1

def via_self() -> U32:
    return self_import.f()
""",
        )
        main = """\
import self_import

x: U32 = self_import.via_self()
assert x == 1
"""
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

    def test_mutual_import(self, fprime_test_api, tmp_path):
        """mod_a and mod_b import each other, and each calls into the other."""
        _write_sequence(
            tmp_path,
            "mod_a",
            """\
import mod_b

def a() -> U32:
    return mod_b.b()

def a_leaf() -> U32:
    return 10
""",
        )
        _write_sequence(
            tmp_path,
            "mod_b",
            """\
import mod_a

def b() -> U32:
    return 7

def b_uses_a() -> U32:
    return mod_a.a_leaf()
""",
        )
        main = """\
import mod_a
import mod_b

x: U32 = mod_a.a()
assert x == 7
y: U32 = mod_b.b_uses_a()
assert y == 10
"""
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

    def test_three_way_cycle(self, fprime_test_api, tmp_path):
        """c1 -> c2 -> c3 -> c1, with a call chain running the length of it."""
        _write_sequence(
            tmp_path,
            "c1",
            """\
import c2

def f() -> U32:
    return c2.g()
""",
        )
        _write_sequence(
            tmp_path,
            "c2",
            """\
import c3

def g() -> U32:
    return c3.h()
""",
        )
        _write_sequence(
            tmp_path,
            "c3",
            """\
import c1

def h() -> U32:
    return 3
""",
        )
        main = """\
import c1

x: U32 = c1.f()
assert x == 3
"""
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

    def test_cycle_compiles_each_sequence_once(self, tmp_path):
        """A sequence in a cycle is compiled once, like any other: the cycle adds
        no duplicate block."""
        _write_sequence(
            tmp_path,
            "cyc_a",
            """\
import cyc_b

def a() -> U32:
    return 1
""",
        )
        _write_sequence(
            tmp_path,
            "cyc_b",
            """\
import cyc_a

def b() -> U32:
    return 2
""",
        )
        main = """\
import cyc_a
import cyc_b

x: U32 = U32(cyc_a.a() + cyc_b.b())
assert x == 3
"""
        state, _, _ = compile_seq(main, import_directories=[str(tmp_path)])
        loaded = {Path(p).stem for p in state.loaded_sequences}
        assert loaded == {"cyc_a", "cyc_b"}
        assert len(state.imported_blocks) == 2


class TestImportMainSequence:
    """ "If D has previously been included in the program's AST, or if its
    sequence is the main sequence, skip it": an import path resolving to the
    input file itself is not included again; it binds against the main
    sequence's own scope."""

    def test_main_sequence_self_import_is_skipped(self, tmp_path):
        main_file = tmp_path / "main.fpy"
        main = """\
import .main

def f() -> U32:
    return 3

x: U32 = main.f()
assert x == 3
"""
        main_file.write_text(main)
        state, _, _ = compile_seq(
            main,
            import_directories=[],
            main_file_dir=str(tmp_path),
            main_file_path=str(main_file.resolve()),
        )
        # The main sequence is skipped, never included as a sibling block.
        assert state.imported_blocks == []


class TestImportDottedPaths:
    """Dotted sequence paths resolve through directories, Pythonically."""

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
        # No expected_warnings: any warning at all would fail the test.
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )

    def test_missing_leaf_in_existing_directory_is_error(
        self, fprime_test_api, tmp_path
    ):
        """The directory exists but the leaf sequence file does not."""
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
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )

    def test_two_sequences_in_same_module_no_collision(self, fprime_test_api, tmp_path):
        """Sibling sequences under one directory get independent sequence symbols."""
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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])


class TestImportFileDirectoryConflict:
    """In a directory D, an identifier I refers to the sequence file or
    directory in D named I, if one exists.  If D contains both, an error is
    raised -- there is no precedence between them."""

    def test_file_and_directory_same_name_is_error(self, fprime_test_api, tmp_path):
        """`foo.fpy` and a `foo/` directory both exist in one directory;
        `foo` refers to both, so `import foo` is an error."""
        _write_sequence(tmp_path, "foo", "def f() -> U32:\n    return 1\n")
        # This also creates the sibling `foo/` directory:
        _write_sequence(tmp_path, "foo.inner", "def g() -> U32:\n    return 2\n")
        main = """\
import foo
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="refers to both a sequence file and a directory",
            import_directories=[str(tmp_path)],
        )

    def test_file_and_directory_conflict_on_dotted_descent(
        self, fprime_test_api, tmp_path
    ):
        """The same conflict hits a dotted path at its first identifier:
        `pkg.fpy` and `pkg/` both exist, so `import pkg.mod` cannot resolve
        `pkg`."""
        _write_sequence(tmp_path, "pkg", "def top() -> U32:\n    return 1\n")
        _write_sequence(tmp_path, "pkg.mod", "def f() -> U32:\n    return 5\n")
        main = """\
import pkg.mod
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="refers to both a sequence file and a directory",
            import_directories=[str(tmp_path)],
        )

    def test_bare_directory_import_is_error(self, fprime_test_api, tmp_path):
        """`import pkg` where only a `pkg/` directory exists (no `pkg.fpy`) is
        an error -- a directory has no sequence file to inline."""
        # Creates `pkg/mod.fpy`, so `pkg/` exists as a directory but `pkg.fpy`
        # does not.
        _write_sequence(tmp_path, "pkg.mod", "def f() -> U32:\n    return 1\n")
        main = """\
import pkg
"""
        assert_compile_failure(
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )

    def test_dotted_leaf_directory_import_is_error(self, fprime_test_api, tmp_path):
        """`import a.b` where `a/b/` is a directory but `a/b.fpy` does not exist
        is likewise an error at the dotted leaf."""
        _write_sequence(tmp_path, "a.b.c", "def f() -> U32:\n    return 1\n")
        main = """\
import a.b
"""
        assert_compile_failure(
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )


class TestImportDirectories:
    """The import directories are an ordered list: resolution of an absolute
    import path's first identifier is attempted in each import directory in
    order until it succeeds.  If it cannot be resolved in any import
    directory, an error is raised."""

    def test_sequence_found_in_later_import_directory(self, fprime_test_api, tmp_path):
        """A sequence present in only one import directory is found, wherever
        that directory sits in the list."""
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
        assert_run_success(fprime_test_api, main, import_directories=[str(d1), str(d2)])

    def test_same_name_in_two_dirs_first_wins(self, fprime_test_api, tmp_path):
        """A sequence name that resolves in two import directories resolves in
        the earlier one: resolution is attempted in each in order until it
        succeeds."""
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        _write_sequence(d1, "lib", "def f() -> U32:\n    return 1\n")
        _write_sequence(d2, "lib", "def f() -> U32:\n    return 2\n")
        main = """\
import lib

x: U32 = lib.f()
assert x == 1
"""
        assert_run_success(fprime_test_api, main, import_directories=[str(d1), str(d2)])

    def test_first_identifier_pins_the_import_directory(
        self, fprime_test_api, tmp_path
    ):
        """The first identifier alone picks the import directory; the rest
        of the path resolves within it.  `a` refers to the directory `a/` in
        the first import directory, which has no `b` -- `d2/a/b.fpy` is
        never consulted, so the import fails."""
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        # Creates d1/a/, holding only `other.fpy`:
        _write_sequence(d1, "a.other", "def f() -> U32:\n    return 1\n")
        _write_sequence(d2, "a.b", "def f() -> U32:\n    return 2\n")
        main = """\
import a.b
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="could not be resolved in directory",
            import_directories=[str(d1), str(d2)],
        )

    def test_dotted_sequence_resolved_across_import_directories(
        self, fprime_test_api, tmp_path
    ):
        """The first identifier is attempted in every import directory:
        `pkg/mod.fpy` lives only in the second."""
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
        assert_run_success(fprime_test_api, main, import_directories=[str(d1), str(d2)])

    def test_no_import_directories_cannot_resolve(self, fprime_test_api, tmp_path):
        """With no import directories, no absolute import can resolve."""
        _write_sequence(tmp_path, "lib", "def f() -> U32:\n    return 1\n")
        main = """\
import lib
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="could not be resolved in any import directory",
            import_directories=[],
        )


class TestImportVariables:
    """A sequence's top-level variable is a side effect (its assignment runs at
    sequence start), so an imported sequence may not declare one -- there is
    no imported variable, only functions."""

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
            import_directories=[str(tmp_path)],
        )


class TestImportMember:
    """An import path must refer to a sequence: the definitions in a
    sequence cannot be named by an import path.  `import seq.func` is an
    error; write `from seq import func`."""

    def test_import_member_is_error(self, fprime_test_api, tmp_path):
        """`import lib.add_one` names a member of the sequence `lib`."""
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
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="an import path cannot name the definitions in it",
            import_directories=[str(tmp_path)],
        )

    def test_import_member_of_dotted_sequence_is_error(self, fprime_test_api, tmp_path):
        """The same error when the sequence's own path is dotted: in
        `import pkg.mod.f`, `pkg.mod` refers to the sequence and `f` to a
        member."""
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
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="an import path cannot name the definitions in it",
            import_directories=[str(tmp_path)],
        )

    def test_error_suggests_from_import(self, fprime_test_api, tmp_path):
        """The error suggests the from-import spelling of the same name."""
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
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="from lib import add_one",
            import_directories=[str(tmp_path)],
        )

    def test_path_past_member_is_error(self, fprime_test_api, tmp_path):
        """`import lib.a.b` fails at `a` already: the path cannot enter the
        sequence at all."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1
""",
        )
        main = """\
import lib.a.b
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="an import path cannot name the definitions in it",
            import_directories=[str(tmp_path)],
        )


class TestImportAlias:
    """An `as` clause introduces no directory names: the alias alone is
    associated with the imported sequence definition."""

    def test_import_sequence_as_alias(self, fprime_test_api, tmp_path):
        """`import lib as L` binds `L` to `lib`'s sequence definition; its
        definitions are reached as `L.name`."""
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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

    def test_dotted_import_as_alias(self, fprime_test_api, tmp_path):
        """`import pkg.mod as m` binds `m` to the sequence `mod`'s symbol."""
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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
            fprime_test_api, main, import_directories=[str(tmp_path)]
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
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )


class TestImportFrom:
    """A `from` statement binds its imported symbols directly in the importing
    sequence's global scope, introducing no modules."""

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

    def test_from_import_reexported_module(self, fprime_test_api, tmp_path):
        """A `from` member may name a symbol the imported sequence itself
        imported (a re-export). `lib` imports `sub`, so `sub`'s sequence symbol
        is in lib's own scope; `from lib import sub` binds that whole symbol,
        and `sub.f()` resolves through it."""
        _write_sequence(tmp_path, "sub", "def f() -> U32:\n    return 7\n")
        _write_sequence(tmp_path, "lib", "import sub\n")
        main = """\
from lib import sub

x: U32 = sub.f()
assert x == 7
"""
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

    def test_from_import_duplicate_member_warns(self, fprime_test_api, tmp_path):
        """One member list binding the same name to the same definition twice
        folds like any two aliases of one definition, but it is a near-certain
        typo: the `import-duplicate` warning, emitted once."""
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

x: U32 = a()
assert x == 1
"""
        expected = {WarningType.IMPORT_DUPLICATE}
        state, _, _ = compile_seq(
            main, import_directories=[str(tmp_path)], expected_warnings=expected
        )
        duplicate_warnings = [
            w for w in state.warnings if w.type == WarningType.IMPORT_DUPLICATE
        ]
        assert (
            len(duplicate_warnings) == 1
        ), f"expected exactly one import-duplicate warning, got {state.warnings}"
        assert_run_success(
            fprime_test_api,
            main,
            import_directories=[str(tmp_path)],
            expected_warnings=expected,
        )

    def test_duplicate_warning_is_per_member_not_per_definition(
        self, fprime_test_api, tmp_path
    ):
        """The import-duplicate warning fires on a repeated member, not on
        two DIFFERENT members that happen to deliver the same definition:
        re-exported under two names, they alias one definition under one
        name and fold silently."""
        _write_sequence(tmp_path, "base", "def f() -> U32:\n    return 7\n")
        _write_sequence(
            tmp_path,
            "lib",
            """\
from base import f
from base import f as f2
""",
        )
        main = """\
from lib import f as x, f2 as x

y: U32 = x()
assert y == 7
"""
        # No expected_warnings: any warning at all would fail the test.
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

    def test_from_import_one_name_two_members_is_error(self, fprime_test_api, tmp_path):
        """One member list binding the same name to two DIFFERENT definitions
        is a collision, not a duplicate warning."""
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
from lib import a as x, b as x
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="collides with an existing definition",
            import_directories=[str(tmp_path)],
        )

    def test_from_import_does_not_introduce_sequence_name(
        self, fprime_test_api, tmp_path
    ):
        """`from lib import add_one` binds no `lib` at all, so `lib.add_one`
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
            fprime_test_api, main, import_directories=[str(tmp_path)]
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
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )

    def test_from_path_past_sequence_file_is_error(self, fprime_test_api, tmp_path):
        """A from-statement's import path cannot name a member either:
        `from a.b import x` where `a` is a sequence fails at `b` (members
        come only from the import list, never the path)."""
        _write_sequence(tmp_path, "a", "def b() -> U32:\n    return 1\n")
        main = """\
from a.b import x
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="an import path cannot name the definitions in it",
            import_directories=[str(tmp_path)],
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
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )

    def test_from_star_collides_across_sequences(self, fprime_test_api, tmp_path):
        """Two `from ... import *` statements deliver `f` from two DIFFERENT
        files: two different definitions on one name collide (only the same
        definition folds)."""
        _write_sequence(tmp_path, "a", "def f() -> U32:\n    return 1\n")
        _write_sequence(tmp_path, "b", "def f() -> U32:\n    return 2\n")
        main = """\
from a import *
from b import *
"""
        assert_compile_failure(
            fprime_test_api, main, import_directories=[str(tmp_path)]
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
            import_directories=[str(tmp_path)],
        )


class TestImportDuplicateSequence:
    """A sequence may be imported more than once in one file. Whether that is
    allowed follows from the collision rule on the names each import binds, and
    the sequence is compiled just once regardless."""

    def test_import_and_from_same_sequence_coexist(self, fprime_test_api, tmp_path):
        """`import lib` binds the sequence symbol `lib`; `from lib import b`
        binds `b`. The names differ, so they coexist."""
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
        # No expected_warnings: any warning at all would fail the test.
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
        # No expected_warnings: any warning at all would fail the test.
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

    def test_from_import_repeated_across_statements_folds(
        self, fprime_test_api, tmp_path
    ):
        """Two statements aliasing the same definition under the same name are
        the same semantic alias definition: the second folds silently."""
        _write_sequence(
            tmp_path,
            "lib",
            """\
def a() -> U32:
    return 1
""",
        )
        main = """\
from lib import a
from lib import a

x: U32 = a()
assert x == 1
"""
        # No expected_warnings: any warning at all would fail the test.
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])


class TestImportDiamond:
    """One definition reaching a scope by two import routes binds once.

    An alias definition is identified by the definition it names, so a facade
    sequence that re-exports a base sequence's definitions composes with a
    direct import of the base: both routes deliver the same definitions, and
    they fold silently. (Definitions from two DIFFERENT files stay a
    collision -- see TestImportFrom::test_from_star_collides_across_sequences.)"""

    def test_star_import_diamond_folds(self, fprime_test_api, tmp_path):
        """main star-imports both a facade (which star-imports base) and base
        itself; base's definitions arrive twice but bind once."""
        _write_sequence(tmp_path, "base", "def f() -> U32:\n    return 7\n")
        _write_sequence(tmp_path, "facade", "from base import *\n")
        main = """\
from facade import *
from base import *

x: U32 = f()
assert x == 7
"""
        # No expected_warnings: any warning at all would fail the test.
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

    def test_member_import_diamond_folds(self, fprime_test_api, tmp_path):
        """The same diamond through named members: `from facade import f` and
        `from base import f` alias one definition."""
        _write_sequence(tmp_path, "base", "def f() -> U32:\n    return 7\n")
        _write_sequence(tmp_path, "facade", "from base import *\n")
        main = """\
from facade import f
from base import f

x: U32 = f()
assert x == 7
"""
        # No expected_warnings: any warning at all would fail the test.
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])


class TestImportRelative:
    """Leading dots make an import relative: one dot anchors at the importing
    sequence's own directory, each extra dot one parent up, and the base
    search path is never consulted.  Absolute imports, conversely, never see
    the importing sequence's own directory.  The dots affect resolution only;
    the bound path is the one after them."""

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
            fprime_test_api, main, import_directories=[str(tmp_path)]
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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

    def test_parent_relative_import(self, fprime_test_api, tmp_path):
        """`import ..util` in `lib/sub/inner.fpy` anchors one parent up,
        resolving `lib/util.fpy` and binding `util`."""
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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

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
            import_directories=[str(tmp_path)],
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
            fprime_test_api, main, import_directories=[], main_file_dir=str(tmp_path)
        )

    def test_imported_lib_relative_import_anchors_at_the_lib(
        self, fprime_test_api, tmp_path
    ):
        """A relative import in an IMPORTED sequence anchors at that sequence's
        own directory, not the original importer's. main lives in `app/` and a
        decoy `helper.fpy` sits beside it; the library `lib/consumer.fpy` does
        `import .helper`, which must bind `lib/helper.fpy` (returning 5), never
        the decoy next to main (returning 99)."""
        d_app = tmp_path / "app"
        d_lib = tmp_path / "lib"
        d_app.mkdir()
        d_lib.mkdir()
        # Decoy: same name as the library's sibling, but next to main. If the
        # anchor were the original importer, this would be picked instead.
        _write_sequence(d_app, "helper", "def h() -> U32:\n    return 99\n")
        _write_sequence(d_lib, "helper", "def h() -> U32:\n    return 5\n")
        _write_sequence(
            d_lib,
            "consumer",
            """\
import .helper

def f() -> U32:
    return helper.h()
""",
        )
        main = """\
import consumer

x: U32 = consumer.f()
assert x == 5
"""
        assert_run_success(
            fprime_test_api,
            main,
            import_directories=[str(d_lib)],
            main_file_dir=str(d_app),
        )

    def test_over_deep_relative_has_no_anchor_directory(
        self, fprime_test_api, tmp_path
    ):
        """A relative import with more dots than the tree is deep has no Nth
        parent directory to anchor at: an error (not an IndexError, and not a
        silent saturation at the filesystem root)."""
        _write_sequence(tmp_path, "util", "def v() -> U32:\n    return 6\n")
        main = "import " + "." * 40 + "util\n"
        assert_compile_failure(
            fprime_test_api,
            main,
            match="parent directory",
            import_directories=[str(tmp_path)],
            main_file_dir=str(tmp_path),
        )

    def test_relative_import_without_location_is_error(self, fprime_test_api, tmp_path):
        """A relative import in a sequence with no containing directory (a
        stream; `main_file_dir=None` here) has no anchor and is an error."""
        _write_sequence(tmp_path, "util", "def v() -> U32:\n    return 6\n")
        main = """\
import .util
"""
        assert_compile_failure(
            fprime_test_api, main, import_directories=[str(tmp_path)]
        )

    def test_relative_and_absolute_same_file_idempotent(
        self, fprime_test_api, tmp_path
    ):
        """An absolute and a relative import resolving to the same file bind the
        same definitions; the second merges idempotently (the file is compiled
        once)."""
        _write_sequence(tmp_path, "util", "def v() -> U32:\n    return 6\n")
        main = """\
import util
import .util
assert util.v() == 6
"""
        assert_run_success(
            fprime_test_api,
            main,
            import_directories=[str(tmp_path)],
            main_file_dir=str(tmp_path),
        )


class TestImportModuleMerging:
    """How definitions that meet on one name in a sequence's scope combine.

    One file or directory is one definition: associations with the SAME
    directory or sequence file fold, so sibling sequences in one directory
    coexist under it, while associations with DIFFERENT directories or
    files collide -- and a directory and a sequence file are never the same
    definition. (Same-file folding is TestImportDuplicateSequence's.)"""

    def test_sibling_modules_merge(self, fprime_test_api, tmp_path):
        """`import pkg.a` and `import pkg.b` associate `pkg` with the same
        directory definition twice, so both sequences are reachable under
        it."""
        _write_sequence(tmp_path, "pkg.a", "def f() -> U32:\n    return 1\n")
        _write_sequence(tmp_path, "pkg.b", "def g() -> U32:\n    return 2\n")
        main = """\
import pkg.a
import pkg.b
assert pkg.a.f() == 1
assert pkg.b.g() == 2
"""
        assert_run_success(fprime_test_api, main, import_directories=[str(tmp_path)])

    def test_file_and_directory_same_name_cannot_even_resolve(
        self, fprime_test_api, tmp_path
    ):
        """`pkg.fpy` and `pkg/` in one directory never reach binding: `pkg`
        refers to both a sequence file and a directory, an error at
        resolution."""
        _write_sequence(tmp_path, "pkg", "def top() -> U32:\n    return 1\n")
        _write_sequence(tmp_path, "pkg.mod", "def f() -> U32:\n    return 5\n")
        main = """\
import pkg
import pkg.mod
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="refers to both a sequence file and a directory",
            import_directories=[str(tmp_path)],
        )

    def test_two_different_directories_same_name_collide(
        self, fprime_test_api, tmp_path
    ):
        """Two DIFFERENT directories bound under one name are two directory
        definitions: a collision, even with no member overlap."""
        d_base = tmp_path / "base"
        d_main = tmp_path / "main"
        d_base.mkdir()
        d_main.mkdir()
        _write_sequence(d_base, "pkg.a", "def f() -> U32:\n    return 1\n")
        _write_sequence(d_main, "pkg.b", "def g() -> U32:\n    return 2\n")
        main = """\
import pkg.a
import .pkg.b
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="collides with an existing imported directory",
            import_directories=[str(d_base)],
            main_file_dir=str(d_main),
        )

    def test_directory_and_sequence_same_name_forbidden(
        self, fprime_test_api, tmp_path
    ):
        """One name cannot be associated with both a directory and a sequence
        file.  The relative `import .pkg.mod` makes `pkg` a directory's
        module; the absolute `import pkg` (a different location's `pkg.fpy`)
        wants the same name for a sequence -- forbidden."""
        d_base = tmp_path / "base"
        d_main = tmp_path / "main"
        d_base.mkdir()
        d_main.mkdir()
        _write_sequence(d_base, "pkg", "def top() -> U32:\n    return 1\n")
        _write_sequence(d_main, "pkg.mod", "def f() -> U32:\n    return 5\n")
        main = """\
import .pkg.mod
import pkg
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="imported both as a directory and as a sequence file",
            import_directories=[str(d_base)],
            main_file_dir=str(d_main),
        )

    def test_two_aliases_same_name_collide(self, fprime_test_api, tmp_path):
        """Two DIFFERENT sequences aliased to one name are two sequence symbols
        naming different files: a collision, even with no member overlap."""
        _write_sequence(tmp_path, "a", "def f() -> U32:\n    return 1\n")
        _write_sequence(tmp_path, "b", "def g() -> U32:\n    return 2\n")
        main = """\
import a as m
import b as m
"""
        assert_compile_failure(
            fprime_test_api,
            main,
            match="collides with an existing imported sequence",
            import_directories=[str(tmp_path)],
        )

    def test_relative_and_absolute_leaf_collision(self, fprime_test_api, tmp_path):
        """An absolute `import util` and a relative `import .util` naming two
        DIFFERENT files both define sequence symbol `util`: a collision."""
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
            match="collides with an existing imported sequence",
            import_directories=[str(d_base)],
            main_file_dir=str(d_main),
        )
