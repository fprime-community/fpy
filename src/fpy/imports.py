"""The import statement.

This module implements SPEC.md's Imports section and is written to mirror it: passes,
methods, and checks are named with the spec's terms, and each carries the
spec text it implements (quoted). Where code must go beyond the spec's words
-- diagnostics such as warnings, or plumbing such as recording work between
compiler passes -- it says so.

The spec's semantics sections map onto two compiler passes:

* `ConstructAst` implements "Constructing the AST" and "Import path
  resolution". It runs on the raw AST, before scopes exist: it resolves each
  import path to a sequence definition (a sequence file in the file system),
  includes that file in the program's AST as a sibling of the main
  sequence's block, and records each import statement for `BindImports`.

* `BindImports` implements "Binding". It runs after `DefineFunctions` /
  `DefineVariables`, once every sequence's definitions exist: each recorded
  import statement associates one or more qualified names with definitions
  in the importing scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fpy.error
from fpy.error import WarningType
from fpy.symbols import (
    CallableSymbol,
    DirectorySymbol,
    ModuleSymbol,
    NameGroup,
    Scope,
    SequenceSymbol,
    VariableSymbol,
)
from fpy.syntax import (
    AstBlock,
    AstDef,
    AstGetAttr,
    AstImport,
    AstSequenceMetadata,
)
from fpy.state import CompileState
from fpy.types import is_instance_compat
from fpy.visitors import Visitor


@dataclass
class SequenceContext:
    """A sequence taking part in the program: the main sequence ("the
    sequence defined by the input file the user passes into the compiler"),
    or the sequence defined by an imported sequence file."""

    file_path: str | None
    """the absolute path of the sequence's file, or None for a main sequence
    that was not read from a file."""
    dir_path: str | None
    """the directory containing the sequence's file: the 1st parent
    directory of its absolute path, from which a relative import statement's
    anchor directory is counted. None when the sequence has no location."""
    block: AstBlock = None
    """the sequence's block in the program's AST: the block B its file was
    parsed to for an imported sequence (set here at inclusion), or the main
    sequence's block (set by _build_root_block)."""
    definition: SequenceSymbol = None
    """the sequence's file as a sequence definition: one file, one
    definition, so every scope that imports the file is associated with this
    one symbol. None for a main sequence that was not read from a file."""


@dataclass
class ResolvedImport:
    """An import statement whose import path has resolved to a sequence
    definition ("Import path resolution"), recorded by `ConstructAst` for
    `BindImports`. Each proper, non-empty prefix of the import path refers
    to a directory definition; the whole path refers to the imported
    sequence definition."""

    node: AstImport
    importing_sequence: SequenceContext
    """"The importing sequence is the sequence containing the import
    statement.\""""
    imported_sequence: SequenceContext
    """the sequence of the imported sequence definition's file."""


@dataclass
class _SequenceDefinition:
    """A sequence definition, as import path resolution refers to it: "Each
    file whose name is of the form `<name>.fpy` is a sequence definition
    with name `name`." One file is one definition, so the file's path is the
    definition's identity; binding realizes the definition as the file's one
    shared SequenceSymbol."""

    path: Path
    """the sequence file."""


@dataclass
class _DirectoryDefinition:
    """A directory definition, as import path resolution refers to it:
    "Each directory is a directory definition with the directory's name."
    One directory is one definition, so the directory's path is the
    definition's identity."""

    path: Path
    """the directory."""


class ConstructAst:
    """Constructing the AST (SPEC.md Imports, Semantics).

    "For each import statement in the AST, including statements added by
    this process:

    1. The import path must resolve to a sequence definition D, otherwise
       an error is raised.
    2. If D has previously been included in the program's AST, or if its
       sequence is the main sequence, skip it.
    3. Otherwise, D is lexed and parsed according to this specification,
       producing a new block B.
    4. If B has top-level statements which may have side effects, an error
       is raised.
    5. B is included in the program's AST as a sibling of the main
       sequence's block."

    "A sequence metadata statement with one or more formal parameters is a
    statement which may have side effects."

    "Cyclical imports are allowed. This is not an issue because import
    statements cannot have side effects."
    """

    def run(self, body: AstBlock, state: CompileState):
        # "Let the main sequence refer to the sequence defined by the input
        # file the user passes into the compiler."
        main_sequence = SequenceContext(
            file_path=state.main_file_path,
            dir_path=state.main_file_dir,
        )
        state.main_sequence = main_sequence
        # Register the main sequence's file so that an import path resolving
        # to it is skipped by step 2 ("or if its sequence is the main
        # sequence") and binds against the main sequence itself.
        if main_sequence.file_path is not None:
            main_sequence.definition = SequenceSymbol(main_sequence.file_path)
            state.loaded_sequences[main_sequence.file_path] = main_sequence

        body.stmts = self._for_each_import_statement(body.stmts, main_sequence, state)

    def _for_each_import_statement(
        self, stmts, containing_sequence: SequenceContext, state: CompileState
    ):
        """Apply steps 1-5 to each import statement among *stmts*, and
        return *stmts* with the import statements removed.

        Removing them is not in the spec but changes nothing observable: an
        import statement only associates names ("Binding"), so it
        contributes nothing at its position, and `BindImports` works from
        the ResolvedImport records made here. Statements added by this
        process (an included block B) are reached through _include's
        recursion back into this method.

        On an error, stop and hand back the statements processed so far."""
        remaining = []
        for stmt in stmts:
            if is_instance_compat(stmt, AstImport):
                self._import_statement(stmt, containing_sequence, state)
                if state.errors:
                    return remaining
            else:
                remaining.append(stmt)
        return remaining

    def _import_statement(
        self, node: AstImport, containing_sequence: SequenceContext, state: CompileState
    ):
        # 1. The import path must resolve to a sequence definition D,
        #    otherwise an error is raised.
        file_path = self._resolve_import_path(node, containing_sequence, state)
        if file_path is None:
            return

        # 2. If D has previously been included in the program's AST, or if
        #    its sequence is the main sequence, skip it.
        imported_sequence = state.loaded_sequences.get(file_path)
        if imported_sequence is None:
            imported_sequence = self._include(file_path, node.meta, state)
            if state.errors:
                return
            assert imported_sequence is not None

        state.resolved_imports.append(
            ResolvedImport(node, containing_sequence, imported_sequence)
        )

    def _include(
        self, file_path: str, meta, state: CompileState
    ) -> SequenceContext | None:
        """Steps 3-5, for a sequence definition that has not previously
        been included in the program's AST.

        *meta* is the position of the import statement that first named the
        file, carried onto the block built from it."""
        # 3. Otherwise, D is lexed and parsed according to this
        #    specification, producing a new block B.
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            state.err(f"Cannot read imported sequence file '{file_path}': {e}", None)
            return None
        parsed = self._lex_and_parse(file_path, text, state)
        if parsed is None:
            return None
        block = AstBlock(meta, parsed.stmts)

        # 4. If B has top-level statements which may have side effects, an
        #    error is raised.
        self._check_side_effects(block, state)
        if state.errors:
            return None

        # 5. B is included in the program's AST as a sibling of the main
        #    sequence's block. (The blocks collected here are installed as
        #    siblings by _build_root_block; CreateScopes then gives the
        #    block its own scope, isolated from every other sequence's.)
        imported_sequence = SequenceContext(
            file_path=file_path,
            dir_path=str(Path(file_path).parent),
            block=block,
            definition=SequenceSymbol(file_path),
        )
        state.loaded_sequences[file_path] = imported_sequence
        state.imported_blocks.append(block)

        # "For each import statement in the AST, including statements added
        # by this process": the block's own import statements are processed
        # in turn. The file is registered above BEFORE this descent, so a
        # cyclical import arriving back at it is skipped by step 2 instead
        # of included again.
        block.stmts = self._for_each_import_statement(
            block.stmts, imported_sequence, state
        )
        return imported_sequence

    def _lex_and_parse(self, file_path: str, text: str, state: CompileState):
        """Lex and parse the text of the imported sequence file under its
        own diagnostic context, so that errors in it point into that file
        rather than into the importing file. Restores the caller's context
        before returning.

        Reports a compile error and returns None on failure."""
        from fpy.compiler import text_to_ast

        with fpy.error.diagnostic_context(file_path):
            try:
                return text_to_ast(text)
            except fpy.error.CompileError as e:
                # str(e) must format here, under the imported file's
                # diagnostic context
                state.err(
                    f"Failed to parse imported sequence file '{file_path}':\n{e}",
                    None,
                )
                return None

    def _check_side_effects(self, block: AstBlock, state: CompileState):
        """4. "If B has top-level statements which may have side effects, an
        error is raised."

        A function definition, an import statement, and a sequence metadata
        statement with no formal parameters are the top-level statements
        that cannot have side effects; every other statement may. "A
        sequence metadata statement with one or more formal parameters is a
        statement which may have side effects": it binds argument values,
        which nothing would supply at an import."""
        for stmt in block.stmts:
            if is_instance_compat(stmt, AstSequenceMetadata):
                if stmt.parameters:
                    state.err(
                        "Cannot import a sequence with sequence arguments",
                        stmt,
                    )
                    return
            elif not is_instance_compat(stmt, (AstDef, AstImport)):
                state.err(
                    "An imported sequence may contain only function definitions "
                    "and imports",
                    stmt,
                )
                return

    # -- Import path resolution --

    def _resolve_import_path(
        self, node: AstImport, containing_sequence: SequenceContext, state: CompileState
    ) -> str | None:
        """Import path resolution: "the process by which the qualified
        identifier `import_path` is resolved to a definition."

        "These rules are applied to `import_path`. It must refer to a
        sequence definition; if it refers to a directory definition, an
        error is raised."

        Returns the sequence definition's file as an absolute path, or None
        with the error reported."""
        # These rules (_resolve_qualified_identifier) are applied to
        # import_path.
        definition = self._resolve_qualified_identifier(
            list(node.path), node, containing_sequence, state
        )
        if definition is None:
            return None
        # "It must refer to a sequence definition; if it refers to a
        # directory definition, an error is raised."
        if is_instance_compat(definition, _DirectoryDefinition):
            state.err(
                f"Import path '{'.'.join(node.path)}' refers to the directory "
                f"'{definition.path}', not a sequence file",
                node,
            )
            return None
        assert is_instance_compat(definition, _SequenceDefinition), definition
        return str(definition.path.resolve())

    def _resolve_qualified_identifier(
        self,
        path: list[str],
        node: AstImport,
        containing_sequence: SequenceContext,
        state: CompileState,
    ):
        """ "To resolve qualified identifier Q.I:
        1. Recursively resolve Q.
        2. If Q refers to a directory definition, resolution of I is
           attempted in its directory. An error is raised if I could not be
           resolved.
        3. Otherwise, Q refers to a sequence definition, and an error is
           raised."

        *path* is the qualified identifier's identifiers. A single
        identifier is the recursion's base case, resolved in the import
        directories or the anchor directory (_resolve_first_identifier).

        Returns the definition the qualified identifier refers to, or None
        with the error reported."""
        if len(path) == 1:
            return self._resolve_first_identifier(
                path[0], node, containing_sequence, state
            )
        identifier = path[-1]  # I
        # 1. Recursively resolve Q.
        definition = self._resolve_qualified_identifier(
            path[:-1], node, containing_sequence, state
        )
        if definition is None:
            return None
        # 2. If Q refers to a directory definition, resolution of I is
        #    attempted in its directory.
        if is_instance_compat(definition, _DirectoryDefinition):
            resolved = self._refers_to(definition.path, identifier, node, state)
            if state.errors:
                return None
            #    An error is raised if I could not be resolved.
            if resolved is None:
                state.err(
                    f"'{identifier}' could not be resolved in directory "
                    f"'{definition.path}'",
                    node,
                )
                return None
            return resolved
        # 3. Otherwise, Q refers to a sequence definition, and an error is
        #    raised.
        assert is_instance_compat(definition, _SequenceDefinition), definition
        qualifier = ".".join(path[:-1])
        state.err(
            f"'{qualifier}' is a sequence; an import path cannot name the "
            f"definitions in it. Write 'from {'.' * node.num_dots}{qualifier} "
            f"import {identifier}'",
            node,
        )
        return None

    def _resolve_first_identifier(
        self,
        identifier: str,
        node: AstImport,
        containing_sequence: SequenceContext,
        state: CompileState,
    ):
        """Resolve a single identifier: the import path's first, the
        recursion's base case.

        "If the import statement is an absolute import statement, resolution
        of I is attempted in each import directory in order until it
        succeeds. If I cannot be resolved in any import directory, an error
        is raised."

        "If the import statement is a relative import statement, resolution
        of I is attempted in the anchor directory. An error is raised if I
        cannot be resolved."
        """
        if node.num_dots > 0:  # a relative import statement
            anchor_directory = self._anchor_directory(node, containing_sequence, state)
            if anchor_directory is None:
                return None
            resolved = self._refers_to(anchor_directory, identifier, node, state)
            if state.errors:
                return None
            if resolved is None:
                state.err(
                    f"'{identifier}' could not be resolved in "
                    f"directory '{anchor_directory}'",
                    node,
                )
            return resolved
        # an absolute import statement: "The import directories are an
        # ordered list of absolute paths of directories provided by the
        # environment in which the compiler is invoked."
        for import_directory in state.import_directories:
            resolved = self._refers_to(Path(import_directory), identifier, node, state)
            if state.errors:
                return None
            if resolved is not None:
                return resolved
        state.err(
            f"'{identifier}' could not be resolved in any import directory",
            node,
        )
        return None

    def _anchor_directory(
        self, node: AstImport, containing_sequence: SequenceContext, state: CompileState
    ) -> Path | None:
        """ "Relative import statements have an anchor directory, which is
        the Nth parent directory of the absolute path of the sequence file
        containing the statement, where N is the number of dots preceding
        `import_path`. If the sequence was not read from a file, or if there
        is no Nth parent directory, an error is raised."
        """
        if containing_sequence.dir_path is None:
            state.err(
                "Relative import statement in a sequence that was not read "
                "from a file",
                node,
            )
            return None
        # The 1st parent directory of the sequence file is the directory
        # containing it; each dot past the first moves one more parent up.
        anchor_directory = Path(containing_sequence.dir_path)
        for _ in range(node.num_dots - 1):
            parent = anchor_directory.parent
            if parent == anchor_directory:
                state.err(
                    f"Relative import statement has no anchor directory: "
                    f"there is no {node.num_dots}th parent directory",
                    node,
                )
                return None
            anchor_directory = parent
        return anchor_directory

    def _refers_to(
        self, directory: Path, identifier: str, node: AstImport, state: CompileState
    ):
        """ "In a directory D, an identifier I refers to the definition in
        D named I. If D contains two definitions named I (a sequence file
        and a subdirectory of one name), an error is raised."

        Returns the definition, or None either when the directory contains
        none (the caller words its "could not be resolved" error) or on the
        two-definitions error (reported here; the caller distinguishes the
        two by state.errors)."""
        sequence_file = directory / (identifier + ".fpy")
        subdirectory = directory / identifier
        has_sequence_file = sequence_file.is_file()
        has_subdirectory = subdirectory.is_dir()
        if has_sequence_file and has_subdirectory:
            state.err(
                f"'{identifier}' refers to both a sequence file and a directory "
                f"in '{directory}'",
                node,
            )
            return None
        if has_sequence_file:
            return _SequenceDefinition(sequence_file)
        if has_subdirectory:
            return _DirectoryDefinition(subdirectory)
        return None


class BindImports:
    """Binding (SPEC.md Imports, Semantics).

    "An import statement associates one or more qualified names with
    definitions in the importing scope."

    Runs on the ResolvedImport records after `DefineFunctions` /
    `DefineVariables`, so that every sequence's definitions exist to be
    associated. The records are in inner-first order: a sequence's own
    import statements bind before an import statement naming that sequence,
    so a definition it re-exports is in its scope by the time an importer
    asks for it.
    """

    def run(self, body, state: CompileState):
        for resolved in state.resolved_imports:
            self._import_statement(resolved, state)
            if state.errors:
                return

    def _import_statement(self, resolved: ResolvedImport, state: CompileState):
        node = resolved.node
        # "The importing sequence is the sequence containing the import
        # statement; the importing scope is its scope."
        importing_scope = state.enclosing_scope[resolved.importing_sequence.block]
        # "The imported sequence definition is the sequence definition the
        # import statement's import path refers to; the imported sequence
        # is its sequence." This is the imported sequence's scope.
        imported_scope = state.enclosing_scope[resolved.imported_sequence.block]

        if node.is_star:
            self._import_star_statement(node, importing_scope, imported_scope, state)
        elif node.is_from:
            self._import_from_statement(node, importing_scope, imported_scope, state)
        else:
            self._direct_import_statement(
                resolved, importing_scope, imported_scope, state
            )

    def _import_star_statement(
        self,
        node: AstImport,
        importing_scope: Scope,
        imported_scope: Scope,
        state: CompileState,
    ):
        """ "For an import-star statement:
        For each definition D with name N in the imported sequence's scope:
        1. If N begins with an underscore, skip it.
        2. Otherwise, associate N with D in the importing scope."
        """
        for name, definition in imported_scope.own_symbols().items():
            # 1. If N begins with an underscore, skip it.
            if name.startswith("_"):
                continue
            # 2. Otherwise, associate N with D in the importing scope.
            self._associate(importing_scope, name, definition, node, state)
            if state.errors:
                return

    def _import_from_statement(
        self,
        node: AstImport,
        importing_scope: Scope,
        imported_scope: Scope,
        state: CompileState,
    ):
        """ "For other import-from statements:
        For each member with name N and optional alias A in the `members`
        list:
        1. If there is no definition named N in the imported sequence's
           scope, an error is raised.
        2. Otherwise, let D be that definition.
        3. If the optional alias A is provided, associate A with D in the
           importing scope.
        4. Otherwise, associate N with D in the importing scope."
        """
        associated: dict[str, str] = {}
        for member_name, alias in node.members:
            # 1. If there is no definition named N in the imported
            #    sequence's scope, an error is raised.
            definition = imported_scope.own_symbols().get(member_name)
            if definition is None:
                state.err(
                    f"The imported sequence has no definition named "
                    f"'{member_name}'",
                    node,
                )
                return
            # 2. Otherwise, let D be that definition.
            self._underscore_warning(member_name, node, state)
            name = alias if alias is not None else member_name
            # Diagnostic beyond the Imports spec: one members list importing one
            # member under one name more than once folds like any
            # re-association, but is a near-certain typo.
            if associated.get(name) == member_name:
                state.warn(
                    WarningType.IMPORT_DUPLICATE,
                    f"'{member_name}' is imported more than once by this statement",
                    node,
                )
            associated[name] = member_name
            # 3. If the optional alias A is provided, associate A with D in
            #    the importing scope.
            # 4. Otherwise, associate N with D in the importing scope.
            self._associate(importing_scope, name, definition, node, state)
            if state.errors:
                return

    def _direct_import_statement(
        self,
        resolved: ResolvedImport,
        importing_scope: Scope,
        imported_scope: Scope,
        state: CompileState,
    ):
        """ "Otherwise, the import statement is a direct import statement.
        Let D be the imported sequence definition:
        1. If the optional alias A is provided, A is associated with D in
           the importing scope.
        2. Otherwise, `import_path` is the qualified name of D in the
           importing sequence: each proper, non-empty prefix of
           `import_path` is associated with the directory definition it
           refers to, and `import_path` is associated with D."
        """
        node = resolved.node
        if node.alias is not None:
            # 1. If the optional alias A is provided, A is associated with D
            #    in the importing scope.
            existing = self._module_or_sequence_associated(importing_scope, node.alias)
            definition = self._sequence_definition(
                existing, node.alias, resolved, imported_scope, state
            )
            if definition is None:
                return
            self._associate(importing_scope, node.alias, definition, node, state)
            return
        # 2. Otherwise, `import_path` is the qualified name of D in the
        #    importing sequence.
        self._associate_qualified_name(resolved, importing_scope, imported_scope, state)

    def _associate_qualified_name(
        self,
        resolved: ResolvedImport,
        importing_scope: Scope,
        imported_scope: Scope,
        state: CompileState,
    ):
        """ "`import_path` is the qualified name of D in the importing
        sequence: each proper, non-empty prefix of `import_path` is
        associated with the directory definition it refers to, and
        `import_path` is associated with D." (step 2 of a direct import
        statement)

        Resolution descended the import path through directories, so the
        prefix of length k refers to the directory k levels below the root:
        the (n-k)th parent of the sequence definition's file, for a path of
        n identifiers."""
        node = resolved.node
        path = node.path
        file_path = resolved.imported_sequence.file_path

        # The first identifier reuses what the importing scope already
        # associates its name with, if that is a directory or sequence
        # symbol (any other occupant is the collision _associate reports
        # below).
        existing = self._module_or_sequence_associated(importing_scope, path[0])
        first_symbol = None
        container = None
        for i, identifier in enumerate(path):
            if i > 0:
                existing = container.get(identifier)
            if i == len(path) - 1:
                symbol = self._sequence_definition(
                    existing, identifier, resolved, imported_scope, state
                )
            else:
                # The prefix ending at this identifier (length i + 1) refers
                # to the directory len(path) - (i + 1) levels above F's own.
                directory = Path(file_path).parents[len(path) - i - 2]
                symbol = self._directory_definition(
                    existing, identifier, directory, node, state
                )
            if symbol is None:
                return
            if container is None:
                first_symbol = symbol
            else:
                container[identifier] = symbol
            container = symbol

        # Associate the first identifier's name in the importing scope, now
        # that the chain below it is complete (the name groups it resides in
        # derive from the definitions it holds -- see _associate).
        self._associate(importing_scope, path[0], first_symbol, node, state)

    def _directory_definition(
        self,
        existing,
        name: str,
        directory: Path,
        node: AstImport,
        state: CompileState,
    ):
        """The symbol for an import-path prefix's name: the directory
        definition the prefix refers to, as associated in the importing
        scope.

        One directory is one definition: an existing symbol for the same
        directory is the same association again and is reused (`import
        pkg.a` and `import pkg.b` share `pkg`); one for a different
        directory, for a sequence, or for any other definition is a
        different definition for the name -- an error."""
        if existing is None:
            return DirectorySymbol(str(directory))
        if is_instance_compat(existing, SequenceSymbol):
            state.err(
                f"'{name}' is imported both as a directory and as a sequence file",
                node,
            )
            return None
        if is_instance_compat(existing, DirectorySymbol):
            if existing.directory != str(directory):
                state.err(
                    f"Import of '{name}' collides with an existing imported "
                    f"directory of the same name",
                    node,
                )
                return None
            return existing
        state.err(
            f"Import of '{name}' collides with an existing definition",
            node,
        )
        return None

    def _sequence_definition(
        self,
        existing,
        name: str,
        resolved: ResolvedImport,
        imported_scope: Scope,
        state: CompileState,
    ):
        """ "Let D be the imported sequence definition": the symbol for the
        name the import statement associates with it (the import path's last
        identifier, or the alias).

        One file is one definition, so D is the imported sequence's one
        shared SequenceSymbol: an existing occupant that IS that symbol is
        the same association again and stands; a different sequence, a
        directory, or any other definition is a different definition for
        the name -- an error.

        The symbol's entries are the definitions in the imported sequence's
        scope, copied in here and refreshed on each association so that
        definitions bound into that scope by earlier bindings (its own
        imports) are carried over."""
        node = resolved.node
        definition = resolved.imported_sequence.definition
        assert definition is not None, "an imported sequence always has a file"
        if existing is not None and existing is not definition:
            if is_instance_compat(existing, SequenceSymbol):
                state.err(
                    f"Import of '{name}' collides with an existing imported "
                    f"sequence of the same name",
                    node,
                )
            elif is_instance_compat(existing, ModuleSymbol):
                state.err(
                    f"'{name}' is imported both as a directory and as a "
                    f"sequence file",
                    node,
                )
            else:
                state.err(
                    f"Import of '{name}' collides with an existing definition",
                    node,
                )
            return None
        definition.update(imported_scope.own_symbols())
        return definition

    def _module_or_sequence_associated(self, scope: Scope, name: str):
        """The directory or sequence symbol *name* is already associated
        with in *scope* itself, or None. Such a symbol may be bound in the
        callable and/or value group; either will do -- one name denotes one
        symbol."""
        for ng in (NameGroup.CALLABLE, NameGroup.VALUE):
            candidate = scope.get(ng, name)
            if is_instance_compat(candidate, ModuleSymbol):
                return candidate
        return None

    def _associate(
        self, scope: Scope, name: str, definition, node: AstImport, state: CompileState
    ):
        """Associate *name* with *definition* in the importing scope.

        "Associating a name with a definition it is already associated with
        changes nothing. Associating a name with a definition different from
        the one it is associated with is an error." An occupant that IS
        *definition* is the same association arriving again -- the same
        definition reached by two import routes, or the chain this statement
        just extended -- and stands; any other occupant in a shared name
        group errors.

        Name groups are beyond the Imports spec (see SPEC.md "Names and
        scopes"): a name is
        associated per name group, and a module or sequence symbol resides
        in the groups of the definitions it (transitively) holds -- a symbol
        holding no definitions resides in no group, and associating it binds
        nothing. A name that is taken only in an enclosing (base) scope is
        shadowed rather than collided with: a warning, not an error."""
        groups = self._name_groups(definition)
        for ng in groups:
            existing = scope.get(ng, name)
            if existing is not None and existing is not definition:
                state.err(
                    f"Import of '{name}' collides with an existing definition",
                    node,
                )
                return
        for ng in groups:
            outer = scope.lookup(ng, name)
            if outer is not None and outer is not definition:
                warning = (
                    WarningType.SHADOW_CALLABLE
                    if ng is NameGroup.CALLABLE
                    else WarningType.SHADOW_VALUE
                )
                state.warn(
                    warning,
                    f"Import of '{name}' shadows an existing definition",
                    node,
                )
            scope.define(ng, name, definition)

    def _name_groups(self, definition, visited: set | None = None) -> set:
        """The name groups *definition* resides in (see _associate). The
        *visited* set guards against a sequence symbol reached through
        itself (a sequence importing itself re-exports its own symbol)."""
        if is_instance_compat(definition, ModuleSymbol):
            if visited is None:
                visited = set()
            if id(definition) in visited:
                return set()
            visited.add(id(definition))
            groups = set()
            for held in definition.values():
                groups |= self._name_groups(held, visited)
            return groups
        if is_instance_compat(definition, CallableSymbol):
            return {NameGroup.CALLABLE}
        if is_instance_compat(definition, VariableSymbol):
            return {NameGroup.VALUE}
        # Any other definition resolves in the value name group.
        return {NameGroup.VALUE}

    def _underscore_warning(self, name: str, node: AstImport, state: CompileState):
        """Diagnostic beyond the Imports spec: an importing sequence
        naming a definition of an imported sequence by a name that begins
        with an underscore emits the import-underscore warning."""
        if name.startswith("_"):
            state.warn(
                WarningType.IMPORT_UNDERSCORE,
                f"'{name}' is a library-internal definition (its name begins "
                f"with an underscore)",
                node,
            )


class WarnImportUnderscore(Visitor):
    """Diagnostic beyond the Imports spec: warn when the importing
    sequence *uses* an underscore-prefixed imported definition via a
    qualified name (`lib._helper`).

    A bare name never needs this: the only import form that associates a
    definition under a bare name without naming it is an import-star
    statement, and that one skips underscore names entirely. Import
    statements that DO name an underscore definition warn as `BindImports`
    processes the statement."""

    def visit_AstGetAttr(self, node: AstGetAttr, state):
        if not node.attr.startswith("_"):
            return
        parent_sym = state.resolved_symbols.get(node.parent)
        if is_instance_compat(parent_sym, ModuleSymbol):
            state.warn(
                WarningType.IMPORT_UNDERSCORE,
                f"'{node.attr}' is a library-internal definition (its name "
                f"begins with an underscore)",
                node,
            )
