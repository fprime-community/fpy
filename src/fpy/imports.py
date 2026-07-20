"""Import statement support.

The importer sequence S imports sequence T by compiling T's definitions alongside
S's and exposing them to S under T's name (or, for `from` imports, directly). An
imported sequence may contain only function definitions and imports -- nothing
that executes -- so importing runs no code, and order does not matter. Nor do
cycles: a sequence may transitively import itself.

Each imported sequence is compiled as its own block, a sibling of the main
program under the shared library root; its scope is a child of the base scopes
the library root owns.

The work is split into two passes:

* `LoadImports` runs first, on the raw AST. It resolves each import to a file,
  recursively parses it, collects its statements as a sibling block in
  state.imported_blocks, and records an `ImportAnalysis` for each import.
  _build_root_block then installs those blocks under the library root.

* `DefineImports` runs after `DefineFunctions`/`DefineVariables` have registered
  every sequence's definitions. It expands each recorded import into the
  syntactic definitions the statement makes -- a module definition per
  enclosing directory on its sequence path, a sequence definition for the
  file, or alias definitions for `from`/aliased-member forms -- and combines
  them, per qualified name, into the semantic symbols the importer's scope
  holds. Two definitions of one qualified name in a shared name group
  collide unless they are semantically equivalent: module definitions always
  are (varying only by name), sequence definitions are iff they name one
  file, and alias definitions iff they name one definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fpy.error
from fpy.error import WarningType
from fpy.symbols import (
    CallableSymbol,
    ModuleSymbol,
    NameGroup,
    Scope,
    SequenceSymbol,
    Symbol,
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
    """A single sequence taking part in a compilation.

    Holds only what an import needs: where the sequence came from (for resolving
    its own relative imports) and the AST block it was compiled into.."""

    file_path: str | None
    """the sequence's resolved file path (realpath), or None for the main
    sequence when it was compiled from a stream."""
    dir_path: str | None
    """the directory the sequence lives in, anchoring its relative imports.
    None when the sequence has no location."""
    block: AstBlock = None
    """the sequence's AST block (the main block, or an imported sequence's
    sibling block). Set once the block exists (_build_root_block for the
    main sequence; at construction for an imported one)."""


@dataclass
class ResolvedImport:
    """The sequence file an import statement names, and how its import path split
    to name it."""

    seq_path: list[str]
    """the sequence path segments (leading dots already stripped) that name the
    file; the dotted path a plain `import` binds (modules for the
    directories, the sequence at the leaf)."""
    member: str | None
    """the trailing member name for `import a.b.c.foo`, else None."""
    file_path: str
    """the named sequence's file (realpath)."""


@dataclass
class ImportAnalysis:
    """A resolved import, recorded for `DefineImports` to turn into scope
    definitions once every sequence's own definitions exist."""

    node: AstImport
    importer: SequenceContext
    target: SequenceContext
    seq_path: list[str]
    """the sequence path segments (leading dots already stripped) that name the
    imported file; the dotted path a plain `import` binds (modules for the
    directories, the sequence at the leaf)."""
    member: str | None
    """the trailing member name for `import a.b.c.foo`, else None."""


class LoadImports:
    """Load every sequence the program transitively imports, and strip the import
    statements naming them out of the AST."""

    def run(self, body: AstBlock, state: CompileState):
        main_ctx = SequenceContext(
            file_path=None,
            dir_path=state.main_file_dir,
        )
        state.main_sequence = main_ctx

        body.stmts = self._load_imports_in(body.stmts, main_ctx, state)

    def _load_imports_in(self, stmts, ctx: SequenceContext, state):
        """Load every sequence the imports in *stmts* name, and return *stmts*
        with those import statements removed. Each named sequence is collected as
        a sibling block in state.imported_blocks.

        On error we stop and hand back the statements stripped so far."""
        result = []
        for stmt in stmts:
            if is_instance_compat(stmt, AstImport):
                self._load_import(stmt, ctx, state)
                if state.errors:
                    return result
            else:
                result.append(stmt)
        return result

    def _load_import(
        self,
        node: AstImport,
        importer: SequenceContext,
        state: CompileState,
    ) -> None:
        """Load the sequence *node* names (if it is not already loaded) and record
        an ImportAnalysis for DefineImports."""
        resolved = self._resolve(node, importer, state)
        if resolved is None:
            return

        # A sequence is compiled once. The first import of a file loads it and
        # installs its block; later imports of the same file (in this sequence or
        # any other) reuse that one SequenceContext, so its definitions are shared,
        # never duplicated. Whether importing the same sequence more than once is
        # allowed then falls to the ordinary name-collision rule in DefineImports.
        target = state.loaded_sequences.get(resolved.file_path)
        if target is None:
            target = self._load_sequence(resolved.file_path, node.meta, state)
            if state.errors:
                return
            assert target is not None

        state.import_analyses.append(
            ImportAnalysis(node, importer, target, resolved.seq_path, resolved.member)
        )

    def _load_sequence(self, file_path: str, meta, state) -> SequenceContext | None:
        """Parse the sequence at *file_path*, check it, register it, and load
        every sequence it imports in turn. Returns its SequenceContext, or None
        with an error reported.

        *meta* is the position of the import that first named the file, used for
        the block we build from it."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            state.err(f"Cannot read imported sequence '{file_path}': {e}", None)
            return None

        parsed = self._parse(file_path, text, state)
        if parsed is None:
            return None

        self._check_metadata(parsed.stmts, state)
        if state.errors:
            return None

        self._check_no_side_effects(parsed.stmts, state)
        if state.errors:
            return None

        # Collect the imported sequence's statements as a sibling block of the
        # main program (installed under the library root by
        # _build_root_block). CreateScopes gives this block its own
        # isolated scope; the sequence's context points at it via .block. The
        # import statement itself contributes nothing at its position -- an
        # imported sequence has only definitions (no side effects), so it
        # never executes inline.
        ctx = SequenceContext(
            file_path=file_path,
            dir_path=str(Path(file_path).parent),
            block=AstBlock(meta, parsed.stmts),
        )
        state.loaded_sequences[file_path] = ctx
        state.imported_blocks.append(ctx.block)

        # Only now descend into this sequence's own imports. Registering it first
        # is what lets an import cycle resolve: an import that comes back around
        # to this file finds the context above and binds against it, instead of
        # loading the file a second time forever. Nothing about a cycle needs
        # rejecting -- an import binds names and runs nothing, and DefineImports
        # does not run until every sequence's definitions exist.
        ctx.block.stmts = self._load_imports_in(parsed.stmts, ctx, state)
        return ctx

    def _parse(self, file_path: str, text: str, state):
        """Parse *text* under its own diagnostic context, so its parse errors
        point into *file_path* rather than into the importing file. Restores the
        caller's context before returning."""
        from fpy.compiler import text_to_ast

        old_file, old_text, old_lines = (
            fpy.error.file_name,
            fpy.error.input_text,
            fpy.error.input_lines,
        )
        fpy.error.file_name = file_path
        try:
            parsed = text_to_ast(text)
        finally:
            fpy.error.file_name = old_file
            fpy.error.input_text = old_text
            fpy.error.input_lines = old_lines

        if parsed is None:
            state.err(f"Failed to parse imported sequence '{file_path}'", None)
        return parsed

    def _check_no_side_effects(self, stmts, state):
        """An imported sequence may contain only function definitions and import
        statements. Any other top-level statement is a side effect that would
        have to execute at the importer's position; since an imported sequence is
        a sibling scope block (it does not run inline), such a statement has no
        place to execute and is an error."""
        for stmt in stmts:
            if not is_instance_compat(stmt, (AstDef, AstImport, AstSequenceMetadata)):
                state.err(
                    "An imported sequence may contain only function definitions "
                    "and imports, not statements that execute",
                    stmt,
                )
                return

    def _check_metadata(self, stmts, state):
        """Sequence arguments make a sequence un-importable: nothing would supply
        them at the import."""
        for stmt in stmts:
            if is_instance_compat(stmt, AstSequenceMetadata) and stmt.parameters:
                state.err(
                    "Cannot import a sequence that declares sequence arguments",
                    stmt,
                )
                return

    def _resolve(
        self, node: AstImport, importer: SequenceContext, state
    ) -> ResolvedImport | None:
        """Resolve *node* against its candidate directories. Reports an error and
        returns None if it names no file, or a file in more than one directory."""
        candidate_dirs = self._candidate_dirs(node, importer, state)
        if candidate_dirs is None:
            return None

        splits = []
        for d in candidate_dirs:
            s = self._split_in_dir(d, node.path, node.is_from)
            if s is not None:
                splits.append(s)

        if len(splits) == 0:
            state.err(
                f"Cannot resolve import '{'.'.join(node.path)}'; no matching sequence found",
                node,
            )
            return None
        if len(splits) > 1:
            state.err(
                f"Ambiguous import '{'.'.join(node.path)}': it resolves in more than "
                f"one search directory",
                node,
            )
            return None
        return splits[0]

    def _candidate_dirs(
        self, node: AstImport, importer: SequenceContext, state
    ) -> list[Path] | None:
        """The directories *node* is searched in: the anchor of a relative import,
        or the base search path of an absolute one. None on error."""
        if node.num_dots > 0:
            if importer.dir_path is None:
                state.err(
                    "Relative import in a sequence with no containing directory",
                    node,
                )
                return None
            # Each dot past the first moves the anchor one directory up. .parent
            # saturates at the root, so an over-deep relative import lands there
            # and simply finds no file.
            anchor = Path(importer.dir_path)
            for _ in range(node.num_dots - 1):
                anchor = anchor.parent
            return [anchor]
        return self._dedupe(state.import_search_dirs)

    def _dedupe(self, dirs) -> list[Path]:
        """Drop duplicate search directories, comparing them resolved. Duplicates
        are dropped rather than rejected: only a path resolving in two DISTINCT
        directories is the ambiguity the error is for, so a doubled `-i` flag must
        not manufacture one."""
        seen = set()
        deduped = []
        for d in dirs:
            path = Path(d)
            if path.resolve() not in seen:
                seen.add(path.resolve())
                deduped.append(path)
        return deduped

    def _split_in_dir(self, d: Path, path, is_from) -> ResolvedImport | None:
        """Return how *path* splits into a sequence path and an optional member in
        *d*, or None if it does not resolve there."""
        whole = self._file_for(d, path)
        if whole is not None:
            return ResolvedImport(list(path), None, whole)
        # `from` paths are always the whole sequence path; only plain/alias
        # imports may split a trailing member off.
        if not is_from and len(path) > 1:
            f = self._file_for(d, path[:-1])
            if f is not None:
                return ResolvedImport(list(path[:-1]), path[-1], f)
        return None

    def _file_for(self, d: Path, segments) -> str | None:
        """The resolved path of `<d>/<segments...>.fpy`, or None if it is not a
        file. Every segment but the last names a directory."""
        assert segments, "an import path always has at least one segment"
        *dirs, leaf = segments
        p = d.joinpath(*dirs, leaf + ".fpy")
        if p.is_file():
            return str(p.resolve())
        return None


@dataclass
class ModuleDef:
    """A syntactic module definition: one enclosing directory on an import's
    sequence path, defining the module symbol at qualified name *path* in the
    importer's global scope.

    Module definitions merge: every syntactic module definition with one
    qualified name defines the same one semantic module symbol, whose members
    are everything the definitions' statements put there."""

    path: tuple[str, ...]


@dataclass
class SequenceDef:
    """A syntactic sequence definition: the sequence file an import statement
    names, defining the sequence symbol at qualified name *path* that holds
    *members* of the file's global scope (all of them, or only the one a
    member import names).

    Syntactic sequence definitions with one qualified name that name the same
    *file* define the same one semantic sequence symbol: the file is compiled
    once, so their members are the same objects and pool idempotently. Two
    that name different files collide."""

    path: tuple[str, ...]
    file: str
    members: dict[str, Symbol]


@dataclass
class AliasDef:
    """A syntactic alias definition: an existing definition, defined again
    under a bare (possibly aliased) name in the importer's global scope --
    what each member of a `from` import and an aliased member import make.

    An alias definition is identified by the definition it names: two with
    one qualified name that name the SAME definition define the same one
    semantic alias and fold idempotently, so one definition reached by two
    import routes (a diamond) binds once. Two that name DIFFERENT
    definitions collide."""

    name: str
    sym: Symbol


class DefineImports:
    """Enter each recorded import's definitions into its importer's scope.

    One import statement is processed in three steps:

    1. `_expand` turns the statement into the syntactic definitions it makes:
       a ModuleDef per directory on its sequence path plus a SequenceDef for
       the file, or AliasDefs for `from`/aliased-member forms.
    2. `_define` installs each definition by combining it with whatever its
       qualified name already denotes: the same semantic definition folds in,
       a different one is a conflict.
    3. `_define_name` enters a symbol into the scope under its name, in every
       name group it resides in. Alias definitions reach it directly from
       `_define`; a dotted import's root symbol reaches it only after the
       whole statement is defined, because the name groups a module or
       sequence symbol resides in derive from the members steps 1-2 give
       it."""

    def run(self, body, state):
        for analysis in state.import_analyses:
            self._define_import(analysis, state)
            if state.errors:
                return

    def _define_import(self, analysis: ImportAnalysis, state):
        importer_scope = state.enclosing_scope[analysis.importer.block]
        defs = self._expand(analysis, state)
        if state.errors:
            return
        # The module/sequence symbol the statement's chain root defines, by
        # name -- held here, off the scope, until every definition of the
        # statement has combined into it (the name groups it binds in derive
        # from the members they contribute).
        roots: dict[str, ModuleSymbol] = {}
        for d in defs:
            self._define(d, importer_scope, roots, analysis.node, state)
            if state.errors:
                return
        for name, sym in roots.items():
            self._define_name(importer_scope, name, sym, analysis.node, state)
            if state.errors:
                return

    def _expand(self, analysis: ImportAnalysis, state) -> list:
        """Expand an import statement into the flat list of syntactic
        definitions it makes, ordered so that every ModuleDef precedes the
        definitions at longer paths it encloses:

          * `from` forms and `import a.b.c.member as x` make AliasDefs;
          * `import a.b.c as x` makes one SequenceDef at path (x,);
          * `import a.b.c[.member]` makes a ModuleDef per proper prefix of the
            sequence path and a SequenceDef at the full path.

        On error, reports it via state.err and returns whatever definitions
        were built so far. The caller checks state.errors before installing
        any of them, so a partial list is never bound."""
        node = analysis.node
        target_scope = state.enclosing_scope[analysis.target.block]

        # `from a.b.c import *` aliases the sequence's public definitions,
        # each under its own name. A star import takes only the
        # public surface: underscore definitions are internal and stay unbound
        if node.is_from and node.is_star:
            return [
                AliasDef(name, sym)
                for name, sym in target_scope.own_symbols().items()
                if not name.startswith("_")
            ]

        # `from a.b.c import m [as n], ...` aliases looked-up definitions
        # under bare names, introducing no modules.
        if node.is_from:
            results = []
            bound: dict[str, Symbol] = {}
            for member_name, alias in node.members:
                sym = self._lookup_definition(target_scope, member_name, node, state)
                if sym is None:
                    return results
                self._maybe_underscore_warn(member_name, node, state)
                name = alias or member_name
                # One member list defining the same alias twice is legal (it
                # folds like any two aliases of one definition) but almost
                # certainly a typo: warn. One name for two DIFFERENT
                # definitions is left to the ordinary collision error.
                if bound.get(name) is sym:
                    state.warn(
                        WarningType.IMPORT_DUPLICATE,
                        f"'{name}' is imported more than once by this statement",
                        node,
                    )
                bound[name] = sym
                results.append(AliasDef(name, sym))
            return results

        # `import a.b.c.member [as alias]`: a single looked-up definition.
        if analysis.member is not None:
            sym = self._lookup_definition(target_scope, analysis.member, node, state)
            if sym is None:
                return []
            self._maybe_underscore_warn(analysis.member, node, state)
            if node.alias is not None:
                # `import a.b.c.member as x` aliases the definition directly
                # under x.
                return [AliasDef(node.alias, sym)]
            # `import a.b.c.member` defines sequence a.b.c holding just `member`.
            return self._chain_defs(analysis, {analysis.member: sym})

        # `import a.b.c [as alias]`: the whole sequence.
        members = target_scope.own_symbols()
        if node.alias is not None:
            # `import a.b.c as x` defines the sequence at path (x,): no
            # enclosing directories are named, so no module definitions.
            return [SequenceDef((node.alias,), analysis.target.file_path, members)]
        return self._chain_defs(analysis, members)

    def _chain_defs(self, analysis: ImportAnalysis, members) -> list:
        """The definitions a chain-introducing import makes: a module
        definition per enclosing directory on the sequence path (each proper,
        non-empty prefix), then the sequence definition at the full path."""
        path = tuple(analysis.seq_path)
        defs = [ModuleDef(path[:i]) for i in range(1, len(path))]
        defs.append(SequenceDef(path, analysis.target.file_path, members))
        return defs

    def _define(self, d, scope: Scope, roots, node, state):
        """Install one syntactic definition.

        An AliasDef binds directly into the importer's scope. A module or
        sequence definition combines with the symbol its qualified name
        already denotes: a root-level name denotes what the scope binds (with
        the result held in *roots* for the caller to bind last), and a deeper
        name denotes a member of the module at its path prefix -- which this
        same statement defined just before it, prefixes first."""
        if is_instance_compat(d, AliasDef):
            self._define_name(scope, d.name, d.sym, node, state)
            return
        name = d.path[-1]
        if len(d.path) == 1:
            combined = self._combine(d, self._existing_module(scope, name), node, state)
            if combined is None:
                return
            roots[name] = combined
            return
        container = roots[d.path[0]]
        for seg in d.path[1:-1]:
            container = container[seg]
            assert is_instance_compat(
                container, ModuleSymbol
            ) and not is_instance_compat(
                container, SequenceSymbol
            ), f"prefix {seg} was defined as a module by an earlier ModuleDef"
        combined = self._combine(d, container.get(name), node, state)
        if combined is None:
            return
        container[name] = combined

    def _combine(self, d, existing, node, state):
        """Combine one syntactic module/sequence definition *d* with
        *existing*, the symbol its qualified name already denotes (None if
        unclaimed), and return the semantic symbol the name denotes after it:

          * an unclaimed name: the definition creates its symbol;
          * module definition meets module symbol: they define the same one
            semantic module, so the existing symbol stands unchanged;
          * sequence definition meets sequence symbol naming the SAME file:
            they define the same one semantic sequence, and this occurrence
            pools its members into it (idempotently -- same objects);
          * sequence definition meets sequence symbol naming a DIFFERENT
            file: a collision;
          * one name claimed as both a module and a sequence: forbidden,
            whichever came first.

        Returns None with the error reported on conflict."""
        if existing is not None:
            assert is_instance_compat(existing, ModuleSymbol), existing
        name = d.path[-1]
        is_seq_def = is_instance_compat(d, SequenceDef)
        if existing is None:
            if not is_seq_def:
                return ModuleSymbol()
            sym = SequenceSymbol(d.file)
            sym.update(d.members)
            return sym
        if is_instance_compat(existing, SequenceSymbol) != is_seq_def:
            state.err(
                f"'{name}' is imported both as a module and as a sequence",
                node,
            )
            return None
        if not is_seq_def:
            return existing
        if existing.source_file != d.file:
            state.err(
                f"Import of '{name}' collides with an existing imported "
                f"sequence of the same name",
                node,
            )
            return None
        existing.update(d.members)
        return existing

    def _lookup_definition(self, target_scope: Scope, name: str, node, state):
        sym = target_scope.own_symbols().get(name)
        if sym is None:
            state.err(
                f"Imported sequence has no definition named '{name}'",
                node,
            )
        return sym

    def _maybe_underscore_warn(self, name: str, node, state):
        if name.startswith("_"):
            state.warn(
                WarningType.IMPORT_UNDERSCORE,
                f"'{name}' is a library-internal definition (its name begins "
                f"with an underscore)",
                node,
            )

    def _symbol_groups(self, sym) -> set:
        """The name groups *sym* resides in: a module or sequence symbol
        resides in the groups of the definitions it (transitively) holds, and
        so exists exactly where its members can collide."""
        if is_instance_compat(sym, ModuleSymbol):
            groups = set()
            for member in sym.values():
                groups |= self._symbol_groups(member)
            return groups
        if is_instance_compat(sym, CallableSymbol):
            return {NameGroup.CALLABLE}
        if is_instance_compat(sym, VariableSymbol):
            return {NameGroup.VALUE}
        # Any other value-like definition resolves in the value name group.
        return {NameGroup.VALUE}

    def _define_name(self, scope: Scope, name, sym, node, state):
        """Bind *sym* under *name* in the importer's scope, in each name group
        *sym* resides in. A symbol holding no definitions resides in no group
        and binds nothing (importing an empty sequence introduces nothing).

        This is the scope-level conflict rule: an occupant that IS *sym* is
        the same semantic definition arriving again -- the chain root this
        statement merged into, or a definition this name already aliases (two
        import routes to one definition) -- and stands, folding idempotently.
        Any other occupant is a second definition of the name in that group:
        a collision. A name that is instead taken only up the parent chain,
        in a base (dictionary/builtin) scope, is shadowed: a warning, not an
        error."""
        groups = self._symbol_groups(sym)
        for ng in groups:
            existing = scope.get(ng, name)
            if existing is not None and existing is not sym:
                state.err(
                    f"Import of '{name}' collides with an existing definition",
                    node,
                )
                return
        for ng in groups:
            outer = scope.lookup(ng, name)
            if outer is not None and outer is not sym:
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
            scope.define(ng, name, sym)

    def _existing_module(self, scope: Scope, name):
        """The semantic module/sequence symbol *name* already denotes in
        *scope*'s own groups, or None. A module can be bound in the callable
        and/or value group; either does -- one name denotes one symbol."""
        for ng in (NameGroup.CALLABLE, NameGroup.VALUE):
            candidate = scope.get(ng, name)
            if is_instance_compat(candidate, ModuleSymbol):
                return candidate
        return None


class WarnImportUnderscore(Visitor):
    """Warn when the importer *uses* an underscore-prefixed imported definition
    via qualified member access (`lib._helper`).

    A bare name never needs this: the only import form that binds a definition
    under a bare name without naming it is `from ... import *`, and that one
    does not bind underscore names at all. Import statements that DO name an
    underscore member warn as `DefineImports` processes the statement."""

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
