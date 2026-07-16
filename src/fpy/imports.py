"""Import statement support.

Importing sequence S imports sequence T by compiling T's definitions alongside
S's and exposing them to S under a module (or, for `from` imports, directly). An
imported sequence may contain only function definitions and imports -- no
top-level statements, so importing runs no code and order does not matter.

Isolation is by construction. Each imported sequence is compiled as its own
block, a sibling of the main program under the shared library root; its callable
and value scopes are children of the base/universe scopes the library root owns.
A sequence's own functions live in its own scopes, so they are invisible to any
other sequence except through the modules an import binds, while dictionary and
builtin names still resolve up the parent chain.

The work is split into two passes:

* `LoadImports` runs first, on the raw AST. It resolves each import to a file
  (via `ImportResolver`), recursively parses it, collects its statements as a
  sibling block in state.imported_blocks, and records an `ImportBinding` for each
  import. _build_compilation_unit then installs those blocks under the library
  root.

* `BindImports` runs after `DefineFunctions`/`DefineVariables` have registered
  every sequence's definitions. It creates the module chains / direct bindings
  each recorded import calls for, handling module merging and name collisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os

import fpy.error
from fpy.error import WarningType
from fpy.symbols import (
    CallableSymbol,
    ModuleSymbol,
    NameGroup,
    Scope,
    SymbolTable,
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
    sibling block). Set once the block exists (_build_compilation_unit for the
    main sequence, _handle_import for an imported one)."""


@dataclass
class ResolvedImport:
    """The sequence file an import statement names, and how its import path split
    to name it."""

    seq_path: list[str]
    """the sequence path segments (leading dots already stripped) that name the
    file; the module chain an `import` introduces."""
    member: str | None
    """the trailing member name for `import a.b.c.foo`, else None."""
    file_path: str
    """the named sequence's file (realpath)."""


@dataclass
class ImportBinding:
    """A resolved import, recorded for `BindImports` to bind once every
    sequence's definitions exist."""

    node: AstImport
    importer: SequenceContext
    target: SequenceContext
    seq_path: list[str]
    """the sequence path segments (leading dots already stripped) that name the
    imported file; the module chain an `import` introduces."""
    member: str | None
    """the trailing member name for `import a.b.c.foo`, else None."""


def _make_module(is_sequence_module: bool) -> ModuleSymbol:
    m = SymbolTable()
    m.is_sequence_module = is_sequence_module
    return m


def _is_sequence_module(m) -> bool:
    return m.is_sequence_module


def _shadow_warning_type(ng: NameGroup) -> WarningType:
    """The shadow warning category for a name group: the callable group warns as
    `shadow-callable`, every other group (value) as `shadow-value`."""
    if ng is NameGroup.CALLABLE:
        return WarningType.SHADOW_CALLABLE
    return WarningType.SHADOW_VALUE


class ImportResolver:
    """Resolve an import statement to the sequence file it names.

    Pure path logic over the filesystem and the search path: it neither reads nor
    writes the AST, and knows nothing of sequences, blocks or scopes. The only
    thing it needs from the importer is the directory that anchors a relative
    import."""

    def resolve(
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
    ) -> list[str] | None:
        """The directories *node* is searched in: the anchor of a relative import,
        or the base search path of an absolute one. None on error."""
        if node.num_dots > 0:
            anchor = importer.dir_path
            if anchor is None:
                state.err(
                    "Relative import in a sequence with no containing directory",
                    node,
                )
                return None
            for _ in range(node.num_dots - 1):
                # FIXME should we use the Path api? is it safer?
                anchor = os.path.dirname(anchor)
            return [anchor]
        return self._dedupe(state.import_search_dirs)

    def _dedupe(self, dirs) -> list[str]:
        """Drop duplicate directories (after resolution) so a repeated search
        dir cannot manufacture an ambiguity."""
        # FIXME I thought that we don't allow duplicate directories? I thought
        # that was a compile error? or i guess its just ambiguous files that we dont allow
        # FIXME we should do a brief TOCTOU analysis just to find out if that's
        # something we need to consider a bit more
        seen = set()
        deduped = []
        for d in dirs:
            rp = os.path.realpath(d)
            if rp not in seen:
                seen.add(rp)
                deduped.append(d)
        return deduped

    def _split_in_dir(self, d, path, is_from) -> ResolvedImport | None:
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

    def _file_for(self, d, segments):
        p = os.path.join(d, *segments) + ".fpy"
        if os.path.isfile(p):
            return os.path.realpath(p)
        return None


class LoadImports:
    """Load every sequence the program transitively imports, and strip the import
    statements naming them out of the AST.

    Resolving an import (a path -> a file, `ImportResolver`) and loading one (a
    file -> a parsed, checked sequence) are separate concerns, but they cannot be
    separate passes: an import is only resolvable once its importer has been
    parsed, and parsing that importer is what surfaces the next imports to
    resolve. So the two interleave here, walking the import graph outward from
    the main sequence."""

    def __init__(self):
        self._resolver = ImportResolver()

    def run(self, body: AstBlock, state):
        main_ctx = SequenceContext(
            file_path=None,
            dir_path=state.main_file_dir,
        )
        state.main_sequence = main_ctx

        body.stmts = self._strip_imports(body.stmts, main_ctx, state, import_stack=[])

    def _strip_imports(self, stmts, ctx: SequenceContext, state, import_stack):
        """Return *stmts* with each import removed, having loaded every sequence
        they name. An import's target is collected as a sibling block in
        state.imported_blocks."""
        result = []
        for stmt in stmts:
            if is_instance_compat(stmt, AstImport):
                self._ensure_id(stmt, state)
                self._handle_import(stmt, ctx, state, import_stack)
                if state.errors:
                    # FIXME how do we know that it's okay to not return some
                    # error condition here?
                    return result
            else:
                result.append(stmt)
        return result

    def _ensure_id(self, node, state):
        # Import nodes are removed before AssignIds runs, but we still use them
        # for diagnostics, so give them a unique id up front.
        # FIXME can we consider assigning ids at parse time instead of in semantics passes?
        if node.id is None:
            node.id = state.next_node_id
            state.next_node_id += 1

    def _handle_import(
        self,
        node: AstImport,
        importer: SequenceContext,
        state: CompileState,
        import_stack,
    ) -> None:
        """Load the sequence *node* names (if it is not already loaded) and record
        an ImportBinding for BindImports."""
        resolved = self._resolver.resolve(node, importer, state)
        if resolved is None:
            return
        file_path = resolved.file_path

        # A file currently on the import stack is mid-load: importing it now is a
        # cycle.
        # FIXME type annotation for import stack?
        if file_path in import_stack:
            # FIXME would it be easy to make a better err msg for circular imports? showing the cycle?
            # If not we can ignore this for now.
            state.err(
                f"Circular import detected: '{os.path.basename(file_path)}'",
                node,
            )
            return

        # A sequence is compiled once. The first import of a file loads it and
        # installs its block; later imports of the same file (in this sequence or
        # any other) reuse that one SequenceContext, so its definitions are shared,
        # never duplicated. Whether importing the same sequence more than once is
        # allowed then falls to the ordinary name-collision rule in BindImports.
        target = state.loaded_sequences.get(file_path)
        if target is None:
            target, target_stmts = self._load_sequence(
                file_path, state, import_stack + [file_path]
            )
            if state.errors:
                return
            state.loaded_sequences[file_path] = target

            # Collect the imported sequence's statements as a sibling block of the
            # main program (installed under the library root by
            # _build_compilation_unit). CreateScopes gives this block its own
            # isolated scope; the sequence's context points at it via .block. The
            # import statement itself contributes nothing at its position -- an
            # imported sequence has only definitions (no side effects), so it
            # never executes inline.
            block = AstBlock(node.meta, target_stmts)
            target.block = block
            state.imported_blocks.append(block)

        state.import_bindings.append(
            ImportBinding(node, importer, target, resolved.seq_path, resolved.member)
        )

    def _load_sequence(self, file_path: str, state, import_stack):
        """Parse the sequence at *file_path*, check it, and load every sequence it
        imports in turn.

        Returns (SequenceContext, its statements with imports stripped)."""
        from fpy.compiler import text_to_ast

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            state.err(f"Cannot read imported sequence '{file_path}': {e}", None)
            return None, []

        ctx = SequenceContext(
            file_path=file_path,
            dir_path=os.path.dirname(file_path),
        )

        # Parse the imported file with its own diagnostic context so parse
        # errors point into it, then restore the caller's context.
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
            return ctx, []

        self._check_metadata(parsed.stmts, state)
        if state.errors:
            return ctx, []

        self._check_no_side_effects(parsed.stmts, state)
        if state.errors:
            return ctx, []

        stmts = self._strip_imports(parsed.stmts, ctx, state, import_stack)
        return ctx, stmts

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
                    # FIXME error msg misleading, astdef and astimport and astsequencemeta are top level statements
                    "and imports, not top-level statements",
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


class BindImports:
    """Bind each recorded import into its importer's scopes.

    Every import form is first *expanded* (_expand) to a flat list of
    (name, sym, mergeable) actions, each installing one definition or one
    synthesized module under a name in the importer's scope."""

    def run(self, body, state):
        for binding in state.import_bindings:
            self._bind(binding, state)
            if state.errors:
                return

    def _bind(self, binding: ImportBinding, state):
        importer_scope = state.enclosing_scope[binding.importer.block]
        for name, sym, mergeable in self._expand(binding, state):
            if state.errors:
                return
            self._bind_one(importer_scope, name, sym, mergeable, binding.node, state)
            if state.errors:
                return

    def _expand(self, binding: ImportBinding, state):
        """Expand an import into a flat list of (name, sym, mergeable) actions,
        each of which installs one thing under *name* in the importer's scope:

          * a synthesized package/sequence module (mergeable=True) -- what a
            plain `import a.b.c` or an aliased `import a.b.c as x` binds; or
          * a single looked-up definition (mergeable=False) -- what a member
            import, an aliased member, or any `from` binds.

        On error, reports it via state.err and returns whatever actions were
        built so far. The caller checks state.errors before installing any of
        them, so a partial list is never bound."""
        node = binding.node
        target_scope = state.enclosing_scope[binding.target.block]

        # `from a.b.c import *` binds the sequence's public definitions, each
        # under its own name. A star import takes only the
        # public surface: underscore definitions are internal and stay unbound
        if node.is_from and node.is_star:
            return [
                (name, sym, False)
                for name, sym in target_scope.own_symbols().items()
                if not name.startswith("_")
            ]

        # `from a.b.c import m [as n], ...` binds looked-up definitions under
        # bare names, introducing no module chain.
        if node.is_from:
            actions = []
            for member_name, alias in node.members:
                sym = self._lookup_definition(target_scope, member_name, node, state)
                if sym is None:
                    return actions
                self._maybe_underscore_warn(member_name, node, state)
                actions.append((alias or member_name, sym, False))
            return actions

        # `import a.b.c[.member] [as alias]`.
        if binding.member is not None:
            sym = self._lookup_definition(target_scope, binding.member, node, state)
            if sym is None:
                return []
            self._maybe_underscore_warn(binding.member, node, state)
            if node.alias is not None:
                # `import a.b.c.member as x` binds the definition directly under x.
                return [(node.alias, sym, False)]
            # `import a.b.c.member` binds a.b.c as a module holding just `member`.
            # FIXME _make_module/_is_sequence_module are the is_sequence_module
            # bool's only real call sites; both collapse into Sequence/Package
            # Module classes when symbols.py:131 lands.
            leaf = _make_module(is_sequence_module=True)
            leaf[binding.member] = sym
        else:
            # Whole sequence: a module of all its definitions.
            leaf = _make_module(is_sequence_module=True)
            for name, sym in target_scope.own_symbols().items():
                leaf[name] = sym
            if node.alias is not None:
                # `import a.b.c as x` binds the whole module under x (no chain).
                return [(node.alias, leaf, True)]

        # Plain `import a.b.c[.member]`: wrap the leaf in its package chain and
        # bind the chain's root name.
        root_name, root = self._build_chain(binding.seq_path, leaf)
        return [(root_name, root, True)]

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

    def _build_chain(self, seq_path, leaf):
        """Wrap *leaf* in package modules for a dotted sequence path, returning
        (root_name, root_module)."""
        current = leaf
        for i in range(len(seq_path) - 2, -1, -1):
            parent = _make_module(is_sequence_module=False)
            parent[seq_path[i + 1]] = current
            current = parent
        return seq_path[0], current

    def _symbol_groups(self, sym) -> set:
        if is_instance_compat(sym, ModuleSymbol):
            return self._module_groups(sym)
        if is_instance_compat(sym, CallableSymbol):
            return {NameGroup.CALLABLE}
        if is_instance_compat(sym, VariableSymbol):
            return {NameGroup.VALUE}
        # Any other value-like definition resolves in the value name group.
        return {NameGroup.VALUE}

    def _module_groups(self, module) -> set:
        groups = set()
        for sym in module.values():
            groups |= self._symbol_groups(sym)
        return groups

    def _bind_one(self, scope: Scope, name, sym, mergeable, node, state):
        """Install *sym* under *name* into *scope* (the importer's).

        *sym* is either a single definition (mergeable=False -- a function, a
        variable, or a re-exported module bound opaquely) or a synthesized
        package/sequence module (mergeable=True). A mergeable module merges with
        a package module THIS sequence already built under *name*; two sequence
        modules on one name collide.

        A name already taken in the importer's OWN scope is a same-scope
        collision (error); a name that only resolves up the parent chain -- a
        dictionary/builtin definition -- is shadowed (warning).
        """
        groups = self._symbol_groups(sym)
        if not groups:
            # An empty sequence's module holds nothing and belongs to no name
            # group; there is nothing to bind.
            return

        # A mergeable module folds into a package module already built here.
        merged_into = None
        if mergeable:
            existing = None
            for ng in (NameGroup.CALLABLE, NameGroup.VALUE):
                candidate = scope.get(ng, name)
                if is_instance_compat(candidate, ModuleSymbol):
                    existing = candidate
                    break
            if existing is not None:
                if _is_sequence_module(existing) and _is_sequence_module(sym):
                    state.err(
                        f"Import of '{name}' collides with an existing imported "
                        f"sequence of the same name",
                        node,
                    )
                    return
                self._merge_modules(existing, sym, node, state)
                if state.errors:
                    return
                sym = existing
                merged_into = existing
                groups = self._symbol_groups(sym)

        # A name occupied in the importer's own scope collides -- unless it is
        # the very module we just merged into.
        for ng in groups:
            own = scope.get(ng, name)
            if own is not None and own is not merged_into:
                state.err(
                    f"Import of '{name}' collides with an existing definition",
                    node,
                )
                return
        # The name is free (or holds the just-merged module). If it still
        # resolves up the parent chain to a base (dictionary) definition, the
        # import shadows it: a warning, not an error.
        for ng in groups:
            outer = scope.lookup(ng, name)
            if outer is not None and outer is not sym:
                state.warn(
                    _shadow_warning_type(ng),
                    f"Import of '{name}' shadows an existing definition",
                    node,
                )
            scope.define(ng, name, sym)

    def _merge_modules(self, existing, incoming, node, state):
        """Merge *incoming* module's members into *existing*."""
        for key, sym in incoming.items():
            if key in existing:
                ex = existing[key]
                if (
                    is_instance_compat(ex, ModuleSymbol)
                    and is_instance_compat(sym, ModuleSymbol)
                    and not (_is_sequence_module(ex) and _is_sequence_module(sym))
                ):
                    self._merge_modules(ex, sym, node, state)
                    if state.errors:
                        return
                else:
                    state.err(
                        f"Imported module member '{key}' collides on merge",
                        node,
                    )
                    return
            else:
                existing[key] = sym
        if _is_sequence_module(incoming):
            existing.is_sequence_module = True


class WarnImportUnderscore(Visitor):
    """Warn when the importer *uses* an underscore-prefixed imported definition
    via `module._helper` member access.

    A bare name never needs this: the only import form that binds a definition
    under a bare name without naming it is `from ... import *`, and that one
    does not bind underscore names at all. Import statements that DO name an
    underscore member warn at bind time in `BindImports`."""

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
