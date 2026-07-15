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

* `InlineImports` runs first, on the raw AST. It resolves each import to a file,
  recursively parses it, collects its statements as a sibling block in
  state.imported_blocks, and records an `ImportBinding` for each import.
  _build_compilation_unit then installs those blocks under the library root.

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
    AstIdent,
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


class InlineImports:
    # FIXME can we split resolving and inlining into separate passes?
    """Resolve and inline every import into the main sequence's AST."""

    def run(self, body: AstBlock, state):
        main_ctx = SequenceContext(
            file_path=None,
            dir_path=state.main_file_dir,
        )
        state.main_sequence = main_ctx

        body.stmts = self._inline_stmts(
            body.stmts, main_ctx, state, import_stack=[], is_main=True
        )

    def _inline_stmts(
        self, stmts, ctx: SequenceContext, state, import_stack, is_main: bool
    ):
        """Return *stmts* with each import removed. Each import's target sequence
        is collected as a sibling block in state.imported_blocks rather than
        spliced in at the import's position."""
        result = []
        for stmt in stmts:
            if is_instance_compat(stmt, AstImport):
                self._ensure_id(stmt, state)
                inlined = self._handle_import(stmt, ctx, state, import_stack)
                if state.errors:
                    # FIXME how do we know that it's okay to not return some
                    # error condition here?
                    return result
                result.extend(inlined)
            elif is_instance_compat(stmt, AstSequenceMetadata):
                # The main sequence keeps its metadata (handled by the normal
                # passes). Imported sequences' metadata was already validated
                # in _load_sequence and is dropped.
                # FIXME why do we drop the metadata from non main seqs? why not just keep it?
                # is it because otherwise we would crash because there could be multiple metadatas
                # in one file? i think the metadata checker pass should just take into account
                # that there could be multiple, one for each block, right?
                if is_main:
                    result.append(stmt)
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
    ):
        resolved = self._resolve(node, importer, state)
        if resolved is None:
            return []
        seq_path, member, file_path = resolved

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
            return []

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
                return []
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
            ImportBinding(node, importer, target, seq_path, member)
        )
        return []

    def _load_sequence(self, file_path: str, state, import_stack):
        """Parse and recursively inline the sequence at *file_path*.

        Returns (SequenceContext, inlined_statements)."""
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

        stmts = self._inline_stmts(
            parsed.stmts, ctx, state, import_stack, is_main=False
        )
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
        """An imported sequence may declare `sequence()` (no args) as its first
        statement; sequence arguments make it un-importable."""
        for i, stmt in enumerate(stmts):
            if not is_instance_compat(stmt, AstSequenceMetadata):
                continue
            if i != 0:
                # FIXME won't this already be caught in other semantic passes?
                # should we move that pass up? should we have it run as a sub pass of this?
                # should this pass handle "importing" the main sequence? i.e. just treat it
                # as yet another block?
                state.err(
                    "sequence() definition must be the first statement in the file",
                    stmt,
                )
                return
            if stmt.parameters:
                state.err(
                    "Cannot import a sequence that declares sequence arguments",
                    stmt,
                )
                return

    # -- resolution -----------------------------------------------------------

    def _resolve(self, node: AstImport, importer: SequenceContext, state):
        """Resolve an import to (seq_path, member, file_path), or None on error."""
        path = node.path
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
            candidate_dirs = [anchor]
        else:
            candidate_dirs = list(state.import_search_dirs)

        # Drop duplicate directories (after resolution) so a repeated search
        # dir cannot manufacture an ambiguity.
        # FIXME I thought that we don't allow duplicate directories? I thought
        # that was a compile error? or i guess its just ambiguous files that we dont allow
        # FIXME we should do a brief TOCTOU analysis just to find out if that's
        # something we need to consider a bit more
        seen = set()
        deduped = []
        for d in candidate_dirs:
            rp = os.path.realpath(d)
            if rp not in seen:
                seen.add(rp)
                deduped.append(d)
        candidate_dirs = deduped

        splits = []
        for d in candidate_dirs:
            s = self._split_in_dir(d, path, node.is_from)
            if s is not None:
                splits.append(s)

        if len(splits) == 0:
            state.err(
                f"Cannot resolve import '{'.'.join(path)}'; no matching sequence found",
                node,
            )
            return None
        if len(splits) > 1:
            state.err(
                f"Ambiguous import '{'.'.join(path)}': it resolves in more than "
                f"one search directory",
                node,
            )
            return None
        return splits[0]

    def _split_in_dir(self, d, path, is_from):
        """Return (seq_path, member, file_path) for how *path* splits in *d*,
        or None if it does not resolve there."""
        whole = self._file_for(d, path)
        if whole is not None:
            return (list(path), None, whole)
        # `from` paths are always the whole sequence path; only plain/alias
        # imports may split a trailing member off.
        if not is_from and len(path) > 1:
            f = self._file_for(d, path[:-1])
            if f is not None:
                return (list(path[:-1]), path[-1], f)
        return None

    def _file_for(self, d, segments):
        p = os.path.join(d, *segments) + ".fpy"
        if os.path.isfile(p):
            return os.path.realpath(p)
        return None


class BindImports:
    """Bind each recorded import into its importer's scopes.

    # FIXME don't call it desguar, that's not actually accurate
    Every import form is first *desugared* (_desugar) to a flat list of
    (name, value, mergeable) actions, each installing one definition or one
    synthesized module under a name in the importer's scope."""

    def run(self, body, state):
        for binding in state.import_bindings:
            self._bind(binding, state)
            if state.errors:
                return

    def _bind(self, binding: ImportBinding, state):
        importer_scope = state.enclosing_scope[binding.importer.block]
        for name, value, mergeable in self._desugar(binding, importer_scope, state):
            if state.errors:
                return
            self._bind_one(importer_scope, name, value, mergeable, binding.node, state)
            if state.errors:
                return

    # -- desugaring -----------------------------------------------------------

    def _desugar(self, binding: ImportBinding, importer_scope: Scope, state):
        """Expand an import into a flat list of (name, value, mergeable) actions,
        each of which installs one thing under *name* in the importer's scope:

          * a synthesized package/sequence module (mergeable=True) -- what a
            plain `import a.b.c` or an aliased `import a.b.c as x` binds; or
          * a single looked-up definition (mergeable=False) -- what a member
            import, an aliased member, or any `from` binds.

        # FIXME document how it returns on error
        Member lookups and their underscore warnings (the source side) happen
        here; collisions, shadows and module merging (the destination side)
        happen in _bind_one."""
        node = binding.node
        target_scope = state.enclosing_scope[binding.target.block]

        # `from a.b.c import ...` binds looked-up definitions under bare names,
        # introducing no module chain.
        # FIXME I wonder if we could write this in a less-indented form. basically
        # cut down on the number of sub blocks here for readability
        if node.is_from:
            if node.is_star:
                actions = []
                for name, sym in target_scope.own_symbols().items():
                    if name.startswith("_"):
                        # FIXME what if * import doesn't import underscored names? then we can remove star_underscore_names
                        importer_scope.star_underscore_names.add(name)
                    actions.append((name, sym, False))
                return actions
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
            # FIXME again wondering if we can drop this func and have a SequenceSymbol...
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

    def _bind_one(self, scope: Scope, name, value, mergeable, node, state):
        """Install *value* under *name* into *scope* (the importer's).

        # FIXME value is misleading, it's not a value, it should just be called sym i think?
        *value* is either a single definition (mergeable=False -- a function, a
        variable, or a re-exported module bound opaquely) or a synthesized
        package/sequence module (mergeable=True). A mergeable module merges with
        a package module THIS sequence already built under *name*; two sequence
        modules on one name collide.

        A name already taken in the importer's OWN scope is a same-scope
        collision (error); a name that only resolves up the parent chain -- a
        dictionary/builtin definition -- is shadowed (warning).
        """
        groups = self._symbol_groups(value)
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
                if _is_sequence_module(existing) and _is_sequence_module(value):
                    state.err(
                        f"Import of '{name}' collides with an existing imported "
                        f"sequence of the same name",
                        node,
                    )
                    return
                self._merge_modules(existing, value, node, state)
                if state.errors:
                    return
                value = existing
                merged_into = existing
                groups = self._symbol_groups(value)

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
            if outer is not None and outer is not value:
                state.warn(
                    _shadow_warning_type(ng),
                    f"Import of '{name}' shadows an existing definition",
                    node,
                )
            scope.define(ng, name, value)

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
    """Warn when the importer *uses* an underscore-prefixed imported definition.

    Covers `module._helper` member access and a bare use of a name that a
    `from ... import *` brought in under an underscore name. Import statements
    that name an underscore member warn at bind time in `BindImports`."""

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

    def visit_AstIdent(self, node: AstIdent, state):
        if not node.name.startswith("_"):
            return
        # star_underscore_names lives on the sequence's root scope -- the scope
        # in this node's chain whose parent is the shared base scope. Walk up to
        # it (a node under no sequence, e.g. a builtin lib def, finds none).
        scope = state.enclosing_scope[node]
        while scope is not None and scope.parent is not state.base_scope:
            scope = scope.parent
        if scope is not None and node.name in scope.star_underscore_names:
            state.warn(
                WarningType.IMPORT_UNDERSCORE,
                f"'{node.name}' is a library-internal definition (its name "
                f"begins with an underscore)",
                node,
            )
