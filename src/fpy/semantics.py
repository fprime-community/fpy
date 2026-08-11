from __future__ import annotations
from dataclasses import fields, replace as dc_replace
from datetime import datetime, timezone
from decimal import Decimal
import decimal
import itertools
from pathlib import Path
import struct
from typing import Union

from fpy.bytecode.assembler import read_bin_arg_specs, resolve_arg_specs

from fpy.error import CompileError
from fpy.macros import TIME_MACRO
from fpy.types import (
    ARBITRARY_PRECISION_TYPES,
    SIGNED_INTEGER_TYPES,
    SPECIFIC_NUMERIC_TYPES,
    TIME_OPS,
    UNSIGNED_INTEGER_TYPES,
    FpyType,
    FpyValue,
    StructMember,
    TypeKind,
    INTEGER,
    FLOAT,
    INTERNAL_STRING,
    RANGE,
    NOTHING,
    SIZED,
    BOOL,
    TIME,
    TIME_BASE,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    SEQ_ARGS,
    ChDef,
    PrmDef,
    is_instance_compat,
)
from fpy.state import (
    CompileState,
    ForLoopAnalysis,
)
from fpy.error import WarningType
from fpy.symbols import (
    BuiltinFuncSymbol,
    CallableSymbol,
    CastSymbol,
    CommandSymbol,
    FieldAccess,
    FunctionSymbol,
    NameGroup,
    ModuleSymbol,
    Scope,
    Symbol,
    TypeCtorSymbol,
    VariableSymbol,
    is_symbol_an_expr,
)
from fpy.visitors import (
    STOP_DESCENT,
    TopDownVisitor,
    Visitor,
)

# In Python 3.10+, the `|` operator creates a `types.UnionType`.
# We need to handle this for forward compatibility, but it won't exist in 3.9.
try:
    from types import UnionType

    UNION_TYPES = (Union, UnionType)
except ImportError:
    UNION_TYPES = (Union,)

from fpy.bytecode.directives import (
    BOOLEAN_OPERATORS,
    COMPARISON_OPS,
    NUMERIC_OPERATORS,
    ArrayIndexType,
    ErrorCodeType,
    LoopVarType,
    BinaryStackOp,
    UnaryStackOp,
)
from fpy.syntax import (
    AstAssert,
    AstAnonStruct,
    AstAnonArray,
    AstBinaryOp,
    AstBoolean,
    AstBreak,
    AstContinue,
    AstDef,
    AstElif,
    AstExpr,
    AstFor,
    AstGetAttr,
    AstIndexExpr,
    AstNamedArgument,
    AstNumber,
    AstPass,
    AstRange,
    AstReference,
    AstReturn,
    AstBlock,
    AstSequenceMetadata,
    AstStmt,
    AstStmtWithExpr,
    AstString,
    Ast,
    AstBlock,
    AstLiteral,
    AstIf,
    AstAssign,
    AstFuncCall,
    AstUnaryOp,
    AstIdent,
    AstWhile,
)


class AssignIds(TopDownVisitor):
    """Assigns a unique id to each node and builds the parent map."""

    def run(self, start: Ast, state: CompileState):
        self._visit(start, state)

        def _descend(node: Ast):
            if not isinstance(node, Ast):
                return
            children = []
            for field in fields(node):
                field_val = getattr(node, field.name)
                if isinstance(field_val, list):
                    if len(field_val) > 0 and isinstance(field_val[0], tuple):
                        field_val = itertools.chain.from_iterable(field_val)
                    children.extend(field_val)
                else:
                    children.append(field_val)

            for child in children:
                if not isinstance(child, Ast):
                    continue
                self._visit(child, state)
                state.parent_map[child] = node
                if len(state.errors) != 0:
                    break
                _descend(child)
                if len(state.errors) != 0:
                    break

        _descend(start)

    def visit_default(self, node, state: CompileState):
        node.id = state.next_node_id
        state.next_node_id += 1


class CreateScopes(TopDownVisitor):
    """Creates the Scope for every AstBlock

    Every block gets a fresh Scope that is a child of its enclosing block's
    scope, so scope nesting exactly follows syntactic nesting. The one exception
    is the library root, which has no enclosing block: it owns the pre-built base
    scope (dictionary/builtin symbols, created before any AST existed).

    Non-block nodes inherit the scope from their enclosing block.
    """

    def visit_default(self, node: Ast, state: CompileState):
        parent = state.parent_map.get(node)
        if parent is None:
            return
        state.enclosing_scope[node] = state.enclosing_scope[parent]

    def visit_AstBlock(self, node: AstBlock, state: CompileState):
        parent = state.parent_map.get(node)

        if parent is None:
            # The base block has no enclosing block, so it cannot make a child
            # scope. It owns the pre-built base scope (dictionary/builtin symbols,
            # created before any AST existed) instead.
            state.enclosing_scope[node] = state.base_scope
            return

        parent_scope = state.enclosing_scope[parent]
        in_function = parent_scope.in_function or isinstance(parent, AstDef)
        scope = Scope(parent=parent_scope, in_function=in_function)
        state.enclosing_scope[node] = scope

        # The main block's scope is the main sequence's scope
        if node is state.main_block:
            state.main_scope = scope


class CheckSequenceMetadataDefinedAtTop(TopDownVisitor):
    """
    Ensure a sequence() statement is the first statement of its file.

    Every sequence's block gets this check.
    """

    def visit_AstBlock(self, node: AstBlock, state: CompileState):
        for stmt in node.stmts:
            if isinstance(stmt, AstSequenceMetadata) and stmt is not node.stmts[0]:
                # sequence() is guaranteed to be only in a top-level block by the
                # grammar, so we know node is a top level block of a sequence
                # (may be an imported sequence)
                state.err(
                    "sequence() definition must be the first statement in the file",
                    stmt,
                )
                return


class CheckAssignSyntax(TopDownVisitor):

    def visit_AstAssign(self, node: AstAssign, state: CompileState):
        if not is_instance_compat(node.lhs, AstReference):
            # trying to assign a value to some complex expression like (1 + 1) = 2
            state.err("Invalid assignment", node.lhs)
            return

        if is_instance_compat(node.lhs, (AstGetAttr, AstIndexExpr)):
            # assigning to a member or array element. don't need to make a new variable,
            # space already exists
            if node.type_ann is not None:
                # type annotation on a field assignment... it already has a type!
                state.err("Cannot specify a type annotation for a field", node.type_ann)
                return
            # otherwise we good
            return

    def visit_AstDef(self, node: AstDef, state: CompileState):

        if node.parameters is None:
            return

        # Check that default arguments come after non-default arguments
        seen_default = False
        for arg in node.parameters:
            arg_name_var, arg_type_name, default_value = arg
            if default_value is not None:
                seen_default = True
            elif seen_default:
                # Non-default argument after default argument
                state.err(
                    f"Non-default parameter '{arg_name_var.name}' follows default parameter",
                    arg_name_var,
                )
                return

    def visit_AstSequenceMetadata(self, node: AstSequenceMetadata, state: CompileState):
        if node.parameters is None:
            return

        if len(node.parameters) > 255:
            state.err(
                f"Too many sequence arguments ({len(node.parameters)}); maximum is 255",
                node,
            )
            return


class DefineFunctions(TopDownVisitor):

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # Functions go in their node's enclosing scope's callable group.
        scope = state.enclosing_scope[node]

        # get(), not lookup(): a name already in THIS scope's callable group is a
        # conflict
        if scope.get(NameGroup.CALLABLE, node.name.name) is not None:
            state.err(
                f"Function '{node.name.name}' has already been defined", node.name
            )
            return

        # If the name resolves to something already, it's a shadow
        if scope.lookup(NameGroup.CALLABLE, node.name.name) is not None:
            state.warn(
                WarningType.SHADOW_CALLABLE,
                f"Function '{node.name.name}' shadows an existing definition",
                node.name,
            )

        func = FunctionSymbol(
            # we know the name
            node.name.name,
            # we don't know the return type yet
            return_type=None,
            # we don't know the arg types yet
            args=None,
            definition=node,
        )

        scope.define(NameGroup.CALLABLE, func.name, func)


class DefineVariables(TopDownVisitor):
    """Finds all variable declarations and adds them to the appropriate scope.

    Function bodies are deferred: the top-down pass first processes every
    non-function-body node so that all global variables and for-loop variables
    are registered.  Then, in a second phase, it descends into each function
    body.  This lets functions reference globals that are declared later in the
    source without needing a separate pre-registration pass.
    """

    def run(self, start: Ast, state: CompileState):
        self._deferred_defs: list[AstDef] = []

        # Phase 1: visit everything; visit_AstDef returns STOP_DESCENT so
        # the framework skips function bodies.
        super().run(start, state)

        # Phase 2: now descend into deferred function bodies.
        for func_node in self._deferred_defs:
            if state.errors:
                break
            super().run(func_node.body, state)

    def define_variable(
        self,
        sym: VariableSymbol,
        scope: Scope,
        state: CompileState,
        variable_kind: str,
        assert_undeclared: bool = False,
    ):
        # A variable is global if it is declared directly in the main sequence's
        # root scope, not in a nested block or function scope. Only the main
        # sequence can declare top-level variables: imported sequences may hold
        # only definitions and imports (no top-level statements) and may not take
        # sequence arguments, so their root scope never gains a variable.
        is_root = scope is state.main_scope

        # get(), not lookup(): a name already in THIS scope's value group is a
        # same-scope redeclaration
        if scope.get(NameGroup.VALUE, sym.name) is not None:
            if assert_undeclared:
                assert False, f"{variable_kind} '{sym.name}' has already been defined"
            state.err(
                f"{variable_kind} '{sym.name}' has already been defined",
                sym.declaration,
            )
            return

        if scope.lookup(NameGroup.VALUE, sym.name) is not None:
            state.warn(
                WarningType.SHADOW_VALUE,
                f"{variable_kind} '{sym.name}' shadows an existing definition",
                sym.declaration,
            )

        sym.is_global = is_root

        # new var. put it in the scope
        scope.define(NameGroup.VALUE, sym.name, sym)

    def visit_AstAssign(self, node: AstAssign, state: CompileState):

        if node.type_ann is None:
            # not a variable definition
            return

        scope = state.enclosing_scope[node]
        # yes a variable definition
        sym = VariableSymbol(node.lhs.name, node.type_ann, node)
        self.define_variable(sym, scope, state, "Variable")

    def visit_AstFor(self, node: AstFor, state: CompileState):
        # The loop variable is always a new declaration in the loop body's scope.
        body_scope = state.enclosing_scope[node.body]

        self.define_variable(
            loop_var := VariableSymbol(node.loop_var.name, None, node, LoopVarType),
            body_scope,
            state,
            "Loop variable",
            assert_undeclared=True,
        )

        # Each loop also defines an implicit upper-bound variable
        self.define_variable(
            upper_bound_var := VariableSymbol(
                state.new_anonymous_variable_name(),
                None,
                node,
                LoopVarType,
            ),
            body_scope,
            state,
            "Loop variable",
            assert_undeclared=True,
        )
        analysis = ForLoopAnalysis(loop_var, upper_bound_var)
        state.for_loops[node] = analysis

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # Parameters go in the function's enclosing scope
        body_scope = state.enclosing_scope[node.body]

        for arg in node.parameters or []:
            arg_name_var, arg_type_name, _ = arg
            sym = VariableSymbol(arg_name_var.name, arg_type_name, node)
            self.define_variable(sym, body_scope, state, "Parameter")

        # Defer traversal of the function body to phase 2, so that all
        # global-scope declarations are visible inside functions regardless
        # of source ordering.
        self._deferred_defs.append(node)
        return STOP_DESCENT

    def visit_AstSequenceMetadata(self, node: AstSequenceMetadata, state: CompileState):
        scope = state.enclosing_scope[node]

        for arg in node.parameters or []:
            arg_name_var, arg_type_name = arg
            sym = VariableSymbol(arg_name_var.name, arg_type_name, node)
            self.define_variable(sym, scope, state, "Sequence argument")


class SetEnclosingLoops(Visitor):
    """sets the enclosing_loop of any break/continue it finds"""

    def __init__(self, loop: Union[AstFor, AstWhile]):
        super().__init__()
        self.loop = loop

    def visit_AstBreak_AstContinue(
        self, node: Union[AstBreak, AstContinue], state: CompileState
    ):
        state.enclosing_loops[node] = self.loop


class CheckBreakAndContinueInLoop(TopDownVisitor):
    def visit_AstFor_AstWhile(self, node: Union[AstFor, AstWhile], state: CompileState):
        SetEnclosingLoops(node).run(node.body, state)

    def visit_AstBreak_AstContinue(
        self, node: Union[AstBreak, AstContinue], state: CompileState
    ):
        if node not in state.enclosing_loops:
            state.err("Cannot break/continue outside of a loop", node)
            return


class SetEnclosingFunction(Visitor):
    def __init__(self, func: AstDef):
        super().__init__()
        self.func = func

    def visit_AstReturn(self, node: AstReturn, state: CompileState):
        state.enclosing_funcs[node] = self.func


class CheckReturnInFunc(TopDownVisitor):
    def visit_AstDef(self, node: AstDef, state: CompileState):
        SetEnclosingFunction(node).run(node.body, state)

    def visit_AstReturn(self, node: AstReturn, state: CompileState):
        if node not in state.enclosing_funcs:
            state.err("Cannot return outside of a function", node)
            return


class AssignNameGroups(Visitor):
    """Record, for every expression, the name group it is used in: a callee is
    CALLABLE, a type annotation is TYPE, an operand / argument / rhs / condition
    is VALUE.

    A *defining* occurrence (a def / parameter / loop-variable name being
    introduced, not referenced) is not resolved, and is deliberately given no
    name group."""

    def visit_AstDef(self, node: AstDef, state: CompileState):
        state.contextual_name_group[node.name] = NameGroup.CALLABLE
        if node.return_type is not None:
            state.contextual_name_group[node.return_type] = NameGroup.TYPE
        if node.parameters is not None:
            for _arg_name, arg_type_name, default_value in node.parameters:
                state.contextual_name_group[arg_type_name] = NameGroup.TYPE
                if default_value is not None:
                    state.contextual_name_group[default_value] = NameGroup.VALUE

    def visit_AstAssign(self, node: AstAssign, state: CompileState):
        if node.type_ann is not None:
            state.contextual_name_group[node.type_ann] = NameGroup.TYPE
        state.contextual_name_group[node.lhs] = NameGroup.VALUE
        state.contextual_name_group[node.rhs] = NameGroup.VALUE

    def visit_AstSequenceMetadata(self, node: AstSequenceMetadata, state: CompileState):
        if node.parameters is None:
            return
        for _arg_name, arg_type_name in node.parameters:
            state.contextual_name_group[arg_type_name] = NameGroup.TYPE

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        state.contextual_name_group[node.func] = NameGroup.CALLABLE
        if node.args is None:
            return
        for arg in node.args:
            value = arg.value if is_instance_compat(arg, AstNamedArgument) else arg
            state.contextual_name_group[value] = NameGroup.VALUE

    def visit_AstIf_AstElif(self, node: Union[AstIf, AstElif], state: CompileState):
        state.contextual_name_group[node.condition] = NameGroup.VALUE

    def visit_AstBinaryOp(self, node: AstBinaryOp, state: CompileState):
        state.contextual_name_group[node.lhs] = NameGroup.VALUE
        state.contextual_name_group[node.rhs] = NameGroup.VALUE

    def visit_AstUnaryOp(self, node: AstUnaryOp, state: CompileState):
        state.contextual_name_group[node.val] = NameGroup.VALUE

    def visit_AstFor(self, node: AstFor, state: CompileState):
        state.contextual_name_group[node.range] = NameGroup.VALUE

    def visit_AstWhile(self, node: AstWhile, state: CompileState):
        state.contextual_name_group[node.condition] = NameGroup.VALUE

    def visit_AstAssert(self, node: AstAssert, state: CompileState):
        state.contextual_name_group[node.condition] = NameGroup.VALUE
        if node.exit_code is not None:
            state.contextual_name_group[node.exit_code] = NameGroup.VALUE

    def visit_AstIndexExpr(self, node: AstIndexExpr, state: CompileState):
        state.contextual_name_group[node.parent] = NameGroup.VALUE
        state.contextual_name_group[node.item] = NameGroup.VALUE

    def visit_AstRange(self, node: AstRange, state: CompileState):
        state.contextual_name_group[node.lower_bound] = NameGroup.VALUE
        state.contextual_name_group[node.upper_bound] = NameGroup.VALUE

    def visit_AstReturn(self, node: AstReturn, state: CompileState):
        if node.value is not None:
            state.contextual_name_group[node.value] = NameGroup.VALUE

    def visit_AstAnonStruct(self, node: AstAnonStruct, state: CompileState):
        for _, value_expr in node.members:
            state.contextual_name_group[value_expr] = NameGroup.VALUE

    def visit_AstAnonArray(self, node: AstAnonArray, state: CompileState):
        for elem_expr in node.elements:
            state.contextual_name_group[elem_expr] = NameGroup.VALUE

    def visit_AstLiteral_AstGetAttr_AstIdent(
        self, node: Union[AstLiteral, AstGetAttr, AstIdent], state: CompileState
    ):
        # A bare literal / getattr / ident does not name a group on its own; the
        # enclosing expression records its name group (or intentionally records
        # none, for a defining occurrence).
        pass

    def visit_default(self, node, state):
        # Every statement/expression that contains an identifier to resolve must
        # be handled above, or those identifiers would never be given a name
        # group and so never resolved. This mirrors the old resolver's safety
        # assertion.
        assert not is_instance_compat(node, AstStmtWithExpr), node


class ResolveQualifiedIdentifiers(TopDownVisitor):
    """Resolve every referenced identifier to its symbol, in the name group
    AssignNameGroups recorded for it. Parameter and loop-variable names are
    *definitions*, not references, so they are bound directly instead."""

    def may_contain_sub_definitions(self, sym: Symbol) -> bool:
        """return True if a symbol may contain other definitions reachable by
        member access. At the moment, only a module does -- a dictionary
        module, or a module or sequence symbol an import defines."""
        return is_instance_compat(sym, ModuleSymbol)

    def get_sub_definition(self, parent_sym: Symbol, name: str) -> Symbol | None:
        assert is_instance_compat(parent_sym, ModuleSymbol), parent_sym
        return parent_sym.get(name)

    # -- resolve any identifier/getattr that was given a name group --

    def visit_AstIdent(self, node: AstIdent, state: CompileState):
        self._resolve(node, state)

    def visit_AstGetAttr(self, node: AstGetAttr, state: CompileState):
        self._resolve(node, state)

    def _resolve(self, node: AstExpr, state: CompileState):
        # Only the outermost expression of a reference is given a name group; the
        # rest of a getattr chain is resolved within this same call, and defining
        # occurrences are never given one. So a node with no name group is one we
        # must not resolve here.
        ng = state.contextual_name_group.get(node)
        if ng is None:
            return

        # Walk down to the leftmost identifier, collecting getattrs (outermost
        # first).
        attrs: list[AstGetAttr] = []
        while is_instance_compat(node, AstGetAttr):
            attrs.append(node)
            node = node.parent

        # The root isn't an identifier, so this isn't a qualified identifier.
        if not is_instance_compat(node, AstIdent):
            return

        # Resolve the root identifier in its enclosing scope, in the name group,
        # walking the parent chain.
        resolved = state.enclosing_scope[node].lookup(ng, node.name)
        if resolved is None:
            state.err(f"Unknown {ng.value} '{node.name}'", node)
            return
        state.resolved_symbols[node] = resolved

        # Walk back down the getattr chain (innermost first) resolving each.
        # Stop when the parent isn't a module -- the rest is a member access
        # (e.g. a struct field) handled later by type checking.
        for getattr_node in reversed(attrs):
            parent_sym = state.resolved_symbols.get(getattr_node.parent)
            if not self.may_contain_sub_definitions(parent_sym):
                # further getattrs cannot be qualified names as
                # the parent symbol definition may not contain
                # other definitions
                return
            attr_sym = self.get_sub_definition(parent_sym, getattr_node.attr)
            if attr_sym is None:
                state.err("Unknown name", getattr_node)
                return
            state.resolved_symbols[getattr_node] = attr_sym

    # -- defining occurrences: bind the introduced name directly --

    def visit_AstDef(self, node: AstDef, state: CompileState):
        if node.parameters is None:
            return
        # Params are defined in the function body's scope by DefineVariables.
        body_values = state.enclosing_scope[node.body].group(NameGroup.VALUE)
        for arg_name_var, _arg_type_name, _default_value in node.parameters:
            state.resolved_symbols[arg_name_var] = body_values[arg_name_var.name]

    def visit_AstFor(self, node: AstFor, state: CompileState):
        # loop_var is defined in the body's scope by DefineVariables
        body_values = state.enclosing_scope[node.body].group(NameGroup.VALUE)
        state.resolved_symbols[node.loop_var] = body_values[node.loop_var.name]

    def visit_AstSequenceMetadata(self, node: AstSequenceMetadata, state: CompileState):
        if node.parameters is None:
            return
        values = state.enclosing_scope[node].group(NameGroup.VALUE)
        for arg_name_var, _arg_type_name in node.parameters:
            state.resolved_symbols[arg_name_var] = values[arg_name_var.name]


class CheckAllUnqualifiedIdentifiersResolved(Visitor):
    """Verify every AstIdent was resolved by ResolveQualifiedIdentifiers. Catches
    an identifier without a contextual name group -- e.g. a bare expression statement
    like `Foo.bar.BAZ`, whose root `Foo` is given no name group and so is never
    resolved -- which would otherwise KeyError in a later pass."""

    def visit_AstIdent(self, node: AstIdent, state: CompileState):
        if node not in state.resolved_symbols:
            state.err("Unknown name", node)


def is_cmd_and_response_unhandled(stmt: Ast, state: CompileState) -> bool:
    """True when *stmt* is a command call whose response is not captured."""
    return is_instance_compat(stmt, AstFuncCall) and is_instance_compat(
        state.resolved_symbols.get(stmt.func), CommandSymbol
    )


def is_type_constant_size(type: FpyType) -> bool:
    """Return true if the type has a statically known size.

    Internal Strings have constant sizes, but runtime strings don't -> They
    can vary in length.
    Also allow concrete constant-size types.
    """
    if type.kind == TypeKind.INTERNAL_STRING:
        return True

    if not type.is_concrete:
        return False

    if type.kind == TypeKind.STRING:
        return False

    if type.kind == TypeKind.ARRAY:
        return is_type_constant_size(type.elem_type)

    if type.kind == TypeKind.STRUCT:
        for m in type.members:
            if not is_type_constant_size(m.type):
                return False
        return True

    return True


class CheckResolvedSymbolKinds(Visitor):
    """Verify each resolved identifier is the KIND its name group requires: a
    callee must be callable, a type annotation must be a type, and a value must
    be a value -- not a module, which is legal only as a member-access
    qualifier (`Fw` in `Fw.Time`, never bare `Fw`).

    The name group comes from AssignNameGroups, so no parent lookup is needed;
    this replaces the old parallel visit methods and the module-as-value check
    that used to live in the (bottom-up) type pass."""

    def visit_default(self, node: Ast, state: CompileState):
        ng = state.contextual_name_group.get(node)
        if ng is None:
            return
        sym = state.resolved_symbols.get(node)
        if sym is None:
            # A type or callable name group must be a single resolved name;
            # anything with no resolution there -- an unresolved identifier, or a
            # literal like `x: True` -- is unknown (the resolver often reports it
            # first). A value name group may instead hold a compound expression
            # or a member access resolved later, which have no resolution here
            # and are fine.
            if ng in (NameGroup.TYPE, NameGroup.CALLABLE):
                state.err(f"Unknown {ng.value}", node)
            return
        if ng == NameGroup.CALLABLE and not is_instance_compat(sym, CallableSymbol):
            state.err(f"Expected a {ng.value}", node)
        elif ng == NameGroup.TYPE and not is_instance_compat(sym, FpyType):
            state.err(f"Expected a {ng.value}", node)
        elif ng == NameGroup.VALUE and not is_symbol_an_expr(sym):
            state.err(f"Expected a {ng.value}", node)


class CheckForConstantSizeTypes(Visitor):

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # Check return type
        if node.return_type is not None:
            return_type = state.resolved_symbols[node.return_type]
            if not is_type_constant_size(return_type):
                state.err(
                    f"Type {return_type.display_name} is not constant-sized (contains strings)",
                    node.return_type,
                )
                return

        # Check parameter types
        if node.parameters is not None:
            for _, arg_type_name, _ in node.parameters:
                arg_type = state.resolved_symbols[arg_type_name]
                if not is_type_constant_size(arg_type):
                    state.err(
                        f"Type {arg_type.display_name} is not constant-sized (contains strings)",
                        arg_type_name,
                    )
                    return

    def visit_AstAssign(self, node: AstAssign, state: CompileState):
        if node.type_ann is None:
            return

        var_type = state.resolved_symbols[node.type_ann]

        if not is_type_constant_size(var_type):
            state.err(
                f"Type {var_type.display_name} is not constant-sized (contains strings)",
                node.type_ann,
            )
            return

    def visit_AstSequenceMetadata(self, node: AstSequenceMetadata, state: CompileState):
        if node.parameters is None:
            return

        for _, arg_type_name in node.parameters:
            arg_type = state.resolved_symbols[arg_type_name]
            if not is_type_constant_size(arg_type):
                state.err(
                    f"Type {arg_type.display_name} is not constant-sized (contains strings)",
                    arg_type_name,
                )
                return


class UpdateStateWithTypes(Visitor):

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # Get the function that was created in DefineFunctions
        func = state.resolved_symbols[node.name]
        assert is_instance_compat(func, FunctionSymbol), func

        # Resolve return type
        if node.return_type is None:
            func.return_type = NOTHING
        else:
            return_type = state.resolved_symbols[node.return_type]
            func.return_type = return_type

        # Resolve parameter types
        args = []
        if node.parameters is not None:
            for arg_name_var, arg_type_name, default_value in node.parameters:
                arg_type = state.resolved_symbols[arg_type_name]
                # update the var type
                arg_var = state.resolved_symbols[arg_name_var]
                assert is_instance_compat(arg_var, VariableSymbol), arg_var
                arg_var.type = arg_type
                args.append((arg_name_var.name, arg_type, default_value))

        func.args = args

    def visit_AstAssign(self, node: AstAssign, state: CompileState):
        if node.type_ann is None:
            return

        var_type = state.resolved_symbols[node.type_ann]

        var = state.resolved_symbols[node.lhs]

        var.type = var_type

    def visit_AstSequenceMetadata(self, node: AstSequenceMetadata, state: CompileState):
        # Resolve parameter types
        if node.parameters is None:
            return

        for arg_name_var, arg_type_name in node.parameters:
            arg_type = state.resolved_symbols[arg_type_name]
            # update the var type
            arg_var = state.resolved_symbols[arg_name_var]
            assert is_instance_compat(arg_var, VariableSymbol), arg_var
            arg_var.type = arg_type
            state.this_seq_arg_specs.append((arg_var.name, arg_type))


class EnsureVariableNotReferenced(Visitor):
    def __init__(self, var: VariableSymbol):
        super().__init__()
        self.var = var

    def visit_AstIdent(self, node: AstIdent, state: CompileState):
        sym = state.resolved_symbols[node]
        if sym == self.var:
            state.err(f"'{node.name}' used before defined", node)
            return


class CheckUseBeforeDefine(TopDownVisitor):
    """
    Checks that variables are not used before they are defined.
    Handles both regular variable assignments (AstAssign) and for loop variables (AstFor).

    Uses TopDownVisitor because for loops need the loop variable to be defined
    before visiting the body. For assignments, we manually check the RHS before
    marking the variable as defined.
    """

    def __init__(self):
        super().__init__()
        self.currently_defined_vars: list[VariableSymbol] = []

    def visit_AstFor(self, node: AstFor, state: CompileState):
        var = state.resolved_symbols[node.loop_var]
        # Check that the loop var isn't referenced in the range (before it's defined)
        EnsureVariableNotReferenced(var).run(node.range, state)
        # Now mark it as defined for the body
        self.currently_defined_vars.append(var)

    def visit_AstAssign(self, node: AstAssign, state: CompileState):
        if not is_instance_compat(node.lhs, AstIdent):
            # definitely not a declaration, it's a field assignment
            return

        var = state.resolved_symbols[node.lhs]

        if var is None or var.declaration != node:
            # either not defined in this scope, or this is not a
            # declaration of this var
            return

        # Before marking as defined, check that the variable isn't used in its own RHS
        EnsureVariableNotReferenced(var).run(node.rhs, state)

        # Now mark this variable as defined
        self.currently_defined_vars.append(var)

    def visit_AstIdent(self, node: AstIdent, state: CompileState):
        sym = state.resolved_symbols[node]
        if not is_instance_compat(sym, VariableSymbol):
            # not a variable, might be a type name or smth
            return

        if is_instance_compat(sym.declaration, (AstDef, AstSequenceMetadata)):
            # function parameters and sequence metadata - no use-before-define
            # check needed this is because if it's in scope, it's defined, as
            # its "declaration" is the start of the scope
            return
        if sym.declaration is None:
            # Built-in variable (e.g., flags) — always defined
            return
        if (
            is_instance_compat(sym.declaration, AstAssign)
            and sym.declaration.lhs == node
        ):
            # this is the declaring reference for an assignment
            return
        if (
            is_instance_compat(sym.declaration, AstFor)
            and sym.declaration.loop_var == node
        ):
            # this is the declaring reference for a for loop variable
            return

        if sym not in self.currently_defined_vars:
            # Global variables referenced from inside a function are always
            # accessible — they are allocated and zero-initialized at sequence
            # start, regardless of textual ordering.
            if sym.is_global and state.enclosing_scope[node].in_function:
                return
            state.err(f"'{node.name}' used before defined", node)
            return


def _add_unique(items: list, item):
    """Append item to items if not already present (by equality)."""
    if item not in items:
        items.append(item)


class FindGlobalUsesInFunction(TopDownVisitor):
    """Scans a single function body, recording the globals it reads and the
    user functions it calls into state, keyed by the function definition.

    Only assignment-declared globals matter: sequence parameters and builtins
    are initialized at sequence start, so they are always defined.
    """

    def __init__(self, func: AstDef):
        super().__init__()
        self.func = func

    def visit_AstIdent(self, node: AstIdent, state: CompileState):
        sym = state.resolved_symbols.get(node)
        if (
            is_instance_compat(sym, VariableSymbol)
            and sym.is_global
            and is_instance_compat(sym.declaration, AstAssign)
        ):
            _add_unique(state.function_global_uses.setdefault(self.func, []), sym)

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        sym = state.resolved_symbols.get(node.func)
        if is_instance_compat(sym, FunctionSymbol):
            _add_unique(
                state.function_callees.setdefault(self.func, []), sym.definition
            )


class CollectFunctionGlobalUses(TopDownVisitor):
    """For each function definition, record the globals its body (and parameter
    defaults) reads directly and the user functions it calls directly."""

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # ensure every function has entries even if it uses/calls nothing
        state.function_global_uses.setdefault(node, [])
        state.function_callees.setdefault(node, [])

        scanner = FindGlobalUsesInFunction(node)
        scanner.run(node.body, state)
        if node.parameters is not None:
            for _ident, _type_ann, default in node.parameters:
                if default is not None:
                    scanner.run(default, state)


class _FindScriptCalls(TopDownVisitor):
    """Collects the definition of every script function called in a subtree,
    without descending into nested function definitions."""

    def __init__(self):
        super().__init__()
        self.called: list[AstDef] = []

    def visit_AstDef(self, node: AstDef, state: CompileState):
        return STOP_DESCENT

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        func = state.resolved_symbols.get(node.func)
        if is_instance_compat(func, FunctionSymbol):
            self.called.append(func.definition)


class CollectUsedFunctions(TopDownVisitor):
    """Collects the set of functions the program can actually call: the ones
    the main sequence's top-level code calls, plus, transitively, the ones
    called from the bodies of used functions. A function whose only callers
    are themselves unused -- including a group of unused functions calling
    each other -- is not used, and no backend generates code for it.
    """

    def run(self, start: Ast, state: CompileState):
        assert start is state.root_block, "must run on the root block"
        self.callees: dict[AstDef, list[AstDef]] = {}
        # The walk fills self.callees with every function's direct callees.
        super().run(start, state)

        # Only the main block's top-level code runs; everything else in the
        # root block (the builtin library, imported sequences) contributes
        # definitions but no executed statements.
        roots = _FindScriptCalls()
        roots.run(state.main_block, state)
        worklist = roots.called
        while worklist:
            func_def = worklist.pop()
            if func_def in state.used_funcs:
                continue
            state.used_funcs.add(func_def)
            worklist.extend(self.callees[func_def])

    def visit_AstDef(self, node: AstDef, state: CompileState):
        scanner = _FindScriptCalls()
        scanner.run(node.body, state)
        self.callees[node] = scanner.called
        return STOP_DESCENT


class ResolveTransitiveGlobalUses:
    """Grows function_global_uses from direct uses to transitive uses too.

    Repeatedly, for each call edge f -> g, fold g's globals into f's, until a
    full pass adds nothing. At this point every call edge is accounted for,
    which (by induction over call chains) is the full transitive closure.
    """

    def run(self, start: Ast, state: CompileState):
        num_funcs = len(state.function_global_uses)
        num_globals = len(
            {id(g) for uses in state.function_global_uses.values() for g in uses}
        )

        for _ in range(num_funcs * num_globals + 1):
            changed = False
            for func, callees in state.function_callees.items():
                uses = state.function_global_uses[func]
                for callee in callees:
                    for g in state.function_global_uses[callee]:
                        if g not in uses:
                            uses.append(g)
                            changed = True
            if not changed:
                return

        assert False, "transitive global-use algo did not converge"


class CheckGlobalsInitializedBeforeCall(Visitor):
    """Ensures a global is initialized before any function that reads it
    (directly or transitively) is called.

    A function may be defined before the globals it uses, but every call to it
    must come after those globals are declared. Because Fpy is block scoped,
    functions can only reference root-scope globals, which are declared
    unconditionally in source order — so a global is "defined" once its
    declaration statement has executed.
    """

    def __init__(self):
        super().__init__()
        self.defined: list[VariableSymbol] = []

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        if state.enclosing_scope[node].in_function:
            # checked transitively at the top-level call that reaches this one
            return
        sym = state.resolved_symbols.get(node.func)
        if not is_instance_compat(sym, FunctionSymbol):
            return
        missing = [
            g
            for g in state.function_global_uses[sym.definition]
            if g not in self.defined
        ]
        if not missing:
            return
        # Report the global declared latest in the source for a stable message.
        var = max(missing, key=lambda v: v.declaration.id)
        state.err(
            f"'{sym.name}' is called here but reads global '{var.name}', "
            f"which is not defined until later",
            node,
        )

    def visit_AstAssign(self, node: AstAssign, state: CompileState):
        if not is_instance_compat(node.lhs, AstIdent):
            return
        sym = state.resolved_symbols.get(node.lhs)
        if (
            is_instance_compat(sym, VariableSymbol)
            and sym.is_global
            and sym.declaration is node
        ):
            _add_unique(self.defined, sym)


class ResolveSequenceDependencies(TopDownVisitor):
    """Discover and resolve all sequence-run dependencies before type checking.

    For each call to a seq-run command with a string-literal filename,
    reads the target .bin header and resolves its argument types.
    Results are stored in state.called_seq_arg_specs so that later passes
    can use them without file I/O.
    """

    def _get_bin_name(self, node: AstFuncCall, state: CompileState) -> str | None:
        """Return the .bin filename for a seq-run-with-args call, or None.

        Reports a compile error if the filename argument is not a string literal.
        """
        func = state.resolved_symbols.get(node.func)
        if not is_instance_compat(func, CommandSymbol) or not func.is_seq_run_with_args:
            return None
        if not node.args or len(node.args) < 1:
            # Missing args will be caught by build_resolved_call_args (too few arguments)
            return None
        file_name_arg = node.args[0]
        if not is_instance_compat(file_name_arg, AstString):
            state.err(
                "Sequence file name must be a string literal",
                file_name_arg,
            )
            return None
        return file_name_arg.value

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        bin_name = self._get_bin_name(node, state)
        if bin_name is None or bin_name in state.called_seq_arg_specs:
            return

        ground_binary_dir = state.ground_binary_dir
        if ground_binary_dir is None:
            state.err(
                "Cannot resolve sequence binary path: no binary directory configured (use --ground-binary-dir / -B)",
                node,
            )
            return

        bin_path = Path(ground_binary_dir) / bin_name
        if not bin_path.exists():
            state.err(
                f"Compiled sequence binary not found: {bin_path}",
                node.args[0],
            )
            return

        try:
            arg_specs = read_bin_arg_specs(bin_path)
        except Exception as e:
            state.err(
                f"Failed to read sequence binary {bin_path}: {e}",
                node.args[0],
            )
            return

        try:
            target_arg_types = resolve_arg_specs(arg_specs, state.type_defs)
        except RuntimeError as e:
            state.err(
                f"Failed to resolve argument types from {bin_path}: {e}",
                node.args[0],
            )
            return

        state.called_seq_arg_specs[bin_name] = target_arg_types

        # Build an extended CommandSymbol that includes the target sequence's
        # parameters so that standard arg resolution works in PickTypes.
        func = state.resolved_symbols.get(node.func)
        extra_args = [(name, t, None) for name, t in target_arg_types]
        extended_func = dc_replace(func, args=func.args + extra_args)
        state.resolved_symbols[node.func] = extended_func


class CollectSequenceDependencies(ResolveSequenceDependencies):
    """Collect .bin filenames from seq-run-with-args calls without reading binaries.

    Use this instead of ResolveSequenceDependencies when you only need the
    dependency list (e.g. the fprime-fpy-depend tool) and the binaries may
    not exist yet.
    """

    def __init__(self):
        super().__init__()
        self.bin_names: list[str] = []
        self._seen: set[str] = set()

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        bin_name = self._get_bin_name(node, state)
        if bin_name is None or bin_name in self._seen:
            return
        self._seen.add(bin_name)
        self.bin_names.append(bin_name)


class PickTypesAndResolveFields(Visitor):

    def can_coerce_type(self, source: FpyType, target: FpyType) -> bool:
        """Returns True if source can be implicitly coerced to target.

        Coercion is allowed when the common type of source and target IS target,
        meaning target can already represent everything source can.
        """
        # The SIZED sentinel accepts any serializable, statically-sized argument.
        if target.kind == TypeKind.SIZED:
            return is_type_constant_size(source)
        return self.find_common_type(source, target) == target

    def coerce_expr_type(
        self, node: AstExpr, type: FpyType, state: CompileState
    ) -> bool:
        unconverted_type = state.synthesized_types[node]
        current_contextual = state.contextual_types[node]

        # Already coerced — idempotent if same target, bug if different
        if current_contextual != unconverted_type:
            assert (
                current_contextual == type
            ), f"double coercion: {unconverted_type} -> {current_contextual} vs {type}"
            return True

        if not self.can_coerce_type(unconverted_type, type):
            state.err(
                f"Expected {type.display_name}, found {unconverted_type.display_name}",
                node,
            )
            return False

        # SIZED is a sentinel; resolve it to the argument's own concrete sized type.
        if type.kind == TypeKind.SIZED:
            if unconverted_type.kind == TypeKind.INTERNAL_STRING:
                # A string literal gets a concrete String[N] sized to the literal.
                assert is_instance_compat(node, AstString), node
                str_len = len(node.value.encode("utf-8"))
                type = FpyType(TypeKind.STRING, f"String_{str_len}", max_length=str_len)
            else:
                type = unconverted_type

        # For anon structs/arrays, recursively coerce children and build resolved_args
        if unconverted_type.kind == TypeKind.ANON_STRUCT:
            return self._coerce_anon_struct(node, type, state)
        if unconverted_type.kind == TypeKind.ANON_ARRAY:
            return self._coerce_anon_array(node, type, state)

        state.contextual_types[node] = type
        return True

    def _coerce_anon_struct(
        self, node: AstAnonStruct, target: FpyType, state: CompileState
    ) -> bool:
        """Recursively coerce each provided member and build resolved_args.

        Called after can_coerce_type has already confirmed structural compatibility.
        """
        provided_members = {name: value_expr for name, value_expr in node.members}

        # Build resolved list in target member order: coerce provided, fill defaults
        resolved_members = []
        for member in target.members:
            if member.name in provided_members:
                value_expr = provided_members[member.name]
                if not self.coerce_expr_type(value_expr, member.type, state):
                    return False
                resolved_members.append(value_expr)
            else:
                resolved_members.append(target.member_defaults[member.name])

        state.resolved_args[node] = resolved_members
        state.contextual_types[node] = target
        return True

    def _coerce_anon_array(
        self, node: AstAnonArray, target: FpyType, state: CompileState
    ) -> bool:
        """Recursively coerce each provided element and build resolved_args.

        Called after can_coerce_type has already confirmed structural compatibility.
        """
        # Coerce each provided element to the target element type
        for elem_expr in node.elements:
            if not self.coerce_expr_type(elem_expr, target.elem_type, state):
                return False

        # Build resolved list: provided elements + defaults for missing positions.
        resolved = list(node.elements)
        for i in range(len(node.elements), target.length):
            resolved.append(target.elem_defaults[i])

        state.resolved_args[node] = resolved
        state.contextual_types[node] = target
        return True

    def find_common_type(
        self, first_type: FpyType, second_type: FpyType
    ) -> FpyType | None:

        # important principles to reduce surprise:

        # type of an operation should be decided by the types of its inputs. let's not do
        # anything clever with trying to inspect the values of consts

        # no common type between signed and unsigned int

        # TODO unit test that this "works either way"
        if first_type == second_type:
            # no coercion necessary
            return second_type

        # Anonymous struct adapts to a compatible concrete struct.
        if (
            first_type.kind == TypeKind.ANON_STRUCT
            or second_type.kind == TypeKind.ANON_STRUCT
        ):
            return self._find_common_type_anon_struct(first_type, second_type)

        # Anonymous array adapts to a compatible concrete array.
        if (
            first_type.kind == TypeKind.ANON_ARRAY
            or second_type.kind == TypeKind.ANON_ARRAY
        ):
            return self._find_common_type_anon_array(first_type, second_type)

        # literal strings adapt to specific strings
        if first_type.is_string and second_type == INTERNAL_STRING:
            return first_type
        if second_type.is_string and first_type == INTERNAL_STRING:
            return second_type

        if not first_type.is_numerical or not second_type.is_numerical:
            # there are no other non numeric types which have a common type
            return None

        second_float = second_type.is_float
        first_float = first_type.is_float

        # common type of int and float is float
        # but arb-precision adapts to specific: if one side is a specific int
        # and the other is an arb-precision float, the result is F64 (not arb float)
        if second_float and not first_float:
            if second_type == FLOAT and first_type not in ARBITRARY_PRECISION_TYPES:
                return F64
            return second_type
        if not second_float and first_float:
            if first_type == FLOAT and second_type not in ARBITRARY_PRECISION_TYPES:
                return F64
            return first_type

        # only case left is that we have both floats, or both ints
        if second_float:
            return self.find_common_float_type(first_type, second_type)

        return self.find_common_integer_type(first_type, second_type)

    def find_common_float_type(
        self, first_type: FpyType, second_type: FpyType
    ) -> FpyType | None:
        # arb precision adapts to specific
        if first_type == FLOAT:
            return second_type
        if second_type == FLOAT:
            return first_type
        # both specific: wider wins
        if max(first_type.bits, second_type.bits) > 32:
            return F64
        return F32

    def find_common_integer_type(
        self, first_type: FpyType, second_type: FpyType
    ) -> FpyType | None:
        # arb precision adapts to specific
        if first_type == INTEGER:
            return second_type
        if second_type == INTEGER:
            return first_type

        # both specific: must have matching signedness
        first_unsigned = first_type in UNSIGNED_INTEGER_TYPES
        second_unsigned = second_type in UNSIGNED_INTEGER_TYPES

        if first_unsigned != second_unsigned:
            return None

        # same signedness: wider wins
        bits = max(first_type.bits, second_type.bits)
        if first_unsigned:
            if bits <= 8:
                return U8
            elif bits <= 16:
                return U16
            elif bits <= 32:
                return U32
            else:
                return U64
        else:
            if bits <= 8:
                return I8
            elif bits <= 16:
                return I16
            elif bits <= 32:
                return I32
            else:
                return I64

    def _find_common_type_anon_struct(self, a: FpyType, b: FpyType) -> FpyType | None:
        """Return the concrete struct type if one side is an anonymous struct
        that is structurally compatible with the other, otherwise None."""
        if a.kind == TypeKind.ANON_STRUCT and b.kind == TypeKind.STRUCT:
            anon, concrete = a, b
        elif b.kind == TypeKind.ANON_STRUCT and a.kind == TypeKind.STRUCT:
            anon, concrete = b, a
        else:
            return None

        if not is_type_constant_size(concrete):
            return None

        target_members = {m.name: m for m in concrete.members}
        seen: set[str] = set()
        for member in anon.members:
            if member.name in seen:
                return None
            seen.add(member.name)
            if member.name not in target_members:
                return None
            if not self.can_coerce_type(member.type, target_members[member.name].type):
                return None
        return concrete

    def _find_common_type_anon_array(self, a: FpyType, b: FpyType) -> FpyType | None:
        """Return the concrete array type if one side is an anonymous array
        that is structurally compatible with the other, otherwise None."""
        if a.kind == TypeKind.ANON_ARRAY and b.kind == TypeKind.ARRAY:
            anon, concrete = a, b
        elif b.kind == TypeKind.ANON_ARRAY and a.kind == TypeKind.ARRAY:
            anon, concrete = b, a
        else:
            return None

        if not is_type_constant_size(concrete):
            return None
        if anon.length > concrete.length:
            return None
        return concrete

    def get_type_of_symbol(self, sym: Symbol) -> FpyType:
        """returns the fprime type of the sym, if it were to be evaluated as an expression"""
        if isinstance(sym, ChDef):
            result_type = sym.ch_type
        elif isinstance(sym, PrmDef):
            result_type = sym.prm_type
        elif isinstance(sym, FpyValue):
            # constant value
            result_type = sym.type
        elif isinstance(sym, VariableSymbol):
            result_type = sym.type
        elif isinstance(sym, FieldAccess):
            result_type = sym.type
        else:
            assert False, sym

        return result_type

    def visit_AstGetAttr(self, node: AstGetAttr, state: CompileState):
        this_sym = state.resolved_symbols.get(node)
        if this_sym is not None:
            # already resolved by ResolveQualifiedNames
            if not is_symbol_an_expr(this_sym):
                # not an expr, doesn't have a type
                return
            # otherwise, this is a qualified name AND an expr.
            # can happen in cases like enum consts
        else:
            # perform member access
            parent_sym = state.resolved_symbols.get(node.parent)
            # theoretically the only thing left should be cases where the parent
            # is some sort of expr

            # either a symbol that is an expr, or something more complex
            assert parent_sym is None or is_symbol_an_expr(parent_sym), parent_sym

            # it may or may not have a compile time value, but it definitely has a type
            parent_type = state.synthesized_types[node.parent]

            if parent_type.kind == TypeKind.ANON_STRUCT:
                # Direct member access on anonymous struct literal
                member_type = None
                for m in parent_type.members:
                    if m.name == node.attr:
                        member_type = m.type
                        break
                if member_type is None:
                    state.err(
                        f"Anonymous struct has no member named '{node.attr}'",
                        node,
                    )
                    return
                this_sym = FieldAccess(
                    is_struct_member=True,
                    parent_expr=node.parent,
                    type=member_type,
                    base_sym=None,
                    name=node.attr,
                )
            elif parent_type.kind == TypeKind.STRUCT:
                if not is_type_constant_size(parent_type):
                    state.err(
                        f"{parent_type.display_name} is not constant-sized (contains strings), cannot access members",
                        node,
                    )
                    return

                # field symbols store their "base symbol", which is the first non-field-symbol parent of
                # the field symbol. this lets you easily check what actual underlying thing (tlm chan, variable, prm)
                # you're talking about a field of
                base_sym = (
                    parent_sym
                    if not is_instance_compat(parent_sym, FieldAccess)
                    else parent_sym.base_sym
                )
                # we also calculate a "base offset" wrt. the start of the base_sym type, so you
                # can easily pick out this field from a value of the base sym type
                base_offset = (
                    0
                    if not is_instance_compat(parent_sym, FieldAccess)
                    else parent_sym.base_offset
                )

                member_list = [(m.name, m.type) for m in parent_type.members]

                offset = 0
                for arg_name, arg_type in member_list:
                    if arg_name == node.attr:
                        this_sym = FieldAccess(
                            is_struct_member=True,
                            parent_expr=node.parent,
                            type=arg_type,
                            base_sym=base_sym,
                            local_offset=offset,
                            base_offset=base_offset,
                            name=arg_name,
                        )
                        break
                    offset += arg_type.max_size
                    if base_offset is not None:
                        base_offset += arg_type.max_size

                if this_sym is None:
                    state.err(
                        f"{parent_type.display_name} has no member named {node.attr}",
                        node,
                    )
                    return
            else:
                state.err(
                    f"{parent_type.display_name} is not a struct, cannot access members",
                    node,
                )
                return

        sym_type = self.get_type_of_symbol(this_sym)

        state.resolved_symbols[node] = this_sym
        state.synthesized_types[node] = sym_type
        state.contextual_types[node] = sym_type

    def visit_AstIndexExpr(self, node: AstIndexExpr, state: CompileState):
        parent_sym = state.resolved_symbols.get(node.parent)

        if parent_sym is not None and not is_symbol_an_expr(parent_sym):
            state.err("Unknown item", node)
            return

        # otherwise, we should definitely have a well-defined type for our parent expr

        parent_type = state.synthesized_types[node.parent]

        if parent_type.kind == TypeKind.ANON_ARRAY:
            # Index access on anonymous array literal
            if parent_type.length == 0:
                state.err("Cannot index into an empty anonymous array", node)
                return

            # coerce the index expression to array index type
            if not self.coerce_expr_type(node.item, ArrayIndexType, state):
                return

            sym = FieldAccess(
                is_array_element=True,
                parent_expr=node.parent,
                type=parent_type.elem_type,
                base_sym=None,
                idx_expr=node.item,
            )

            state.resolved_symbols[node] = sym
            state.synthesized_types[node] = parent_type.elem_type
            state.contextual_types[node] = parent_type.elem_type
            return

        if parent_type.kind != TypeKind.ARRAY:
            state.err(f"{parent_type.display_name} is not an array", node)
            return

        if not is_type_constant_size(parent_type):
            state.err(
                f"{parent_type.display_name} is not constant-sized (contains strings), cannot access items",
                node,
            )
            return

        # coerce the index expression to array index type
        if not self.coerce_expr_type(node.item, ArrayIndexType, state):
            return

        base_sym = (
            parent_sym
            if not is_instance_compat(parent_sym, FieldAccess)
            else parent_sym.base_sym
        )

        sym = FieldAccess(
            is_array_element=True,
            parent_expr=node.parent,
            type=parent_type.elem_type,
            base_sym=base_sym,
            idx_expr=node.item,
        )

        state.resolved_symbols[node] = sym
        state.synthesized_types[node] = parent_type.elem_type
        state.contextual_types[node] = parent_type.elem_type

    def visit_AstIdent(self, node: AstIdent, state: CompileState):
        # already been resolved
        sym = state.resolved_symbols[node]
        if sym is None:
            return
        if not is_symbol_an_expr(sym):
            # A non-value symbol reaching here is a module serving as a member-
            # access qualifier (`Fw` in `Fw.Time`); it has no value type of its
            # own to synthesize. A module misused as a bare value was already
            # rejected by ResolveQualifiedIdentifiers.
            return

        sym_type = self.get_type_of_symbol(sym)

        state.synthesized_types[node] = sym_type
        state.contextual_types[node] = sym_type

    def visit_AstNumber(self, node: AstNumber, state: CompileState):
        # give a best guess as to the final type of this node. we don't actually know
        # its bitwidth or signedness yet
        if is_instance_compat(node.value, Decimal):
            result_type = FLOAT
        else:
            result_type = INTEGER

        state.synthesized_types[node] = result_type
        state.contextual_types[node] = result_type

    def widen_to_64(self, common_type: FpyType) -> FpyType:
        """Widen a specific numeric type to its 64-bit counterpart for VM execution.
        Returns the type unchanged if it's already 64-bit or arb precision.
        """
        if common_type in ARBITRARY_PRECISION_TYPES:
            return common_type
        if common_type.is_float:
            return F64
        if common_type in UNSIGNED_INTEGER_TYPES:
            return U64
        if common_type in SIGNED_INTEGER_TYPES:
            return I64
        assert False, common_type

    def pick_intermediate_type(
        self,
        arg_types: list[FpyType],
        op: BinaryStackOp | UnaryStackOp,
    ) -> FpyType | None:
        """Determine the intermediate type for an operator.

        Uses find_common_type as the base, then applies op-specific
        overrides and widens to 64-bit for runtime VM execution.

        Returns None if the operation is invalid for the given types.
        """
        if op in BOOLEAN_OPERATORS:
            return BOOL

        # for == and !=, non-numeric same-type comparisons are valid
        if op in (BinaryStackOp.EQUAL, BinaryStackOp.NOT_EQUAL):
            if len(arg_types) == 2 and arg_types[0] == arg_types[1]:
                if not arg_types[0].is_numerical:
                    # non-numeric equality (struct, array, enum, time)
                    return arg_types[0]

        # from here, all args must be numeric
        if not all(t.is_numerical for t in arg_types):
            return None

        # division and exponentiation always operate over floats
        if op in (BinaryStackOp.DIVIDE, BinaryStackOp.EXPONENT):
            if all(t in ARBITRARY_PRECISION_TYPES for t in arg_types):
                return FLOAT
            return F64

        # for everything else, find the common type then widen to 64-bit
        common = (
            self.find_common_type(*arg_types) if len(arg_types) == 2 else arg_types[0]
        )
        if common is None:
            return None

        return self.widen_to_64(common)

    def pick_result_type(
        self,
        intermediate_type: FpyType,
        op: BinaryStackOp | UnaryStackOp,
    ) -> FpyType:
        """Derive the result type from the intermediate type (excluding time ops).

        For comparisons and boolean ops, the result is always bool.
        For numeric ops, the result type equals the intermediate type.
        This avoids data loss from truncating back to a narrower type
        (e.g. U32 * literal computes in U64 and stays U64).
        """
        if op in BOOLEAN_OPERATORS or op in COMPARISON_OPS:
            return BOOL

        # all other cases, result is a number
        assert op in NUMERIC_OPERATORS

        return intermediate_type

    def _resolve_time_op(
        self,
        lhs_type: FpyType,
        rhs_type: FpyType,
        op: BinaryStackOp,
    ) -> tuple[FpyType, FpyType, FpyType, FpyType, str, bool] | None:
        """Look up a TIME_OPS entry, resolving anonymous structs if needed.

        Returns (resolved_lhs, resolved_rhs, common_type, result_type, func_name, is_cmp)
        or None if no match.

        When multiple entries match (e.g. an anon struct can coerce to both TIME
        and TIME_INTERVAL), prefer the entry whose operand types have fewer
        unmatched members — i.e., the most specific structural match.
        """
        matches: list[tuple[FpyType, FpyType, FpyType, FpyType, str, bool]] = []
        for (l, r, o), (
            common_type,
            result_type,
            func_name,
            is_cmp,
        ) in TIME_OPS.items():
            if o != op:
                continue
            if self.can_coerce_type(lhs_type, l) and self.can_coerce_type(rhs_type, r):
                matches.append((l, r, common_type, result_type, func_name, is_cmp))

        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]

        # Multiple matches — pick the most specific (fewest defaulted members).
        def _extra_member_count(source: FpyType, target: FpyType) -> int:
            if source.kind == TypeKind.ANON_STRUCT and target.kind == TypeKind.STRUCT:
                return len(target.members) - len(source.members)
            return 0

        def _specificity(m: tuple) -> int:
            return _extra_member_count(lhs_type, m[0]) + _extra_member_count(
                rhs_type, m[1]
            )

        matches.sort(key=_specificity)
        assert _specificity(matches[0]) < _specificity(matches[1]), (
            f"Ambiguous time op: {lhs_type} {op} {rhs_type} "
            f"matches {matches[0][0]},{matches[0][1]} and {matches[1][0]},{matches[1][1]} "
            f"with equal specificity"
        )
        return matches[0]

    def visit_AstBinaryOp(self, node: AstBinaryOp, state: CompileState):
        lhs_type = state.synthesized_types[node.lhs]
        rhs_type = state.synthesized_types[node.rhs]
        arg_types = [lhs_type, rhs_type]

        # Check for time/interval operator overloads (with anon struct resolution)
        resolved = self._resolve_time_op(lhs_type, rhs_type, node.op)
        if resolved is not None:
            resolved_lhs, resolved_rhs, common_type, result_type, _, _ = resolved
            # _resolve_time_op already confirmed coercibility
            assert self.coerce_expr_type(node.lhs, resolved_lhs, state)
            assert self.coerce_expr_type(node.rhs, resolved_rhs, state)
            state.op_intermediate_types[node] = common_type
            state.synthesized_types[node] = result_type
            state.contextual_types[node] = result_type
            return

        # pick_intermediate_type uses find_common_type internally,
        # then applies op overrides and widens to 64-bit for runtime
        intermediate_type = self.pick_intermediate_type(arg_types, node.op)
        if intermediate_type is None:
            state.err(
                f"Op {node.op} undefined for {lhs_type.display_name}, {rhs_type.display_name}",
                node,
            )
            return

        # coerce both operands to the intermediate type
        if not self.coerce_expr_type(node.lhs, intermediate_type, state):
            return
        if not self.coerce_expr_type(node.rhs, intermediate_type, state):
            return

        result_type = self.pick_result_type(intermediate_type, node.op)

        state.op_intermediate_types[node] = intermediate_type
        state.synthesized_types[node] = result_type
        state.contextual_types[node] = result_type

    def visit_AstUnaryOp(self, node: AstUnaryOp, state: CompileState):
        val_type = state.synthesized_types[node.val]
        arg_types = [val_type]

        intermediate_type = self.pick_intermediate_type(arg_types, node.op)
        if intermediate_type is None:
            state.err(f"Op {node.op} undefined for {val_type.display_name}", node)
            return

        if not self.coerce_expr_type(node.val, intermediate_type, state):
            return

        result_type = self.pick_result_type(intermediate_type, node.op)

        state.op_intermediate_types[node] = intermediate_type
        state.synthesized_types[node] = result_type
        state.contextual_types[node] = result_type

    def visit_AstString(self, node: AstString, state: CompileState):
        state.synthesized_types[node] = INTERNAL_STRING
        state.contextual_types[node] = INTERNAL_STRING

    def visit_AstBoolean(self, node: AstBoolean, state: CompileState):
        state.synthesized_types[node] = BOOL
        state.contextual_types[node] = BOOL

    def visit_AstAnonStruct(self, node: AstAnonStruct, state: CompileState):
        # Check for duplicate member names
        seen_names: set[str] = set()
        for name, _ in node.members:
            if name in seen_names:
                state.err(f"Duplicate member '{name}' in anonymous struct", node)
                return
            seen_names.add(name)

        # Synthesize an anonymous struct type from the member expressions
        members = tuple(
            StructMember(name, state.synthesized_types[value_expr])
            for name, value_expr in node.members
        )
        anon_type = FpyType(
            TypeKind.ANON_STRUCT,
            f"$AnonStruct({', '.join(m.name for m in members)})",
            members=members,
        )
        state.synthesized_types[node] = anon_type
        state.contextual_types[node] = anon_type

    def visit_AstAnonArray(self, node: AstAnonArray, state: CompileState):
        # Synthesize an anonymous array type from the element expressions
        elem_types = [state.synthesized_types[elem] for elem in node.elements]
        # Compute common element type
        common_elem_type = None
        if len(elem_types) > 0:
            common_elem_type = elem_types[0]
            for et in elem_types[1:]:
                common_elem_type = self.find_common_type(common_elem_type, et)
                if common_elem_type is None:
                    state.err("Array elements have no common type", node)
                    return
        anon_type = FpyType(
            TypeKind.ANON_ARRAY,
            f"$AnonArray[{len(elem_types)}]",
            length=len(node.elements),
            elem_type=common_elem_type,
        )
        state.synthesized_types[node] = anon_type
        state.contextual_types[node] = anon_type

    def resolve_args(
        self,
        node: AstFuncCall,
        func: CallableSymbol,
        node_args: list,
        state: CompileState,
    ) -> list[AstExpr] | CompileError:
        """Resolve a function call's arguments.

        Reorders named arguments to positional order, fills in default values
        for missing optional arguments, checks for missing required arguments,
        and validates argument types are compatible.

        Returns assigned_args on success.
        Returns a CompileError if there's an issue with the arguments.
        """
        func_args = func.args
        param_name_to_idx = {a[0]: i for i, a in enumerate(func_args)}

        # Validate: no positional args after named args
        seen_named = False
        for arg in node_args:
            if is_instance_compat(arg, AstNamedArgument):
                seen_named = True
            elif seen_named:
                return CompileError(
                    "Positional argument cannot follow named argument",
                    arg if is_instance_compat(arg, Ast) else node,
                )

        assigned: list[AstExpr | None] = [None] * len(func_args)

        # Process positional args (guaranteed to come before any named args)
        for i, arg in enumerate(node_args):
            if is_instance_compat(arg, AstNamedArgument):
                break
            if i < len(func_args):
                assigned[i] = arg
            else:
                return CompileError(
                    f"Too many arguments (expected {len(func_args)})",
                    node,
                )

        # Process named args
        for arg in node_args:
            if not is_instance_compat(arg, AstNamedArgument):
                continue
            if arg.name not in param_name_to_idx:
                return CompileError(
                    f"Unknown argument '{arg.name}'",
                    arg,
                )
            idx = param_name_to_idx[arg.name]
            if assigned[idx] is not None:
                return CompileError(
                    f"Argument '{arg.name}' specified multiple times",
                    arg,
                )
            assigned[idx] = arg.value

        # Fill in default values for missing arguments, error on missing required args
        for i, arg_expr in enumerate(assigned):
            if arg_expr is None:
                default_value = func_args[i][2]
                if default_value is not None:
                    assigned[i] = default_value
                else:
                    return CompileError(
                        f"Missing required argument '{func_args[i][0]}'",
                        node,
                    )

        # Type check resolved args against the function signature
        if is_instance_compat(func, CastSymbol):
            # casts do not follow coercion rules, because casting is the counterpart of coercion!
            # coercion is implicit, casting is explicit. if they say they want to cast, we let them
            node_arg = assigned[0]
            input_type = state.synthesized_types[node_arg]
            output_type = func.to_type
            # right now we only have casting to numbers
            assert output_type in SPECIFIC_NUMERIC_TYPES
            if not input_type.is_numerical:
                # cannot convert a non-numeric type to a numeric type
                return CompileError(
                    f"Expected a number, found {input_type.display_name}", node_arg
                )
        else:
            for value_expr, arg in zip(assigned, func_args):
                arg_type = arg[1]

                # Skip type check for default values that are FpyValue instances
                # this can happen if the value is hardcoded from a builtin func
                # or from dictionary defaults for type constructors
                if not is_instance_compat(value_expr, Ast):
                    assert is_instance_compat(
                        func, (BuiltinFuncSymbol, TypeCtorSymbol)
                    ), func
                    assert is_instance_compat(value_expr, FpyValue), value_expr
                    continue

                # Skip type check for default values from forward-called functions.
                # These expressions haven't been visited yet, so they're not in
                # synthesized_types. Their type compatibility is verified when
                # the function definition is visited.
                if value_expr not in state.synthesized_types:
                    continue

                unconverted_type = state.synthesized_types[value_expr]
                if not self.can_coerce_type(unconverted_type, arg_type):
                    return CompileError(
                        f"Expected {arg_type.display_name}, found {unconverted_type.display_name}",
                        value_expr if is_instance_compat(value_expr, Ast) else node,
                    )

        return assigned

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        func = state.resolved_symbols.get(node.func)
        if func is None:
            # if it were a reference to a callable, it would have already been resolved
            # if it were a symbol to something else, it would have already errored
            # so it's not even a symbol, just some expr
            state.err(f"Unknown function", node.func)
            return

        # Check that type constructors are for constant-sized types
        if is_instance_compat(func, TypeCtorSymbol):
            if not is_type_constant_size(func.type):
                state.err(
                    f"Type {func.type.display_name} is not constant-sized (contains strings)",
                    node.func,
                )
                return

        node_args = node.args if node.args else []

        # Resolve args: reorder named args, fill in defaults, check types
        result = self.resolve_args(node, func, node_args, state)
        if is_instance_compat(result, CompileError):
            state.errors.append(result)
            return

        resolved_args = result
        state.resolved_args[node] = resolved_args

        # go handle coercion/casting
        if is_instance_compat(func, CastSymbol):
            node_arg = resolved_args[0]
            output_type = func.to_type
            # we're going from input_type to output type, and we're going to ignore
            # the coercion rules
            state.contextual_types[node_arg] = output_type
            # keep track of which ones we explicitly cast. this will
            # let us turn off some checks for boundaries later when we do const folding
            # we turn off the checks because the user is asking us to force this!
            state.expr_explicit_casts.append(node_arg)
        else:
            for value_expr, arg in zip(resolved_args, func.args):
                target_type = arg[1]
                # Skip coercion for FpyValue defaults from builtins or type constructors
                if not is_instance_compat(value_expr, Ast):
                    continue
                # Skip coercion for default values from forward-called functions.
                # These will be coerced when the function definition is visited.
                if value_expr not in state.synthesized_types:
                    continue
                assert self.coerce_expr_type(value_expr, target_type, state)

        state.synthesized_types[node] = func.return_type
        state.contextual_types[node] = func.return_type

    def visit_AstRange(self, node: AstRange, state: CompileState):
        if not self.coerce_expr_type(node.lower_bound, LoopVarType, state):
            return
        if not self.coerce_expr_type(node.upper_bound, LoopVarType, state):
            return

        state.synthesized_types[node] = RANGE
        state.contextual_types[node] = RANGE

    def visit_AstAssign(self, node: AstAssign, state: CompileState):
        # should be present in resolved refs because we only let it through if
        # variable is attr, item or var
        lhs_sym = state.resolved_symbols[node.lhs]
        if not is_instance_compat(lhs_sym, (VariableSymbol, FieldAccess)):
            # assigning to a scope or something
            state.err("Invalid assignment", node.lhs)
            return

        lhs_type = None
        if is_instance_compat(lhs_sym, VariableSymbol):
            lhs_type = lhs_sym.type
        else:
            # reference to a field. make sure that the field is a field of
            # a variable and not like a field of some tlm chan (we can't modify tlm)
            if not is_instance_compat(lhs_sym.base_sym, VariableSymbol):
                state.err("Can only assign variables", node.lhs)
                return
            assert state.contextual_types[node.lhs] == state.synthesized_types[node.lhs]
            lhs_type = state.contextual_types[node.lhs]

        # coerce the rhs into the lhs type
        if not self.coerce_expr_type(node.rhs, lhs_type, state):
            return

    def visit_AstAssert(self, node: AstAssert, state: CompileState):
        if not self.coerce_expr_type(node.condition, BOOL, state):
            return
        if node.exit_code is not None:
            if not self.coerce_expr_type(node.exit_code, ErrorCodeType, state):
                return

    def visit_AstFor(self, node: AstFor, state: CompileState):
        # range must coerce to a range!
        if not self.coerce_expr_type(node.range, RANGE, state):
            return

    def visit_AstWhile(self, node: AstWhile, state: CompileState):
        if not self.coerce_expr_type(node.condition, BOOL, state):
            return

    def visit_AstIf_AstElif(self, node: Union[AstIf, AstElif], state: CompileState):
        if not self.coerce_expr_type(node.condition, BOOL, state):
            return

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # Validate that default argument types are compatible with parameter types
        if node.parameters is None:
            return

        func = state.resolved_symbols[node.name]
        assert is_instance_compat(func, FunctionSymbol), func

        for (arg_name_var, arg_type_name, default_value), (_, arg_type, _) in zip(
            node.parameters, func.args
        ):
            if default_value is not None:
                # Check that default value's type can be coerced to parameter type
                if not self.coerce_expr_type(default_value, arg_type, state):
                    return

    def visit_AstReturn(self, node: AstReturn, state: CompileState):
        func = state.enclosing_funcs[node]
        func = state.resolved_symbols[func.name]
        if func.return_type is NOTHING and node.value is not None:
            state.err("Expected no return value", node.value)
            return
        if func.return_type is not NOTHING and node.value is None:
            state.err(
                f"Expected a return value of type {func.return_type.display_name}",
                node.value,
            )
            return
        if node.value is not None:
            if not self.coerce_expr_type(node.value, func.return_type, state):
                return

    def visit_default(self, node, state):
        # coding error, missed an expr
        assert not is_instance_compat(node, AstStmtWithExpr), node


class CalculateDefaultArgConstValues(Visitor):
    """Pass that calculates const values for default argument expressions.

    This must run before CalculateConstExprValues because function call sites may
    reference functions defined later in the source. When we visit a call site that
    uses default arguments, we need the default value's const value to be available.

    This pass also enforces that default values are const expressions.
    """

    def visit_AstDef(self, node: AstDef, state: CompileState):
        if node.parameters is None:
            return

        for arg_name_var, _, default_value in node.parameters:
            if default_value is None:
                continue

            # Run the full CalculateConstExprValues pass on just this default expr
            CalculateConstExprValues().run(default_value, state)
            if len(state.errors) != 0:
                return

            # Check that the default value is a const expression
            const_value = state.const_expr_values.get(default_value)
            if const_value is None:
                state.err(
                    f"Default value for argument '{arg_name_var.name}' must be a constant expression",
                    default_value,
                )
                return


class CalculateConstExprValues(Visitor):
    """for each expr, try to calculate its constant value and store it in a map. stores None if no value could be
    calculated at compile time, and NothingType if the expr had no value"""

    @staticmethod
    def _round_float_to_type(value: float, to_type: FpyType) -> float | None:
        from fpy.types import _PRIMITIVE_FORMATS

        fmt = _PRIMITIVE_FORMATS.get(to_type.kind)
        assert fmt is not None, to_type
        try:
            packed = struct.pack(fmt, value)
        except OverflowError:
            return None

        return struct.unpack(fmt, packed)[0]

    @staticmethod
    def _parse_time_string(
        time_str: str, time_base: str, time_context: int, node: Ast, state: CompileState
    ) -> FpyValue | None:
        """Parse an ISO 8601 timestamp string into an FpyValue(TIME, ...).

        Accepts formats like:
        - "2025-12-19T14:30:00Z"
        - "2025-12-19T14:30:00.123456Z"

        Returns FpyValue(TIME, ...) with the provided timeBase (a TimeBase enum constant name)
        and timeContext, and the parsed seconds/microseconds since Unix epoch.
        """
        try:
            # Try parsing with microseconds first
            try:
                dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            except ValueError:
                # Fall back to no microseconds
                dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")

            # Convert to UTC timestamp
            dt = dt.replace(tzinfo=timezone.utc)
            timestamp = dt.timestamp()

            # Split into seconds and microseconds
            seconds = int(timestamp)
            useconds = int((timestamp - seconds) * 1_000_000)

            # Validate ranges for U32
            if seconds < 0:
                state.err(
                    f"Time string '{time_str}' results in negative seconds ({seconds})",
                    node,
                )
                return None
            if seconds > 0xFFFFFFFF:
                state.err(
                    f"Time string '{time_str}' results in seconds ({seconds}) exceeding U32 max",
                    node,
                )
                return None

            return FpyValue(
                TIME,
                {
                    "timeBase": FpyValue(TIME_BASE, time_base),
                    "timeContext": FpyValue(U8, time_context),
                    "seconds": FpyValue(U32, seconds),
                    "useconds": FpyValue(U32, useconds),
                },
            )

        except ValueError as e:
            state.err(
                f"Invalid time string '{time_str}': expected ISO 8601 format "
                "(e.g., '2025-12-19T14:30:00Z' or '2025-12-19T14:30:00.123456Z')",
                node,
            )
            return None

    @staticmethod
    def const_convert_type(
        from_val: FpyValue,
        to_type: FpyType,
        node: Ast,
        state: CompileState,
        skip_range_check: bool = False,
    ) -> FpyValue | None:
        try:
            from_type = from_val.type

            if from_type == to_type:
                # no conversion necessary
                return from_val

            if to_type.is_string:
                assert from_type == INTERNAL_STRING, from_type
                if to_type.max_length is not None:
                    encoded = from_val.val.encode("utf-8")
                    if len(encoded) > to_type.max_length:
                        state.err(
                            f"String literal is too long for type {to_type.display_name}: "
                            f"{len(encoded)} bytes exceeds max length {to_type.max_length}",
                            node,
                        )
                        return None
                return FpyValue(to_type, from_val.val)

            if to_type.is_float:
                assert from_type.is_numerical, from_type
                raw_val = from_val.val

                if to_type == FLOAT:
                    # arbitrary precision
                    # decimal constructor should handle all cases: int, float, or other Decimal
                    return FpyValue(FLOAT, Decimal(raw_val))

                # otherwise, we're going to a finite bitwidth float type
                try:
                    coerced_value = float(raw_val)
                except OverflowError:
                    state.err(
                        f"{raw_val} is out of range for type {to_type.display_name}",
                        node,
                    )
                    return None

                rounded_value = CalculateConstExprValues._round_float_to_type(
                    coerced_value, to_type
                )
                if rounded_value is None:
                    state.err(
                        f"{raw_val} is out of range for type {to_type.display_name}",
                        node,
                    )
                    return None

                converted = FpyValue(to_type, rounded_value)
                try:
                    # catch if we would crash the struct packing lib
                    converted.serialize()
                except OverflowError:
                    state.err(
                        f"{raw_val} is out of range for type {to_type.display_name}",
                        node,
                    )
                    return None
                return converted
            if to_type.is_integer:
                assert from_type.is_numerical, from_type
                raw_val = from_val.val

                if to_type == INTEGER:
                    # arbitrary precision
                    # int constructor should handle all cases: int, float, or Decimal
                    return FpyValue(INTEGER, int(raw_val))

                # otherwise going to a finite bitwidth integer type

                if not skip_range_check:
                    # does it fit within bounds?
                    # check that the value can fit in the dest type
                    dest_min, dest_max = to_type.value_range()
                    if raw_val < dest_min or raw_val > dest_max:
                        state.err(
                            f"{raw_val} is out of range for type {to_type.display_name}",
                            node,
                        )
                        return None

                    # just convert it
                    raw_val = int(raw_val)
                else:
                    # we skipped the range check, but it's still gotta fit. cut it down

                    # handle narrowing, if necessary
                    raw_val = int(raw_val)
                    # if signed, convert to unsigned (bit representation should be the same)
                    # first cut down to bitwidth. performed in two's complement
                    mask = (1 << to_type.bits) - 1
                    # this also implicitly converts value to an unsigned number
                    raw_val &= mask
                    if to_type in SIGNED_INTEGER_TYPES:
                        # now if the target was signed:
                        sign_bit = 1 << (to_type.bits - 1)
                        if raw_val & sign_bit:
                            # the sign bit is set, the result should be negative
                            # subtract the max value as this is how two's complement works
                            raw_val -= 1 << to_type.bits

                # okay, we either checked that the value fits in the dest, or we've skipped
                # the check and changed the value to fit
                return FpyValue(to_type, raw_val)

            assert False, (from_val, from_type, to_type)
        except (ValueError, struct.error) as e:
            state.err(f"For type {from_type.display_name}: {e}", node)
            return None

    def visit_AstLiteral(self, node: AstLiteral, state: CompileState):
        unconverted_type = state.synthesized_types[node]

        try:
            expr_value = FpyValue(unconverted_type, node.value)
        except (ValueError, struct.error) as e:
            # TODO can this be reached any more? maybe for string types
            state.err(f"For type {unconverted_type.display_name}: {e}", node)
            return

        skip_range_check = node in state.expr_explicit_casts
        converted_type = state.contextual_types[node]
        if converted_type != unconverted_type:
            expr_value = self.const_convert_type(
                expr_value, converted_type, node, state, skip_range_check
            )
            if expr_value is None:
                return

        state.const_expr_values[node] = expr_value

    def visit_AstGetAttr(self, node: AstGetAttr, state: CompileState):
        sym = state.resolved_symbols[node]
        if not is_symbol_an_expr(sym):
            return
        unconverted_type = state.synthesized_types[node]
        converted_type = state.contextual_types[node]
        expr_value = None
        if is_instance_compat(sym, (ChDef, PrmDef, VariableSymbol)):
            # has a value but won't try to calc at compile time
            state.const_expr_values[node] = None
            return
        elif is_instance_compat(sym, FpyValue):
            expr_value = sym
        elif is_instance_compat(sym, FieldAccess):
            parent_value = state.const_expr_values[node.parent]
            if parent_value is None:
                # Parent is not const. For anon struct, try getting the member
                # expression's const value directly.
                if is_instance_compat(node.parent, AstAnonStruct):
                    for name, value_expr in node.parent.members:
                        if name == node.attr:
                            member_val = state.const_expr_values.get(value_expr)
                            if member_val is not None:
                                expr_value = member_val
                            break
                if expr_value is None:
                    state.const_expr_values[node] = None
                    return
            else:
                # we are accessing an attribute of something with an fprime value at compile time
                # we must be getting a member
                if isinstance(parent_value, FpyValue) and parent_value.type.kind in (
                    TypeKind.STRUCT,
                    TypeKind.ANON_STRUCT,
                ):
                    expr_value = parent_value.val[node.attr]
                else:
                    assert False, parent_value

        assert expr_value is not None

        assert (
            isinstance(expr_value, FpyValue) and expr_value.type == unconverted_type
        ), (
            expr_value,
            unconverted_type,
        )

        skip_range_check = node in state.expr_explicit_casts
        if converted_type != unconverted_type:
            expr_value = self.const_convert_type(
                expr_value, converted_type, node, state, skip_range_check
            )
            if expr_value is None:
                return
        state.const_expr_values[node] = expr_value

    def visit_AstIndexExpr(self, node: AstIndexExpr, state: CompileState):
        sym = state.resolved_symbols[node]
        # index expression can only be a field symbol
        assert is_instance_compat(sym, FieldAccess), sym

        parent_value = state.const_expr_values[node.parent]

        if parent_value is None:
            state.const_expr_values[node] = None
            return

        assert isinstance(parent_value, FpyValue) and parent_value.type.kind in (
            TypeKind.ARRAY,
            TypeKind.ANON_ARRAY,
        ), parent_value

        idx = state.const_expr_values.get(node.item)
        if idx is None:
            # no compile time constant value for our index
            state.const_expr_values[node] = None
            return

        assert isinstance(idx, FpyValue)

        if idx.val < 0 or idx.val >= len(parent_value.val):
            # Out of bounds — CheckConstArrayAccesses will report the error
            state.const_expr_values[node] = None
            return

        expr_value = parent_value.val[idx.val]

        unconverted_type = state.synthesized_types[node]
        assert (
            isinstance(expr_value, FpyValue) and expr_value.type == unconverted_type
        ), (
            expr_value,
            unconverted_type,
        )

        skip_range_check = node in state.expr_explicit_casts
        converted_type = state.contextual_types[node]
        if converted_type != unconverted_type:
            expr_value = self.const_convert_type(
                expr_value, converted_type, node, state, skip_range_check
            )
            if expr_value is None:
                return
        state.const_expr_values[node] = expr_value

    def visit_AstIdent(self, node: AstIdent, state: CompileState):
        sym = state.resolved_symbols[node]
        if not is_symbol_an_expr(sym):
            return
        unconverted_type = state.synthesized_types[node]
        converted_type = state.contextual_types[node]
        expr_value = None
        if is_instance_compat(sym, (ChDef, PrmDef, VariableSymbol)):
            # Has a value but we don't try to calculate it at compile time.
            # NOTE: If you ever add const-folding for VariableSymbol here, you must also
            # update CalculateDefaultArgConstValues. That pass runs CalculateConstExprValues
            # on default argument expressions BEFORE this pass runs on variable assignments.
            # So if a default value references a variable, the variable's const value won't
            # be available yet, and the default value will incorrectly be rejected as non-const.
            state.const_expr_values[node] = None
            return
        elif is_instance_compat(sym, FpyValue):
            expr_value = sym
        else:
            assert False, sym

        assert expr_value is not None

        assert (
            isinstance(expr_value, FpyValue) and expr_value.type == unconverted_type
        ), (
            expr_value,
            unconverted_type,
        )

        skip_range_check = node in state.expr_explicit_casts
        if converted_type != unconverted_type:
            expr_value = self.const_convert_type(
                expr_value, converted_type, node, state, skip_range_check
            )
            if expr_value is None:
                return
        state.const_expr_values[node] = expr_value

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        func = state.resolved_symbols[node.func]
        assert is_instance_compat(func, CallableSymbol)

        # Use resolved args from semantic analysis (already in positional order,
        # with defaults filled in)
        # This is guaranteed to be set by PickTypesAndResolveAttrsAndItems
        resolved_args = state.resolved_args[node]

        # Gather arg values. Since defaults are already filled in, we just need
        # to look up each arg's const value. For FpyValue defaults from builtins,
        # use the value directly.
        arg_values = []
        for arg_expr in resolved_args:
            if is_instance_compat(arg_expr, Ast):
                arg_values.append(state.const_expr_values.get(arg_expr))
            else:
                # It's a raw FpyValue default from a builtin
                arg_values.append(arg_expr)

        unknown_value = any(v is None for v in arg_values)

        # Check that any args required to be compile-time constants actually are,
        # even if other args are unknown (those will be evaluated at runtime).
        if is_instance_compat(func, BuiltinFuncSymbol):
            for i in func.const_arg_indices:
                if arg_values[i] is None:
                    state.errors.append(
                        CompileError(
                            f"Argument '{func.args[i][0]}' of '{func.name}' must be a compile-time constant",
                            resolved_args[i],
                        )
                    )
                    return

        if unknown_value:
            # we will have to calculate this at runtime
            state.const_expr_values[node] = None
            return

        expr_value = None

        if is_instance_compat(func, TypeCtorSymbol):
            # actually construct the type
            if func.type.kind == TypeKind.STRUCT:
                # pass in args as a dict
                arg_dict = {m.name: v for m, v in zip(func.type.members, arg_values)}
                expr_value = FpyValue(func.type, arg_dict)

            elif func.type.kind == TypeKind.ARRAY:
                expr_value = FpyValue(func.type, arg_values)

            else:
                # no other FpyTypes have ctors
                assert False, func.return_type
        elif is_instance_compat(func, CastSymbol):
            # should only be one value. it should be of some numeric type
            # our const convert type func will convert it for us
            expr_value = arg_values[0]
        elif func is TIME_MACRO:
            # time() builtin parses ISO 8601 timestamps at compile time
            timestamp_str = arg_values[0].val
            timeBase = arg_values[1].val
            timeContext = arg_values[2].val
            expr_value = self._parse_time_string(
                timestamp_str, timeBase, timeContext, node, state
            )
            if expr_value is None:
                return
        else:
            # don't try to calculate the value of this function call
            # it's something like a user defined func, cmd or builtin
            state.const_expr_values[node] = None
            return

        unconverted_type = state.synthesized_types[node]
        assert (
            isinstance(expr_value, FpyValue) and expr_value.type == unconverted_type
        ), (
            expr_value,
            unconverted_type,
        )

        skip_range_check = node in state.expr_explicit_casts
        converted_type = state.contextual_types[node]
        if converted_type != unconverted_type:
            expr_value = self.const_convert_type(
                expr_value, converted_type, node, state, skip_range_check
            )
            if expr_value is None:
                return

        state.const_expr_values[node] = expr_value

    def visit_AstBinaryOp(self, node: AstBinaryOp, state: CompileState):
        # Check if both left-hand side (lhs) and right-hand side (rhs) are constants
        lhs_value: FpyValue = state.const_expr_values.get(node.lhs)
        rhs_value: FpyValue = state.const_expr_values.get(node.rhs)

        if lhs_value is None or rhs_value is None:
            state.const_expr_values[node] = None
            return

        # Time operations are desugared to function calls; skip constant folding.
        lhs_type = state.contextual_types[node.lhs]
        rhs_type = state.contextual_types[node.rhs]
        if (lhs_type, rhs_type, node.op) in TIME_OPS:
            state.const_expr_values[node] = None
            return

        # Both sides are constants, evaluate the operation if the operator is supported
        # get the actual pythonic value from the fpy type
        lhs_value = lhs_value.val
        rhs_value = rhs_value.val

        folded_value = None
        # Arithmetic operations
        try:
            if node.op == BinaryStackOp.ADD:
                folded_value = lhs_value + rhs_value
            elif node.op == BinaryStackOp.SUBTRACT:
                folded_value = lhs_value - rhs_value
            elif node.op == BinaryStackOp.MULTIPLY:
                folded_value = lhs_value * rhs_value
            elif node.op == BinaryStackOp.DIVIDE:
                folded_value = lhs_value / rhs_value
            elif node.op == BinaryStackOp.EXPONENT:
                folded_value = lhs_value**rhs_value
            elif node.op == BinaryStackOp.FLOOR_DIVIDE:
                # Floor toward -inf (Python `//`), matching the runtime backends.
                if isinstance(lhs_value, int) and isinstance(rhs_value, int):
                    folded_value = lhs_value // rhs_value
                elif isinstance(lhs_value, Decimal):
                    folded_value = (lhs_value / rhs_value).to_integral_value(
                        rounding=decimal.ROUND_FLOOR
                    )
                else:
                    folded_value = Decimal(
                        str(lhs_value / rhs_value)
                    ).to_integral_value(rounding=decimal.ROUND_FLOOR)
            elif node.op == BinaryStackOp.MODULUS:
                folded_value = lhs_value % rhs_value
            # Boolean logic operations
            elif node.op == BinaryStackOp.AND:
                folded_value = lhs_value and rhs_value
            elif node.op == BinaryStackOp.OR:
                folded_value = lhs_value or rhs_value
            # Inequalities
            elif node.op == BinaryStackOp.GREATER_THAN:
                folded_value = lhs_value > rhs_value
            elif node.op == BinaryStackOp.GREATER_THAN_OR_EQUAL:
                folded_value = lhs_value >= rhs_value
            elif node.op == BinaryStackOp.LESS_THAN:
                folded_value = lhs_value < rhs_value
            elif node.op == BinaryStackOp.LESS_THAN_OR_EQUAL:
                folded_value = lhs_value <= rhs_value
            # Equality Checking
            elif node.op == BinaryStackOp.EQUAL:
                folded_value = lhs_value == rhs_value
            elif node.op == BinaryStackOp.NOT_EQUAL:
                folded_value = lhs_value != rhs_value
            else:
                # missing an operation
                assert False, node.op
        except ZeroDivisionError:
            # also catches decimal.DivisionByZero (a ZeroDivisionError subclass)
            state.err("Divide by zero error", node)
            return
        except (OverflowError, decimal.Overflow):
            # decimal.Overflow is a sibling of the builtin OverflowError
            # (both are ArithmeticError), not a subclass, so it must be listed
            # explicitly or it escapes as an uncaught compiler crash.
            state.err("Overflow error", node)
            return
        except ValueError as err:
            state.err(str(err) if str(err) else "Domain error", node)
            return
        except decimal.DecimalException:
            # any other Decimal arithmetic error (InvalidOperation, etc.)
            state.err("Domain error", node)
            return

        assert folded_value is not None

        if type(folded_value) == int:
            folded_value = FpyValue(INTEGER, folded_value)
        elif type(folded_value) == float:
            # can happen when operands were previously const-converted to
            # specific float types (F32/F64) whose .val is a
            # Python float, or from int / int (true division) in Python
            folded_value = FpyValue(FLOAT, Decimal(folded_value))
        elif type(folded_value) == Decimal:
            folded_value = FpyValue(FLOAT, folded_value)
        elif type(folded_value) == bool:
            folded_value = FpyValue(BOOL, folded_value)
        else:
            assert False, folded_value

        # first fold, store the result in arbitrary precision

        # then if the expression is some other type, convert:
        skip_range_check = node in state.expr_explicit_casts
        unconverted_type = state.synthesized_types.get(node)
        # the intent of this is to handle situations where we're constant folding and the results cannot be arbitrary precision
        folded_value = self.const_convert_type(
            folded_value, unconverted_type, node, state, skip_range_check=False
        )

        converted_type = state.contextual_types.get(node)
        # okay and now perform type coercion/casting
        if converted_type != unconverted_type:
            folded_value = self.const_convert_type(
                folded_value, converted_type, node, state, skip_range_check
            )
            if folded_value is None:
                return
        state.const_expr_values[node] = folded_value

    def visit_AstUnaryOp(self, node: AstUnaryOp, state: CompileState):
        value: FpyValue = state.const_expr_values.get(node.val)

        if value is None:
            state.const_expr_values[node] = None
            return

        # input is constant, evaluate the operation if the operator is supported

        # get the actual pythonic value from the fpy type
        value = value.val
        folded_value = None

        if node.op == UnaryStackOp.NEGATE:
            # Decimal.__neg__ follows the decimal spec and returns +0 for any
            # zero, which would fold the literal -0.0 to +0.0. copy_negate
            # flips the sign unconditionally, matching runtime negation
            # (llvm fneg / the VM's multiply by -1.0).
            folded_value = value.copy_negate() if type(value) == Decimal else -value
        elif node.op == UnaryStackOp.IDENTITY:
            folded_value = value
        elif node.op == UnaryStackOp.NOT:
            folded_value = not value
        else:
            # missing an operation
            assert False, node.op

        assert folded_value is not None

        if type(folded_value) == int:
            folded_value = FpyValue(INTEGER, folded_value)
        elif type(folded_value) == float:
            folded_value = FpyValue(FLOAT, Decimal(folded_value))
        elif type(folded_value) == Decimal:
            folded_value = FpyValue(FLOAT, folded_value)
        elif type(folded_value) == bool:
            folded_value = FpyValue(BOOL, folded_value)
        else:
            assert False, folded_value

        # first fold, store the result in arbitrary precision

        # then if the expression is some other type, convert:
        skip_range_check = node in state.expr_explicit_casts
        unconverted_type = state.synthesized_types.get(node)
        # the intent of this is to handle situations where we're constant folding and the results cannot be arbitrary precision
        folded_value = self.const_convert_type(
            folded_value, unconverted_type, node, state, skip_range_check=False
        )

        converted_type = state.contextual_types.get(node)
        if converted_type != unconverted_type:
            folded_value = self.const_convert_type(
                folded_value, converted_type, node, state, skip_range_check
            )
            if folded_value is None:
                return
        state.const_expr_values[node] = folded_value

    def visit_AstRange(self, node: AstRange, state: CompileState):
        # ranges don't really end up having a value, they kinda just exist as a type
        state.const_expr_values[node] = None

    def visit_AstAnonStruct(self, node: AstAnonStruct, state: CompileState):
        converted_type = state.contextual_types[node]

        if converted_type.kind == TypeKind.ANON_STRUCT:
            exprs = [value_expr for _, value_expr in node.members]
            names = [name for name, _ in node.members]
        else:
            assert converted_type.kind == TypeKind.STRUCT, converted_type
            exprs = state.resolved_args[node]
            names = [m.name for m in converted_type.members]

        values = []
        for expr in exprs:
            if is_instance_compat(expr, Ast):
                val = state.const_expr_values.get(expr)
                if val is None:
                    state.const_expr_values[node] = None
                    return
                values.append(val)
            else:
                values.append(expr)

        state.const_expr_values[node] = FpyValue(
            converted_type, dict(zip(names, values))
        )

    def visit_AstAnonArray(self, node: AstAnonArray, state: CompileState):
        converted_type = state.contextual_types[node]

        if converted_type.kind == TypeKind.ANON_ARRAY:
            exprs = list(node.elements)
        else:
            assert converted_type.kind == TypeKind.ARRAY, converted_type
            exprs = state.resolved_args[node]

        values = []
        for expr in exprs:
            if is_instance_compat(expr, Ast):
                val = state.const_expr_values.get(expr)
                if val is None:
                    state.const_expr_values[node] = None
                    return
                values.append(val)
            else:
                values.append(expr)

        state.const_expr_values[node] = FpyValue(converted_type, values)

    def visit_default(self, node, state):
        # coding error, missed an expr
        assert not is_instance_compat(node, AstExpr), node


class CheckAllBranchesReturn(Visitor):
    def visit_AstReturn(self, node: AstReturn, state: CompileState):
        state.does_return[node] = True

    def visit_AstBlock(self, node: AstBlock, state: CompileState):
        state.does_return[node] = any(state.does_return[n] for n in node.stmts)

    def visit_AstIf(self, node: AstIf, state: CompileState):
        # an if statement returns if all of its branches return
        branch_returns = [state.does_return[node.body]]

        for _elif in node.elifs:
            branch_returns.append(state.does_return[_elif])

        if node.els is not None:
            branch_returns.append(state.does_return[node.els])
        else:
            # implicit else branch that falls through without returning
            branch_returns.append(False)

        state.does_return[node] = all(branch_returns)

    def visit_AstElif(self, node: Union[AstElif], state: CompileState):
        state.does_return[node] = state.does_return[node.body]

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # if we found another func def inside this body, it definitely doesn't return
        state.does_return[node] = False

    def visit_AstAssign_AstPass_AstAssert_AstContinue_AstBreak_AstWhile_AstFor(
        self,
        node: Union[
            AstAssign, AstPass, AstAssert, AstContinue, AstBreak, AstWhile, AstFor
        ],
        state: CompileState,
    ):
        # while and for do not return because we don't know if their body
        # will actually execute.
        # we could do some analysis to figure this out but it would only work
        # for constants
        state.does_return[node] = False

    def visit_AstExpr(self, node: AstExpr, state: CompileState):
        # expressions do not return, except exit
        if not is_instance_compat(node, AstFuncCall):
            state.does_return[node] = False
            return
        func = state.resolved_symbols[node.func]
        if not is_instance_compat(func, BuiltinFuncSymbol) or not func.name == "exit":
            state.does_return[node] = False
            return
        # builtin exit "returns" (really just ends call stack entirely)
        state.does_return[node] = True

    def visit_default(self, node, state):
        assert not is_instance_compat(node, AstStmt)


class CheckFunctionReturns(Visitor):
    def visit_AstDef(self, node: AstDef, state: CompileState):
        CheckAllBranchesReturn().run(node.body, state)
        if node.return_type is None:
            # don't need to return explicitly
            return
        if not state.does_return[node.body]:
            state.err(
                f"Function '{node.name.name}' does not always return a value", node
            )
            return


class CheckConstArrayAccesses(Visitor):
    def visit_AstIndexExpr(self, node: AstIndexExpr, state: CompileState):
        # if the index is a const, we should be able to check if it's in bounds
        idx_value = state.const_expr_values.get(node.item)

        parent_type = state.contextual_types[node.parent]
        assert parent_type.kind in (TypeKind.ARRAY, TypeKind.ANON_ARRAY), parent_type

        if idx_value is None:
            # can't check at compile time
            if parent_type.kind == TypeKind.ANON_ARRAY:
                state.err(
                    "Index on anonymous array must be a compile-time constant",
                    node.item,
                )
            return

        if idx_value.val < 0 or idx_value.val >= parent_type.length:
            state.err(
                f"Index {idx_value.val} out of bounds for array type {parent_type.display_name} with length {parent_type.length}",
                node.item,
            )
            return


class WarnRangesAreNotEmpty(Visitor):
    def visit_AstRange(self, node: AstRange, state: CompileState):
        # if the index is a const, we should be able to check if it's in bounds
        lower_value: FpyValue = state.const_expr_values.get(node.lower_bound)
        upper_value: FpyValue = state.const_expr_values.get(node.upper_bound)
        if lower_value is None or upper_value is None:
            # cannot check at compile time
            return

        if lower_value.val >= upper_value.val:
            state.warn(WarningType.EMPTY_RANGE, "Range is empty", node)


class CheckSequenceArgs(Visitor):
    """Check sequence argument constraints:
    - vararg data fits in the SeqArgs buffer
    - arg count fits in a u8 (max 255)
    - arg names and type names fit in a pascal string (max 255 UTF-8 bytes)
    """

    def visit_AstSequenceMetadata(self, node: AstSequenceMetadata, state: CompileState):
        if node.parameters is None:
            return

        if len(node.parameters) > 255:
            state.err(
                f"Too many sequence arguments ({len(node.parameters)}); max is 255",
                node,
            )
            return

        # mirrors the sequencer's load-time check against
        # Svc.Fpy.MAX_SEQUENCE_ARG_COUNT (TooManySequenceArgs)
        if len(node.parameters) > state.max_seq_arg_count:
            state.err(
                f"Too many sequence arguments ({len(node.parameters)}); max is "
                f"{state.max_seq_arg_count} (Svc.Fpy.MAX_SEQUENCE_ARG_COUNT)",
                node,
            )
            return

        for arg_name_var, arg_type_name in node.parameters:
            arg_var = state.resolved_symbols[arg_name_var]
            arg_type = state.resolved_symbols[arg_type_name]

            name_len = len(arg_var.name.encode("utf-8"))
            if name_len > 255:
                state.err(
                    f"Sequence argument name '{arg_var.name}' is too long "
                    f"({name_len} UTF-8 bytes); max is 255",
                    arg_name_var,
                )
                return

            type_name_len = len(arg_type.name.encode("utf-8"))
            if type_name_len > 255:
                state.err(
                    f"Sequence argument type name '{arg_type.name}' is too long "
                    f"({type_name_len} UTF-8 bytes); max is 255",
                    arg_type_name,
                )
                return

        total_arg_size = sum(
            state.resolved_symbols[arg_type_name].max_size
            for _, arg_type_name in node.parameters
        )

        # sequence arguments only ever arrive through a Svc.SeqArgs buffer
        # (RUN_ARGS/VALIDATE_ARGS from the ground, or a seq-run command from
        # another sequence), so they must fit in its buffer
        seq_args_capacity = SEQ_ARGS.members[1].type.length
        if total_arg_size > seq_args_capacity:
            state.err(
                f"Total size of sequence arguments ({total_arg_size} bytes) "
                f"exceeds Svc.SeqArgs buffer capacity ({seq_args_capacity} bytes)",
                node,
            )
            return

        # mirrors the sequencer's load-time check against
        # Svc.Fpy.MAX_STACK_SIZE (ArgTotalSizeExceedsStackLimit)
        if total_arg_size > state.max_stack_size:
            state.err(
                f"Total size of sequence arguments ({total_arg_size} bytes) "
                f"exceeds Svc.Fpy.MAX_STACK_SIZE ({state.max_stack_size} bytes)",
                node,
            )
            return

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        func = state.resolved_symbols.get(node.func)
        if not is_instance_compat(func, CommandSymbol) or not func.is_seq_run_with_args:
            return

        bin_name = state.resolved_args[node][0].value
        seq_arg_types = [t for _, t in state.called_seq_arg_specs[bin_name]]
        vararg_data_size = sum(t.max_size for t in seq_arg_types)
        buffer_size = SEQ_ARGS.members[1].type.length
        if vararg_data_size > buffer_size:
            state.err(
                f"Sequence arguments ({vararg_data_size} bytes) exceed "
                f"Svc.SeqArgs buffer capacity ({buffer_size} bytes)",
                node,
            )
            return
