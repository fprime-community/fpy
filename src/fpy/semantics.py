from __future__ import annotations
from dataclasses import fields
from datetime import datetime, timezone
from decimal import Decimal
import decimal
import struct
from typing import Union

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
    BOOL,
    TIME,
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
    is_instance_compat,
)
from fpy.state import (
    BuiltinFuncSymbol,
    CallableSymbol,
    CastSymbol,
    CompileState,
    FieldAccess,
    ForLoopAnalysis,
    FunctionSymbol,
    NameGroup,
    Symbol,
    SymbolTable,
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
    LoopVarType,
    BinaryStackOp,
    UnaryStackOp,
)
from fpy.state import ChDef, PrmDef
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
    """assigns a unique id to each node to allow it to be indexed in a dict"""

    def visit_default(self, node, state: CompileState):
        node.id = state.next_node_id
        state.next_node_id += 1


class CreateScopes:
    """Creates block-level scopes for all AstBlocks.

    Each AstBlock creates a new scope that is a child of the enclosing scope.
    Function bodies create a scope with in_function=True.
    """

    def run(self, start: Ast, state: CompileState):
        self._walk(start, state, state.global_value_scope)

    def _walk(self, node, state: CompileState, scope: SymbolTable):
        if not isinstance(node, Ast):
            return

        if isinstance(node, AstDef):
            self._walk_def(node, state, scope)
            return

        if isinstance(node, AstFor):
            self._walk_for(node, state, scope)
            return

        if isinstance(node, AstBlock):
            self._walk_block(node, state, scope)
            return

        # For all other nodes, set the scope and walk children
        state.enclosing_value_scope[node] = scope
        self._walk_children(node, state, scope)

    def _walk_def(self, node: AstDef, state: CompileState, scope: SymbolTable):
        # Per grammar, defs only appear at the top level, so scope is always global.
        assert scope is state.global_value_scope
        state.enclosing_value_scope[node] = scope

        # Name reference is in the enclosing scope
        self._walk(node.name, state, scope)

        # Return type annotation is in the enclosing scope
        if node.return_type is not None:
            self._walk(node.return_type, state, scope)

        # Create the function body scope
        func_body_scope = SymbolTable(parent=scope)
        func_body_scope.in_function = True

        # Parameters are in the function body scope
        if node.parameters is not None:
            for arg_name_var, arg_type_name, default_value in node.parameters:
                self._walk(arg_name_var, state, func_body_scope)
                self._walk(arg_type_name, state, func_body_scope)
                # Default values are evaluated at definition site (enclosing scope)
                if default_value is not None:
                    self._walk(default_value, state, scope)

        # Set the body scope directly (don't create a child scope for it)
        state.enclosing_value_scope[node.body] = func_body_scope
        for stmt in node.body.stmts:
            self._walk(stmt, state, func_body_scope)

    def _walk_for(self, node: AstFor, state: CompileState, scope: SymbolTable):
        state.enclosing_value_scope[node] = scope

        # Range is evaluated in the parent scope
        self._walk(node.range, state, scope)

        # Body creates a new scope; loop_var lives inside it
        body_scope = SymbolTable(parent=scope)
        state.enclosing_value_scope[node.body] = body_scope

        # Loop variable is in the body scope
        self._walk(node.loop_var, state, body_scope)

        # Walk body statements
        for stmt in node.body.stmts:
            self._walk(stmt, state, body_scope)

    def _walk_block(self, node: AstBlock, state: CompileState, scope: SymbolTable):
        # Check if the scope was pre-set (e.g., function body)
        pre_set = state.enclosing_value_scope.get(node)
        if pre_set is not None:
            block_scope = pre_set
        elif node is state.root:
            block_scope = scope
        else:
            # Each indentation block creates a new child scope
            block_scope = SymbolTable(parent=scope)

        state.enclosing_value_scope[node] = block_scope
        for stmt in node.stmts:
            self._walk(stmt, state, block_scope)

    def _walk_children(self, node, state: CompileState, scope: SymbolTable):
        for f in fields(node):
            val = getattr(node, f.name)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, Ast):
                        self._walk(item, state, scope)
                    elif isinstance(item, tuple):
                        for elem in item:
                            if isinstance(elem, Ast):
                                self._walk(elem, state, scope)
            elif isinstance(val, Ast):
                self._walk(val, state, scope)


class CreateVariablesAndFuncs(TopDownVisitor):
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

        assert is_instance_compat(node.lhs, AstIdent), node.lhs
        # variable decl or assign
        scope = state.enclosing_value_scope[node]

        if node.type_ann is not None:
            # new variable declaration
            # make sure it isn't defined in this scope (shadowing parent scopes is ok)
            existing_local = scope.get(node.lhs.name)
            if existing_local is not None:
                # redeclaring an existing variable in the SAME scope
                state.err(f"Variable '{node.lhs.name}' has already been defined", node)
                return
            # okay, define the var
            is_global = not scope.in_function
            var = VariableSymbol(
                node.lhs.name, node.type_ann, node, is_global=is_global
            )
            # new var. put it in the scope
            scope[node.lhs.name] = var
        else:
            # otherwise, it's a reference to an existing var
            # walk up the scope chain to find it
            sym = scope.lookup(node.lhs.name)
            if sym is None:
                # unable to find this symbol
                state.err(
                    f"Variable '{node.lhs.name}' used before defined",
                    node.lhs,
                )
                return
            # okay, we were able to resolve it

    def visit_AstFor(self, node: AstFor, state: CompileState):
        # The loop variable is always a new declaration in the loop body's scope.
        body_scope = state.enclosing_value_scope[node.body]
        is_global = not body_scope.in_function

        loop_var = VariableSymbol(
            node.loop_var.name, None, node, LoopVarType, is_global=is_global
        )
        body_scope[loop_var.name] = loop_var

        # Each loop also defines an implicit upper-bound variable
        upper_bound_var = VariableSymbol(
            state.new_anonymous_variable_name(),
            None,
            node,
            LoopVarType,
            is_global=is_global,
        )
        body_scope[upper_bound_var.name] = upper_bound_var
        analysis = ForLoopAnalysis(loop_var, upper_bound_var)
        state.for_loops[node] = analysis

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # Functions always go in the global callable scope
        existing_func = state.global_callable_scope.get(node.name.name)
        if existing_func is not None:
            state.err(
                f"Function '{node.name.name}' has already been defined", node.name
            )
            return STOP_DESCENT

        func = FunctionSymbol(
            # we know the name
            node.name.name,
            # we don't know the return type yet
            return_type=None,
            # we don't know the arg types yet
            args=None,
            definition=node,
        )

        state.global_callable_scope[func.name] = func

        if node.parameters is None:
            # no arguments
            self._deferred_defs.append(node)
            return STOP_DESCENT

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
                return STOP_DESCENT

        # Parameters go in the function's value scope
        func_scope = state.enclosing_value_scope[node.body]
        for arg in node.parameters:
            arg_name_var, arg_type_name, default_value = arg
            existing_local = func_scope.get(arg_name_var.name)
            if existing_local is not None:
                # two args with the same name
                state.err(
                    f"Parameter '{arg_name_var.name}' has already been defined",
                    arg_name_var,
                )
                return STOP_DESCENT
            arg_var = VariableSymbol(arg_name_var.name, arg_type_name, node)
            func_scope[arg_name_var.name] = arg_var

        # Defer traversal of the function body to phase 2, so that all
        # global-scope declarations are visible inside functions regardless
        # of source ordering.
        self._deferred_defs.append(node)
        return STOP_DESCENT


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


class ResolveQualifiedNames(TopDownVisitor):

    def try_resolve_name(
        self, node: Ast, group: NameGroup, state: CompileState
    ) -> bool:
        """resolves the root name of a qualified name, return True if able to resolve, False
        if an error was raised.
        if the node is not a qualified name, return True"""
        # first check that this is a fully qualified name
        # list of attrs, most specific attrs first
        attrs = []
        leaf_node = node
        root_node = node
        while is_instance_compat(root_node, AstGetAttr):
            attrs.append(root_node)
            root_node = root_node.parent

        if not is_instance_compat(root_node, AstIdent):
            # not a qualified name
            # skip for now
            return True

        root_symbol = None
        # it is a qualified name
        # look up the root name in the appropriate scope
        if group == NameGroup.CALLABLE:
            root_symbol = state.global_callable_scope.get(root_node.name)
        elif group == NameGroup.TYPE:
            root_symbol = state.global_type_scope.get(root_node.name)
        else:
            # Walk up the scope chain to find the value
            root_symbol = state.enclosing_value_scope[root_node].lookup(root_node.name)

        # the node which corresponds to the entire qualified name
        # note this does not include nodes which are member accesses
        qualified_name_node = root_node

        # the parent of the attr that we're about to resolve, as we iterate
        # through the list of attrs
        current_parent_symbol = root_symbol

        # okay, now we just have to perform attribute resolution
        while len(attrs) > 0:
            if current_parent_symbol is None:
                state.err(f"Unknown {group}", leaf_node)
                return False

            state.resolved_symbols[qualified_name_node] = current_parent_symbol

            if is_symbol_an_expr(current_parent_symbol):
                # this is member access
                # stop here
                break

            if not is_instance_compat(current_parent_symbol, SymbolTable):
                # it's not member access and it's not namespace access
                state.err(f"Unknown {group}", leaf_node)
                return False

            attr = attrs.pop()
            qualified_name_node = attr
            current_parent_symbol = current_parent_symbol.get(attr.attr)

        # has it resolved?
        if current_parent_symbol is None:
            state.err(f"Unknown {group}", leaf_node)
            return False

        # but has it actually resolved to a non-namespace symbol?
        if is_instance_compat(current_parent_symbol, SymbolTable):
            state.err(f"Unknown {group}", leaf_node)
            return False

        state.resolved_symbols[qualified_name_node] = current_parent_symbol
        return True

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # all callables are always resolved in callable scope
        if not self.try_resolve_name(node.name, NameGroup.CALLABLE, state):
            return
        if node.return_type is not None:
            # all types always in type scope
            if not self.try_resolve_name(node.return_type, NameGroup.TYPE, state):
                return

        if node.parameters is not None:
            for arg_name_var, arg_type_name, default_value in node.parameters:
                if not self.try_resolve_name(arg_type_name, NameGroup.TYPE, state):
                    return
                # arg names become vars in func scope, so resolve them in func scope
                if not self.try_resolve_name(arg_name_var, NameGroup.VALUE, state):
                    return
                if default_value is not None:
                    # TODO make sure that we test that default vals cant access vars inside of func
                    # default values are calculated outside of func scope
                    if not self.try_resolve_name(default_value, NameGroup.VALUE, state):
                        return

    def visit_AstAssign(self, node: AstAssign, state: CompileState):
        if node.type_ann is not None:
            if not self.try_resolve_name(node.type_ann, NameGroup.TYPE, state):
                return

        if not self.try_resolve_name(node.lhs, NameGroup.VALUE, state):
            return
        if not self.try_resolve_name(node.rhs, NameGroup.VALUE, state):
            return

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        if not self.try_resolve_name(node.func, NameGroup.CALLABLE, state):
            return

        if node.args is None:
            return

        for arg in node.args:
            if is_instance_compat(arg, AstNamedArgument):
                if not self.try_resolve_name(arg.value, NameGroup.VALUE, state):
                    return
            else:
                if not self.try_resolve_name(arg, NameGroup.VALUE, state):
                    return

    def visit_AstIf_AstElif(self, node: Union[AstIf, AstElif], state: CompileState):
        if not self.try_resolve_name(node.condition, NameGroup.VALUE, state):
            return

    def visit_AstBinaryOp(self, node: AstBinaryOp, state: CompileState):
        # lhs/rhs side of stack op, if they are refs, must be refs to "runtime vals"
        if not self.try_resolve_name(node.lhs, NameGroup.VALUE, state):
            return
        if not self.try_resolve_name(node.rhs, NameGroup.VALUE, state):
            return

    def visit_AstUnaryOp(self, node: AstUnaryOp, state: CompileState):
        if not self.try_resolve_name(node.val, NameGroup.VALUE, state):
            return

    def visit_AstFor(self, node: AstFor, state: CompileState):
        if not self.try_resolve_name(node.loop_var, NameGroup.VALUE, state):
            return

        # this really shouldn't be possible to be a var right now
        # but this is future proof
        if not self.try_resolve_name(node.range, NameGroup.VALUE, state):
            return

    def visit_AstWhile(self, node: AstWhile, state: CompileState):
        if not self.try_resolve_name(node.condition, NameGroup.VALUE, state):
            return

    def visit_AstAssert(self, node: AstAssert, state: CompileState):
        if not self.try_resolve_name(node.condition, NameGroup.VALUE, state):
            return
        if node.exit_code is not None:
            if not self.try_resolve_name(node.exit_code, NameGroup.VALUE, state):
                return

    def visit_AstIndexExpr(self, node: AstIndexExpr, state: CompileState):
        if not self.try_resolve_name(node.parent, NameGroup.VALUE, state):
            return
        if not self.try_resolve_name(node.item, NameGroup.VALUE, state):
            return

    def visit_AstRange(self, node: AstRange, state: CompileState):
        if not self.try_resolve_name(node.lower_bound, NameGroup.VALUE, state):
            return
        if not self.try_resolve_name(node.upper_bound, NameGroup.VALUE, state):
            return

    def visit_AstReturn(self, node: AstReturn, state: CompileState):
        if node.value is not None:
            if not self.try_resolve_name(node.value, NameGroup.VALUE, state):
                return

    def visit_AstLiteral_AstGetAttr(
        self, node: Union[AstLiteral, AstGetAttr], state: CompileState
    ):
        # this is because they do not imply anything about the context in which an AstIdent should get
        # don't need to do anything for literals or getattr, but just have this here for completion's sake
        # resolved
        pass

    def visit_AstAnonStruct(self, node: AstAnonStruct, state: CompileState):
        for _name, value_expr in node.members:
            if not self.try_resolve_name(value_expr, NameGroup.VALUE, state):
                return

    def visit_AstAnonArray(self, node: AstAnonArray, state: CompileState):
        for elem_expr in node.elements:
            if not self.try_resolve_name(elem_expr, NameGroup.VALUE, state):
                return

    def visit_AstIdent(self, node: AstIdent, state: CompileState):
        if node in state.resolved_symbols:
            # it exists in a context where we can resolve it
            return

        # exists outside of a context where we can resolve it.
        # probably just throw an error?
        state.err(f"Name '{node.name}' cannot be resolved without more context", node)

    def visit_default(self, node, state):
        # coding error, missed an expr
        assert not is_instance_compat(node, AstStmtWithExpr), node


def is_type_constant_size(type: FpyType) -> bool:
    """Return true if the type has a statically known size.

    Types with strings (directly or nested) don't have constant size because
    strings can vary in length.
    """
    if type.kind in (TypeKind.STRING, TypeKind.INTERNAL_STRING):
        return False

    if type.kind == TypeKind.ARRAY:
        return is_type_constant_size(type.elem_type)

    if type.kind == TypeKind.STRUCT:
        for m in type.members:
            if not is_type_constant_size(m.type):
                return False
        return True

    return True


class UpdateTypesAndFuncs(Visitor):

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # Get the function that was created in CreateVariablesAndFuncs
        func = state.resolved_symbols[node.name]
        assert is_instance_compat(func, FunctionSymbol), func

        # Resolve return type
        if node.return_type is None:
            func.return_type = NOTHING
        else:
            return_type = state.resolved_symbols[node.return_type]
            if not is_type_constant_size(return_type):
                state.err(
                    f"Type {return_type.display_name} is not constant-sized (contains strings)",
                    node.return_type,
                )
                return
            func.return_type = return_type

        # Resolve parameter types
        args = []
        if node.parameters is not None:
            for arg_name_var, arg_type_name, default_value in node.parameters:
                arg_type = state.resolved_symbols[arg_type_name]
                if not is_type_constant_size(arg_type):
                    state.err(
                        f"Type {arg_type.display_name} is not constant-sized (contains strings)",
                        arg_type_name,
                    )
                    return
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

        if not is_type_constant_size(var_type):
            state.err(
                f"Type {var_type.display_name} is not constant-sized (contains strings)",
                node.type_ann,
            )
            return

        var = state.resolved_symbols[node.lhs]

        var.type = var_type


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

        if is_instance_compat(sym.declaration, AstDef):
            # function parameters - no use-before-define check needed
            # this is because if it's in scope, it's defined, as its
            # "declaration" is the start of the scope
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
            if sym.is_global and state.enclosing_value_scope[node].in_function:
                return
            state.err(f"'{node.name}' used before defined", node)
            return


class PickTypesAndResolveFields(Visitor):

    def can_coerce_type(self, source: FpyType, target: FpyType) -> bool:
        """Returns True if source can be implicitly coerced to target.

        Coercion is allowed when the common type of source and target IS target,
        meaning target can already represent everything source can.
        Anonymous types are optimistically coercible to their matching concrete
        kind; actual member/element validation happens in _coerce_anon_*.
        """
        if source.kind == TypeKind.ANON_STRUCT and target.kind == TypeKind.STRUCT:
            return True
        if source.kind == TypeKind.ANON_ARRAY and target.kind == TypeKind.ARRAY:
            return True
        return self.find_common_type(source, target) == target

    def coerce_expr_type(
        self, node: AstExpr, type: FpyType, state: CompileState
    ) -> bool:
        unconverted_type = state.synthesized_types[node]
        # make sure it isn't already being coerced
        assert unconverted_type == state.contextual_types[node], (
            unconverted_type,
            state.contextual_types[node],
        )

        # Special handling for anonymous struct coercion
        if unconverted_type.kind == TypeKind.ANON_STRUCT:
            return self._coerce_anon_struct(node, type, state)

        # Special handling for anonymous array coercion
        if unconverted_type.kind == TypeKind.ANON_ARRAY:
            return self._coerce_anon_array(node, type, state)

        if self.can_coerce_type(unconverted_type, type):
            state.contextual_types[node] = type
            return True
        state.err(
            f"Expected {type.display_name}, found {unconverted_type.display_name}", node
        )
        return False

    def _coerce_anon_struct(
        self, node: AstAnonStruct, target: FpyType, state: CompileState
    ) -> bool:
        """Coerce an anonymous struct expression to a concrete struct type."""
        if target.kind != TypeKind.STRUCT:
            state.err(
                f"Expected {target.display_name}, found anonymous struct", node
            )
            return False

        if not is_type_constant_size(target):
            state.err(
                f"Type {target.display_name} is not constant-sized (contains strings)",
                node,
            )
            return False

        # Build a map of provided member names to their value expressions
        provided_members: dict[str, AstExpr] = {}
        for name, value_expr in node.members:
            if name in provided_members:
                state.err(f"Duplicate member '{name}' in anonymous struct", node)
                return False
            provided_members[name] = value_expr

        # Check that all provided members exist in the target type
        target_member_names = {m.name for m in target.members}
        for name in provided_members:
            if name not in target_member_names:
                state.err(
                    f"Member '{name}' does not exist in {target.display_name}", node
                )
                return False

        # Look up the TypeCtorSymbol for defaults
        ctor = state.global_callable_scope.lookup_qualified(target.name)
        ctor_defaults: dict[str, object] = {}
        if is_instance_compat(ctor, TypeCtorSymbol):
            for arg_name, _, default_val in ctor.args:
                if default_val is not None:
                    ctor_defaults[arg_name] = default_val

        # Build the resolved member list in target struct order
        resolved_members = []
        for member in target.members:
            if member.name in provided_members:
                value_expr = provided_members[member.name]
                # Coerce the member value to the target member type
                if not self.coerce_expr_type(value_expr, member.type, state):
                    return False
                resolved_members.append(value_expr)
            elif member.name in ctor_defaults:
                resolved_members.append(ctor_defaults[member.name])
            else:
                state.err(
                    f"Missing member '{member.name}' in anonymous struct "
                    f"(no default available for {target.display_name}.{member.name})",
                    node,
                )
                return False

        state.resolved_func_args[node] = resolved_members
        state.contextual_types[node] = target
        return True

    def _coerce_anon_array(
        self, node: AstAnonArray, target: FpyType, state: CompileState
    ) -> bool:
        """Coerce an anonymous array expression to a concrete array type."""
        if target.kind != TypeKind.ARRAY:
            state.err(
                f"Expected {target.display_name}, found anonymous array", node
            )
            return False

        if not is_type_constant_size(target):
            state.err(
                f"Type {target.display_name} is not constant-sized (contains strings)",
                node,
            )
            return False

        if len(node.elements) > target.length:
            state.err(
                f"Anonymous array has {len(node.elements)} elements, "
                f"but {target.display_name} expects {target.length}",
                node,
            )
            return False

        # Coerce each provided element to the target element type
        for elem_expr in node.elements:
            if not self.coerce_expr_type(elem_expr, target.elem_type, state):
                return False

        # Fill remaining positions with defaults from the TypeCtorSymbol
        resolved = list(node.elements)
        if len(node.elements) < target.length:
            ctor = state.global_callable_scope.lookup_qualified(target.name)
            ctor_defaults: list[object] = []
            if is_instance_compat(ctor, TypeCtorSymbol):
                ctor_defaults = [default_val for _, _, default_val in ctor.args]

            for i in range(len(node.elements), target.length):
                if i < len(ctor_defaults) and ctor_defaults[i] is not None:
                    resolved.append(ctor_defaults[i])
                else:
                    state.err(
                        f"Anonymous array has {len(node.elements)} elements, "
                        f"but {target.display_name} expects {target.length} "
                        f"(no default available for element {i})",
                        node,
                    )
                    return False

        state.resolved_func_args[node] = resolved
        state.contextual_types[node] = target
        return True

    def find_common_type(self, first_type: FpyType, second_type: FpyType) -> FpyType | None:

        # important principles to reduce surprise:

        # type of an operation should be decided by the types of its inputs. let's not do
        # anything clever with trying to inspect the values of consts

        # no common type between signed and unsigned int

        # TODO unit test that this "works either way"
        if first_type == second_type:
            # no coercion necessary
            return second_type

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

            if parent_type.kind != TypeKind.STRUCT:
                state.err(
                    f"{parent_type.display_name} is not a struct, cannot access members",
                    node,
                )
                return

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

        if not is_type_constant_size(parent_type):
            state.err(
                f"{parent_type.display_name} is not constant-sized (contains strings), cannot access items",
                node,
            )
            return

        if parent_type.kind != TypeKind.ARRAY:
            state.err(f"{parent_type.display_name} is not an array", node)
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
        common = self.find_common_type(*arg_types) if len(arg_types) == 2 else arg_types[0]
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

    def visit_AstBinaryOp(self, node: AstBinaryOp, state: CompileState):
        lhs_type = state.synthesized_types[node.lhs]
        rhs_type = state.synthesized_types[node.rhs]
        arg_types = [lhs_type, rhs_type]

        # Check for time/interval operator overloads
        time_op = TIME_OPS.get((lhs_type, rhs_type, node.op))
        if time_op is not None:
            common_type, result_type, _, _ = time_op
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
        anon_type = FpyType(
            TypeKind.ANON_ARRAY,
            f"$AnonArray[{len(elem_types)}]",
            length=len(elem_types),
        )
        state.synthesized_types[node] = anon_type
        state.contextual_types[node] = anon_type

    def build_resolved_call_args(
        self,
        node: AstFuncCall,
        func: CallableSymbol,
        node_args: list,
    ) -> list[AstExpr] | CompileError:
        """Build a complete list of argument expressions for a function call.

        This function:
        1. Reorders named arguments to positional order
        2. Fills in default values for missing optional arguments
        3. Checks for missing required arguments

        Returns a list of argument expressions in positional order.
        Returns a CompileError if there's an issue with the arguments.
        """
        func_args = func.args

        # Build a map of parameter name to index
        param_name_to_idx = {arg[0]: i for i, arg in enumerate(func_args)}

        # Track which arguments have been assigned
        assigned_args: list[AstExpr | None] = [None] * len(func_args)
        seen_named = False
        positional_count = 0

        for arg in node_args:
            if is_instance_compat(arg, AstNamedArgument):
                seen_named = True
                # Check if the name is valid
                if arg.name not in param_name_to_idx:
                    return CompileError(
                        f"Unknown argument name '{arg.name}'",
                        arg,
                    )
                idx = param_name_to_idx[arg.name]
                # Check if the argument was already assigned
                if assigned_args[idx] is not None:
                    return CompileError(
                        f"Argument '{arg.name}' specified multiple times",
                        arg,
                    )
                assigned_args[idx] = arg.value
            else:
                # Positional argument
                if seen_named:
                    return CompileError(
                        "Positional argument cannot follow named argument",
                        arg,
                    )
                if positional_count >= len(func_args):
                    return CompileError(
                        f"Too many arguments (expected at most {len(func_args)})",
                        node,
                    )
                # Check if already assigned (shouldn't happen for positional-only case)
                if assigned_args[positional_count] is not None:
                    # This would happen if named arg came before positional
                    return CompileError(
                        f"Argument '{func_args[positional_count][0]}' specified multiple times",
                        arg,
                    )
                assigned_args[positional_count] = arg
                positional_count += 1

        # Fill in default values for missing arguments, error on missing required args
        for i, arg_expr in enumerate(assigned_args):
            if arg_expr is None:
                default_value = func_args[i][2]
                if default_value is not None:
                    assigned_args[i] = default_value
                else:
                    return CompileError(
                        f"Missing required argument '{func_args[i][0]}'",
                        node,
                    )

        return assigned_args

    def check_arg_types_compatible_with_func(
        self,
        node: AstFuncCall,
        func: CallableSymbol,
        resolved_args: list[AstExpr],
        state: CompileState,
    ) -> CompileError | None:
        """Check if a function call's arguments have compatible types.

        Given args must be coercible to expected args, with a special case for casting
        where any numeric type is accepted.
        resolved_args must be in positional order with all values present (defaults filled in).
        Returns a compile error if types don't match, otherwise None.
        """
        func_args = func.args

        if is_instance_compat(func, CastSymbol):
            # casts do not follow coercion rules, because casting is the counterpart of coercion!
            # coercion is implicit, casting is explicit. if they say they want to cast, we let them
            node_arg = resolved_args[0]
            input_type = state.synthesized_types[node_arg]
            output_type = func.to_type
            # right now we only have casting to numbers
            assert output_type in SPECIFIC_NUMERIC_TYPES
            if not input_type.is_numerical:
                # cannot convert a non-numeric type to a numeric type
                return CompileError(
                    f"Expected a number, found {input_type.display_name}", node_arg
                )
            # no error! looks good to me
            return

        # Check provided args against expected
        for value_expr, arg in zip(resolved_args, func_args):
            arg_type = arg[1]

            # Skip type check for default values that are FpyValue instances
            # this can happen if the value is hardcoded from a builtin func
            # or from dictionary defaults for type constructors
            if not is_instance_compat(value_expr, Ast):
                assert is_instance_compat(func, (BuiltinFuncSymbol, TypeCtorSymbol)), func
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
        # all args r good
        return

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

        # Build resolved args: reorder named args, fill in defaults, check for missing required
        resolved_args = self.build_resolved_call_args(node, func, node_args)
        if is_instance_compat(resolved_args, CompileError):
            state.errors.append(resolved_args)
            return

        # Store the resolved args for use in desugaring and codegen
        state.resolved_func_args[node] = resolved_args

        error_or_none = self.check_arg_types_compatible_with_func(
            node, func, resolved_args, state
        )
        if is_instance_compat(error_or_none, CompileError):
            state.errors.append(error_or_none)
            return
        # otherwise, no error, we're good!

        # okay, we've made sure that the func is possible
        # to call with these args

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
                # Skip coercion for FpyValue defaults from builtins or type constructors
                if not is_instance_compat(value_expr, Ast):
                    assert is_instance_compat(func, (BuiltinFuncSymbol, TypeCtorSymbol)), func
                    continue
                # Skip coercion for default values from forward-called functions.
                # These will be coerced when the function definition is visited.
                if value_expr not in state.synthesized_types:
                    continue
                # Skip default values already coerced by visit_AstDef.
                # When a function's default value AST node is reused in resolved_args,
                # it may already have been coerced during the function definition visit.
                if state.contextual_types[value_expr] != state.synthesized_types[value_expr]:
                    continue
                arg_type = arg[1]
                if not self.coerce_expr_type(value_expr, arg_type, state):
                    return

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
            if not self.coerce_expr_type(node.exit_code, U8, state):
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
        if not is_instance_compat(func, FunctionSymbol):
            return

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
        time_str: str, time_base: int, time_context: int, node: Ast, state: CompileState
    ) -> FpyValue | None:
        """Parse an ISO 8601 timestamp string into an FpyValue(TIME, ...).

        Accepts formats like:
        - "2025-12-19T14:30:00Z"
        - "2025-12-19T14:30:00.123456Z"

        Returns FpyValue(TIME, ...) with the provided time_base and time_context, and the parsed
        seconds/microseconds since Unix epoch.
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
                    f"Time string '{time_str}' results in negative seconds ({seconds}), "
                    "which cannot be represented in Fw.Time",
                    node,
                )
                return None
            if seconds > 0xFFFFFFFF:
                state.err(
                    f"Time string '{time_str}' results in seconds ({seconds}) exceeding U32 max",
                    node,
                )
                return None

            return FpyValue(TIME, {
                "time_base": FpyValue(U16, time_base),
                "time_context": FpyValue(U8, time_context),
                "seconds": FpyValue(U32, seconds),
                "useconds": FpyValue(U32, useconds),
            })

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
                # no compile time constant value for our parent here
                state.const_expr_values[node] = None
                return

            # we are accessing an attribute of something with an fprime value at compile time
            # we must be getting a member
            if isinstance(parent_value, FpyValue) and parent_value.type.kind == TypeKind.STRUCT:
                expr_value = parent_value.val[node.attr]
            else:
                assert False, parent_value

        assert expr_value is not None

        assert isinstance(expr_value, FpyValue) and expr_value.type == unconverted_type, (
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
            # no compile time constant value for our parent here
            state.const_expr_values[node] = None
            return

        assert isinstance(parent_value, FpyValue) and parent_value.type.kind == TypeKind.ARRAY, parent_value

        idx = state.const_expr_values.get(node.item)
        if idx is None:
            # no compile time constant value for our index
            state.const_expr_values[node] = None
            return

        assert isinstance(idx, FpyValue)

        expr_value = parent_value.val[idx.val]

        unconverted_type = state.synthesized_types[node]
        assert isinstance(expr_value, FpyValue) and expr_value.type == unconverted_type, (
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

        assert isinstance(expr_value, FpyValue) and expr_value.type == unconverted_type, (
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
        resolved_args = state.resolved_func_args[node]

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
                    state.errors.append(CompileError(
                        f"Argument '{func.args[i][0]}' of '{func.name}' must be a compile-time constant",
                        resolved_args[i],
                    ))
                    return

        if unknown_value:
            # we will have to calculate this at runtime
            state.const_expr_values[node] = None
            return

        expr_value = None

        # whether the conversion that will happen is due to an explicit cast
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
            time_base = arg_values[1].val
            time_context = arg_values[2].val
            expr_value = self._parse_time_string(
                timestamp_str, time_base, time_context, node, state
            )
            if expr_value is None:
                return
        else:
            # don't try to calculate the value of this function call
            # it's something like a user defined func, cmd or builtin
            state.const_expr_values[node] = None
            return

        unconverted_type = state.synthesized_types[node]
        assert isinstance(expr_value, FpyValue) and expr_value.type == unconverted_type, (
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

        # Both sides are constants, evaluate the operation if the operator is supported

        if lhs_value.type == TIME:
            # Time values don't have simple .val primitives; use as-is
            assert rhs_value.type == TIME, (lhs_value, rhs_value)
        else:
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
                folded_value = lhs_value // rhs_value
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
            state.err("Divide by zero error", node)
            return
        except OverflowError:
            state.err("Overflow error", node)
            return
        except ValueError as err:
            state.err(str(err) if str(err) else "Domain error", node)
            return
        except decimal.InvalidOperation:
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
            folded_value = -value
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
        assert converted_type.kind == TypeKind.STRUCT, converted_type

        resolved_members = state.resolved_func_args[node]

        # Gather member values
        member_values = []
        for member_expr in resolved_members:
            if is_instance_compat(member_expr, Ast):
                val = state.const_expr_values.get(member_expr)
                if val is None:
                    state.const_expr_values[node] = None
                    return
                member_values.append(val)
            else:
                # It's an FpyValue default
                member_values.append(member_expr)

        # Build the struct value
        arg_dict = {m.name: v for m, v in zip(converted_type.members, member_values)}
        expr_value = FpyValue(converted_type, arg_dict)
        state.const_expr_values[node] = expr_value

    def visit_AstAnonArray(self, node: AstAnonArray, state: CompileState):
        converted_type = state.contextual_types[node]
        assert converted_type.kind == TypeKind.ARRAY, converted_type

        resolved_elements = state.resolved_func_args[node]

        # Gather element values
        elem_values = []
        for elem_expr in resolved_elements:
            if is_instance_compat(elem_expr, Ast):
                val = state.const_expr_values.get(elem_expr)
                if val is None:
                    state.const_expr_values[node] = None
                    return
                elem_values.append(val)
            else:
                # It's an FpyValue default
                elem_values.append(elem_expr)

        expr_value = FpyValue(converted_type, elem_values)
        state.const_expr_values[node] = expr_value

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
        if idx_value is None:
            # can't check at compile time
            return

        parent_type = state.contextual_types[node.parent]
        assert parent_type.kind == TypeKind.ARRAY, parent_type

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
            state.warn("Range is empty", node)
