from __future__ import annotations
from decimal import Decimal
import decimal
import struct
from numbers import Number
from typing import Union

from fpy.error import CompileError
from fpy.types import (
    ARBITRARY_PRECISION_TYPES,
    SIGNED_INTEGER_TYPES,
    SPECIFIC_NUMERIC_TYPES,
    UNSIGNED_INTEGER_TYPES,
    CompileState,
    FieldSymbol,
    ForLoopAnalysis,
    FppType,
    CallableSymbol,
    CastSymbol,
    FpyFloatValue,
    FunctionSymbol,
    Symbol,
    SymbolTable,
    TypeCtorSymbol,
    VariableSymbol,
    FpyIntegerValue,
    FpyStringValue,
    NothingValue,
    RangeValue,
    TopDownVisitor,
    Visitor,
    is_instance_compat,
    lookup_symbol,
    typename,
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
    NUMERIC_OPERATORS,
    ArrayIndexType,
    LoopVarType,
    BinaryStackOp,
    UnaryStackOp,
)
from fprime_gds.common.templates.ch_template import ChTemplate
from fprime_gds.common.templates.prm_template import PrmTemplate
from fprime_gds.common.models.serialize.time_type import TimeType as TimeValue
from fprime_gds.common.models.serialize.type_base import ValueType
from fprime_gds.common.models.serialize.serializable_type import (
    SerializableType as StructValue,
)
from fprime_gds.common.models.serialize.array_type import ArrayType as ArrayValue
from fprime_gds.common.models.serialize.type_exceptions import TypeException
from fprime_gds.common.models.serialize.numerical_types import (
    U8Type as U8Value,
    U16Type as U16Value,
    U32Type as U32Value,
    U64Type as U64Value,
    I64Type as I64Value,
    F64Type as F64Value,
    FloatType as FloatValue,
    IntegerType as IntegerValue,
    NumericalType as NumericalValue,
)
from fprime_gds.common.models.serialize.string_type import StringType as StringValue
from fprime_gds.common.models.serialize.bool_type import BoolType as BoolValue
from fpy.syntax import (
    AstAssert,
    AstBinaryOp,
    AstStmtList,
    AstBoolean,
    AstBreak,
    AstContinue,
    AstDef,
    AstElif,
    AstExpr,
    AstFor,
    AstMemberAccess,
    AstIndexExpr,
    AstTypeExpr,
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
    AstVar,
    AstWhile,
)
from fprime_gds.common.models.serialize.type_base import BaseType as FppValue


class AssignIds(TopDownVisitor):
    """assigns a unique id to each node to allow it to be indexed in a dict"""

    def visit_default(self, node, state: CompileState):
        node.id = state.next_node_id
        state.next_node_id += 1


class SetLocalScope(Visitor):
    def __init__(self, scope: SymbolTable):
        super().__init__()
        self.scope = scope

    def visit_default(self, node: Ast, state: CompileState):
        state.local_scopes[node] = self.scope


class AssignLocalScopes(TopDownVisitor):

    def visit_AstBlock(self, node: AstBlock, state: CompileState):
        if node is not state.root:
            # only handle the root node this way
            return
        # make a new scope
        scope = SymbolTable()
        state.scope_parents[scope] = None
        # TODO ask rob there must be a better way to do this, that isn't as slow
        SetLocalScope(scope).run(node, state)

    def visit_AstDef(self, node: AstDef, state: CompileState):
        parent_scope = state.local_scopes[node]
        # make a new scope for the function body
        scope = SymbolTable()
        state.scope_parents[scope] = parent_scope

        # The function body gets the new scope
        SetLocalScope(scope).run(node.body, state)

        # Parameter names and type annotations are in the body scope
        # (they're declared inside the function)
        if node.parameters is not None:
            for arg_name_var, arg_type_expr, default_value in node.parameters:
                state.local_scopes[arg_name_var] = scope
                state.local_scopes[arg_type_expr] = scope
                # Default values stay in parent scope - they're evaluated at call site,
                # not inside the function body
                if default_value is not None:
                    SetLocalScope(parent_scope).run(default_value, state)

        # Return type annotation is in parent scope (it references types visible at def site)
        if node.return_type is not None:
            state.local_scopes[node.return_type] = parent_scope

        # The def node itself is in the parent scope
        state.local_scopes[node] = parent_scope
        # Function name is in parent scope
        state.local_scopes[node.name] = parent_scope


class CreateVariablesAndFuncs(TopDownVisitor):
    """finds all variable declarations and adds them to the variable scope"""

    def visit_AstAssign(self, node: AstAssign, state: CompileState):
        if not is_instance_compat(node.lhs, AstReference):
            # trying to assign a value to some complex expression like (1 + 1) = 2
            state.err("Invalid assignment", node.lhs)
            return

        if not is_instance_compat(node.lhs, AstVar):
            # assigning to a member or array element. don't need to make a new variable,
            # space already exists
            if node.type_ann is not None:
                # type annotation on a field assignment... it already has a type!
                state.err("Cannot specify a type annotation for a field", node.type_ann)
                return
            # otherwise we good
            return

        if node.type_ann is not None:
            # new variable declaration
            # make sure it isn't defined in this scope
            # TODO shadowing check
            existing_local = state.local_scopes[node].get(node.lhs.var)
            if existing_local is not None:
                # redeclaring an existing variable
                state.err(f"'{node.lhs.var}' has already been declared", node)
                return
            # okay, declare the var
            # Check if we're in the root (global) scope
            current_scope = state.local_scopes[node]
            is_global = state.scope_parents.get(current_scope) is None
            var = VariableSymbol(node.lhs.var, node.type_ann, node, is_global=is_global)
            # new var. put it in the table under this scope
            state.local_scopes[node][node.lhs.var] = var
        else:
            # otherwise, it's a reference to an existing var
            sym = lookup_symbol(node, node.lhs.var, state)
            if sym is None:
                # unable to find this symbol
                state.err(
                    f"'{node.lhs.var}' has not been declared",
                    node.lhs,
                )
                return
            # okay, we were able to resolve it

    def visit_AstFor(self, node: AstFor, state: CompileState):
        # for loops have an implicit loop variable that they can declare
        # if it isn't already declared in the local scope
        loop_var = state.local_scopes[node].get(node.loop_var.var)

        reuse_existing_loop_var = False
        if loop_var is not None:
            # this is okay as long as the variable is of the same type

            # what follows is a bit of a hack
            # there are two cases: either loop_var has been declared before but we only know the type expr (if it was an AstAssign decl)
            # or loop_var has been declared before and we only know the type, but have no type expr

            # case 1 is easy, just check the type == LoopVarType
            # case 2 is harder, we have to check if the type expr is an AstTypeExpr with a single part
            # that matches the canonical name of the LoopVarType

            # the alternative to this is that we do some primitive type resolution in the same pass as variable creation
            # i'm doing this hack because we're going to switch to type inference for variables later and that will make this go away

            if (loop_var.type_ref is None and loop_var.type != LoopVarType) or (
                loop_var.type is None
                and not (
                    isinstance(loop_var.type_ref, AstTypeExpr)
                    and loop_var.type_ref.parts == [LoopVarType.get_canonical_name()]
                )
            ):
                state.err(
                    f"'{node.loop_var.var}' has already been declared as a type other than {typename(LoopVarType)}",
                    node,
                )
                return
            reuse_existing_loop_var = True
        else:
            # new var. put it in the table under this scope
            loop_var = VariableSymbol(node.loop_var.var, None, node, LoopVarType)
            state.local_scopes[node][loop_var.name] = loop_var

        # each loop also declares an implicit ub variable
        # type of ub var is same as loop var type
        upper_bound_var = VariableSymbol(
            state.new_anonymous_variable_name(), None, node, LoopVarType
        )
        state.local_scopes[node][upper_bound_var.name] = upper_bound_var
        analysis = ForLoopAnalysis(loop_var, upper_bound_var, reuse_existing_loop_var)
        state.for_loops[node] = analysis

    def visit_AstDef(self, node: AstDef, state: CompileState):
        existing_func = state.local_scopes[node].get(node.name.var)
        if existing_func is not None:
            state.err(f"'{node.name.var}' has already been declared", node.name)
            return

        func = FunctionSymbol(
            # we know the name
            node.name.var,
            # we don't know the return type yet
            return_type=None,
            # we don't know the arg types yet
            args=None,
            definition=node,
        )

        state.local_scopes[node][func.name] = func

        if node.parameters is None:
            # no arguments
            return

        # Check that default arguments come after non-default arguments
        seen_default = False
        for arg in node.parameters:
            arg_name_var, arg_type_expr, default_value = arg
            if default_value is not None:
                seen_default = True
            elif seen_default:
                # Non-default argument after default argument
                state.err(
                    f"Non-default argument '{arg_name_var.var}' follows default argument",
                    arg_name_var,
                )
                return

        for arg in node.parameters:
            arg_name_var, arg_type_expr, default_value = arg
            existing_local = state.local_scopes[node.body].get(arg_name_var.var)
            if existing_local is not None:
                # redeclaring an existing variable
                state.err(
                    f"'{arg_name_var.var}' has already been declared", arg_name_var
                )
                return
            arg_var = VariableSymbol(arg_name_var.var, arg_type_expr, node)
            state.local_scopes[node.body][arg_name_var.var] = arg_var


class SetEnclosingLoops(Visitor):
    """sets or clears the enclosing_loop dict of any break/continue it finds"""

    def __init__(self, loop: Union[AstFor, AstWhile]):
        """if loop is None, remove the visited break/continue from enclosing_loop dict"""
        super().__init__()
        self.loop = loop

    def visit_AstBreak_AstContinue(
        self, node: Union[AstBreak, AstContinue], state: CompileState
    ):
        if self.loop is None:
            del state.enclosing_loops[node]
        else:
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

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # going inside of a func def "resets" our loop context. this prevents the following scenario:
        # for x in 0..2:
        #     def test():
        #         break

        # in this case, the break should fail to compile
        SetEnclosingLoops(None).run(node.body, state)


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


class ResolveTypeNames(TopDownVisitor):
    """
    Resolves type annotations (AstTypeExpr) to actual types.
    This runs before ResolveVars so that variable types are known
    when we start resolving references.
    """

    def resolve_type_name(
        self, node: AstTypeExpr, state: CompileState
    ) -> type | None:
        """
        Fully resolves a type name to the actual type.
        Returns None if the type could not be resolved (error already reported).
        """
        # Start from the first part
        sym = state.types.get(node.parts[0])
        if sym is None:
            state.err("Unknown type", node)
            return None

        # Walk through the remaining parts
        for part in node.parts[1:]:
            if not is_instance_compat(sym, dict):
                state.err("Unknown type", node)
                return None
            sym = sym.get(part)
            if sym is None:
                state.err("Unknown type", node)
                return None

        if not is_instance_compat(sym, type):
            state.err("Unknown type", node)
            return None

        state.resolved_symbols[node] = sym
        return sym

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # Get the function that was created in CreateVariablesAndFuncs
        func = state.local_scopes[node].get(node.name.var)
        assert is_instance_compat(func, FunctionSymbol), func

        # Resolve return type
        if node.return_type is not None:
            return_type = self.resolve_type_name(node.return_type, state)
            if return_type is None:
                return
            func.return_type = return_type
        else:
            func.return_type = NothingValue

        # Resolve parameter types
        args = []
        if node.parameters is not None:
            for arg_name_var, arg_type_expr, default_value in node.parameters:
                arg_type = self.resolve_type_name(arg_type_expr, state)
                if arg_type is None:
                    return

                # Get the variable that was created for this parameter
                arg_var = state.local_scopes[node.body].get(arg_name_var.var)
                assert is_instance_compat(arg_var, VariableSymbol), arg_var
                arg_var.type = arg_type
                args.append((arg_name_var.var, arg_type, default_value))

        func.args = args

    def visit_AstAssign(self, node: AstAssign, state: CompileState):
        if node.type_ann is None:
            return

        var_type = self.resolve_type_name(node.type_ann, state)
        if var_type is None:
            return

        # Get the variable - it should be in the local scope
        if not is_instance_compat(node.lhs, AstVar):
            # Type annotations only make sense on simple variable assignments
            state.err("Type annotation can only be on simple variable assignment", node)
            return

        var = state.local_scopes[node].get(node.lhs.var)
        assert is_instance_compat(var, VariableSymbol), f"Variable {node.lhs.var} should have been created by CreateVariablesAndFuncs"

        var.type = var_type


class ResolveVars(TopDownVisitor):
    """
    Resolves all variable references (AstVar) in local and global scopes.
    Also fully resolves function references for function calls.
    Types are already resolved by ResolveTypeNames before this pass.
    """

    def resolve_local_name(
        self, node: AstVar, state: CompileState
    ) -> Symbol | None:
        """resolves a name in local scope only. return None if could not be resolved"""
        local_scope = state.local_scopes[node]
        sym = None
        while local_scope is not None and sym is None:
            sym = local_scope.get(node.var)
            local_scope = state.scope_parents[local_scope]

        if sym is not None:
            state.resolved_symbols[node] = sym
        return sym

    def try_resolve_root_ref(
        self,
        node: Ast,
        global_scope: SymbolTable,
        global_scope_name: str,
        state: CompileState,
        search_local_scope: bool = True,
    ) -> bool:
        """
        recursively tries to resolve the root of a reference. if any node in the chain is not a reference, return True.
        Otherwise recurse up the sym until you find the root, and try resolving
        it in the local scope, and then in the given global scope. Return False if couldn't be resolved in local and global
        """
        if not is_instance_compat(node, AstReference):
            # not a reference, nothing to resolve
            return True

        if not is_instance_compat(node, AstVar):
            # it is a reference but it's not a var
            # recurse until we find the var
            return self.try_resolve_root_ref(
                node.parent, global_scope, global_scope_name, state
            )

        # okay now we have a var
        # see if it's something defined in the script
        sym = None
        if search_local_scope:
            sym = self.resolve_local_name(node, state)

        if sym is None:
            # unable to find this symbol in the hierarchy of local scopes
            # look it up in the global scope
            sym = global_scope.get(node.var)

        if sym is None:
            state.err(f"Unknown {global_scope_name}", node)
            return False

        state.resolved_symbols[node] = sym
        return True

    def finish_resolving_func(
        self,
        node: Ast,
        state: CompileState,
    ) -> CallableSymbol | None:
        """
        Finishes resolving a function reference (AstVar/AstMemberAccess chain) to a callable.
        The root AstVar should already be resolved by try_resolve_root_ref.
        Returns None if the reference could not be resolved (error already reported).
        """

        def resolve(n: Ast, expect_callable: bool) -> CallableSymbol | dict | None:
            """
            Recursively resolve the reference chain.
            expect_callable=True for the final node (must be CallableSymbol),
            expect_callable=False for intermediate nodes (must be dict/namespace).
            """
            if not is_instance_compat(n, (AstVar, AstMemberAccess)):
                state.err("Unknown function", n)
                return None

            if is_instance_compat(n, AstVar):
                sym = state.resolved_symbols.get(n)
                if sym is None:
                    state.err("Unknown function", n)
                    return None
                expected_type = CallableSymbol if expect_callable else dict
                if not is_instance_compat(sym, expected_type):
                    state.err("Unknown function", n)
                    return None
                return sym

            # It's an AstMemberAccess - resolve the parent first (always expecting a namespace)
            parent_scope = resolve(n.parent, expect_callable=False)
            if parent_scope is None:
                return None

            sym = parent_scope.get(n.attr)
            if sym is None:
                state.err("Unknown function", n)
                return None

            expected_type = CallableSymbol if expect_callable else dict
            if not is_instance_compat(sym, expected_type):
                state.err("Unknown function", n)
                return None

            state.resolved_symbols[n] = sym
            return sym

        return resolve(node, expect_callable=True)

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        # First resolve the root of the function reference
        if not self.try_resolve_root_ref(node.func, state.callables, "function", state):
            return

        # Then finish resolving the full chain to get the callable
        if not self.finish_resolving_func(node.func, state):
            return

        for arg in node.args if node.args is not None else []:
            # Handle both positional args (AstExpr) and named args (AstNamedArgument)
            if is_instance_compat(arg, AstNamedArgument):
                # For named arguments, resolve the value expression
                if not self.try_resolve_root_ref(
                    arg.value, state.runtime_values, "value", state
                ):
                    return
            else:
                # arg value refs must have values at runtime
                if not self.try_resolve_root_ref(
                    arg, state.runtime_values, "value", state
                ):
                    return

    def visit_AstIf_AstElif(self, node: Union[AstIf, AstElif], state: CompileState):
        # if condition expr refs must be "runtime values" (tlm/prm/const/etc)
        if not self.try_resolve_root_ref(
            node.condition, state.runtime_values, "value", state
        ):
            return

    def visit_AstBinaryOp(self, node: AstBinaryOp, state: CompileState):
        # lhs/rhs side of stack op, if they are refs, must be refs to "runtime vals"
        if not self.try_resolve_root_ref(
            node.lhs, state.runtime_values, "value", state
        ):
            return
        if not self.try_resolve_root_ref(
            node.rhs, state.runtime_values, "value", state
        ):
            return

    def visit_AstUnaryOp(self, node: AstUnaryOp, state: CompileState):
        if not self.try_resolve_root_ref(
            node.val, state.runtime_values, "value", state
        ):
            return

    def visit_AstAssign(self, node: AstAssign, state: CompileState):
        if not self.try_resolve_root_ref(
            node.lhs, state.runtime_values, "value", state
        ):
            return

        # Type annotation is resolved by ResolveTypeNames pass

        if not self.try_resolve_root_ref(
            node.rhs, state.runtime_values, "value", state
        ):
            return

    def visit_AstFor(self, node: AstFor, state: CompileState):
        if not self.try_resolve_root_ref(
            node.loop_var, state.runtime_values, "value", state
        ):
            return

        # this really shouldn't be possible to be a var right now
        # but this is future proof
        if not self.try_resolve_root_ref(
            node.range, state.runtime_values, "value", state
        ):
            return

    def visit_AstWhile(self, node: AstWhile, state: CompileState):
        if not self.try_resolve_root_ref(
            node.condition, state.runtime_values, "value", state
        ):
            return

    def visit_AstAssert(self, node: AstAssert, state: CompileState):
        if not self.try_resolve_root_ref(
            node.condition, state.runtime_values, "value", state
        ):
            return
        if node.exit_code is not None:
            if not self.try_resolve_root_ref(
                node.exit_code, state.runtime_values, "value", state
            ):
                return

    def visit_AstVar(self, node: AstVar, state: CompileState):
        # if this var isn't resolved, that means it wasn't used in
        # any of the other places that it could. it must be alone on its line
        # this is a strange choice by the dev but rule of thumb says start
        # by allowing everything, then add warnings later
        if node in state.resolved_symbols:
            return

        if self.resolve_local_name(node, state) is None:
            state.err("Unknown symbol", node)
            return

    def visit_AstIndexExpr(self, node: AstIndexExpr, state: CompileState):
        if not self.try_resolve_root_ref(
            node.item, state.runtime_values, "value", state
        ):
            return

    def visit_AstRange(self, node: AstRange, state: CompileState):
        if not self.try_resolve_root_ref(
            node.lower_bound, state.runtime_values, "value", state
        ):
            return
        if not self.try_resolve_root_ref(
            node.upper_bound, state.runtime_values, "value", state
        ):
            return

    def visit_AstDef(self, node: AstDef, state: CompileState):
        if not self.try_resolve_root_ref(node.name, state.callables, "function", state):
            return

        # Return type and parameter types are resolved by ResolveTypeNames pass

        if node.parameters is not None:
            for arg_name_var, arg_type_expr, default_value in node.parameters:
                if not self.try_resolve_root_ref(
                    arg_name_var, state.runtime_values, "value", state
                ):
                    return

                # Type is resolved by ResolveTypeNames pass

                # Resolve default value if present
                if default_value is not None:
                    if not self.try_resolve_root_ref(
                        default_value, state.runtime_values, "value", state
                    ):
                        return

    def visit_AstReturn(self, node: AstReturn, state: CompileState):
        if not self.try_resolve_root_ref(
            node.value, state.runtime_values, "value", state
        ):
            return

    def visit_AstLiteral_AstMemberAccess(
        self, node: Union[AstLiteral, AstMemberAccess], state: CompileState
    ):
        # don't need to do anything for literals or getattr, but just have this here for completion's sake
        # the reason we don't need to do anything for getattr is because the point of this
        # pass is explicitly not to resolve getattr, just resolve vars (and types), not general
        # getattr
        # we don't do this now because we need to resolve all expr types first
        # before we can figure out whether a getattr is correct, and this pass lets
        # us now do that (i.e. after this pass, we have enough info about the program
        # that we can decide types of any expr)
        pass

    def visit_default(self, node, state):
        # coding error, missed an expr
        assert not is_instance_compat(node, AstStmtWithExpr), node


class CheckUseBeforeDeclare(TopDownVisitor):
    """
    Checks that variables are not used before they are declared.
    Handles both regular variable assignments (AstAssign) and for loop variables (AstFor).
    
    Uses TopDownVisitor because for loops need the loop variable to be declared
    before visiting the body. For assignments, we manually check the RHS before
    marking the variable as declared.
    """

    def __init__(self):
        super().__init__()
        self.currently_declared_vars: list[VariableSymbol] = []

    def visit_AstFor(self, node: AstFor, state: CompileState):
        var = state.resolved_symbols[node.loop_var]
        # Check that the loop var isn't referenced in the range (before it's declared)
        EnsureVariableNotReferenced(var).run(node.range, state)
        # Now mark it as declared for the body
        self.currently_declared_vars.append(var)

    def visit_AstAssign(self, node: AstAssign, state: CompileState):
        if not is_instance_compat(node.lhs, AstVar):
            # definitely not a declaration, it's a field assignment
            return

        var = state.resolved_symbols[node.lhs]

        if var is None or var.declaration != node:
            # either not declared in this scope, or this is not a
            # declaration of this var
            return

        # Before marking as declared, check that the variable isn't used in its own RHS
        EnsureVariableNotReferenced(var).run(node.rhs, state)

        # Now mark this variable as declared
        self.currently_declared_vars.append(var)

    def visit_AstVar(self, node: AstVar, state: CompileState):
        sym = state.resolved_symbols[node]
        if not is_instance_compat(sym, VariableSymbol):
            # not a variable, might be a type name or smth
            return

        if is_instance_compat(sym.declaration, AstDef):
            # function parameters - no use-before-declare check needed
            # this is because if it's in scope, it's declared, as its
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

        if sym not in self.currently_declared_vars:
            state.err(f"'{node.var}' used before declared", node)
            return


class EnsureVariableNotReferenced(Visitor):
    def __init__(self, var: VariableSymbol):
        super().__init__()
        self.var = var

    def visit_AstVar(self, node: AstVar, state: CompileState):
        sym = state.resolved_symbols[node]
        if sym == self.var:
            state.err(f"'{node.var}' used before declared", node)
            return


class PickTypesAndResolveAttrsAndItems(Visitor):

    def coerce_expr_type(
        self, node: AstExpr, type: FppType, state: CompileState
    ) -> bool:
        unconverted_type = state.synthesized_types[node]
        # make sure it isn't already being coerced
        assert unconverted_type == state.contextual_types[node], (
            unconverted_type,
            state.contextual_types[node],
        )
        if self.can_coerce_type(unconverted_type, type):
            state.contextual_types[node] = type
            return True
        state.err(
            f"Expected {typename(type)}, found {typename(unconverted_type)}", node
        )
        return False

    def can_coerce_type(self, from_type: FppType, to_type: FppType) -> bool:
        """return True if the type coercion rules allow from_type to be implicitly converted to to_type"""
        if from_type == to_type:
            # no coercion necessary
            return True
        if from_type == FpyStringValue and issubclass(to_type, StringValue):
            # we can convert the literal String type to any string type
            return True
        if not issubclass(from_type, NumericalValue) or not issubclass(
            to_type, NumericalValue
        ):
            # if one of the src or dest aren't numerical, we can't coerce
            return False

        # now we must answer:
        # are all values of from_type representable in the destination type?

        # if going from float to integer, definitely not
        if issubclass(from_type, FloatValue) and issubclass(to_type, IntegerValue):
            return False

        # in general: if either src or dest is one of our FpyXYZValue types, which are
        # arb precision, we allow this coercion.
        # it's easy to argue we should allow converting to arb precision. but why would
        # we allow arb precision to go to an 8 bit type, e.g.?
        # we have a big advantage: the arb precision types are only used for constants. that
        # means we actually know what the value is, so we can actually check!
        # however, we won't perform that check here. That will happen later in the
        # const_convert_type func in the CalcConstExprValues
        # for now, we will let the compilation proceed if either side is arb precision

        if (
            from_type in ARBITRARY_PRECISION_TYPES
            or to_type in ARBITRARY_PRECISION_TYPES
        ):
            return True

        # otherwise, both src and dest have finite bits

        # if we currently have a float
        if issubclass(from_type, FloatValue):
            # the dest must be a float and must be >= width
            return (
                issubclass(to_type, FloatValue)
                and to_type.get_bits() >= from_type.get_bits()
            )

        # otherwise must be an int
        assert issubclass(from_type, IntegerValue)
        # int to float is allowed in any case.
        # this is the big exception to our rule about full representation. this can cause loss of precision
        # for large integer values
        if issubclass(to_type, FloatValue):
            return True

        # the dest must be an int with the same signedness and >= width
        from_unsigned = from_type in UNSIGNED_INTEGER_TYPES
        to_unsigned = to_type in UNSIGNED_INTEGER_TYPES
        return (
            from_unsigned == to_unsigned and to_type.get_bits() >= from_type.get_bits()
        )

    def pick_intermediate_type(
        self, arg_types: list[FppType], op: BinaryStackOp | UnaryStackOp
    ) -> FppType:
        """return the intermediate type that all arguments should be converted to for the given operator"""

        if op in BOOLEAN_OPERATORS:
            return BoolValue

        non_numeric = any(not issubclass(t, NumericalValue) for t in arg_types)

        if (op == BinaryStackOp.EQUAL or op == BinaryStackOp.NOT_EQUAL) and non_numeric:
            # comparison of complex types (structs/strings/arrays/enum consts)
            if len(set(arg_types)) != 1:
                # can only compare equality between the same types
                return None
            return arg_types[0]

        # all other cases require that arguments are numeric
        if non_numeric:
            return None

        # we split this algo up into two stages: picking the type category (float, uint or int), and picking the type bitwidth

        # pick the type category:
        type_category = None
        if op == BinaryStackOp.DIVIDE or op == BinaryStackOp.EXPONENT:
            # always do true division and exponentiation over floats, python style
            # this is because, for the given op, even with integer inputs, we might get
            # float outputs
            type_category = "float"
            # TODO problem: this means that if we do I32 / F32, it doesn't work b/c we can't convert I32 to F32
        elif any(issubclass(t, FloatValue) for t in arg_types):
            # otherwise if any args are floats, use float
            type_category = "float"
        elif any(t in UNSIGNED_INTEGER_TYPES for t in arg_types):
            # otherwise if any args are unsigned, use unsigned
            type_category = "uint"
        else:
            # otherwise use signed int
            type_category = "int"

        # pick the bitwidth
        # we only use the arb precision types for constants, so if theyre all arb precision, they're consts
        constants = all(t in ARBITRARY_PRECISION_TYPES for t in arg_types)

        if constants:
            # we can constant fold this, so use infinite bitwidth
            if type_category == "float":
                return FpyFloatValue
            assert type_category == "int" or type_category == "uint"
            return FpyIntegerValue

        # can't const fold
        if type_category == "float":
            return F64Value
        if type_category == "uint":
            return U64Value
        assert type_category == "int"
        return I64Value

    def is_type_constant_size(self, type: FppType) -> bool:
        """return true if the type is statically sized"""
        if issubclass(type, StringValue):
            return False

        if issubclass(type, ArrayValue):
            return self.is_type_constant_size(type.MEMBER_TYPE)

        if issubclass(type, StructValue):
            for _, arg_type, _, _ in type.MEMBER_LIST:
                if not self.is_type_constant_size(arg_type):
                    return False
            return True

        return True

    def get_members(
        self, node: Ast, parent_type: FppType, state: CompileState
    ) -> list[tuple[str, FppType]] | None:
        if not issubclass(parent_type, (StructValue, TimeValue)):
            return {}

        if not self.is_type_constant_size(parent_type):
            state.err(
                f"{parent_type} has non-constant sized members, cannot access members",
                node,
            )
            return None

        member_list: list[tuple[str, FppType]] = None
        if issubclass(parent_type, StructValue):
            member_list = [t[0:2] for t in parent_type.MEMBER_LIST]
        else:
            # if it is a time type, there are some "implied" members
            member_list = []
            member_list.append(("time_base", U16Value))
            member_list.append(("time_context", U8Value))
            member_list.append(("seconds", U32Value))
            member_list.append(("useconds", U32Value))
        return member_list

    def get_sym_type(self, sym: Symbol) -> FppType:
        """returns the fprime type of the sym, if it were to be evaluated as an expression"""
        if isinstance(sym, ChTemplate):
            result_type = sym.ch_type_obj
        elif isinstance(sym, PrmTemplate):
            result_type = sym.prm_type_obj
        elif isinstance(sym, FppValue):
            # constant value
            result_type = type(sym)
        elif isinstance(sym, CallableSymbol):
            # a reference to a callable isn't a type in and of itself
            # it has a return type but you have to call it (with an AstFuncCall)
            # consider making a separate "reference" type
            result_type = NothingValue
        elif isinstance(sym, VariableSymbol):
            result_type = sym.type
        elif isinstance(sym, type):
            # a reference to a type doesn't have a value, and so doesn't have a type,
            # in and of itself. if this were a function call to the type's ctor then
            # it would have a value and thus a type
            result_type = NothingValue
        elif isinstance(sym, FieldSymbol):
            result_type = sym.type
        elif isinstance(sym, dict):
            # reference to a scope. scopes don't have values
            result_type = NothingValue
        else:
            assert False, sym

        return result_type

    def visit_AstMemberAccess(self, node: AstMemberAccess, state: CompileState):
        parent_sym = state.resolved_symbols.get(node.parent)

        if is_instance_compat(parent_sym, (type, CallableSymbol)):
            state.err("Unknown attribute", node)
            return

        sym = None
        if is_instance_compat(parent_sym, dict):
            # getattr of a namespace
            # parent won't actually have a type
            sym = parent_sym.get(node.attr)
            if sym is None:
                state.err("Unknown attribute", node)
                return
            # GetAttr should never resolve to a lexical variable; variables are accessed directly
            assert not is_instance_compat(
                sym, VariableSymbol
            ), "Field resolution unexpectedly found a local variable"
        else:
            # in all other cases, parent has at least some sort of type
            # sym may be None (if parent is some complex expr), or it may be
            # a tlm chan or var or etc...
            # it may or may not have a compile time value, but it definitely has a type
            parent_type = state.synthesized_types[node.parent]

            # field symbols store their "base symbol", which is the first non-field-symbol parent of
            # the field symbol. this lets you easily check what actual underlying thing (tlm chan, variable, prm)
            # you're talking about a field of
            base_sym = (
                parent_sym
                if not is_instance_compat(parent_sym, FieldSymbol)
                else parent_sym.base_sym
            )
            # we also calculate a "base offset" wrt. the start of the base_sym type, so you
            # can easily pick out this field from a value of the base sym type
            base_offset = (
                0
                if not is_instance_compat(parent_sym, FieldSymbol)
                else parent_sym.base_offset
            )

            member_list = self.get_members(node, parent_type, state)
            if member_list is None:
                return

            offset = 0
            for arg_name, arg_type in member_list:
                if arg_name == node.attr:
                    sym = FieldSymbol(
                        is_struct_member=True,
                        parent_expr=node.parent,
                        type=arg_type,
                        base_sym=base_sym,
                        local_offset=offset,
                        base_offset=base_offset,
                        name=arg_name,
                    )
                    break
                offset += arg_type.getMaxSize()
                base_offset += arg_type.getMaxSize()

        if sym is None:
            state.err(
                f"{typename(parent_type)} has no member named {node.attr}",
                node,
            )
            return

        sym_type = self.get_sym_type(sym)

        state.resolved_symbols[node] = sym
        state.synthesized_types[node] = sym_type
        state.contextual_types[node] = sym_type

    def visit_AstIndexExpr(self, node: AstIndexExpr, state: CompileState):
        parent_sym = state.resolved_symbols.get(node.parent)

        if is_instance_compat(parent_sym, (type, CallableSymbol, dict)):
            state.err("Unknown item", node)
            return

        # otherwise, we should definitely have a well-defined type for our parent expr

        parent_type = state.synthesized_types[node.parent]

        if not self.is_type_constant_size(parent_type):
            state.err(
                f"{typename(parent_type)} has non-constant sized members, cannot access items",
                node,
            )
            return

        if not issubclass(parent_type, ArrayValue):
            state.err(f"{typename(parent_type)} is not an array", node)
            return

        # coerce the index expression to array index type
        if not self.coerce_expr_type(node.item, ArrayIndexType, state):
            return

        base_sym = (
            parent_sym
            if not is_instance_compat(parent_sym, FieldSymbol)
            else parent_sym.base_sym
        )

        sym = FieldSymbol(
            is_array_element=True,
            parent_expr=node.parent,
            type=parent_type.MEMBER_TYPE,
            base_sym=base_sym,
            idx_expr=node.item,
        )

        state.resolved_symbols[node] = sym
        state.synthesized_types[node] = parent_type.MEMBER_TYPE
        state.contextual_types[node] = parent_type.MEMBER_TYPE

    def visit_AstVar(self, node: AstVar, state: CompileState):
        # already been resolved by SetScopes pass
        sym = state.resolved_symbols[node]
        if sym is None:
            return
        sym_type = self.get_sym_type(sym)

        state.synthesized_types[node] = sym_type
        state.contextual_types[node] = sym_type

    def visit_AstNumber(self, node: AstNumber, state: CompileState):
        # give a best guess as to the final type of this node. we don't actually know
        # its bitwidth or signedness yet
        if is_instance_compat(node.value, Decimal):
            result_type = FpyFloatValue
        else:
            result_type = FpyIntegerValue

        state.synthesized_types[node] = result_type
        state.contextual_types[node] = result_type

    def visit_AstBinaryOp(self, node: AstBinaryOp, state: CompileState):
        lhs_type = state.synthesized_types[node.lhs]
        rhs_type = state.synthesized_types[node.rhs]

        intermediate_type = self.pick_intermediate_type([lhs_type, rhs_type], node.op)
        if intermediate_type is None:
            state.err(
                f"Op {node.op} undefined for {typename(lhs_type)}, {typename(rhs_type)}",
                node,
            )
            return

        if not self.coerce_expr_type(node.lhs, intermediate_type, state):
            return
        if not self.coerce_expr_type(node.rhs, intermediate_type, state):
            return

        result_type = None
        if node.op in NUMERIC_OPERATORS:
            result_type = intermediate_type
        else:
            result_type = BoolValue

        state.op_intermediate_types[node] = intermediate_type
        state.synthesized_types[node] = result_type
        state.contextual_types[node] = result_type

    def visit_AstUnaryOp(self, node: AstUnaryOp, state: CompileState):
        val_type = state.synthesized_types[node.val]

        intermediate_type = self.pick_intermediate_type([val_type], node.op)
        if intermediate_type is None:
            state.err(f"Op {node.op} undefined for {typename(val_type)}", node)
            return

        if not self.coerce_expr_type(node.val, intermediate_type, state):
            return

        result_type = None
        if node.op in NUMERIC_OPERATORS:
            result_type = intermediate_type
        else:
            result_type = BoolValue

        state.op_intermediate_types[node] = intermediate_type
        state.synthesized_types[node] = result_type
        state.contextual_types[node] = result_type

    def visit_AstString(self, node: AstString, state: CompileState):
        state.synthesized_types[node] = FpyStringValue
        state.contextual_types[node] = FpyStringValue

    def visit_AstBoolean(self, node: AstBoolean, state: CompileState):
        state.synthesized_types[node] = BoolValue
        state.contextual_types[node] = BoolValue

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
            if not issubclass(input_type, NumericalValue):
                # cannot convert a non-numeric type to a numeric type
                return CompileError(
                    f"Expected a number, found {typename(input_type)}", node_arg
                )
            # no error! looks good to me
            return

        # Check provided args against expected
        for value_expr, arg in zip(resolved_args, func_args):
            arg_type = arg[1]

            # Skip type check for default values from forward-called functions.
            # These expressions haven't been visited yet, so they're not in
            # synthesized_types. Their type compatibility is verified when
            # the function definition is visited.
            if value_expr not in state.synthesized_types:
                continue

            unconverted_type = state.synthesized_types[value_expr]
            if not self.can_coerce_type(unconverted_type, arg_type):
                return CompileError(
                    f"Expected {typename(arg_type)}, found {typename(unconverted_type)}",
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
                # Skip coercion for default values from forward-called functions.
                # These will be coerced when the function definition is visited.
                if value_expr not in state.synthesized_types:
                    continue
                arg_type = arg[1]
                # should be good 2 go based on the check func above
                state.contextual_types[value_expr] = arg_type

        state.synthesized_types[node] = func.return_type
        state.contextual_types[node] = func.return_type

    def visit_AstRange(self, node: AstRange, state: CompileState):
        if not self.coerce_expr_type(node.lower_bound, LoopVarType, state):
            return
        if not self.coerce_expr_type(node.upper_bound, LoopVarType, state):
            return

        state.synthesized_types[node] = RangeValue
        state.contextual_types[node] = RangeValue

    def visit_AstAssign(self, node: AstAssign, state: CompileState):
        # should be present in resolved refs because we only let it through if
        # variable is attr, item or var
        lhs_sym = state.resolved_symbols[node.lhs]
        if not is_instance_compat(lhs_sym, (VariableSymbol, FieldSymbol)):
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
            assert (
                state.contextual_types[node.lhs]
                == state.synthesized_types[node.lhs]
            )
            lhs_type = state.contextual_types[node.lhs]

        # coerce the rhs into the lhs type
        if not self.coerce_expr_type(node.rhs, lhs_type, state):
            return

    def visit_AstAssert(self, node: AstAssert, state: CompileState):
        if not self.coerce_expr_type(node.condition, BoolValue, state):
            return
        if node.exit_code is not None:
            if not self.coerce_expr_type(node.exit_code, U8Value, state):
                return

    def visit_AstFor(self, node: AstFor, state: CompileState):
        # range must coerce to a range!
        if not self.coerce_expr_type(node.range, RangeValue, state):
            return

    def visit_AstWhile(self, node: AstWhile, state: CompileState):
        if not self.coerce_expr_type(node.condition, BoolValue, state):
            return

    def visit_AstIf_AstElif(self, node: Union[AstIf, AstElif], state: CompileState):
        if not self.coerce_expr_type(node.condition, BoolValue, state):
            return

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # Validate that default argument types are compatible with parameter types
        if node.parameters is None:
            return

        func = state.resolved_symbols[node.name]
        if not is_instance_compat(func, FunctionSymbol):
            return

        for (arg_name_var, arg_type_expr, default_value), (_, arg_type, _) in zip(
            node.parameters, func.args
        ):
            if default_value is not None:
                # Check that default value's type can be coerced to parameter type
                if not self.coerce_expr_type(default_value, arg_type, state):
                    return

    def visit_AstReturn(self, node: AstReturn, state: CompileState):
        func = state.enclosing_funcs[node]
        func = state.resolved_symbols[func.name]
        if func.return_type is NothingValue and node.value is not None:
            state.err("Expected no return value", node.value)
            return
        if func.return_type is not NothingValue and node.value is None:
            state.err(
                f"Expected a return value of type {typename(func.return_type)}",
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
            const_value = state.contextual_values.get(default_value)
            if const_value is None:
                state.err(
                    f"Default value for argument '{arg_name_var.var}' must be a constant expression",
                    default_value,
                )
                return


class CalculateConstExprValues(Visitor):
    """for each expr, try to calculate its constant value and store it in a map. stores None if no value could be
    calculated at compile time, and NothingType if the expr had no value"""

    @staticmethod
    def _round_float_to_type(value: float, to_type: type[FloatValue]) -> float | None:
        fmt = to_type.get_serialize_format()
        assert fmt is not None, to_type
        try:
            packed = struct.pack(fmt, value)
        except OverflowError:
            return None

        return struct.unpack(fmt, packed)[0]

    @staticmethod
    def const_convert_type(
        from_val: FppValue,
        to_type: FppType,
        node: Ast,
        state: CompileState,
        skip_range_check: bool = False,
    ) -> FppValue | None:
        try:
            from_type = type(from_val)

            if from_type == to_type:
                # no conversion necessary
                return from_val

            if issubclass(to_type, StringValue):
                assert from_type == FpyStringValue, from_type
                return to_type(from_val.val)

            if issubclass(to_type, FloatValue):
                assert issubclass(from_type, NumericalValue), from_type
                from_val = from_val.val

                if to_type == FpyFloatValue:
                    # arbitrary precision
                    # decimal constructor should handle all cases: int, float, or other Decimal
                    return FpyFloatValue(Decimal(from_val))

                # otherwise, we're going to a finite bitwidth float type
                try:
                    coerced_value = float(from_val)
                except OverflowError:
                    state.err(
                        f"{from_val} is out of range for type {typename(to_type)}",
                        node,
                    )
                    return None

                rounded_value = CalculateConstExprValues._round_float_to_type(
                    coerced_value, to_type
                )
                if rounded_value is None:
                    state.err(
                        f"{from_val} is out of range for type {typename(to_type)}",
                        node,
                    )
                    return None

                converted = to_type(rounded_value)
                try:
                    # catch if we would crash the struct packing lib
                    converted.serialize()
                except OverflowError:
                    state.err(
                        f"{from_val} is out of range for type {typename(to_type)}",
                        node,
                    )
                    return None
                return converted
            if issubclass(to_type, IntegerValue):
                assert issubclass(from_type, NumericalValue), from_type
                from_val = from_val.val

                if to_type == FpyIntegerValue:
                    # arbitrary precision
                    # int constructor should handle all cases: int, float, or Decimal
                    return FpyIntegerValue(int(from_val))

                # otherwise going to a finite bitwidth integer type

                if not skip_range_check:
                    # does it fit within bounds?
                    # check that the value can fit in the dest type
                    dest_min, dest_max = to_type.range()
                    if from_val < dest_min or from_val > dest_max:
                        state.err(
                            f"{from_val} is out of range for type {typename(to_type)}",
                            node,
                        )
                        return None

                    # just convert it
                    from_val = int(from_val)
                else:
                    # we skipped the range check, but it's still gotta fit. cut it down

                    # handle narrowing, if necessary
                    from_val = int(from_val)
                    # if signed, convert to unsigned (bit representation should be the same)
                    # first cut down to bitwidth. performed in two's complement
                    mask = (1 << to_type.get_bits()) - 1
                    # this also implicitly converts value to an unsigned number
                    from_val &= mask
                    if to_type in SIGNED_INTEGER_TYPES:
                        # now if the target was signed:
                        sign_bit = 1 << (to_type.get_bits() - 1)
                        if from_val & sign_bit:
                            # the sign bit is set, the result should be negative
                            # subtract the max value as this is how two's complement works
                            from_val -= 1 << to_type.get_bits()

                # okay, we either checked that the value fits in the dest, or we've skipped
                # the check and changed the value to fit
                return to_type(from_val)

            assert False, (from_val, from_type, to_type)
        except TypeException as e:
            state.err(f"For type {typename(from_type)}: {e}", node)
            return None

    def visit_AstLiteral(self, node: AstLiteral, state: CompileState):
        unconverted_type = state.synthesized_types[node]

        try:
            expr_value = unconverted_type(node.value)
        except TypeException as e:
            # TODO can this be reached any more? maybe for string types
            state.err(f"For type {typename(unconverted_type)}: {e}", node)
            return

        skip_range_check = node in state.expr_explicit_casts
        converted_type = state.contextual_types[node]
        if converted_type != unconverted_type:
            expr_value = self.const_convert_type(
                expr_value, converted_type, node, state, skip_range_check
            )
            if expr_value is None:
                return

        state.contextual_values[node] = expr_value

    def visit_AstMemberAccess(self, node: AstMemberAccess, state: CompileState):
        unconverted_type = state.synthesized_types[node]
        converted_type = state.contextual_types[node]
        sym = state.resolved_symbols[node]
        expr_value = None
        if is_instance_compat(sym, (type, dict, CallableSymbol)):
            # these types have no value
            state.contextual_values[node] = NothingValue()
            assert unconverted_type == converted_type, (
                unconverted_type,
                converted_type,
            )
            return
        elif is_instance_compat(sym, (ChTemplate, PrmTemplate, VariableSymbol)):
            # has a value but won't try to calc at compile time
            state.contextual_values[node] = None
            return
        elif is_instance_compat(sym, FppValue):
            expr_value = sym
        elif is_instance_compat(sym, FieldSymbol):
            parent_value = state.contextual_values[node.parent]
            if parent_value is None:
                # no compile time constant value for our parent here
                state.contextual_values[node] = None
                return

            # we are accessing an attribute of something with an fprime value at compile time
            # we must be getting a member
            if is_instance_compat(parent_value, StructValue):
                expr_value = parent_value._val[node.attr]
            elif is_instance_compat(parent_value, TimeValue):
                if node.attr == "seconds":
                    expr_value = U32Value(parent_value.seconds)
                elif node.attr == "useconds":
                    expr_value = U32Value(parent_value.useconds)
                elif node.attr == "time_base":
                    expr_value = U16Value(parent_value.timeBase)
                elif node.attr == "time_context":
                    expr_value = U8Value(parent_value.timeContext)
                else:
                    assert False, node.attr
            else:
                assert False, parent_value

        assert expr_value is not None

        assert is_instance_compat(expr_value, unconverted_type), (
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
        state.contextual_values[node] = expr_value

    def visit_AstIndexExpr(self, node: AstIndexExpr, state: CompileState):
        sym = state.resolved_symbols[node]
        # index expression can only be a field symbol
        assert is_instance_compat(sym, FieldSymbol), sym

        parent_value = state.contextual_values[node.parent]

        if parent_value is None:
            # no compile time constant value for our parent here
            state.contextual_values[node] = None
            return

        assert is_instance_compat(parent_value, ArrayValue), parent_value

        idx = state.contextual_values.get(node.item)
        if idx is None:
            # no compile time constant value for our index
            state.contextual_values[node] = None
            return

        assert is_instance_compat(idx, ArrayIndexType)

        expr_value = parent_value._val[idx._val]

        unconverted_type = state.synthesized_types[node]
        assert is_instance_compat(expr_value, unconverted_type), (
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
        state.contextual_values[node] = expr_value

    def visit_AstVar(self, node: AstVar, state: CompileState):
        unconverted_type = state.synthesized_types[node]
        converted_type = state.contextual_types[node]
        sym = state.resolved_symbols[node]
        expr_value = None
        if is_instance_compat(sym, (type, dict, CallableSymbol)):
            # these types have no value
            state.contextual_values[node] = NothingValue()
            assert unconverted_type == converted_type, (
                unconverted_type,
                converted_type,
            )
            return
        elif is_instance_compat(sym, (ChTemplate, PrmTemplate, VariableSymbol)):
            # Has a value but we don't try to calculate it at compile time.
            # NOTE: If you ever add const-folding for VariableSymbol here, you must also
            # update CalculateDefaultArgConstValues. That pass runs CalculateConstExprValues
            # on default argument expressions BEFORE this pass runs on variable assignments.
            # So if a default value references a variable, the variable's const value won't
            # be available yet, and the default value will incorrectly be rejected as non-const.
            state.contextual_values[node] = None
            return
        elif is_instance_compat(sym, FppValue):
            expr_value = sym
        elif is_instance_compat(sym, FieldSymbol):
            assert False, sym

        assert expr_value is not None

        assert is_instance_compat(expr_value, unconverted_type), (
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
        state.contextual_values[node] = expr_value

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        func = state.resolved_symbols[node.func]
        assert is_instance_compat(func, CallableSymbol)

        # Use resolved args from semantic analysis (already in positional order,
        # with defaults filled in)
        # This is guaranteed to be set by PickTypesAndResolveAttrsAndItems
        resolved_args = state.resolved_func_args[node]

        # Gather arg values. Since defaults are already filled in, we just need
        # to look up each arg's const value.
        arg_values = [
            state.contextual_values.get(arg_expr) for arg_expr in resolved_args
        ]

        unknown_value = any(v is None for v in arg_values)
        if unknown_value:
            # we will have to calculate this at runtime
            state.contextual_values[node] = None
            return

        expr_value = None

        # whether the conversion that will happen is due to an explicit cast
        if is_instance_compat(func, TypeCtorSymbol):
            # actually construct the type
            if issubclass(func.type, StructValue):
                instance = func.type()
                # pass in args as a dict
                # t[0] is the arg name
                arg_dict = {t[0]: v for t, v in zip(func.type.MEMBER_LIST, arg_values)}
                instance._val = arg_dict
                expr_value = instance

            elif issubclass(func.type, ArrayValue):
                instance = func.type()
                instance._val = arg_values
                expr_value = instance

            elif func.type == TimeValue:
                expr_value = TimeValue(*[val.val for val in arg_values])

            else:
                # no other FppTypees have ctors
                assert False, func.return_type
        elif is_instance_compat(func, CastSymbol):
            # should only be one value. it should be of some numeric type
            # our const convert type func will convert it for us
            expr_value = arg_values[0]
        else:
            # don't try to calculate the value of this function call
            # it's something like a user defined func, cmd or builtin
            state.contextual_values[node] = None
            return

        unconverted_type = state.synthesized_types[node]
        assert is_instance_compat(expr_value, unconverted_type), (
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

        state.contextual_values[node] = expr_value

    def visit_AstBinaryOp(self, node: AstBinaryOp, state: CompileState):
        # Check if both left-hand side (lhs) and right-hand side (rhs) are constants
        lhs_value: FppValue = state.contextual_values.get(node.lhs)
        rhs_value: FppValue = state.contextual_values.get(node.rhs)

        if lhs_value is None or rhs_value is None:
            state.contextual_values[node] = None
            return

        # Both sides are constants, evaluate the operation if the operator is supported

        if not is_instance_compat(lhs_value, ValueType) or not is_instance_compat(
            rhs_value, ValueType
        ):
            # if one of them isn't a ValueType, assume it must be TimeValue
            assert type(lhs_value) == type(rhs_value) and is_instance_compat(
                lhs_value, TimeValue
            ), (
                lhs_value,
                rhs_value,
            )
        else:
            # get the actual pythonic value from the fpp type
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
                if not is_instance_compat(lhs_value, Number):
                    # comparing two complex types
                    assert type(lhs_value) == type(rhs_value), (lhs_value, rhs_value)
                    # for now we don't fold this
                    folded_value = None
                else:
                    folded_value = lhs_value == rhs_value
            elif node.op == BinaryStackOp.NOT_EQUAL:
                if not is_instance_compat(lhs_value, Number):
                    # comparing two complex types
                    assert type(lhs_value) == type(rhs_value), (lhs_value, rhs_value)
                    # for now we don't fold this
                    folded_value = None
                else:
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

        if folded_value is None:
            # give up, don't try to calculate the value of this expr at compile time
            state.contextual_values[node] = None
            return

        if type(folded_value) == int:
            folded_value = FpyIntegerValue(folded_value)
        elif type(folded_value) == Decimal:
            folded_value = FpyFloatValue(folded_value)
        elif type(folded_value) == bool:
            folded_value = BoolValue(folded_value)
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
        state.contextual_values[node] = folded_value

    def visit_AstUnaryOp(self, node: AstUnaryOp, state: CompileState):
        value: FppValue = state.contextual_values.get(node.val)

        if value is None:
            state.contextual_values[node] = None
            return

        # input is constant, evaluate the operation if the operator is supported
        assert is_instance_compat(value, ValueType), value

        # get the actual pythonic value from the fpp type
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
            folded_value = FpyIntegerValue(folded_value)
        elif type(folded_value) == Decimal:
            folded_value = FpyFloatValue(folded_value)
        elif type(folded_value) == bool:
            folded_value = BoolValue(folded_value)
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
        state.contextual_values[node] = folded_value

    def visit_AstRange(self, node: AstRange, state: CompileState):
        # ranges don't really end up having a value, they kinda just exist as a type
        state.contextual_values[node] = None

    def visit_default(self, node, state):
        # coding error, missed an expr
        assert not is_instance_compat(node, AstExpr), node


class CheckAllBranchesReturn(Visitor):
    def visit_AstReturn(self, node: AstReturn, state: CompileState):
        state.does_return[node] = True

    def visit_AstStmtList(self, node: Union[AstStmtList, AstBlock], state: CompileState):
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
        state.does_return[node] = False

    def visit_AstExpr(self, node: AstExpr, state: CompileState):
        # expressions do not return
        state.does_return[node] = False

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
                f"Function '{node.name.var}' does not always return a value", node
            )
            return


class CheckConstArrayAccesses(Visitor):
    def visit_AstIndexExpr(self, node: AstIndexExpr, state: CompileState):
        # if the index is a const, we should be able to check if it's in bounds
        idx_value = state.contextual_values.get(node.item)
        if idx_value is None:
            # can't check at compile time
            return

        parent_type = state.contextual_types[node.parent]
        assert issubclass(parent_type, ArrayValue), parent_type

        if idx_value.val < 0 or idx_value.val >= parent_type.LENGTH:
            state.err(
                f"Index {idx_value.val} out of bounds for array type {typename(parent_type)} with length {parent_type.LENGTH}",
                node.item,
            )
            return


class WarnRangesAreNotEmpty(Visitor):
    def visit_AstRange(self, node: AstRange, state: CompileState):
        # if the index is a const, we should be able to check if it's in bounds
        lower_value: LoopVarType = state.contextual_values.get(node.lower_bound)
        upper_value: LoopVarType = state.contextual_values.get(node.upper_bound)
        if lower_value is None or upper_value is None:
            # cannot check at compile time
            return

        if lower_value.val >= upper_value.val:
            state.warn("Range is empty", node)
