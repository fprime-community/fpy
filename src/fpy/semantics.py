from __future__ import annotations
from datetime import datetime, timezone
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
    BuiltinFuncSymbol,
    CompileState,
    FieldAccess,
    ForLoopAnalysis,
    FppType,
    CallableSymbol,
    CastSymbol,
    FpyFloatValue,
    FunctionSymbol,
    NameGroup,
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
    is_symbol_an_expr,
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
    COMPARISON_OPS,
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
    I8Type as I8Value,
    I16Type as I16Value,
    I32Type as I32Value,
    I64Type as I64Value,
    F32Type as F32Value,
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
from fprime_gds.common.models.serialize.type_base import BaseType as FppValue


class AssignIds(TopDownVisitor):
    """assigns a unique id to each node to allow it to be indexed in a dict"""

    def visit_default(self, node, state: CompileState):
        node.id = state.next_node_id
        state.next_node_id += 1


class SetEnclosingValueScope(Visitor):
    """Sets the enclosing value scope for all visited nodes."""

    def __init__(self, scope: SymbolTable):
        super().__init__()
        self.scope = scope

    def visit_default(self, node: Ast, state: CompileState):
        state.enclosing_value_scope[node] = self.scope


class CreateFunctionScopes(TopDownVisitor):
    """Creates all function value scopes. Every other scope is global, so is already made at init
    Assigns enclosing_value_scope for all nodes.
    """

    def visit_AstBlock(self, node: AstBlock, state: CompileState):
        if node is not state.root:
            # only handle the root node this way
            return
        # Global nodes use global_value_scope directly
        SetEnclosingValueScope(state.global_value_scope).run(node, state)

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # Make a new scope for the function body
        func_scope = SymbolTable()

        # The function body gets the new scope
        SetEnclosingValueScope(func_scope).run(node.body, state)

        # Parameter names and type annotations are in the function scope
        # (they're defined inside the function)
        if node.parameters is not None:
            for arg_name_var, arg_type_name, default_value in node.parameters:
                state.enclosing_value_scope[arg_name_var] = func_scope
                state.enclosing_value_scope[arg_type_name] = func_scope
                # Default values are evaluated at definition site, so they use the parent scope


class CreateVariablesAndFuncs(TopDownVisitor):
    """finds all variable declarations and adds them to the appropriate scope"""

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
            # make sure it isn't defined in this scope
            # TODO shadowing check
            existing_local = scope.get(node.lhs.name)
            if existing_local is not None:
                # redeclaring an existing variable
                state.err(f"Variable '{node.lhs.name}' has already been defined", node)
                return
            # okay, define the var
            is_global = state.enclosing_value_scope[node] is state.global_value_scope
            var = VariableSymbol(
                node.lhs.name, node.type_ann, node, is_global=is_global
            )
            # new var. put it in the scope
            scope[node.lhs.name] = var
        else:
            # otherwise, it's a reference to an existing var
            # may be in this scope or outer scope
            sym = scope.get(node.lhs.name) or state.global_value_scope.get(
                node.lhs.name
            )
            if sym is None:
                # unable to find this symbol
                state.err(
                    f"Variable '{node.lhs.name}' used before defined",
                    node.lhs,
                )
                return
            # okay, we were able to resolve it

    def visit_AstFor(self, node: AstFor, state: CompileState):
        # for loops have an implicit loop variable that they can define
        # if it isn't already defined in the local scope
        scope = state.enclosing_value_scope[node]
        loop_var = scope.get(node.loop_var.name)

        reuse_existing_loop_var = False
        if loop_var is not None:
            # this is okay as long as the variable is of the same type

            # what follows is a bit of a hack
            # there are two cases: either loop_var has been defined before but we only know the type expr (if it was an AstAssign decl)
            # or loop_var has been defined before and we only know the type, but have no type expr (from some other for loop)

            # case 1 is easy, just check the type == LoopVarType
            # case 2 is harder, we have to check if the type ident is an AstIdent
            # that matches the canonical name of the LoopVarType

            # the alternative to this is that we do some primitive type resolution in the same pass as variable creation
            # i'm doing this hack because we're going to switch to type inference for variables later and that will make this go away

            if (loop_var.type_ref is None and loop_var.type != LoopVarType) or (
                loop_var.type is None
                and not (
                    is_instance_compat(loop_var.type_ref, AstIdent)
                    and loop_var.type_ref.name == LoopVarType.get_canonical_name()
                )
            ):
                state.err(
                    f"Variable '{node.loop_var.name}' has already been defined as a type other than {typename(LoopVarType)}",
                    node,
                )
                return
            reuse_existing_loop_var = True
        else:
            # new var. put it in the scope
            is_global = state.enclosing_value_scope[node] is state.global_value_scope
            loop_var = VariableSymbol(
                node.loop_var.name, None, node, LoopVarType, is_global=is_global
            )
            scope[loop_var.name] = loop_var

        # each loop also defines an implicit ub variable
        # type of ub var is same as loop var type
        is_global = state.enclosing_value_scope[node] is state.global_value_scope
        upper_bound_var = VariableSymbol(
            state.new_anonymous_variable_name(),
            None,
            node,
            LoopVarType,
            is_global=is_global,
        )
        scope[upper_bound_var.name] = upper_bound_var
        analysis = ForLoopAnalysis(loop_var, upper_bound_var, reuse_existing_loop_var)
        state.for_loops[node] = analysis

    def visit_AstDef(self, node: AstDef, state: CompileState):
        # Functions always go in the global callable scope
        existing_func = state.global_callable_scope.get(node.name.name)
        if existing_func is not None:
            state.err(
                f"Function '{node.name.name}' has already been defined", node.name
            )
            return

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
                return
            arg_var = VariableSymbol(arg_name_var.name, arg_type_name, node)
            func_scope[arg_name_var.name] = arg_var


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
            root_symbol = state.enclosing_value_scope[root_node].get(root_node.name)
            if (
                root_symbol is None
                and state.enclosing_value_scope[root_node]
                is not state.global_value_scope
            ):
                # if we just checked and failed in a function value scope, check the global value scope
                root_symbol = state.global_value_scope.get(root_node.name)

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


def is_type_constant_size(type: FppType) -> bool:
    """Return true if the type has a statically known size.
    
    Types with strings (directly or nested) don't have constant size because
    strings can vary in length.
    """
    if issubclass(type, StringValue):
        return False

    if issubclass(type, ArrayValue):
        return is_type_constant_size(type.MEMBER_TYPE)

    if issubclass(type, StructValue):
        for _, arg_type, _, _ in type.MEMBER_LIST:
            if not is_type_constant_size(arg_type):
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
            func.return_type = NothingValue
        else:
            return_type = state.resolved_symbols[node.return_type]
            if not is_type_constant_size(return_type):
                state.err(
                    f"Type {typename(return_type)} is not constant-sized (contains strings)",
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
                        f"Type {typename(arg_type)} is not constant-sized (contains strings)",
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
                f"Type {typename(var_type)} is not constant-sized (contains strings)",
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
            state.err(f"'{node.name}' used before defined", node)
            return


class PickTypesAndResolveMembersAndElements(Visitor):

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

    def can_coerce_type(self, from_type: FppType, from_const: bool, to_type: FppType, to_const: bool) -> bool:
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

        # if we have a const, we know its value, we can check later if the value fits in the dest.
        # for now, permit it?
        
        # if we have a non-const, and we're going to const, we will fail

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

    def pick_time_intermediate_type(
        self, arg_types: list[FppType], op: BinaryStackOp, state: CompileState
    ) -> FppType | None:
        """Return intermediate type for time/interval operations, or None if not a time operation."""
        if len(arg_types) != 2:
            return None
        lhs_type, rhs_type = arg_types
        lhs_is_time = lhs_type is not None and issubclass(lhs_type, TimeValue)
        rhs_is_time = rhs_type is not None and issubclass(rhs_type, TimeValue)
        lhs_is_interval = getattr(lhs_type, '__name__', None) == "Fw.TimeIntervalValue"
        rhs_is_interval = getattr(rhs_type, '__name__', None) == "Fw.TimeIntervalValue"
        
        # Time - Time -> TimeIntervalValue (via time_sub)
        if lhs_is_time and rhs_is_time and op == BinaryStackOp.SUBTRACT:
            return TimeValue  # intermediate type is Time, result will be interval
        
        # Time + TimeInterval -> Time (via time_add)
        if lhs_is_time and rhs_is_interval and op == BinaryStackOp.ADD:
            return TimeValue  # intermediate type doesn't need conversion
        
        # Time comparisons
        if lhs_is_time and rhs_is_time and op in COMPARISON_OPS:
            return TimeValue
        
        # TimeInterval + TimeInterval -> TimeInterval (via time_interval_add)
        if lhs_is_interval and rhs_is_interval and op == BinaryStackOp.ADD:
            return state.time_interval_type
        
        # TimeInterval - TimeInterval -> TimeInterval (via time_interval_sub)
        if lhs_is_interval and rhs_is_interval and op == BinaryStackOp.SUBTRACT:
            return state.time_interval_type
        
        # TimeInterval comparisons
        if lhs_is_interval and rhs_is_interval and op in COMPARISON_OPS:
            return state.time_interval_type
        
        return None

    def _pick_numeric_type_category(
        self, arg_types: list[FppType], op: BinaryStackOp | UnaryStackOp
    ) -> str:
        """Determine the type category (float, uint, int) for a numeric op intermediate or result type."""
        if op == BinaryStackOp.DIVIDE or op == BinaryStackOp.EXPONENT:
            # always do true division and exponentiation over floats, python style
            return "float"
        elif any(issubclass(t, FloatValue) for t in arg_types):
            return "float"
        elif any(t in UNSIGNED_INTEGER_TYPES for t in arg_types):
            return "uint"
        else:
            return "int"

    def _select_type_for_category_and_bits(self, type_category: str, bits: int) -> FppType:
        """Select the appropriate concrete type given a category and bitwidth."""
        if type_category == "float":
            return F64Value if bits > 32 else F32Value
        if type_category == "uint":
            if bits <= 8:
                return U8Value
            elif bits <= 16:
                return U16Value
            elif bits <= 32:
                return U32Value
            else:
                return U64Value
        assert type_category == "int"
        if bits <= 8:
            return I8Value
        elif bits <= 16:
            return I16Value
        elif bits <= 32:
            return I32Value
        else:
            return I64Value

    def pick_intermediate_type(
        self,
        arg_types: list[FppType],
        arg_consts: list[bool],
        op: BinaryStackOp | UnaryStackOp,
    ) -> FppType | None:
        """Return the intermediate type for an operation (excluding time ops).
        
        This always returns 64-bit types for non-constant numerics, as required by the VM.
        Returns None if the operation is invalid for the given types.
        """

        # intermediate type is really about two things:
        # 1: we must pick a type for which a bytecode impl, or compile time const impl, exists
        # okay, so we know which types work for bytecode ops. but which types work for compile time consts?
        # any type? i guess so. why not pick I8 for all?
        # for compile time consts, do they even need to be converted?
        # it doesn't really matter what the intermediate type is for compile time consts.
        # the result type matters, but intermediate type, well, as long as inputs are consts, we 

        # well, what is intermed type used for? it's used to coerce both args into that. so if we skip it
        # for consts, then yeah i mean that's what we're already pretty much trying to do. 

        # so maybe skip this step for compile time consts?

        # 2: we must pick a type to which both args can be coerced

        # perhaps we focus on 2 first. are there common types to which both args can be coerced?

        # we can decide the intermediate type based on the following properties, and
        # also the operator itself
        non_numeric = any(not issubclass(t, NumericalValue) for t in arg_types)
        has_finite_precision = any(t not in ARBITRARY_PRECISION_TYPES for t in arg_types)
        has_arbitrary_precision = any(t in ARBITRARY_PRECISION_TYPES for t in arg_types)
        has_float = any(issubclass(t, FloatValue) for t in arg_types)
        has_unsigned = any(t in UNSIGNED_INTEGER_TYPES for t in arg_types)
        more_than_one_input_type = len(set(arg_types)) > 1
        
        # all boolean operators operate on bool type
        if op in BOOLEAN_OPERATORS:
            return BoolValue

        if non_numeric:
            # non numeric arg types are only valid for == and !=
            if (op == BinaryStackOp.EQUAL or op == BinaryStackOp.NOT_EQUAL):
                # comparison of complex types (structs/strings/arrays/enum consts)
                # only valid if == 1 unique input type (cannot compare dissimilar types)
                if more_than_one_input_type:
                    return None
                return arg_types[0]
            return None

        

        # Check for mixed signed/unsigned integers - no valid common type exists
        # (floats are fine to mix with any integer, and arbitrary precision can convert to anything)
        if not has_float and len(concrete_types) >= 2:
            has_signed = any(t in SIGNED_INTEGER_TYPES for t in concrete_types)
            has_unsigned = any(t in UNSIGNED_INTEGER_TYPES for t in concrete_types)
            if has_signed and has_unsigned:
                return None

        type_category = self._pick_numeric_type_category(arg_types, op)

        # If all operands are arbitrary precision constants, we can constant fold
        if all(t in ARBITRARY_PRECISION_TYPES for t in arg_types):
            if type_category == "float":
                return FpyFloatValue
            return FpyIntegerValue

        return self._select_type_for_category_and_bits(type_category, 64)

    def pick_result_type(
        self, arg_types: list[FppType], intermediate_type: FppType, op: BinaryStackOp | UnaryStackOp
    ) -> FppType:
        """Derive the result type from the intermediate type (excluding time ops).

        For numerics, shrinks to max input bitwidth.
        Arbitrary precision literals are treated as 64-bit, so e.g. `U32 * literal` results in U64.
        """
        if op in BOOLEAN_OPERATORS or op in COMPARISON_OPS:
            return BoolValue

        # all other cases, result is a number
        assert op in NUMERIC_OPERATORS

        # if all args are arb precision consts, keep it that way
        if intermediate_type in (FpyIntegerValue, FpyFloatValue):
            return intermediate_type

        # Compute max bitwidth, treating arbitrary precision as 64-bit
        def get_bits(t: FppType) -> int:
            return 64 if t in ARBITRARY_PRECISION_TYPES else t.get_bits()

        bits = max(get_bits(t) for t in arg_types)

        # Determine category from intermediate type
        if issubclass(intermediate_type, FloatValue):
            type_category = "float"
        elif intermediate_type in UNSIGNED_INTEGER_TYPES:
            type_category = "uint"
        else:
            type_category = "int"

        return self._select_type_for_category_and_bits(type_category, bits)

    def get_type_of_symbol(self, sym: Symbol) -> FppType:
        """returns the fprime type of the sym, if it were to be evaluated as an expression"""
        if isinstance(sym, ChTemplate):
            result_type = sym.ch_type_obj
        elif isinstance(sym, PrmTemplate):
            result_type = sym.prm_type_obj
        elif isinstance(sym, FppValue):
            # constant value
            result_type = type(sym)
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

            if not issubclass(parent_type, (StructValue, TimeValue)):
                state.err(f"{typename(parent_type)} is not a struct, cannot access members", node)
                return

            if not is_type_constant_size(parent_type):
                state.err(
                    f"{typename(parent_type)} is not constant-sized (contains strings), cannot access members",
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
                offset += arg_type.getMaxSize()
                base_offset += arg_type.getMaxSize()

            if this_sym is None:
                state.err(
                    f"{typename(parent_type)} has no member named {node.attr}",
                    node,
                )
                return

        sym_type = self.get_type_of_symbol(this_sym)

        is_const = False

        if is_instance_compat(this_sym, FppValue):
            is_const = True
        elif is_instance_compat(this_sym, FieldAccess):
            is_const = node.parent in state.const_exprs

        if is_const:
            state.const_exprs.add(node)
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
                f"{typename(parent_type)} is not constant-sized (contains strings), cannot access items",
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
            if not is_instance_compat(parent_sym, FieldAccess)
            else parent_sym.base_sym
        )

        sym = FieldAccess(
            is_array_element=True,
            parent_expr=node.parent,
            type=parent_type.MEMBER_TYPE,
            base_sym=base_sym,
            idx_expr=node.item,
        )

        if node.parent in state.const_exprs and node.item in state.const_exprs:
            # this is a const expr if parent and item are consts
            state.const_exprs.add(node)
        state.resolved_symbols[node] = sym
        state.synthesized_types[node] = parent_type.MEMBER_TYPE
        state.contextual_types[node] = parent_type.MEMBER_TYPE

    def visit_AstIdent(self, node: AstIdent, state: CompileState):
        # already been resolved
        sym = state.resolved_symbols[node]
        if sym is None:
            return
        if not is_symbol_an_expr(sym):
            return

        sym_type = self.get_type_of_symbol(sym)

        if is_instance_compat(sym, FppValue):
            state.const_exprs.add(node)
        state.synthesized_types[node] = sym_type
        state.contextual_types[node] = sym_type

    def visit_AstNumber(self, node: AstNumber, state: CompileState):
        # give a best guess as to the final type of this node. we don't actually know
        # its bitwidth or signedness yet
        if is_instance_compat(node.value, Decimal):
            result_type = FpyFloatValue
        else:
            result_type = FpyIntegerValue

        state.const_exprs.add(node)
        state.synthesized_types[node] = result_type
        state.contextual_types[node] = result_type

    def pick_time_result_type(
        self, lhs_type: type[FppValue], rhs_type: type[FppValue], op: BinaryStackOp, state: CompileState
    ) -> type[FppValue] | None:
        """Return the result type for time/interval operations, or None if not a time operation."""
        lhs_is_time = issubclass(lhs_type, TimeValue)
        rhs_is_time = issubclass(rhs_type, TimeValue)
        lhs_is_interval = getattr(lhs_type, '__name__', None) == "Fw.TimeIntervalValue"
        rhs_is_interval = getattr(rhs_type, '__name__', None) == "Fw.TimeIntervalValue"

        if lhs_is_time and rhs_is_time and op == BinaryStackOp.SUBTRACT:
            return state.time_interval_type
        if lhs_is_time and rhs_is_time and op in COMPARISON_OPS:
            return BoolValue
        if lhs_is_time and rhs_is_interval and op == BinaryStackOp.ADD:
            return TimeValue
        if lhs_is_interval and rhs_is_interval and op in (BinaryStackOp.ADD, BinaryStackOp.SUBTRACT):
            return state.time_interval_type
        if lhs_is_interval and rhs_is_interval and op in COMPARISON_OPS:
            return BoolValue
        return None

    def visit_AstBinaryOp(self, node: AstBinaryOp, state: CompileState):
        lhs_type = state.synthesized_types[node.lhs]
        rhs_type = state.synthesized_types[node.rhs]
        arg_types = [lhs_type, rhs_type]

        if node.lhs in state.const_exprs and node.rhs in state.const_exprs:
            state.const_exprs.add(node)

        # Time/interval operations are desugared to function calls
        time_intermediate = self.pick_time_intermediate_type(arg_types, node.op, state)
        if time_intermediate is not None:
            time_result = self.pick_time_result_type(lhs_type, rhs_type, node.op, state)
            assert time_result is not None
            state.op_intermediate_types[node] = time_intermediate
            state.synthesized_types[node] = time_result
            state.contextual_types[node] = time_result
            return

        intermediate_type = self.pick_intermediate_type(arg_types, node.op)
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

        result_type = self.pick_result_type(arg_types, intermediate_type, node.op)

        state.op_intermediate_types[node] = intermediate_type
        state.synthesized_types[node] = result_type
        state.contextual_types[node] = result_type

    def visit_AstUnaryOp(self, node: AstUnaryOp, state: CompileState):
        val_type = state.synthesized_types[node.val]
        arg_types = [val_type]

        intermediate_type = self.pick_intermediate_type(arg_types, node.op)
        if intermediate_type is None:
            state.err(f"Op {node.op} undefined for {typename(val_type)}", node)
            return

        if not self.coerce_expr_type(node.val, intermediate_type, state):
            return

        result_type = self.pick_result_type(arg_types, intermediate_type, node.op)

        if node.val in state.const_exprs:
            state.const_exprs.add(node)
        state.op_intermediate_types[node] = intermediate_type
        state.synthesized_types[node] = result_type
        state.contextual_types[node] = result_type

    def visit_AstString(self, node: AstString, state: CompileState):
        state.const_exprs.add(node)
        state.synthesized_types[node] = FpyStringValue
        state.contextual_types[node] = FpyStringValue

    def visit_AstBoolean(self, node: AstBoolean, state: CompileState):
        state.const_exprs.add(node)
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

            # Skip type check for default values that are FppValue instances
            # this can only happen if the value is hardcoded into Fpy from a builtin func
            if not is_instance_compat(value_expr, Ast):
                assert is_instance_compat(func, BuiltinFuncSymbol), func
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

        # Check that type constructors are for constant-sized types
        if is_instance_compat(func, TypeCtorSymbol):
            if not is_type_constant_size(func.type):
                state.err(
                    f"Type {typename(func.type)} is not constant-sized (contains strings)",
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
                # Skip coercion for FppValue defaults from builtins
                if not is_instance_compat(value_expr, Ast):
                    assert is_instance_compat(func, BuiltinFuncSymbol), func
                    continue
                # Skip coercion for default values from forward-called functions.
                # These will be coerced when the function definition is visited.
                if value_expr not in state.synthesized_types:
                    continue
                arg_type = arg[1]
                # should be good 2 go based on the check func above
                state.contextual_types[value_expr] = arg_type

        # Note: if you're going to make function default arguments non-const exprs, then you will have
        # to update this line
        if all(arg in state.const_exprs for arg in node_args):
            # a function has a const value if all non-default args are const
            # (default args must be const so don't have to check them)
            state.const_exprs.add(node)
        state.synthesized_types[node] = func.return_type
        state.contextual_types[node] = func.return_type

    def visit_AstRange(self, node: AstRange, state: CompileState):
        if not self.coerce_expr_type(node.lower_bound, LoopVarType, state):
            return
        if not self.coerce_expr_type(node.upper_bound, LoopVarType, state):
            return

        if node.lower_bound in state.const_exprs and node.upper_bound in state.const_exprs:
            state.const_exprs.add(node)
        state.synthesized_types[node] = RangeValue
        state.contextual_types[node] = RangeValue

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
    def _round_float_to_type(value: float, to_type: type[FloatValue]) -> float | None:
        fmt = to_type.get_serialize_format()
        assert fmt is not None, to_type
        try:
            packed = struct.pack(fmt, value)
        except OverflowError:
            return None

        return struct.unpack(fmt, packed)[0]

    @staticmethod
    def _parse_time_string(
        time_str: str, time_base: int, time_context: int, node: Ast, state: CompileState
    ) -> TimeValue | None:
        """Parse an ISO 8601 timestamp string into a TimeValue.

        Accepts formats like:
        - "2025-12-19T14:30:00Z"
        - "2025-12-19T14:30:00.123456Z"

        Returns TimeValue with the provided time_base and time_context, and the parsed
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

            return TimeValue(time_base=time_base, time_context=time_context, seconds=seconds, useconds=useconds)

        except ValueError as e:
            state.err(
                f"Invalid time string '{time_str}': expected ISO 8601 format "
                "(e.g., '2025-12-19T14:30:00Z' or '2025-12-19T14:30:00.123456Z')",
                node,
            )
            return None

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

        state.const_expr_values[node] = expr_value

    def visit_AstGetAttr(self, node: AstGetAttr, state: CompileState):
        sym = state.resolved_symbols[node]
        if not is_symbol_an_expr(sym):
            return
        unconverted_type = state.synthesized_types[node]
        converted_type = state.contextual_types[node]
        expr_value = None
        if is_instance_compat(sym, (ChTemplate, PrmTemplate, VariableSymbol)):
            # has a value but won't try to calc at compile time
            state.const_expr_values[node] = None
            return
        elif is_instance_compat(sym, FppValue):
            expr_value = sym
        elif is_instance_compat(sym, FieldAccess):
            parent_value = state.const_expr_values[node.parent]
            if parent_value is None:
                # no compile time constant value for our parent here
                state.const_expr_values[node] = None
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

        assert is_instance_compat(parent_value, ArrayValue), parent_value

        idx = state.const_expr_values.get(node.item)
        if idx is None:
            # no compile time constant value for our index
            state.const_expr_values[node] = None
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
        state.const_expr_values[node] = expr_value

    def visit_AstIdent(self, node: AstIdent, state: CompileState):
        sym = state.resolved_symbols[node]
        if not is_symbol_an_expr(sym):
            return
        unconverted_type = state.synthesized_types[node]
        converted_type = state.contextual_types[node]
        expr_value = None
        if is_instance_compat(sym, (ChTemplate, PrmTemplate, VariableSymbol)):
            # Has a value but we don't try to calculate it at compile time.
            # NOTE: If you ever add const-folding for VariableSymbol here, you must also
            # update CalculateDefaultArgConstValues. That pass runs CalculateConstExprValues
            # on default argument expressions BEFORE this pass runs on variable assignments.
            # So if a default value references a variable, the variable's const value won't
            # be available yet, and the default value will incorrectly be rejected as non-const.
            state.const_expr_values[node] = None
            return
        elif is_instance_compat(sym, FppValue):
            expr_value = sym
        else:
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
        state.const_expr_values[node] = expr_value

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        func = state.resolved_symbols[node.func]
        assert is_instance_compat(func, CallableSymbol)

        # Use resolved args from semantic analysis (already in positional order,
        # with defaults filled in)
        # This is guaranteed to be set by PickTypesAndResolveAttrsAndItems
        resolved_args = state.resolved_func_args[node]

        # Gather arg values. Since defaults are already filled in, we just need
        # to look up each arg's const value. For FppValue defaults from builtins,
        # use the value directly.
        arg_values = []
        for arg_expr in resolved_args:
            if is_instance_compat(arg_expr, Ast):
                arg_values.append(state.const_expr_values.get(arg_expr))
            else:
                # It's a raw FppValue default from a builtin
                arg_values.append(arg_expr)

        unknown_value = any(v is None for v in arg_values)
        if unknown_value:
            # we will have to calculate this at runtime
            state.const_expr_values[node] = None
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
        elif is_instance_compat(func, BuiltinFuncSymbol) and func.name == "time":
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

        state.const_expr_values[node] = expr_value

    def visit_AstBinaryOp(self, node: AstBinaryOp, state: CompileState):
        # Check if both left-hand side (lhs) and right-hand side (rhs) are constants
        lhs_value: FppValue = state.const_expr_values.get(node.lhs)
        rhs_value: FppValue = state.const_expr_values.get(node.rhs)

        if lhs_value is None or rhs_value is None:
            state.const_expr_values[node] = None
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
        state.const_expr_values[node] = folded_value

    def visit_AstUnaryOp(self, node: AstUnaryOp, state: CompileState):
        value: FppValue = state.const_expr_values.get(node.val)

        if value is None:
            state.const_expr_values[node] = None
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
        state.const_expr_values[node] = folded_value

    def visit_AstRange(self, node: AstRange, state: CompileState):
        # ranges don't really end up having a value, they kinda just exist as a type
        state.const_expr_values[node] = None

    def visit_default(self, node, state):
        # coding error, missed an expr
        assert not is_instance_compat(node, AstExpr), node


class CheckAllBranchesReturn(Visitor):
    def visit_AstReturn(self, node: AstReturn, state: CompileState):
        state.does_return[node] = True

    def visit_AstBlock(
        self, node: AstBlock, state: CompileState
    ):
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
        lower_value: LoopVarType = state.const_expr_values.get(node.lower_bound)
        upper_value: LoopVarType = state.const_expr_values.get(node.upper_bound)
        if lower_value is None or upper_value is None:
            # cannot check at compile time
            return

        if lower_value.val >= upper_value.val:
            state.warn("Range is empty", node)
