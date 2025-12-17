from __future__ import annotations
import copy
from fpy.bytecode.directives import BinaryStackOp, Directive, LoopVarType
from fpy.syntax import (
    Ast,
    AstAssign,
    AstBinaryOp,
    AstBlock,
    AstBoolean,
    AstBreak,
    AstCheck,
    AstFor,
    AstFuncCall,
    AstIf,
    AstMemberAccess,
    AstNumber,
    AstRange,
    AstStmtList,
    AstTypeExpr,
    AstUnaryOp,
    AstVar,
    AstWhile,
)
from fpy.types import (
    CompileState,
    ForLoopAnalysis,
    FppType,
    Symbol,
    FpyIntegerValue,
    Transformer,
    TopDownVisitor,
    is_instance_compat,
    lookup_symbol,
)
from fprime_gds.common.models.serialize.type_base import BaseType as FppValue
from fprime_gds.common.models.serialize.bool_type import BoolType as BoolValue


class DesugarForLoops(Transformer):

    # this function forces you to give values for all of these dicts
    # really the point is to make sure i don't forget to consider this
    # for one of the new nodes
    def new(
        self,
        state: CompileState,
        node: Ast,
        contextual_type: FppType | None,
        synthesized_type: FppType | None,
        contextual_value: FppValue | None,
        op_intermediate_type: type[Directive] | None,
        resolved_symbol: Symbol | None,
    ) -> Ast:
        node.id = state.next_node_id
        state.next_node_id += 1
        state.contextual_types[node] = contextual_type
        state.synthesized_types[node] = synthesized_type
        state.contextual_values[node] = contextual_value
        state.op_intermediate_types[node] = op_intermediate_type
        state.resolved_symbols[node] = resolved_symbol
        return node

    def initialize_loop_var(
        self, state: CompileState, loop_node: AstFor, loop_info: ForLoopAnalysis
    ) -> Ast:
        # 1 <node.loop_var>: LoopVarType = <node.range.lower_bound>
        # OR (depending on whether redeclaring or not)
        # 1 <node.loop_var> = <node.range.lower_bound>

        if loop_info.reuse_existing_loop_var:
            loop_var_type_var = None
        else:
            loop_var_type_name = LoopVarType.get_canonical_name()
            # create a new node for the type_ann
            loop_var_type_var = self.new(
                state,
                AstTypeExpr(None, [loop_var_type_name]),
                contextual_type=None,
                synthesized_type=None,
                contextual_value=None,
                op_intermediate_type=None,
                resolved_symbol=LoopVarType,
            )

        lhs = loop_node.loop_var
        rhs = loop_node.range.lower_bound
        return self.new(
            state,
            AstAssign(None, lhs, loop_var_type_var, rhs),
            contextual_type=None,
            synthesized_type=None,
            contextual_value=None,
            op_intermediate_type=None,
            resolved_symbol=None,
        )

    def declare_upper_bound_var(
        self, state: CompileState, loop_node: AstFor, loop_info: ForLoopAnalysis
    ) -> Ast:
        # 2 $upper_bound_var: LoopVarType = <node.range.upper_bound>

        # ub var for use in assignment
        # type is gonna be loop var type
        # gonna be used in the astassign lhs, no need assign a type in the dict
        upper_bound_var: AstVar = self.new(
            state,
            AstVar(None, loop_info.upper_bound_var.name),
            contextual_type=None,
            synthesized_type=None,
            contextual_value=None,
            op_intermediate_type=None,
            resolved_symbol=loop_info.upper_bound_var,
        )

        loop_var_type_name = LoopVarType.get_canonical_name()
        # create a new node for the type_ann
        loop_var_type_var = self.new(
            state,
            AstTypeExpr(None, [loop_var_type_name]),
            contextual_type=None,
            synthesized_type=None,
            contextual_value=None,
            op_intermediate_type=None,
            resolved_symbol=LoopVarType,
        )

        # assign ub to ub var
        # not an expr, not a symbol
        return self.new(
            state,
            AstAssign(
                None, upper_bound_var, loop_var_type_var, loop_node.range.upper_bound
            ),
            contextual_type=None,
            synthesized_type=None,
            contextual_value=None,
            op_intermediate_type=None,
            resolved_symbol=None,
        )

    def loop_var_plus_one(
        self, state: CompileState, loop_node: AstFor, loop_info: ForLoopAnalysis
    ):
        # <node.loop_var> + 1
        # the expression adding one to the lv
        # will have conv type of lv, unconverted type depends what the addition intermediate type is
        # we've already determined the dir
        lhs = self.new(
            state,
            AstVar(None, loop_info.loop_var.name),
            contextual_type=LoopVarType,
            synthesized_type=LoopVarType,
            contextual_value=None,
            op_intermediate_type=None,
            resolved_symbol=loop_info.loop_var,
        )
        rhs = self.new(
            state,
            AstNumber(None, 1),
            contextual_type=LoopVarType,
            synthesized_type=FpyIntegerValue,
            contextual_value=LoopVarType(1),
            op_intermediate_type=None,
            resolved_symbol=None,
        )

        return self.new(
            state,
            AstBinaryOp(None, lhs, BinaryStackOp.ADD, rhs),
            contextual_type=LoopVarType,
            synthesized_type=LoopVarType,
            contextual_value=None,
            op_intermediate_type=LoopVarType,
            resolved_symbol=None,
        )

    def increment_loop_var(
        self, state: CompileState, loop_node: AstFor, loop_info: ForLoopAnalysis
    ) -> Ast:
        # <node.loop_var> = <node.loop_var> + 1

        # create a new loop var symbol for use in lhs of loop var inc
        lhs = self.new(
            state,
            AstVar(None, loop_info.loop_var.name),
            contextual_type=None,
            synthesized_type=None,
            contextual_value=None,
            op_intermediate_type=None,
            resolved_symbol=loop_info.loop_var,
        )

        rhs = self.loop_var_plus_one(state, loop_node, loop_info)

        return self.new(
            state,
            AstAssign(None, lhs, None, rhs),
            contextual_type=None,
            synthesized_type=None,
            contextual_value=None,
            op_intermediate_type=None,
            resolved_symbol=None,
        )

    def while_loop_condition(
        self, state: CompileState, loop_node: AstFor, loop_info: ForLoopAnalysis
    ) -> Ast:
        # <node.loop_var> < $upper_bound_var
        # create a new loop var symbol for use in lhs
        lhs = self.new(
            state,
            AstVar(None, loop_info.loop_var.name),
            contextual_type=LoopVarType,
            synthesized_type=LoopVarType,
            contextual_value=None,
            op_intermediate_type=None,
            resolved_symbol=loop_info.loop_var,
        )
        rhs = self.new(
            state,
            AstVar(None, loop_info.upper_bound_var.name),
            contextual_type=LoopVarType,
            synthesized_type=LoopVarType,
            contextual_value=None,
            op_intermediate_type=None,
            resolved_symbol=loop_info.upper_bound_var,
        )

        return self.new(
            state,
            AstBinaryOp(None, lhs, BinaryStackOp.LESS_THAN, rhs),
            contextual_type=BoolValue,
            synthesized_type=BoolValue,
            contextual_value=None,
            op_intermediate_type=LoopVarType,
            resolved_symbol=None,
        )

    def while_loop(
        self, state: CompileState, loop_node: AstFor, loop_info: ForLoopAnalysis
    ) -> Ast:
        #  while <node.loop_var> < $upper_bound_var:
        #     <node.body>
        #     <node.loop_var> = <node.loop_var> + 1

        condition = self.while_loop_condition(state, loop_node, loop_info)
        increment = self.increment_loop_var(state, loop_node, loop_info)

        body = loop_node.body

        body.stmts.append(increment)

        return self.new(
            state,
            AstWhile(None, condition, body),
            contextual_type=None,
            synthesized_type=None,
            contextual_value=None,
            op_intermediate_type=None,
            resolved_symbol=None,
        )

    def visit_AstFor(self, node: AstFor, state: CompileState):
        assert isinstance(node.range, AstRange), node.range

        # transform this:

        # for <node.loop_var> in <node.range>:
        #     <node.body>

        # to:

        # 1 <node.loop_var>: LoopVarType = <node.range.lower_bound>
        # OR (depending on whether redeclaring or not)
        # 1 <node.loop_var> = <node.range.lower_bound>
        # 2 $upper_bound_var: LoopVarType = <node.range.upper_bound>
        # 3 while <node.loop_var> < $upper_bound_var:
        #      <node.body>
        #      <node.loop_var> = <node.loop_var> + 1

        loop_info = state.for_loops[node]

        # 1
        initialize_loop_var = self.initialize_loop_var(state, node, loop_info)
        # 2
        declare_upper_bound_var = self.declare_upper_bound_var(state, node, loop_info)
        # 3
        while_loop: AstWhile = self.while_loop(state, node, loop_info)

        # this is the first and so far only piece of code in the compiler itself written by AI
        # Update any break/continue statements in the body to point to new while loop
        # instead of the original for loop
        for key, value in list(state.enclosing_loops.items()):
            if value == node:  # If a break/continue was pointing to our for loop
                state.enclosing_loops[key] = (
                    while_loop  # Point it to while loop instead
                )

        state.desugared_for_loops[while_loop] = node

        # turn one node into three
        return [initialize_loop_var, declare_upper_bound_var, while_loop]


class DesugarDefaultArgs(Transformer):
    """
    Desugars function calls with named or missing arguments by:
    1. Reordering named arguments to positional order
    2. Filling in default values for missing arguments

    For example, if we have:
        def foo(a: U8, b: U8 = 5, c: U8 = 10):
            pass
        foo(c=15, a=1)

    This becomes:
        foo(1, 5, 15)

    Note: The type coercion for default values is handled during semantic analysis
    in PickTypesAndResolveAttrsAndItems.visit_AstDef. By the time this desugaring
    runs, contextual_types already has the correct coerced types for default
    value expressions.
    """

    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        # Get the resolved arguments from semantic analysis.
        # This list is already in positional order with defaults filled in.
        resolved_args = state.resolved_func_args.get(node)
        assert resolved_args is not None, (
            f"No resolved args for function call {node}. "
            f"This should have been set by PickTypesAndResolveAttrsAndItems."
        )

        # Update the node's args with the resolved arguments
        node.args = resolved_args

        return node


class ResolveTimeoutPlaceholders(Transformer):
    """
    Resolves $timeout_to_absolute(timeout_expr) placeholder calls.
    
    After semantic analysis, we know the type of timeout_expr:
    - If Fw.Time (absolute): replace with just timeout_expr
    - If Fw.TimeIntervalValue (relative): replace with time_add(now(), timeout_expr)
    
    This enables check statements to accept both absolute and relative timeouts
    while keeping the early desugaring simple.
    """
    
    def visit_AstFuncCall(self, node: AstFuncCall, state: CompileState):
        from fpy.types import BuiltinFuncSymbol
        from fprime_gds.common.models.serialize.time_type import TimeType as TimeValue
        
        func = state.resolved_symbols.get(node.func)
        if not is_instance_compat(func, BuiltinFuncSymbol):
            return node
        if func.name != "$timeout_to_absolute":
            return node
        
        # Get the timeout argument
        timeout_arg = node.args[0]
        timeout_type = state.synthesized_types.get(timeout_arg)
        
        # Check if it's absolute (Fw.Time) or relative (Fw.TimeIntervalValue)
        is_absolute = issubclass(timeout_type, TimeValue)
        
        if is_absolute:
            # Just return the argument directly - it's already an absolute time
            return timeout_arg
        else:
            # It's relative (TimeIntervalValue), wrap with time_add(now(), timeout)
            # We need to find the time_add function and now function
            time_add_func = lookup_symbol(node, "time_add", state)
            now_func = state.callables.get("now")
            
            assert time_add_func is not None, "time_add function not found"
            assert now_func is not None, "now function not found"
            
            # Create now() call
            now_var = AstVar(None, "now")
            now_var.id = state.next_node_id
            state.next_node_id += 1
            state.resolved_symbols[now_var] = now_func
            
            now_call = AstFuncCall(None, now_var, [])
            now_call.id = state.next_node_id
            state.next_node_id += 1
            state.resolved_symbols[now_call] = now_func
            state.resolved_func_args[now_call] = []
            state.synthesized_types[now_call] = TimeValue
            state.contextual_types[now_call] = TimeValue
            
            # Create time_add(now(), timeout) call
            time_add_var = AstVar(None, "time_add")
            time_add_var.id = state.next_node_id
            state.next_node_id += 1
            state.resolved_symbols[time_add_var] = time_add_func
            
            time_add_call = AstFuncCall(None, time_add_var, [now_call, timeout_arg])
            time_add_call.id = state.next_node_id
            state.next_node_id += 1
            state.resolved_symbols[time_add_call] = time_add_func
            state.resolved_func_args[time_add_call] = [now_call, timeout_arg]
            state.synthesized_types[time_add_call] = TimeValue
            state.contextual_types[time_add_call] = TimeValue
            
            return time_add_call


class DesugarCheckStatements:
    """
    Desugars check statements into while loops BEFORE semantic analysis.
    
    This runs before AssignIds, so we don't need to worry about node IDs.
    The generated AST nodes will go through normal semantic analysis.
    
    A check statement:
        check <condition> timeout <timeout> persist <persist> every <every>:
            <body>
        timeout:
            <timeout_body>
    
    Gets desugared into (roughly):
        $check_state: $CheckState = $CheckState(...)
        while True:
            $current_time: Fw.Time = now()
            $timed_out: I8 = time_cmp($current_time, $check_state.timeout)
            assert $timed_out != 2, 1
            if $timed_out == 1:
                break
            if <condition>:
                if not $check_state.last_was_true:
                    $check_state.last_time_true = $current_time
                    $check_state.last_was_true = True
                $succeeded: I8 = time_interval_cmp(time_sub($current_time, $check_state.last_time_true), $check_state.persist)
                assert $succeeded != 2, 1
                if $succeeded == 1 or $succeeded == 0:
                    $check_state.result = True
                    break
            else:
                $check_state.last_was_true = False
            sleep($check_state.every.seconds, $check_state.every.useconds)
        if $check_state.result:
            <body>
        else:
            <timeout_body>
    """
    
    def __init__(self):
        self.var_counter = 0
        self.meta = None  # Will be set when desugaring each check statement
    
    def new_var_name(self) -> str:
        """Generate a unique anonymous variable name."""
        name = f"$check{self.var_counter}"
        self.var_counter += 1
        return name
    
    def var(self, name: str) -> AstVar:
        """Create a variable reference node."""
        return AstVar(self.meta, name)
    
    def type_expr(self, *parts: str) -> AstTypeExpr:
        """Create a type expression node."""
        return AstTypeExpr(self.meta, list(parts))
    
    def number(self, val: int) -> AstNumber:
        """Create a number literal node."""
        return AstNumber(self.meta, val)
    
    def boolean(self, val: bool) -> AstBoolean:
        """Create a boolean literal node."""
        return AstBoolean(self.meta, val)
    
    def member(self, parent, attr: str) -> AstMemberAccess:
        """Create a member access node."""
        return AstMemberAccess(self.meta, parent, attr)
    
    def callable_ref(self, *parts: str):
        """Create a callable reference from parts (e.g., 'Fw', 'Time' -> Fw.Time).
        
        For function calls, we need AstVar or AstMemberAccess, not AstTypeExpr.
        """
        if len(parts) == 0:
            raise ValueError("callable_ref requires at least one part")
        
        result = self.var(parts[0])
        for part in parts[1:]:
            result = self.member(result, part)
        return result
    
    def call(self, func_name: str, *args) -> AstFuncCall:
        """Create a function call node."""
        func = self.var(func_name)
        return AstFuncCall(self.meta, func, list(args) if args else [])
    
    def call_parts(self, func_parts: list[str], *args) -> AstFuncCall:
        """Create a function call node with a multi-part function name."""
        func = self.callable_ref(*func_parts)
        return AstFuncCall(self.meta, func, list(args) if args else [])
    
    def call_expr(self, func_expr, *args) -> AstFuncCall:
        """Create a function call node with an expression as the function."""
        return AstFuncCall(self.meta, func_expr, list(args) if args else [])
    
    def binop(self, lhs, op: str, rhs) -> AstBinaryOp:
        """Create a binary operation node."""
        return AstBinaryOp(self.meta, lhs, op, rhs)
    
    def unary(self, op: str, val) -> AstUnaryOp:
        """Create a unary operation node."""
        return AstUnaryOp(self.meta, op, val)
    
    def assign(self, lhs, rhs, type_ann=None) -> AstAssign:
        """Create an assignment node."""
        return AstAssign(self.meta, lhs, type_ann, rhs)
    
    def stmt_list(self, *stmts) -> AstStmtList:
        """Create a statement list node."""
        return AstStmtList(self.meta, list(stmts))
    
    def if_stmt(self, cond, body_stmts, else_stmts=None) -> AstIf:
        """Create an if statement node."""
        body = self.stmt_list(*body_stmts)
        els = self.stmt_list(*else_stmts) if else_stmts else None
        return AstIf(self.meta, cond, body, [], els)
    
    def while_stmt(self, cond, body_stmts) -> AstWhile:
        """Create a while statement node."""
        body = self.stmt_list(*body_stmts)
        return AstWhile(self.meta, cond, body)
    
    def break_stmt(self) -> AstBreak:
        """Create a break statement node."""
        return AstBreak(self.meta)
    
    def desugar_check(self, node: AstCheck) -> list[Ast]:
        """
        Desugar a single check statement into a list of statements.
        """
        # Use the check node's meta for all generated nodes (for error reporting)
        self.meta = node.meta
        
        # Generate unique variable names for this check statement
        check_state_name = self.new_var_name()
        current_time_name = self.new_var_name()
        timed_out_name = self.new_var_name()
        succeeded_name = self.new_var_name()
        
        # Helper to reference check_state members
        def cs(attr: str):
            return self.member(self.var(check_state_name), attr)
        
        # Build the CheckState constructor call
        # $CheckState(persist=<persist>, timeout=<timeout_or_time_add>, every=<every>, 
        #             result=False, last_was_true=False, last_time_true=Fw.Time(0,0,0,0), time_started=now())
        
        # For timeout, we use a placeholder function $timeout_to_absolute that will be
        # resolved in a late pass after semantic analysis determines the type.
        # If timeout_expr is Fw.Time (absolute), it gets used directly.
        # If timeout_expr is Fw.TimeIntervalValue (relative), it becomes time_add(now(), timeout_expr).
        timeout_expr_to_use = self.call("$timeout_to_absolute", copy.deepcopy(node.timeout))
        
        check_state_init = self.call_expr(
            self.callable_ref("$CheckState"),               # Use callable_ref, not type_expr
            copy.deepcopy(node.persist),                    # persist
            timeout_expr_to_use,                            # timeout (converted to absolute)
            copy.deepcopy(node.every),                      # every
            self.boolean(False),                            # result
            self.boolean(False),                            # last_was_true
            self.call_parts(                                # last_time_true = Fw.Time(0,0,0,0)
                ["Fw", "Time"],
                self.number(0), self.number(0), self.number(0), self.number(0)
            ),
            self.call("now"),                               # time_started
        )
        
        # 1. $check_state: $CheckState = $CheckState(...)
        init_check_state = self.assign(
            self.var(check_state_name),
            check_state_init,
            self.type_expr("$CheckState")
        )
        
        # Build the while loop body
        # 2. $current_time: Fw.Time = now()
        get_current_time = self.assign(
            self.var(current_time_name),
            self.call("now"),
            self.type_expr("Fw", "Time")
        )
        
        # 3. $timed_out: I8 = time_cmp($current_time, $check_state.timeout)
        check_timeout = self.assign(
            self.var(timed_out_name),
            self.call("time_cmp", self.var(current_time_name), cs("timeout")),
            self.type_expr("I8")
        )
        
        # 4. assert $timed_out != 2, 1
        from fpy.syntax import AstAssert
        assert_comparable = AstAssert(
            self.meta,
            self.binop(self.var(timed_out_name), "!=", self.number(2)),
            self.number(1)
        )
        
        # 5. if $timed_out == 1: break
        timeout_break = self.if_stmt(
            self.binop(self.var(timed_out_name), "==", self.number(1)),
            [self.break_stmt()]
        )
        
        # 6. Build the condition check block
        # if <condition>:
        #     if not $check_state.last_was_true:
        #         $check_state.last_time_true = $current_time
        #         $check_state.last_was_true = True
        #     $succeeded: I8 = time_interval_cmp(time_sub($current_time, $check_state.last_time_true), $check_state.persist)
        #     assert $succeeded != 2, 1
        #     if $succeeded == 1 or $succeeded == 0:
        #         $check_state.result = True
        #         break
        # else:
        #     $check_state.last_was_true = False
        
        # Inner if: not last_was_true
        update_last_true = self.if_stmt(
            self.unary("not", cs("last_was_true")),
            [
                self.assign(cs("last_time_true"), self.var(current_time_name)),
                self.assign(cs("last_was_true"), self.boolean(True)),
            ]
        )
        
        # Check if condition has persisted long enough
        check_persist = self.assign(
            self.var(succeeded_name),
            self.call(
                "time_interval_cmp",
                self.call("time_sub", self.var(current_time_name), cs("last_time_true")),
                cs("persist")
            ),
            self.type_expr("I8")
        )
        
        assert_persist_comparable = AstAssert(
            self.meta,
            self.binop(self.var(succeeded_name), "!=", self.number(2)),
            self.number(1)
        )
        
        # if succeeded >= 0 (i.e., succeeded == 0 or succeeded == 1)
        success_check = self.if_stmt(
            self.binop(
                self.binop(self.var(succeeded_name), "==", self.number(1)),
                "or",
                self.binop(self.var(succeeded_name), "==", self.number(0))
            ),
            [
                self.assign(cs("result"), self.boolean(True)),
                self.break_stmt(),
            ]
        )
        
        # Main condition check if/else
        condition_check = self.if_stmt(
            copy.deepcopy(node.condition),
            [update_last_true, check_persist, assert_persist_comparable, success_check],
            [self.assign(cs("last_was_true"), self.boolean(False))]
        )
        
        # 7. sleep($check_state.every.seconds, $check_state.every.useconds)
        sleep_call = self.call(
            "sleep",
            self.member(cs("every"), "seconds"),
            self.member(cs("every"), "useconds")
        )
        
        # Build the while loop
        while_loop = self.while_stmt(
            self.boolean(True),
            [
                get_current_time,
                check_timeout,
                assert_comparable,
                timeout_break,
                condition_check,
                sleep_call,
            ]
        )
        
        # 8. Final if/else to run body or timeout_body
        # if $check_state.result:
        #     <body>
        # else:
        #     <timeout_body>
        final_if = AstIf(
            self.meta,
            cs("result"),
            node.body,           # Use original body
            [],                  # No elifs
            node.timeout_body    # Use original timeout_body
        )
        
        return [init_check_state, while_loop, final_if]
    
    def run(self, body: AstBlock):
        """
        Run the desugaring pass to transform all check statements.
        
        Args:
            body: The user's AST root block (with builtin functions already prepended)
        """
        self._desugar_all_checks(body)
    
    def _desugar_all_checks(self, node):
        """Recursively find and desugar all check statements."""
        if not isinstance(node, Ast):
            return
        
        from dataclasses import fields
        
        for field in fields(node):
            val = getattr(node, field.name)
            if isinstance(val, list):
                new_list = []
                for item in val:
                    if isinstance(item, AstCheck):
                        # Replace check statement with desugared statements
                        new_list.extend(self.desugar_check(item))
                    else:
                        self._desugar_all_checks(item)
                        new_list.append(item)
                setattr(node, field.name, new_list)
            else:
                self._desugar_all_checks(val)
