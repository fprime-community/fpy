from __future__ import annotations
import typing
from typing import Callable, Union, get_args, get_origin
from dataclasses import dataclass, field
from enum import Enum

from fpy.error import CompileError
from fpy.ir import Ir, IrLabel
from fpy.syntax import (
    Ast,
    AstBreak,
    AstContinue,
    AstDef,
    AstExpr,
    AstFor,
    AstFuncCall,
    AstGetAttr,
    AstIndexExpr,
    AstOp,
    AstReference,
    AstReturn,
    AstBlock,
    AstWhile,
)
from fpy.types import (
    DEFAULT_MAX_DIRECTIVES_COUNT,
    DEFAULT_MAX_DIRECTIVE_SIZE,
    FpyType,
    FpyValue,
    NOTHING,
    CmdDef,
    ChDef,
    PrmDef,
    is_instance_compat,
)
from fpy.bytecode.directives import Directive


@dataclass
class CallableSymbol:
    name: str
    return_type: FpyType
    # args is a list of (name, type, default_value) tuples
    # default_value is an AstExpr or None if no default is provided
    args: list[tuple[str, FpyType, AstExpr | None]]


@dataclass
class CommandSymbol(CallableSymbol):
    cmd: CmdDef


@dataclass
class BuiltinFuncSymbol(CallableSymbol):
    generate: Callable[[AstFuncCall, dict[int, FpyValue]], list[Directive]]
    """a function which instantiates the builtin given the calling node and
    a dict mapping const_arg_indices to their compile-time values"""
    const_arg_indices: frozenset[int] = field(default_factory=frozenset)
    """indices of args that must be compile-time constants and are NOT pushed
    to the stack; instead their values are passed to generate()"""


@dataclass
class FunctionSymbol(CallableSymbol):
    definition: AstDef


@dataclass
class TypeCtorSymbol(CallableSymbol):
    type: FpyType


@dataclass
class CastSymbol(CallableSymbol):
    to_type: FpyType


@dataclass
class FieldAccess:
    """a reference to a member/element of an fprime struct/array type"""

    parent_expr: AstExpr
    """the complete qualifier"""
    base_sym: Union[Symbol, None]
    """the base symbol, up through all the layers of field symbols, or None if parent at some point is not a symbol at all"""
    type: FpyType
    """the fprime type of this reference"""
    is_struct_member: bool = False
    """True if this is a struct member reference"""
    is_array_element: bool = False
    """True if this is an array element reference"""
    base_offset: int = None
    """the constant offset in the base symbol type, or None if unknown at compile time"""
    local_offset: int = None
    """the constant offset in the parent type at which to find this field
    or None if unknown at compile time"""
    name: str = None
    """the name of the field, if applicable"""
    idx_expr: AstExpr = None
    """the expression that evaluates to the index in the parent array of the field, if applicable"""


# named variables can be tlm chans, prms, callables, or directly referenced consts (usually enums)
@dataclass
class VariableSymbol:
    """a mutable, typed value stored on the stack referenced by an unqualified name"""

    name: str
    type_ref: AstExpr | None
    """the AST node denoting the var's type"""
    declaration: Ast
    """the node where this var is declared"""
    type: FpyType | None = None
    """the resolved type of the variable. None if type unsure at the moment"""
    frame_offset: int | None = None
    """the offset in the lvar array where this var is stored"""
    is_global: bool = False
    """whether this variable is a top-level (global) variable"""


@dataclass
class ForLoopAnalysis:
    loop_var: VariableSymbol
    upper_bound_var: VariableSymbol


next_symbol_table_id = 0


class NameGroup(str, Enum):
    TYPE = "type"
    CALLABLE = "callable"
    VALUE = "value"


class SymbolTable(dict):
    def __init__(self, parent: SymbolTable | None = None):
        global next_symbol_table_id
        super().__init__()
        self.id = next_symbol_table_id
        next_symbol_table_id += 1
        self.parent = parent
        self.in_function = parent.in_function if parent is not None else False

    def __getitem__(self, key: str) -> Symbol:
        return super().__getitem__(key)

    def get(self, key) -> Symbol | None:
        return super().get(key, None)

    def lookup(self, key: str) -> Symbol | None:
        """Look up a key in this scope and all ancestor scopes."""
        val = self.get(key)
        if val is not None:
            return val
        if self.parent is not None:
            return self.parent.lookup(key)
        return None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, value):
        return isinstance(value, SymbolTable) and value.id == self.id

    def copy(self):
        """Return a shallow copy that preserves SymbolTable metadata."""
        new = SymbolTable(parent=self.parent)
        new.in_function = self.in_function
        new.update(self)
        return new


def create_symbol_table(
    symbols: dict[str, Symbol]
) -> SymbolTable:
    """from a flat dict of strs to symbols, creates a hierarchical symbol table.
    no two leaf nodes may have the same name"""

    base = SymbolTable()

    for fqn, sym in symbols.items():
        names_strs = fqn.split(".")

        ns = base
        while len(names_strs) > 1:
            existing_child = ns.get(names_strs[0])
            if existing_child is None:
                # this symbol table is not defined atm
                existing_child = SymbolTable()
                ns[names_strs[0]] = existing_child

            if not isinstance(existing_child, dict):
                # something else already has this name
                break

            ns = existing_child
            names_strs = names_strs[1:]

        if len(names_strs) != 1:
            # broke early. skip this loop
            continue

        # okay, now ns is the complete scope of the attribute
        # i.e. everything up until the last '.'
        name = names_strs[0]

        existing_child = ns.get(name)

        if existing_child is not None:
            # uh oh, something already had this name with a diff value
            continue

        ns[name] = sym

    return base


def merge_symbol_tables(lhs: SymbolTable, rhs: SymbolTable) -> SymbolTable:
    """returns the two symbol tables, joined into one. if there is a conflict, chooses lhs over rhs"""
    lhs_keys = set(lhs.keys())
    rhs_keys = set(rhs.keys())
    common_keys = lhs_keys.intersection(rhs_keys)

    only_lhs_keys = lhs_keys.difference(common_keys)
    only_rhs_keys = rhs_keys.difference(common_keys)

    new = SymbolTable()

    for key in common_keys:
        if not isinstance(lhs[key], dict) or not isinstance(rhs[key], dict):
            # cannot be merged cleanly. one of the two is not a symbol table
            new[key] = lhs[key]
            continue

        new[key] = merge_symbol_tables(lhs[key], rhs[key])

    for key in only_lhs_keys:
        new[key] = lhs[key]
    for key in only_rhs_keys:
        new[key] = rhs[key]

    return new


def is_symbol_an_expr(symbol: Symbol) -> bool:
    """return True if the symbol is a valid expr (can be evaluated)"""
    return is_instance_compat(
        symbol,
        (
            ChDef,
            PrmDef,
            FpyValue,
            VariableSymbol,
            FieldAccess
        ),
    )

Symbol = typing.Union[
    ChDef,
    PrmDef,
    FpyValue,
    CallableSymbol,
    FpyType,
    VariableSymbol,
    SymbolTable,
    FieldAccess
]
"""a named entity in fpy that can be looked up in a symbol table"""


@dataclass
class CompileState:
    """a collection of input, internal and output state variables and maps"""

    global_type_scope: SymbolTable
    """The global type scope: a symbol table whose leaf nodes are FpyType instances."""
    global_callable_scope: SymbolTable
    """The global callable scope: a symbol table whose leaf nodes are CallableSymbol instances."""
    global_value_scope: SymbolTable
    """The global value scope: a symbol table whose leaf nodes are runtime values
    (telemetry channels, parameters, enum constants, variables)."""

    compile_args: dict = field(default_factory=dict)
    
    # Sequence limits loaded from dictionary (or defaults if not specified)
    max_directives_count: int = DEFAULT_MAX_DIRECTIVES_COUNT
    max_directive_size: int = DEFAULT_MAX_DIRECTIVE_SIZE

    next_node_id: int = 0
    root: AstBlock = None
    enclosing_value_scope: dict[Ast, SymbolTable] = field(
        default_factory=dict, repr=False
    )
    """map of node to its enclosing value scope (block scope, function scope, or global_value_scope)"""
    for_loops: dict[AstFor, ForLoopAnalysis] = field(default_factory=dict)
    """map of for loops to a ForLoopAnalysis struct, which contains additional info about the loops"""
    enclosing_loops: dict[Union[AstBreak, AstContinue], Union[AstFor, AstWhile]] = (
        field(default_factory=dict)
    )
    """map of break/continue to the loop which contains the break/continue"""
    desugared_for_loops: dict[AstWhile, AstFor] = field(default_factory=dict)
    """mapping of while loops which are desugared for loops, to the original node from which they came"""

    enclosing_funcs: dict[AstReturn, AstDef] = field(default_factory=dict)

    resolved_symbols: dict[AstReference, Symbol] = field(
        default_factory=dict, repr=False
    )
    """reference to its singular resolution"""

    synthesized_types: dict[AstExpr, FpyType] = field(
        default_factory=dict
    )
    """expr to its fprime type, before type conversions are applied"""

    op_intermediate_types: dict[AstOp, FpyType] = field(default_factory=dict)
    """the intermediate type that all args should be converted to for the given op"""

    expr_explicit_casts: list[AstExpr] = field(default_factory=list)
    """a list of nodes which are explicit casts"""
    contextual_types: dict[AstExpr, FpyType] = field(default_factory=dict)
    """expr to fprime type it will end up being on the stack after type conversions"""

    const_expr_values: dict[AstExpr, FpyValue | None] = field(
        default_factory=dict
    )
    """expr to the fprime value it will end up being on the stack after type conversions.
    None if unsure at compile time.  NOTHING_VALUE for void expressions."""

    resolved_func_args: dict[AstFuncCall, list[AstExpr]] = field(default_factory=dict)
    """function call to resolved arguments in positional order.
    Default values are filled in for arguments not provided at the call site."""

    while_loop_end_labels: dict[AstWhile, IrLabel] = field(default_factory=dict)
    """while loop node mapped to the label pointing to the end of the loop"""
    while_loop_start_labels: dict[AstWhile, IrLabel] = field(default_factory=dict)
    """while loop node mapped to the label pointing to the start of the loop, just before the conditional"""
    # store keys as while because for loops are desugared to while
    for_loop_inc_labels: dict[AstWhile, IrLabel] = field(default_factory=dict)
    """for loop node (desugared into a while) mapped to a label pointing to its increment stmt"""

    does_return: dict[Ast, bool] = field(default_factory=dict)

    used_funcs: set[AstDef] = field(default_factory=set)
    """set of function definitions that are actually called and need code generated"""

    func_entry_labels: dict[AstDef, IrLabel] = field(default_factory=dict)
    """function to entry point label"""

    generated_funcs: dict[AstDef, list[Directive | Ir]] = field(default_factory=dict)

    frame_sizes: dict[Ast, int] = field(default_factory=dict)
    """map of frame root node to the total size in bytes of all local variables in that frame"""

    errors: list[CompileError] = field(default_factory=list)
    """a list of all compile exceptions generated by passes"""

    warnings: list[CompileError] = field(default_factory=list)
    """a list of all compiler warnings generated by passes"""

    next_anon_var_id: int = 0

    def new_anonymous_variable_name(self) -> str:
        id = self.next_anon_var_id
        self.next_anon_var_id += 1
        return f"$value{id}"

    def err(self, msg, n):
        """adds a compile exception to internal state"""
        self.errors.append(CompileError(msg, n))

    def warn(self, msg, n):
        self.warnings.append(CompileError("Warning: " + msg, n))
