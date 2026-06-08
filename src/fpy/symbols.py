from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Union
import typing

from fpy.bytecode.directives import Directive
from fpy.syntax import Ast, AstDef, AstExpr, AstFuncCall
from fpy.types import ChDef, CmdDef, FpyType, FpyValue, PrmDef, is_instance_compat


@dataclass
class CallableSymbol:
    name: str
    return_type: FpyType
    # args is a list of (name, type, default_value) tuples
    # default_value is AstExpr for user-defined functions, FpyValue for builtin funcs
    # including constructors, or None if no default is provided.
    args: list[tuple[str, FpyType, AstExpr | FpyValue | None]]


@dataclass
class CommandSymbol(CallableSymbol):
    cmd: CmdDef
    is_seq_run_with_args: bool = False


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
    base_sym: Union["Symbol", None]
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



next_symbol_table_id = 0


class NameGroup(str, Enum):
    TYPE = "type"
    CALLABLE = "callable"
    VALUE = "value"


class SymbolTable(dict):
    def __init__(self, parent: "SymbolTable" | None = None):
        global next_symbol_table_id
        super().__init__()
        self.id = next_symbol_table_id
        next_symbol_table_id += 1
        self.parent = parent
        self.in_function = parent.in_function if parent is not None else False

    def __getitem__(self, key: str) -> "Symbol":
        return super().__getitem__(key)

    def get(self, key) -> "Symbol" | None:
        return super().get(key, None)

    def lookup(self, key: str) -> "Symbol" | None:
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


def create_symbol_table(symbols: dict[str, "Symbol"]) -> SymbolTable:
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


def is_symbol_an_expr(symbol: "Symbol") -> bool:
    """return True if the symbol is a valid expr (can be evaluated)"""
    return is_instance_compat(
        symbol,
        (ChDef, PrmDef, FpyValue, VariableSymbol, FieldAccess),
    )


Symbol = typing.Union[
    ChDef,
    PrmDef,
    FpyValue,
    CallableSymbol,
    FpyType,
    VariableSymbol,
    SymbolTable,
    FieldAccess,
]
"""a named entity in fpy that can be looked up in a symbol table"""
