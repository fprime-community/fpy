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


def _generate_llvm_unsupported(builder, args):
    """Default LLVM lowering: a builtin that hasn't been taught the llvm/wasm
    backend yet. Raises rather than silently miscompiling."""
    raise NotImplementedError("this builtin has no LLVM/wasm lowering yet")


@dataclass
class BuiltinFuncSymbol(CallableSymbol):
    generate_fpybc: Callable[[AstFuncCall, dict[int, FpyValue]], list[Directive]]
    """fpybc backend: builds bytecode directives given the calling node and a
    dict mapping const_arg_indices to their compile-time values. Non-const args
    are already pushed on the stack by the caller."""
    generate_llvm: Callable = _generate_llvm_unsupported
    """llvm/wasm backend: builds the call's LLVM IR. Called as
    generate_llvm(builder, args), where args is a list of (ir.Value or None,
    FpyValue or None) pairs -- each argument's emitted value alongside its
    compile-time constant value (or None if it isn't constant). Args in
    const_arg_indices are never emitted and arrive as (None, value). Returns the
    result ir.Value (or None for a NOTHING-typed builtin). Defaults to raising
    'not lowered yet'."""
    const_arg_indices: frozenset[int] = field(default_factory=frozenset)
    """indices of args that must be compile-time constants and are NOT pushed
    to the stack; instead their values are passed to generate_fpybc()"""


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
@dataclass(eq=False)
class VariableSymbol:
    """a mutable, typed value stored on the stack referenced by an unqualified
    name. One declaration is one variable, so variables compare and hash by
    identity."""

    name: str
    type_ref: AstExpr | None
    """the AST node denoting the var's type"""
    declaration: Ast
    """the node where this var is declared"""
    type: FpyType | None = None
    """the resolved type of the variable. None if type unsure at the moment"""
    is_global: bool = False
    """whether this variable is a top-level (global) variable"""


next_symbol_table_id = 0


class NameGroup(str, Enum):
    TYPE = "type"
    CALLABLE = "callable"
    VALUE = "value"


class SymbolTable(dict):
    def __init__(self, parent=None):
        global next_symbol_table_id
        super().__init__()
        self.id = next_symbol_table_id
        next_symbol_table_id += 1
        self.parent = parent
        self.in_function = parent.in_function if parent is not None else False

    def __getitem__(self, key: str) -> "Symbol":
        return super().__getitem__(key)

    def get(self, key):
        return super().get(key, None)

    def lookup(self, key: str):
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
        """Return a shallow copy that preserves the table's concrete class and
        metadata."""
        new = self.__class__(parent=self.parent)
        new.in_function = self.in_function
        new.update(self)
        return new


class Scope:
    """A lexical scope.

    Name resolution is name-group-directed: a use site knows whether it wants a
    type, a callable, or a value, and the same identifier text may name a
    different symbol in each group. A Scope therefore holds an independent
    SymbolTable table per NameGroup plus a parent link; a lookup consults
    the requested group, walking the parent chain."""

    def __init__(self, parent: "Scope | None" = None, in_function: bool | None = None):
        self.parent = parent
        if in_function is None:
            in_function = parent.in_function if parent is not None else False
        self.in_function = in_function
        self._groups: dict[NameGroup, dict[str, "Symbol"]] = {
            ng: {} for ng in NameGroup
        }

    def group(self, ng: NameGroup) -> dict[str, "Symbol"]:
        """The name->symbol dict for one name group (mutable)."""
        return self._groups[ng]

    def own_symbols(self) -> dict[str, "Symbol"]:
        """The symbols defined directly in this scope (no parent walk), flattened
        across name groups. Used to enumerate a sequence's own top-level
        definitions -- functions, globals, and re-exported modules -- when an
        import binds the whole sequence."""
        merged = dict(self._groups[NameGroup.CALLABLE])
        merged.update(self._groups[NameGroup.VALUE])
        return merged

    def define(self, ng: NameGroup, name: str, sym: "Symbol"):
        """Bind *name* to *sym* in this scope's *ng* group."""
        self._groups[ng][name] = sym

    def get(self, ng: NameGroup, name: str):
        """Look up *name* in this scope's *ng* group only (no parent walk)."""
        return self._groups[ng].get(name)

    def lookup(self, ng: NameGroup, name: str):
        """Look up *name* in the *ng* group of this scope and its ancestors."""
        scope = self
        while scope is not None:
            sym = scope._groups[ng].get(name)
            if sym is not None:
                return sym
            scope = scope.parent
        return None


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


ModuleSymbol = SymbolTable
"""a table which may contain sub definitions (a dictionary module, or an
import-bound directory or sequence symbol)."""


class SequenceSymbol(SymbolTable):
    """A sequence definition: a sequence file, as a definition (SPEC.md
    "File system definitions").

    One file is one definition, so there is one SequenceSymbol per sequence
    file, shared by every scope that imports it. Its entries are the
    definitions in the scope of its sequence, filled in by BindImports once
    the sequence is compiled."""

    def __init__(self, source_file=None, parent=None):
        super().__init__(parent=parent)
        self.source_file = source_file


class DirectorySymbol(SymbolTable):
    """A directory definition -- a directory, as a definition -- as
    associated in one importing scope (SPEC.md "File system
    definitions").

    One directory is one definition, so two directory symbols stand for the
    same definition iff their `directory` is the same path. Each importing
    scope holds its own DirectorySymbol per bound name, whose entries are
    only the associations that scope's import statements made at longer
    qualified names -- not every definition in the directory."""

    def __init__(self, directory=None, parent=None):
        super().__init__(parent=parent)
        self.directory = directory


Symbol = typing.Union[
    ChDef,
    PrmDef,
    FpyValue,
    CallableSymbol,
    FpyType,
    VariableSymbol,
    ModuleSymbol,
    FieldAccess,
]
"""a named entity in fpy that can be looked up in a symbol table"""
