from __future__ import annotations
from abc import ABC
from decimal import Decimal
from enum import Enum
from importlib.metadata import version
import inspect
from dataclasses import astuple, dataclass, field, fields
import math
import struct
import typing
from typing import Callable, Iterable, Union, get_args, get_origin
import itertools
import zlib

from fpy.error import CompileError
from fpy.ir import Ir, IrLabel

# In Python 3.10+, the `|` operator creates a `types.UnionType`.
# We need to handle this for forward compatibility, but it won't exist in 3.9.
try:
    from types import UnionType

    UNION_TYPES = (Union, UnionType)
except ImportError:
    UNION_TYPES = (Union,)

from fpy.bytecode.directives import (
    BinaryStackOp,
    COMPARISON_OPS,
    Directive,
)
from fprime_gds.common.templates.ch_template import ChTemplate
from fprime_gds.common.templates.cmd_template import CmdTemplate
from fprime_gds.common.templates.prm_template import PrmTemplate
from fprime_gds.common.models.serialize.serializable_type import (
    SerializableType as StructValue,
)
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
    IntegerType as IntegerValue,
    FloatType as FloatValue,
    NumericalType as NumericalValue,
)
from fprime_gds.common.models.serialize.string_type import StringType as StringValue
from fprime_gds.common.models.serialize.time_type import TimeType as TimeValue
from fprime_gds.common.models.serialize.bool_type import BoolType as BoolValue
from fpy.syntax import (
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
    Ast,
    AstReturn,
    AstBlock,
    AstWhile,
)
from fprime_gds.common.models.serialize.type_base import BaseType as FppValue

# Default values for sequence limits - may be overridden by dictionary constants
DEFAULT_MAX_DIRECTIVES_COUNT = 1024
DEFAULT_MAX_DIRECTIVE_SIZE = 2048

# Keep old names as aliases for backward compatibility
MAX_DIRECTIVES_COUNT = DEFAULT_MAX_DIRECTIVES_COUNT
MAX_DIRECTIVE_SIZE = DEFAULT_MAX_DIRECTIVE_SIZE

COMPILER_MAX_STRING_SIZE = 128


def typename(typ: FppType) -> str:
    if typ == FpyIntegerValue:
        return "Integer"
    if typ == FpyFloatValue:
        return "Float"
    if issubclass(typ, NumericalValue):
        return typ.get_canonical_name()
    if typ == FpyStringValue:
        return "String"
    if typ == RangeValue:
        return "Range"
    return typ.__name__


# this is the "internal" integer type that integer literals have by
# default. it is arbitrary precision. it is also only used in places where
# we know the value is constant
class FpyIntegerValue(IntegerValue):
    @classmethod
    def range(cls):
        return (-math.inf, math.inf)

    @staticmethod
    def get_serialize_format():
        raise NotImplementedError()

    @classmethod
    def get_bits(cls):
        return math.inf

    @classmethod
    def validate(cls, val):
        if not isinstance(val, int):
            raise RuntimeError()


# this is the "internal" float type that float literals have by
# default. it is arbitrary precision. it is also only used in places where
# we know the value is constant
class FpyFloatValue(FloatValue):
    @staticmethod
    def get_serialize_format():
        raise NotImplementedError()

    @classmethod
    def get_bits(cls):
        return math.inf

    @classmethod
    def validate(cls, val):
        if not isinstance(val, Decimal):
            raise RuntimeError()


class RangeValue(FppValue):
    """the type produced by range expressions `X .. Y`"""

    def serialize(self):
        raise NotImplementedError()

    def deserialize(self, data, offset):
        raise NotImplementedError()

    def getSize(self):
        raise NotImplementedError()

    @classmethod
    def getMaxSize(cls):
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__

    def to_jsonable(self):
        raise NotImplementedError()


# this is the "internal" string type that string literals have by
# default. it is arbitrary length. it is also only used in places where
# we know the value is constant
FpyStringValue = StringValue.construct_type("FpyStringValue", None)

SPECIFIC_NUMERIC_TYPES = (
    U32Value,
    U16Value,
    U64Value,
    U8Value,
    I16Value,
    I32Value,
    I64Value,
    I8Value,
    F32Value,
    F64Value,
)
SPECIFIC_INTEGER_TYPES = (
    U32Value,
    U16Value,
    U64Value,
    U8Value,
    I16Value,
    I32Value,
    I64Value,
    I8Value,
)
SIGNED_INTEGER_TYPES = (
    I16Value,
    I32Value,
    I64Value,
    I8Value,
)
UNSIGNED_INTEGER_TYPES = (
    U32Value,
    U16Value,
    U64Value,
    U8Value,
)
SPECIFIC_FLOAT_TYPES = (
    F32Value,
    F64Value,
)
ARBITRARY_PRECISION_TYPES = (FpyFloatValue, FpyIntegerValue)

from fprime_gds.common.models.serialize.enum_type import EnumType as EnumValue

# The canonical Svc.Fpy.FlagId enum type
# This must match the dictionary's Svc.Fpy.FlagId definition
FlagIdValue = EnumValue.construct_type(
    "Svc.Fpy.FlagId",
    {"EXIT_ON_CMD_FAIL": 0},
    "U8",
)

# The canonical Fw.TimeIntervalValue struct type
# This must match the dictionary's Fw.TimeIntervalValue definition
# The format string '{}' and empty description match what the GDS JSON loader produces
TimeIntervalValue = StructValue.construct_type(
    "Fw.TimeIntervalValue",
    [
        ("seconds", U32Value, "{}", ""),
        ("useconds", U32Value, "{}", ""),
    ],
)

# Time operator overloads: maps (lhs_type, rhs_type, op) -> (intermediate_type, result_type, func_name, is_comparison)
TIME_OPS: dict[tuple[type, type, BinaryStackOp], tuple[type, type, str, bool]] = {
    # Time - Time -> TimeInterval
    (TimeValue, TimeValue, BinaryStackOp.SUBTRACT): (TimeValue, TimeIntervalValue, "time_sub", False),
    # Time + TimeInterval -> Time  
    (TimeValue, TimeIntervalValue, BinaryStackOp.ADD): (TimeValue, TimeValue, "time_add", False),
    # TimeInterval +/- TimeInterval -> TimeInterval
    (TimeIntervalValue, TimeIntervalValue, BinaryStackOp.ADD): (TimeIntervalValue, TimeIntervalValue, "time_interval_add", False),
    (TimeIntervalValue, TimeIntervalValue, BinaryStackOp.SUBTRACT): (TimeIntervalValue, TimeIntervalValue, "time_interval_sub", False),
    # Time comparisons -> Bool
    **{(TimeValue, TimeValue, op): (TimeValue, BoolValue, "time_cmp_assert_comparable", True) for op in COMPARISON_OPS},
    # TimeInterval comparisons -> Bool
    **{(TimeIntervalValue, TimeIntervalValue, op): (TimeIntervalValue, BoolValue, "time_interval_cmp", True) for op in COMPARISON_OPS},
}


def is_instance_compat(obj, cls):
    """
    A wrapper for isinstance() that correctly handles Union types in Python 3.9+.

    Args:
        obj: The object to check.
        cls: The class, tuple of classes, or Union type to check against.

    Returns:
        True if the object is an instance of the class or any type in the Union.
    """
    origin = get_origin(cls)
    if origin in UNION_TYPES:
        # It's a Union type, so get its arguments.
        # e.g., get_args(Union[int, str]) returns (int, str)
        return isinstance(obj, get_args(cls))

    # It's not a Union, so it's a regular type (like int) or a
    # tuple of types ((int, str)), which isinstance handles natively.
    return isinstance(obj, cls)


# a value of type FppType is a Python `type` object representing
# the type of an Fprime value
FppType = type[FppValue]


class NothingValue(ABC):
    """a type which has no valid values in fprime. used to denote
    a function which doesn't return a value"""

    @classmethod
    def __subclasscheck__(cls, subclass):
        return False


# the `type` object representing the NothingType class
NothingType = type[NothingValue]


@dataclass
class CallableSymbol:
    name: str
    return_type: FppType | NothingType
    # args is a list of (name, type, default_value) tuples
    # default_value is an AstExpr or None if no default is provided
    args: list[tuple[str, FppType, "AstExpr | None"]]


@dataclass
class CommandSymbol(CallableSymbol):
    cmd: CmdTemplate


@dataclass
class BuiltinFuncSymbol(CallableSymbol):
    generate: Callable[[AstFuncCall, dict[int, FppValue]], list[Directive]]
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
    type: FppType


@dataclass
class CastSymbol(CallableSymbol):
    to_type: FppType


@dataclass
class FieldAccess:
    """a reference to a member/element of an fprime struct/array type"""

    parent_expr: AstExpr
    """the complete qualifier"""
    base_sym: Union["Symbol", None]
    """the base symbol, up through all the layers of field symbols, or None if parent at some point is not a symbol at all"""
    type: FppType
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
    type: FppType | None = None
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
    def __init__(self, parent: "SymbolTable | None" = None):
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

    def lookup(self, key: str) -> "Symbol | None":
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
    symbols: dict[str, "Symbol"]
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
            ChTemplate,
            PrmTemplate,
            FppValue,
            VariableSymbol,
            FieldAccess
        ),
    )

Symbol = typing.Union[
    ChTemplate,
    PrmTemplate,
    FppValue,
    CallableSymbol,
    FppType,
    VariableSymbol,
    SymbolTable,
    FieldAccess
]
"""a named entity in fpy that can be looked up in a symbol table"""


@dataclass
class CompileState:
    """a collection of input, internal and output state variables and maps"""

    global_type_scope: SymbolTable
    """The global type scope: a symbol table whose leaf nodes are types"""
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

    synthesized_types: dict[AstExpr, FppType | NothingType] = field(
        default_factory=dict
    )
    """expr to its fprime type, before type conversions are applied"""

    op_intermediate_types: dict[AstOp, FppType] = field(default_factory=dict)
    """the intermediate type that all args should be converted to for the given op"""

    expr_explicit_casts: list[AstExpr] = field(default_factory=list)
    """a list of nodes which are explicit casts"""
    contextual_types: dict[AstExpr, FppType] = field(default_factory=dict)
    """expr to fprime type it will end up being on the stack after type conversions"""

    const_expr_values: dict[AstExpr, FppValue | NothingValue | None] = field(
        default_factory=dict
    )
    """expr to the fprime value it will end up being on the stack after type conversions.
    None if unsure at compile time"""

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


# Cache for visitor method mappings, keyed by visitor class
_visitor_cache: dict[type, dict[type, str]] = {}


class _StopDescent:
    """Sentinel returned by a visit_* method to prevent the framework from
    descending into the visited node's children."""
    __slots__ = ()
    def __repr__(self):
        return "STOP_DESCENT"

STOP_DESCENT = _StopDescent()


class Visitor:
    """visits each class, calling a custom visit function, if one is defined, for each
    node type"""

    def __init__(self):
        self.visitors: dict[type[Ast], Callable] = {}
        """dict of node type to handler function"""
        self.build_visitor_dict()

    def build_visitor_dict(self):
        cls = type(self)
        # Check if this class's visitor mapping is already cached
        if cls in _visitor_cache:
            # Use cached mapping (maps node type -> method name)
            for node_type, method_name in _visitor_cache[cls].items():
                self.visitors[node_type] = getattr(self, method_name)
            return

        # Build the mapping and cache it
        class_cache: dict[type, str] = {}
        for name, func in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("visit") or name == "visit_default":
                # not a visitor, or the default visit func
                continue
            signature = inspect.signature(func)
            params = list(signature.parameters.values())
            assert len(params) == 3
            assert params[1].annotation is not None
            annotations = typing.get_type_hints(func)
            param_type = annotations[params[1].name]

            origin = get_origin(param_type)
            if origin in UNION_TYPES:
                # It's a Union type, so get its arguments.
                for t in get_args(param_type):
                    class_cache[t] = name
                    self.visitors[t] = getattr(self, name)
            else:
                # It's not a Union, so it's a regular type
                class_cache[param_type] = name
                self.visitors[param_type] = getattr(self, name)

        _visitor_cache[cls] = class_cache

    def _visit(self, node: Ast, state: CompileState):
        visit_func = self.visitors.get(type(node), self.visit_default)
        return visit_func(node, state)

    def visit_default(self, node: Ast, state: CompileState):
        pass

    def run(self, start: Ast, state: CompileState):
        """runs the visitor, starting at the given node, descending depth-first"""

        def _descend(node: Ast):
            if not isinstance(node, Ast):
                return
            children = []
            for field in fields(node):
                field_val = getattr(node, field.name)
                if isinstance(field_val, list):
                    # also handle the one case where we have a list of tuples
                    if len(field_val) > 0 and isinstance(field_val[0], tuple):
                        field_val = itertools.chain.from_iterable(field_val)
                    children.extend(field_val)
                else:
                    children.append(field_val)

            for child in children:
                if not isinstance(child, Ast):
                    continue
                _descend(child)
                if len(state.errors) != 0:
                    break
                self._visit(child, state)
                if len(state.errors) != 0:
                    break

        _descend(start)
        self._visit(start, state)


class TopDownVisitor(Visitor):
    """Like Visitor, but visits parent before children (top-down / pre-order)."""

    def run(self, start: Ast, state: CompileState):
        """runs the visitor, starting at the given node, descending breadth-first"""

        def _descend(node: Ast):
            if not isinstance(node, Ast):
                return
            children = []
            for field in fields(node):
                field_val = getattr(node, field.name)
                if isinstance(field_val, list):
                    # also handle the one case where we have a list of tuples
                    if len(field_val) > 0 and isinstance(field_val[0], tuple):
                        field_val = itertools.chain.from_iterable(field_val)
                    children.extend(field_val)
                else:
                    children.append(field_val)

            for child in children:
                if not isinstance(child, Ast):
                    continue
                result = self._visit(child, state)
                if len(state.errors) != 0:
                    break
                if result is not STOP_DESCENT:
                    _descend(child)
                if len(state.errors) != 0:
                    break

        result = self._visit(start, state)
        if result is not STOP_DESCENT:
            _descend(start)


class Transformer(Visitor):

    class Delete:
        pass

    def run(self, start: Ast, state: CompileState):

        def _descend(node):
            if not isinstance(node, Ast):
                return
            for field in fields(node):
                field_val = getattr(node, field.name)
                if isinstance(field_val, list):
                    # child is a list, iterate over each member of the list
                    # use a copy so we can remove as we traverse, also so
                    # we don't visit things that we added

                    #
                    idx = -1
                    for child in field_val[:]:
                        idx += 1
                        if not isinstance(child, Ast):
                            continue
                        _descend(child)
                        if len(state.errors) != 0:
                            break
                        transformed = self._visit(child, state)
                        if len(state.errors) != 0:
                            break
                        if isinstance(transformed, Iterable):
                            assert all(
                                isinstance(n, Ast) for n in transformed
                            ), transformed
                            # func split one node into many
                            # remove the original child and add the new ones
                            # insert them in the place where the child used to be, in the right order
                            field_val.remove(child)
                            for new_child_idx, new_child in enumerate(transformed):
                                field_val.insert(idx + new_child_idx, new_child)
                            # make sure that we maintain insertion order by updating the idx
                            # accounting for our removal of an original node
                            # if we don't do this, then if we were to insert into list after this based on idx,
                            # the positions could be swapped around
                            idx += len(transformed) - 1
                        elif isinstance(transformed, Ast):
                            field_val.remove(child)
                            field_val.insert(idx, transformed)
                        elif transformed is Transformer.Delete:
                            # just delete it
                            field_val.remove(child)
                        else:
                            assert transformed is None, transformed
                            # don't do anything, didn't return anything
                    if len(state.errors) != 0:
                        # need a second check here to get out of the enclosing loop
                        break
                    # don't need to update the field, it was a ptr to a list so should
                    # already be updated
                else:
                    _descend(field_val)
                    if len(state.errors) != 0:
                        break
                    transformed = self._visit(field_val, state)
                    if len(state.errors) != 0:
                        break
                    if isinstance(transformed, Ast):
                        setattr(node, field.name, transformed)
                    elif transformed is Transformer.Delete:
                        # just delete it
                        setattr(node, field.name, None)
                    else:
                        # cannot return a list if the original attr wasn't a list
                        assert transformed is None, transformed
                        # don't do anything, didn't return anything

        _descend(start)
        self._visit(start, state)


# Cache for emitter method mappings, keyed by emitter class
_emitter_cache: dict[type, dict[type, str]] = {}


class Emitter:
    # Default: not in a function (top-level code)
    # Subclasses override this to indicate function body context
    in_function = False

    def __init__(self):
        self.emitters: dict[type[Ast], Callable] = {}
        """dict of node type to handler function"""
        self.build_emitter_dict()

    def build_emitter_dict(self):
        cls = type(self)
        # Check if this class's emitter mapping is already cached
        if cls in _emitter_cache:
            # Use cached mapping (maps node type -> method name)
            for node_type, method_name in _emitter_cache[cls].items():
                self.emitters[node_type] = getattr(self, method_name)
            return

        # Build the mapping and cache it
        class_cache: dict[type, str] = {}
        for name, func in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("emit_"):
                # not an emitter
                continue
            signature = inspect.signature(func)
            params = list(signature.parameters.values())
            assert len(params) == 3
            assert params[1].annotation is not None
            annotations = typing.get_type_hints(func)
            param_type = annotations[params[1].name]

            origin = get_origin(param_type)
            if origin in UNION_TYPES:
                # It's a Union type, so get its arguments.
                for t in get_args(param_type):
                    class_cache[t] = name
                    self.emitters[t] = getattr(self, name)
            else:
                # It's not a Union, so it's a regular type
                class_cache[param_type] = name
                self.emitters[param_type] = getattr(self, name)

        _emitter_cache[cls] = class_cache

    def emit(self, node: Ast, state: CompileState) -> list[Directive | Ir]:
        return self.emitters[type(node)](node, state)


def _get_version_tuple() -> tuple[int, int, int]:
    try:
        import re
        v = version("fprime-fpy")
        # Handle versions like "0.0.1a3.dev103+g244fdeadc"
        # Extract just the major.minor.patch part
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", v)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return (0, 0, 0)
    except Exception:
        return (0, 0, 0)


MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION = _get_version_tuple()
SCHEMA_VERSION = 4

HEADER_FORMAT = "!BBBBBHI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


@dataclass
class Header:
    majorVersion: int
    minorVersion: int
    patchVersion: int
    schemaVersion: int
    argumentCount: int
    statementCount: int
    bodySize: int


FOOTER_FORMAT = "!I"
FOOTER_SIZE = struct.calcsize(FOOTER_FORMAT)


@dataclass
class Footer:
    crc: int


def deserialize_directives(bytes: bytes) -> list[Directive]:
    header = Header(*struct.unpack_from(HEADER_FORMAT, bytes))

    if header.schemaVersion != SCHEMA_VERSION:
        raise RuntimeError(
            f"Schema version wrong (expected {SCHEMA_VERSION} found {header.schemaVersion})"
        )

    dirs = []
    idx = 0
    offset = HEADER_SIZE
    while idx < header.statementCount:
        offset_and_dir = Directive.deserialize(bytes, offset)
        if offset_and_dir is None:
            raise RuntimeError("Unable to deserialize sequence")
        offset, dir = offset_and_dir
        dirs.append(dir)
        idx += 1

    if offset != len(bytes) - FOOTER_SIZE:
        raise RuntimeError(
            f"{len(bytes) - FOOTER_SIZE - offset} extra bytes at end of sequence"
        )

    return dirs


def serialize_directives(dirs: list[Directive]) -> tuple[bytes, int]:
    output_bytes = bytes()

    for dir in dirs:
        dir_bytes = dir.serialize()
        if len(dir_bytes) > MAX_DIRECTIVE_SIZE:
            print(
                CompileError(
                    f"Directive {dir} in sequence too large (expected less than {MAX_DIRECTIVE_SIZE}, was {len(dir_bytes)})"
                )
            )
            exit(1)
        output_bytes += dir_bytes

    header = Header(
        MAJOR_VERSION,
        MINOR_VERSION,
        PATCH_VERSION,
        SCHEMA_VERSION,
        0,
        len(dirs),
        len(output_bytes),
    )
    output_bytes = struct.pack(HEADER_FORMAT, *astuple(header)) + output_bytes

    crc = zlib.crc32(output_bytes) % (1 << 32)
    footer = Footer(crc)
    output_bytes += struct.pack(FOOTER_FORMAT, *astuple(footer))

    return output_bytes, crc
