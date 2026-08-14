from abc import ABC
from enum import Enum
from pathlib import Path
from typing import (
    Generic,
    Literal,
    Mapping,
    NamedTuple,
    Sequence,
    TypeVar,
    cast,
)


class Error(Enum):
    QUALIFIED_IDENTIFIER_RESOLVE_FAILED = 0
    QUALIFIER_IDENT_RESOLVED_TO_NON_QUALIFIER_SYMBOL = 1


class CompileError(Exception):
    def __init__(self, error: Error):
        super().__init__(error)


# syntax

type Id = int

class Ast(ABC, NamedTuple):
    id: Id

class AstIdent(Ast):
    text: str


# hmmm... qualified identifier. would be any chain of period separated identifiers
# purely a syntactic construct
# a dot expr is any expr plus a period plus an ident. must know semantic info to know whether qual ident is a dot expr or 


# if we were going to transform this into a simpler language, one of the first things we'd do is transform the dot exprs and qual names into 
# separate syntactic constructs. so let's do that!

def is_dot_expr(expr: AstExpr) -> bool:
    if isinstance(expr, AstPeriodSeparator):
        pass



type AstQualifiedIdent = AstProperQualifiedIdent | AstIdent


class AstProperQualifiedIdent(Ast):
    qualifier: AstQualifiedIdent
    ident: AstIdent


class AstVarDef(Ast):
    ident: AstIdent
    type_expr: AstExpr
    initial_value: AstExpr


class AstFuncDef(Ast):
    ident: AstIdent



class AstString(Ast):
    value: str


# store floats without regard to precision
type FloatToken = str


class AstNumber(Ast):
    value: int | FloatToken


class AstBoolean(Ast):
    value: Literal[True] | Literal[False]


type AstLiteral = AstString | AstNumber | AstBoolean


class AstGetAttr(Ast):
    parent: AstExpr
    attr: str


class AstIndexExpr(Ast):
    parent: AstExpr
    item: AstExpr


class AstNamedArgument(Ast):
    name: str
    value: "AstExpr"


class AstFuncCall(Ast):
    func: "AstExpr"
    # args can contain both positional (AstExpr) and named arguments (AstNamedArgument)
    args: list[Union["AstExpr", AstNamedArgument]] | None


class AstPass(Ast):
    pass  # ha ha


class AstBinaryOp(Ast):
    lhs: AstExpr
    op: str
    rhs: AstExpr


class AstUnaryOp(Ast):
    op: str
    val: AstExpr


class AstRange(Ast):
    lower_bound: AstExpr
    op: str
    upper_bound: AstExpr


class AstAnonStruct(Ast):
    members: list[tuple[str, "AstExpr"]]


class AstAnonArray(Ast):
    elements: list["AstExpr"]


AstOp = Union[AstBinaryOp, AstUnaryOp]

type AstExpr = AstFuncCall | AstLiteral | AstGetAttr | AstIndexExpr | AstIdent | AstOp | AstRange | AstAnonStruct | AstAnonArray



class AstAssign(Ast):
    lhs: AstExpr
    type_ann: AstExpr | None
    rhs: AstExpr


class AstAugAssign(Ast):
    """An augmented assignment (lhs op= rhs). Desugared into
    AstAssign(lhs, None, AstBinaryOp(lhs, op, rhs)) before semantic analysis."""

    lhs: AstExpr
    op: str
    rhs: AstExpr


class AstElif(Ast):
    condition: AstExpr
    body: "AstBlock"


class AstIf(Ast):
    condition: AstExpr
    body: "AstBlock"
    elifs: list[AstElif]
    els: Union["AstBlock", None]


class AstFor(Ast):
    loop_var: AstIdent
    range: AstExpr
    body: AstBlock


class AstWhile(Ast):
    condition: AstExpr
    body: AstBlock


class AstCheck(Ast):
    condition: AstExpr
    timeout: Union[AstExpr, None]  # The timeout interval, or None if `never`/absent
    persist: Union[AstExpr, None]  # Default: 0 second interval
    period: Union[AstExpr, None]  # Default: 1 second interval
    body: Union["AstBlock", None]  # None for body-less check
    timeout_body: Union["AstBlock", None] = None
    timeout_never: bool = False  # True if the timeout clause is `never`


class AstAssert(Ast):
    condition: AstExpr
    exit_code: Union[AstExpr, None]


class AstBreak(Ast):
    pass


class AstContinue(Ast):
    pass


class AstReturn(Ast):
    value: Union[AstExpr, None]


class AstDef(Ast):
    name: AstIdent
    # parameters is a list of (ident, type, default_value) tuples
    # default_value is None if no default is provided
    parameters: Union[list[tuple[AstIdent, AstExpr, AstExpr | None]], None]
    return_type: Union[AstExpr, None]
    body: AstBlock


class AstSequenceMetadata(Ast):
    parameters: Union[list[tuple[AstIdent, AstExpr]], None]


class AstImport(Ast):
    """An import statement. Only valid as a top-level statement.

    `import [dots] a.b.c [as alias]` and
    `from [dots] a.b.c import (* | m1 [as x], ...)`.
    """

    is_from: bool
    """True for a `from` import, False for a plain `import`."""
    num_dots: int
    """Number of leading dots. 0 means absolute; >0 means relative."""
    path: list[str]
    """The dotted path segments after any leading dots (at least one)."""
    alias: Union[str, None]
    """The `as` alias for a plain `import ... as alias`, else None."""
    members: Union[list[tuple[str, Union[str, None]]], None]
    """For a `from` import: list of (member_name, alias_or_None). None for a
    plain import. Empty/None when `is_star` is True."""
    is_star: bool
    """True for `from ... import *`."""


AstStmt = Union[
    AstExpr,
    AstAssign,
    AstAugAssign,
    AstPass,
    AstIf,
    AstElif,
    AstFor,
    AstBreak,
    AstContinue,
    AstWhile,
    AstCheck,
    AstAssert,
    AstDef,
    AstSequenceMetadata,
    AstReturn,
]
AstStmtWithExpr = Union[
    AstExpr,
    AstAssign,
    AstAugAssign,
    AstIf,
    AstElif,
    AstFor,
    AstWhile,
    AstCheck,
    AstAssert,
    AstDef,
    AstReturn,
]
AstNodeWithSideEffects = Union[
    AstFuncCall,
    AstAssign,
    AstAugAssign,
    AstIf,
    AstElif,
    AstFor,
    AstWhile,
    AstCheck,
    AstAssert,
    AstBreak,
    AstContinue,
    AstDef,
    AstReturn,
]


class AstBlock(Ast):
    stmts: list[AstStmt]


class ModuleDef(Ast):
    ident: AstIdent


class SequenceDef(Ast):
    path: Path


class DirectoryDef(Ast):
    path: Path


# from dict
class TypeDef(NamedTuple):
    ident: AstIdent


# from dict
class EnumConstantDef(NamedTuple):
    ident: AstIdent


# from dict
class TypeCtorDef(NamedTuple):
    ident: AstIdent


# from dict
class CommandDef(NamedTuple):
    ident: AstIdent
# semantics


class VarSymbol(NamedTuple):
    defn: AstVarDef


class FuncSymbol(NamedTuple):
    defn: AstFuncDef


class ModuleSymbol(NamedTuple):
    # a module may be defined by several syntactic defs
    defns: Sequence[ModuleDef]


class TypeSymbol(NamedTuple):
    defn: TypeDef


class EnumConstantSymbol(NamedTuple):
    defn: EnumConstantDef


class TypeCtorSymbol(NamedTuple):
    defn: TypeCtorDef


class DirectorySymbol(NamedTuple):
    defn: DirectoryDef


class SequenceSymbol(NamedTuple):
    defn: SequenceDef


class CommandSymbol(NamedTuple):
    defn: CommandDef


type ValueSymbol = VarSymbol | EnumConstantSymbol

type CallableSymbol = FuncSymbol | TypeCtorSymbol | CommandSymbol

type QualifierSymbol = ModuleSymbol | DirectorySymbol | SequenceSymbol

type Symbol = ValueSymbol | CallableSymbol | TypeSymbol | QualifierSymbol


type Name = str


class ProperQualifiedName(NamedTuple):
    qualifier: "QualifiedName"
    name: Name


type QualifiedName = Name | ProperQualifiedName


class NameGroup[S: Symbol]:
    # a category of names
    pass


ValueNameGroup = NameGroup[ValueSymbol | QualifierSymbol]()
CallableNameGroup = NameGroup[CallableSymbol | QualifierSymbol]()
TypeNameGroup = NameGroup[TypeSymbol | QualifierSymbol]()


class Scope(NamedTuple):
    # the name groups _are_ the keys of this map
    # names _are_ the keys of the inner map
    names: Mapping[NameGroup[Symbol], Mapping[Name, Symbol]]
    parent: "Scope|None"

    def names_in_group[S: Symbol](self, group: NameGroup[S]) -> Mapping[Name, S]:
        return cast(Mapping[Name, S], self.names[group])


class CompileState(NamedTuple):
    parent_symbol: Mapping[Symbol, Symbol]
    symbol_scope: Mapping[QualifierSymbol, Scope]
    use_def: Mapping[Id, Symbol]
    type: Mapping[Id, TypeSymbol]
    value: Mapping[Id, ValueSymbol]


# def get_qualifier_scope(sym: QualifierSymbol) -> Scope:
#     parent_scope =


def resolve_ident[S: Symbol](
    ident: AstIdent, scope: Scope, name_group: NameGroup[S], state: CompileState
) -> S:
    names_in_group_in_scope = scope.names_in_group(name_group)
    for name, sym in names_in_group_in_scope.items():
        if name == ident.text:
            return sym

    if scope.parent is None:
        raise CompileError(Error.QUALIFIED_IDENTIFIER_RESOLVE_FAILED)

    return resolve_ident(ident, scope.parent, name_group, state)


def resolve_proper_qual_ident[S: Symbol](
    qi: AstProperQualifiedIdent,
    scope: Scope,
    name_group: NameGroup[S],
    state: CompileState,
) -> S:
    qualifier_sym = resolve_qual_ident(qi.qualifier, scope, name_group)

    if not isinstance(qualifier_sym, QualifierSymbol):
        raise CompileError(Error.QUALIFIER_IDENT_RESOLVED_TO_NON_QUALIFIER_SYMBOL)

    qualifier_scope = state.symbol_scope[qualifier_sym]

    return resolve_ident(qi.ident, qualifier_scope, name_group, state)


def resolve_qual_ident[S: Symbol](
    qi: AstQualifiedIdent, scope: Scope, name_group: NameGroup[S], state: CompileState
) -> S:
    if isinstance(qi, AstIdent):
        return resolve_ident(qi, scope, name_group, state)
    return resolve_proper_qual_ident(qi, scope, name_group, state)

