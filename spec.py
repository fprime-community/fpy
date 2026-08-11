from enum import Enum
from pathlib import Path
from typing import (
    Generic,
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


class Ident(NamedTuple):
    id: Id
    text: str


type QualifiedIdent = ProperQualifiedIdent | Ident


class ProperQualifiedIdent(NamedTuple):
    id: Id
    qualifier: QualifiedIdent
    ident: Ident


class VarDef(NamedTuple):
    id: Id
    ident: Ident


class FuncDef(NamedTuple):
    id: Id
    ident: Ident


class ModuleDef(NamedTuple):
    id: Id
    ident: Ident


class SequenceDef(NamedTuple):
    id: Id
    path: Path


class DirectoryDef(NamedTuple):
    id: Id
    path: Path


# from dict
class TypeDef(NamedTuple):
    id: Id
    ident: Ident


# from dict
class EnumConstantDef(NamedTuple):
    id: Id
    ident: Ident


# from dict
class TypeCtorDef(NamedTuple):
    id: Id
    ident: Ident


# from dict
class CommandDef(NamedTuple):
    id: Id
    ident: Ident


# semantics


class VarSymbol(NamedTuple):
    defn: VarDef


class FuncSymbol(NamedTuple):
    defn: FuncDef


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
    ident: Ident, scope: Scope, name_group: NameGroup[S], state: CompileState
) -> S:
    names_in_group_in_scope = scope.names_in_group(name_group)
    for name, sym in names_in_group_in_scope.items():
        if name == ident.text:
            return sym

    if scope.parent is None:
        raise CompileError(Error.QUALIFIED_IDENTIFIER_RESOLVE_FAILED)

    return resolve_ident(ident, scope.parent, name_group, state)


def resolve_proper_qual_ident[S: Symbol](
    qi: ProperQualifiedIdent,
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
    qi: QualifiedIdent, scope: Scope, name_group: NameGroup[S], state: CompileState
) -> S:
    if isinstance(qi, Ident):
        return resolve_ident(qi, scope, name_group, state)
    return resolve_proper_qual_ident(qi, scope, name_group, state)
