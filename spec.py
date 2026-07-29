from dataclasses import dataclass
from enum import Enum
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


class Ident(NamedTuple):
    text: str


type QualifiedIdent = ProperQualifiedIdent | Ident


class ProperQualifiedIdent(NamedTuple):
    qualifier: QualifiedIdent
    ident: Ident


class VarDef(NamedTuple):
    ident: Ident


class FuncDef(NamedTuple):
    ident: Ident


class ModuleDef(NamedTuple):
    ident: Ident


class TypeDef(NamedTuple):
    ident: Ident


class EnumConstantDef(NamedTuple):
    ident: Ident


class TypeCtorDef(NamedTuple):
    ident: Ident


type Definition = VarDef | FuncDef | ModuleDef | TypeDef | EnumConstantDef | TypeCtorDef


# semantics

# Symbols are entities: one symbol exists per definition (per module, for
# ModuleSymbol), and equality is object identity. Syntax compares
# structurally; symbols do not.


@dataclass(eq=False)
class VarSymbol:
    definition: VarDef


@dataclass(eq=False)
class FuncSymbol:
    definition: FuncDef


@dataclass(eq=False)
class ModuleSymbol:
    # a module may be defined by several syntactic defs
    definitions: Sequence[ModuleDef]


@dataclass(eq=False)
class TypeSymbol:
    definition: TypeDef


@dataclass(eq=False)
class EnumConstantSymbol:
    definition: EnumConstantDef


@dataclass(eq=False)
class TypeCtorSymbol:
    definition: TypeCtorDef


type Symbol = VarSymbol | FuncSymbol | ModuleSymbol | TypeSymbol | EnumConstantSymbol | TypeCtorSymbol

type ValueSymbol = VarSymbol | EnumConstantSymbol

type CallableSymbol = FuncSymbol | TypeCtorSymbol

type QualifierSymbol = ModuleSymbol | TypeSymbol


# it's a name when it refers to something
S = TypeVar("S", bound=Symbol, covariant=True)


class Name(NamedTuple, Generic[S]):
    """A name referring to a symbol of type S"""

    ident: Ident
    referent: S


# TODO I'd really like to be able to say that:
# for S in ValueSymbol, the Qualifier refs to a Mod or Type (to support enum )
# everywhere else, Qualifier refs to just a Mod
type QualifiedName[S: Symbol] = tuple[QualifiedName[QualifierSymbol], Name[S]] | Name[S]


class NameGroup[S: Symbol]:
    # a category of names
    pass


ValueNameGroup = NameGroup[ValueSymbol | QualifierSymbol]()
CallableNameGroup = NameGroup[CallableSymbol | QualifierSymbol]()
TypeNameGroup = NameGroup[TypeSymbol | QualifierSymbol]()


class Scope(NamedTuple):
    # cannot express that the S in ng[S] should be sq[name[S]] (per key) so we
    # define that in the names_in_group function
    names: Mapping[NameGroup[Symbol], Sequence[Name[Symbol]]]
    parent: "Scope|None"

    def names_in_group[S: Symbol](self, group: NameGroup[S]) -> Sequence[Name[S]]:
        return cast(Sequence[Name[S]], self.names[group])


def resolve_ident[S: Symbol](
    ident: Ident, scope: Scope, name_group: NameGroup[S]
) -> Name[S]:
    names_in_group_in_scope = scope.names_in_group(name_group)
    for name in names_in_group_in_scope:
        if name.ident == ident:
            return name

    if scope.parent is None:
        raise CompileError(Error.QUALIFIED_IDENTIFIER_RESOLVE_FAILED)

    return resolve_ident(ident, scope.parent, name_group)


def resolve_proper_qual_ident[S: Symbol](
    qi: ProperQualifiedIdent, scope: Scope, name_group: NameGroup[S]
) -> QualifiedName[S]:
    qualifier_name = resolve_qual_ident(qi.qualifier, scope, name_group)

    if not isinstance(qualifier_name, QualifierSymbol):
        raise CompileError(Error.QUALIFIER_IDENT_RESOLVED_TO_NON_QUALIFIER_SYMBOL)


def resolve_qual_ident[S: Symbol](
    qi: QualifiedIdent, scope: Scope, name_group: NameGroup[S]
) -> QualifiedName[S]:
    if isinstance(qi, Ident):
        return resolve_ident(qi, scope, name_group)
    return resolve_proper_qual_ident(qi, scope, name_group)
