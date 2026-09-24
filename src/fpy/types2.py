# for some types A, it's easier to code the function A_is_subtype_of_B
# for some types, it's easier to code A is supertype of B


from dataclasses import dataclass
from decimal import Decimal

type EnumConstant = tuple[str, int]
type Struct = dict
type Array = list

# all values in the universe of Fpy are represented as values of this (non-Fpy) type:
type LogicalValue = Decimal | int | str | EnumConstant | Struct | Array | bool | object


@dataclass(frozen=True)
class Type:
    name: str

    def is_subtype_of(self, other: "Type") -> bool:
        raise NotImplementedError

    def is_supertype_of(self, other: "Type") -> bool:
        raise NotImplementedError

    def contains_value(self, value: LogicalValue) -> bool:
        """returns True if the given value is a member of this type"""
        raise NotImplementedError


@dataclass(frozen=True)
class SingletonType(Type):

    value: LogicalValue

    def is_subtype_of(self, other: Type) -> bool:
        return other.contains_value(self.value)

    def is_supertype_of(self, other: Type) -> bool:
        # only true if other is a singleton with the same value
        if not isinstance(other, SingletonType):
            return False

        return self.value == other.value

    def contains_value(self, value):
        return value == self.value


Unit = SingletonType("Unit", object())


@dataclass(frozen=True)
class UnionType(Type):
    members: frozenset[Type]

    def is_subtype_of(self, other: Type) -> bool:
        # all of these member types must be subtypes of the super type for
        # the whole union type to be a subtype of the super type
        return all(t.is_subtype_of(other) for t in self.members)

    def is_supertype_of(self, other: Type) -> bool:
        # if other is a subtype of any of the member types, then
        # we are a supertype of other

        # note this misses the case in which other is a subtype of various combinations
        # of the member types. e.g. bool would not be considered
        # a subtype of Literal[True] | Literal[False] even though it absolutely is.
        # but in order to calculate that case, we'd have to work with the sets of values
        # in each type. it gets complicated and we don't really need to worry about this
        # as it's a relatively rare case
        return any(other.is_subtype_of(t) for t in self.members)

    def contains_value(self, value):
        return any(t.contains_value(value) for t in self.members)


Never = UnionType("Never", [])


def union_of(name: str, types: set[Type]) -> Type:
    pass  # TODO


class AnyType(Type):
    def is_subtype_of(self, other):
        return isinstance(other, AnyType)

    def is_supertype_of(self, other):
        return True

    def contains_value(self, value):
        return True


Any = AnyType("Any")


@dataclass(frozen=True)
class EnumType(UnionType):
    rep_type: IntegerType


def enum_of(
    name: str, rep_type: IntegerType, constants: set[tuple[str, int]]
) -> EnumType:
    enum_constant_types = set()
    for const_name, const_value in constants:
        fq_const_name = name + "." + const_name
        const_type = SingletonType(fq_const_name, (fq_const_name, const_value))
        enum_constant_types.add(const_type)

    return EnumType(name, enum_constant_types, rep_type)


class NumericType(Type):
    def is_supertype_of(self, other):
        return super().is_supertype_of(other)


@dataclass(frozen=True)
class IntegerType(Type):
    def is_subtype_of(self, other):
        # all values of self are values of other
        return super().is_subtype_of(other)

    def is_supertype_of(self, other):
        # all values of other are values of self

        pass

    def contains_value(self, value):
        return isinstance(value, int)
