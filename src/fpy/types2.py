from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

type EnumConstant = tuple[str, int]
type Struct = dict
type Array = list

# all values in the universe of Fpy are represented as values of this (non-Fpy) type:
type UniverseValue = Decimal | int | str | EnumConstant | Struct | Array | bool | object


@dataclass(frozen=True)
class Type:
    name: str

    def contains_value(self, value: UniverseValue) -> bool:
        """returns True if the given universe value is a member of this type"""
        raise NotImplementedError


@dataclass
class Value:
    type: Type
    value: UniverseValue


@dataclass(frozen=True)
class SingletonType(Type):

    value: UniverseValue

    def contains_value(self, value):
        return value == self.value


def is_singleton_type_subtype_of(self, other: Type) -> bool:
    return other.contains_value(self.value)


def is_singleton_type_supertype_of(self, other: Type) -> bool:
    # only true if other is a singleton with the same value
    if not isinstance(other, SingletonType):
        return False

    return self.value == other.value


Unit = SingletonType("Unit", object())


@dataclass(frozen=True)
class UnionType(Type):
    members: frozenset[Type]

    def contains_value(self, value):
        return any(t.contains_value(value) for t in self.members)


def is_union_type_subtype_of(self, other: Type) -> bool:
    # all of these member types must be subtypes of the super type for
    # the whole union type to be a subtype of the super type
    return all(t.is_subtype_of(other) for t in self.members)


def is_union_type_supertype_of(self, other: Type) -> bool:
    # if other is a subtype of any of the member types, then
    # we are a supertype of other

    # note this misses the case in which other is a subtype of various combinations
    # of the member types. e.g. bool would not be considered
    # a subtype of Literal[True] | Literal[False] even though it absolutely is.
    # but in order to calculate that case, we'd have to work with the sets of values
    # in each type. it gets complicated and we don't really need to worry about this
    # as it's a relatively rare case
    return any(other.is_subtype_of(t) for t in self.members)


Never = UnionType("Never", [])


def union_of(name: str, types: set[Type]) -> Type:
    pass  # TODO


class AnyType(Type):
    def contains_value(self, value):
        return True


def is_any_type_subtype_of(self, other: Type) -> bool:
    return isinstance(other, AnyType)


def is_any_type_supertype_of(self, other: Type) -> bool:
    return True


Any = AnyType("Any")


@dataclass(frozen=True)
class IntegerType(Type):
    bits: int
    signed: bool

    def get_max_value(self) -> int:
        return (2**self.bits - 1) if not self.signed else (2 ** (self.bits - 1) - 1)

    def get_min_value(self) -> int:
        return 0 if not self.signed else -(2 ** (self.bits - 1))

    def is_subtype_of(self, other):
        if isinstance(other, AnyType):
            return True
        if not isinstance(other, IntegerType):
            return False
        # check that all values in this type are in the other integer type
        return (
            self.get_min_value() >= other.get_min_value()
            and self.get_max_value() <= other.get_max_value()
        )

    def is_supertype_of(self, other):
        # all values of other are values of self
        pass

    def contains_value(self, value):
        return isinstance(value, int)


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


@dataclass(frozen=True)
class FunctionType(Type):
    argument_types: Sequence[Type]
    return_type: Type


def is_subtype_of(self, super_type: FpyType) -> bool:
    """True if all values of self are values of super_type, with some minor exceptions. See comments"""
    if self == super_type:
        return True

    if self.kind == TypeKind.UNION:
        # all of these member types must be subtypes of the super type for
        # the whole union type to be a subtype of the super type
        return all(t.is_subtype_of(super_type) for t in self.union_types)

    if self.kind == TypeKind.ANY:
        # any type is only a subtype of itself
        return False

    if super_type.kind == TypeKind.ANY:
        # any type is a supertype of all types
        return True

    if super_type.kind == TypeKind.UNION:
        # if we are a subtype of any one of the member types of the super union type,
        # then we are a subtype of the whole super union type

        # note this misses the case in which we are a subtype of various combinations
        # of the member types of the super union type. e.g. bool would not be considered
        # a subtype of Literal[True] | Literal[False] even though it absolutely is.
        # but in order to calculate that case, we'd have to work with the sets of values
        # in each type. it gets complicated and we don't really need to worry about this
        # as it's a relatively rare case
        return any(self.is_subtype_of(t) for t in super_type.union_types)

    if super_type.kind == TypeKind.SIZED:
        # sized is a super type of any type which is serializable, and has a statically-known
        # serialized size
        return self.is_serializable and self.has_static_serialized_size

    if super_type.is_literal:
        # super type is a literal, so it has one value
        # the only way this could be a subtype of it is if
        # it is equal to it, which was already checked
        return False

    if super_type.kind == TypeKind.ENUM:
        # only way an enum type can be a subtype of another enum type
        # is if they are the same type. i.e. no enum constant is shared
        # between two enum types
        return False

    if super_type.kind == TypeKind.RANGE:
        return False

    if super_type.kind == TypeKind.UNIT:
        return False

    if super_type.kind == TypeKind.BOOL:
        # only the bool type is a subtype of the bool type
        return self.kind == TypeKind.BOOL

    if super_type.is_float:
        if self.is_numerical and self.is_literal:
            # a literal numeric type is a subtype of any type containing the literal type's value
            return super_type.is_numeric_value_in_type(self.singleton_val)

        if self.is_float:
            return self.bits <= super_type.bits

        if self.is_integer:
            # okay, this one is the integer, super_type is a float
            if self.bits >= 64:
                # if I64, U64, cannot fit in any float
                return False

            if self.bits == 32:
                # a 32 bit u/i int can only fit into an f64
                return super_type.kind == TypeKind.F64

            return True

        return False

    if super_type.is_integer:
        if self.is_numerical and self.is_literal:
            # a literal numeric type is a subtype of any type containing the literal type's value
            return super_type.is_numeric_value_in_type(self.singleton_val)

        if self.is_float:
            # no non-literal float type is a subtype of an integer type
            return False

        if self.is_integer:
            # an int type B is a subtype of another int type A if all B's values are contained within
            # the value range of A
            this_min, this_max = self.value_range
            super_min, super_max = super_type.value_range

            return this_max <= super_max and this_min >= super_min

        return False

    if super_type.is_array:
        if self.kind == TypeKind.ANON_ARRAY:
            # an anonymous array may only be a subtype of an array (anonymous or not)
            # with the same length
            if self.length != super_type.length:
                return False

            # the element type of this array type must be a subtype of the super type's element type
            return self.elem_type.is_subtype_of(super_type.elem_type)

        return False

    if super_type.is_struct:
        if self.kind == TypeKind.ANON_STRUCT:
            # if there's a member in the super type which isn't present in this
            # type, this type cannot be a subtype
            for super_member in super_type.members:
                sub_member = next(
                    (m for m in self.members if m.name == super_member.name), None
                )
                if sub_member is None:
                    return False
                # the member of this type is not a subtype of the member of the potential super type
                if not sub_member.type.is_subtype_of(super_member.type):
                    return False
            return True

        return False

    if super_type.is_string:
        if self.is_string:
            # this handles string literals too
            return self.max_length <= super_type.max_length
        return False

    assert False, super_type.kind
