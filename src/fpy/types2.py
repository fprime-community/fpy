# for some types A, it's easier to code the function A_is_subtype_of_B
# for some types, it's easier to code A is supertype of B


from dataclasses import dataclass


@dataclass(frozen=True)
class Type:
    name: str

    def is_subtype_of(self, other: "Type") -> bool:
        raise NotImplementedError

    def is_supertype_of(self, other: "Type") -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class SingletonType(Type):

    value: object

    def is_subtype_of(self, other: Type) -> bool:
        # return true if the singleton's value is in the super type

        pass

    def is_supertype_of(self, other: Type) -> bool:
        # only true if other is a singleton with the same value
        if not isinstance(other, SingletonType):
            return False

        return self.value == other.value


Unit = SingletonType("Unit", object())


@dataclass(frozen=True)
class UnionType(Type):
    members: list[Type]

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


Never = UnionType("Never", [])


class AnyType(Type):
    def is_subtype_of(self, other):
        return isinstance(other, AnyType)

    def is_supertype_of(self, other):
        return True


Any = AnyType("Any")


@dataclass(frozen=True)
class EnumType(Type):
    pass
