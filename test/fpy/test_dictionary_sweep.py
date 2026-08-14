"""Dictionary-wide sweeps.

Every type, telemetry channel, parameter, constant, command and enum member in
the dictionary is exercised here, so a symbol the compiler cannot handle is
reported by name instead of going unnoticed until a sequence happens to use it.
Cases are generated from the dictionary, so a symbol added to it is swept
without editing this file.

A value may be referred to in a sequence only when its type is constant-sized.
String-bearing types are therefore expected to be rejected, and each sweep
asserts that rejection rather than skipping the symbol.
"""

import pytest

from fpy.dictionary import load_dictionary
from fpy.test_helpers import (
    assert_compile_failure,
    assert_compile_success,
    default_dictionary,
)
from fpy.types import TypeKind

_DICT = load_dictionary(default_dictionary)

TYPE_DEFS = _DICT["type_defs"]
CHANNELS = _DICT["ch_name_dict"]
PARAMETERS = _DICT["prm_name_dict"]
COMMANDS = _DICT["cmd_name_dict"]
CONSTANTS = _DICT["constants"]

_INTEGER_KINDS = {
    TypeKind.U8,
    TypeKind.U16,
    TypeKind.U32,
    TypeKind.U64,
    TypeKind.I8,
    TypeKind.I16,
    TypeKind.I32,
    TypeKind.I64,
}

# A type whose name the front end does not accept, or that holds a string, is
# rejected with one of these.
NOT_CONSTANT_SIZED = r"is not constant-sized \(contains strings\)|Unknown type"

# Sequence-run commands resolve their argument against the ground binary
# directory, which these sweeps do not configure.
NEEDS_BINARY_DIR = "no binary directory configured"


def bears_string(fpy_type) -> bool:
    """Whether *fpy_type* holds a string anywhere inside it."""
    if fpy_type.kind in (TypeKind.STRING, TypeKind.INTERNAL_STRING):
        return True
    if fpy_type.kind is TypeKind.STRUCT:
        return any(bears_string(m.type) for m in fpy_type.members)
    if fpy_type.kind is TypeKind.ARRAY:
        return bears_string(fpy_type.elem_type)
    return False


def public_enum_members(fpy_type) -> list[str]:
    """The enum's members that a sequence may name (an underscore-prefixed
    member is library-internal)."""
    return [name for name in fpy_type.enum_dict if not name.startswith("_")]


def literal(fpy_type) -> str:
    """A sequence expression evaluating to a value of *fpy_type*."""
    kind = fpy_type.kind
    if kind in _INTEGER_KINDS:
        return "1"
    if kind in (TypeKind.F32, TypeKind.F64):
        return "1.0"
    if kind is TypeKind.BOOL:
        return "True"
    if kind in (TypeKind.STRING, TypeKind.INTERNAL_STRING):
        return '"x"'
    if kind is TypeKind.ENUM:
        return f"{fpy_type.name}.{public_enum_members(fpy_type)[0]}"
    if kind is TypeKind.STRUCT:
        args = ", ".join(literal(m.type) for m in fpy_type.members)
        return f"{fpy_type.name}({args})"
    if kind is TypeKind.ARRAY:
        args = ", ".join([literal(fpy_type.elem_type)] * fpy_type.length)
        return f"{fpy_type.name}({args})"
    raise AssertionError(f"no literal for {fpy_type.name} ({kind})")


def assert_declares(fprime_test_api, fpy_type, expression: str):
    """Declaring a variable of *fpy_type* from *expression* compiles, unless the
    type bears a string, in which case it is rejected for not being
    constant-sized."""
    seq = f"x: {fpy_type.name} = {expression}\n"
    if bears_string(fpy_type):
        assert_compile_failure(fprime_test_api, seq, match=NOT_CONSTANT_SIZED)
    else:
        assert_compile_success(fprime_test_api, seq)


class TestTypeSweep:
    """Every type in the dictionary can be constructed and stored."""

    @pytest.mark.parametrize("type_name", sorted(TYPE_DEFS), ids=str)
    def test_construct_and_store(self, fprime_test_api, type_name):
        fpy_type = TYPE_DEFS[type_name]
        assert_declares(fprime_test_api, fpy_type, literal(fpy_type))

    @pytest.mark.parametrize(
        "type_name",
        sorted(n for n, t in TYPE_DEFS.items() if t.kind is TypeKind.ENUM),
        ids=str,
    )
    def test_every_enum_member_resolves(self, fprime_test_api, type_name):
        fpy_type = TYPE_DEFS[type_name]
        members = public_enum_members(fpy_type)
        seq = "".join(
            f"x{i}: {type_name} = {type_name}.{member}\n"
            for i, member in enumerate(members)
        )
        assert_compile_success(fprime_test_api, seq)

    @pytest.mark.parametrize(
        "type_name",
        sorted(
            n
            for n, t in TYPE_DEFS.items()
            if t.kind is TypeKind.ENUM
            and len(public_enum_members(t)) != len(t.enum_dict)
        ),
        ids=str,
    )
    def test_internal_enum_member_is_rejected(self, fprime_test_api, type_name):
        fpy_type = TYPE_DEFS[type_name]
        internal = next(n for n in fpy_type.enum_dict if n.startswith("_"))
        assert_compile_failure(
            fprime_test_api,
            f"x: {type_name} = {type_name}.{internal}\n",
            match="library-internal definition",
        )


class TestTelemetrySweep:
    """Every telemetry channel in the dictionary can be read."""

    @pytest.mark.parametrize("channel", sorted(CHANNELS), ids=str)
    def test_read_channel(self, fprime_test_api, channel):
        assert_declares(fprime_test_api, CHANNELS[channel].ch_type, channel)


class TestParameterSweep:
    """Every parameter in the dictionary can be read."""

    @pytest.mark.parametrize("parameter", sorted(PARAMETERS), ids=str)
    def test_read_parameter(self, fprime_test_api, parameter):
        assert_declares(fprime_test_api, PARAMETERS[parameter].prm_type, parameter)


class TestConstantSweep:
    """Every constant in the dictionary can be read."""

    @pytest.mark.parametrize("constant", sorted(CONSTANTS), ids=str)
    def test_read_constant(self, fprime_test_api, constant):
        assert_declares(fprime_test_api, CONSTANTS[constant].type, constant)


class TestCommandSweep:
    """Every command in the dictionary can be called with arguments of its
    declared types."""

    @pytest.mark.parametrize("command", sorted(COMMANDS), ids=str)
    def test_call_command(self, fprime_test_api, command):
        cmd = COMMANDS[command]
        args = ", ".join(literal(arg_type) for _, _, arg_type in cmd.args)
        seq = f"{command}({args})\n"
        if command.endswith("RUN_ARGS"):
            assert_compile_failure(fprime_test_api, seq, match=NEEDS_BINARY_DIR)
        else:
            assert_compile_success(fprime_test_api, seq)

    @pytest.mark.parametrize(
        "command",
        # A sequence-run command resolves its path argument before its arity is
        # checked, so it cannot report a missing argument here.
        sorted(c for c, d in COMMANDS.items() if d.args and not c.endswith("RUN_ARGS")),
        ids=str,
    )
    def test_call_command_with_too_few_args(self, fprime_test_api, command):
        cmd = COMMANDS[command]
        args = ", ".join(literal(arg_type) for _, _, arg_type in cmd.args[:-1])
        assert_compile_failure(
            fprime_test_api, f"{command}({args})\n", match="Missing required argument"
        )
