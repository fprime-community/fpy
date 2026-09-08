import json
from pathlib import Path

import pytest

import fpy.error
from fpy.bytecode.directives import Directive, PopSerializableDirective
from fpy.compiler import text_to_ast, analyze_ast, analysis_to_fpybc_directives
from fpy.state import _build_global_scopes, get_base_compile_state
from fpy.bytecode.directives import DirectiveErrorCode
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_success,
    compile_seq,
    default_dictionary,
)

# The testbed wires each FpySequencer serialOut port to a typed check port on
# Ref.fpySerialRcvr that FW_ASSERTs the received value, so on a live deployment
# a write must carry exactly the expected value of the connected port's types:
#   EXAMPLE_PORT_0  U32 42
#   EXAMPLE_PORT_1  string "hello world"
#   EXAMPLE_PORT_2  Ref.FpyExampleStruct(flag=True, count=123, ratio=0.5)
#   EXAMPLE_PORT_3  Ref.FpyExampleArray(1, 2, 3)
#   EXAMPLE_PORT_4  Ref.FpyExampleMulti(Ref.FpyExampleEnum.ON, True, 2.5)
# Tests that write anything else are compile-only.


def pop_dirs(seq: str) -> list[PopSerializableDirective]:
    _, directives, _ = compile_seq(seq)
    return [d for d in directives if isinstance(d, PopSerializableDirective)]


class TestWriteToPort:

    def test_write_int(self, fprime_test_api):
        seq = """
value: U32 = 42
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, value)
"""
        assert_run_success(fprime_test_api, seq)

    def test_write_string(self, fprime_test_api):
        seq = """
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_1, "hello world")
"""
        assert_run_success(fprime_test_api, seq)

    def test_write_struct(self, fprime_test_api):
        seq = """
v: Ref.FpyExampleStruct = Ref.FpyExampleStruct(flag=True, count=123, ratio=0.5)
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_2, v)
"""
        assert_run_success(fprime_test_api, seq)

    def test_write_array(self, fprime_test_api):
        seq = """
v: Ref.FpyExampleArray = Ref.FpyExampleArray(1, 2, 3)
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_3, v)
"""
        assert_run_success(fprime_test_api, seq)

    def test_write_multi(self, fprime_test_api):
        # EXAMPLE_PORT_4's port takes three args (enum, bool, F64);
        # Ref.FpyExampleMulti serializes to the same byte layout
        seq = """
v: Ref.FpyExampleMulti = Ref.FpyExampleMulti(Ref.FpyExampleEnum.ON, True, 2.5)
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_4, v)
"""
        assert_run_success(fprime_test_api, seq)

    def test_multiple_writes(self, fprime_test_api):
        seq = """
value: U32 = 42
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, value)
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_1, "hello world")
s: Ref.FpyExampleStruct = Ref.FpyExampleStruct(flag=True, count=123, ratio=0.5)
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_2, s)
a: Ref.FpyExampleArray = Ref.FpyExampleArray(1, 2, 3)
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_3, a)
m: Ref.FpyExampleMulti = Ref.FpyExampleMulti(Ref.FpyExampleEnum.ON, True, 2.5)
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_4, m)
"""
        dirs = pop_dirs(seq)
        assert [d.portIndex for d in dirs] == [0, 1, 2, 3, 4]
        assert [d.size for d in dirs] == [4, 13, 9, 12, 10]
        assert_run_success(fprime_test_api, seq)

    def test_emits_pop_serializable_directive(self, fprime_test_api):
        seq = """
value: U32 = 42
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, value)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        assert dirs[0].portIndex == 0
        assert dirs[0].size == 4  # U32 is 4 bytes

    def test_correct_size_for_different_types(self, fprime_test_api):
        # size should match the byte width of the value
        seq = """
v1: U8 = 1
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v1)
v2: U32 = 2
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_1, v2)
v3: F64 = 3.0
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_2, v3)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 3
        assert dirs[0].size == 1  # U8
        assert dirs[1].size == 4  # U32
        assert dirs[2].size == 8  # F64

    def test_signed_int_values(self, fprime_test_api):
        # signed ints serialize at their byte width (I8=1, I16=2, I32=4, I64=8)
        seq = """
v1: I8 = -5
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v1)
v2: I16 = -5
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_1, v2)
v3: I32 = -5
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_2, v3)
v4: I64 = -5
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_3, v4)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 4
        assert dirs[0].size == 1  # I8
        assert dirs[1].size == 2  # I16
        assert dirs[2].size == 4  # I32
        assert dirs[3].size == 8  # I64

    def test_bool_value(self, fprime_test_api):
        # a bool serializes to a single byte
        seq = """
v: bool = True
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        assert dirs[0].size == 1

    def test_max_port_index(self, fprime_test_api):
        seq = """
value: U32 = 42
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_4, value)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        assert dirs[0].portIndex == 4

    def test_non_constant_port_rejected(self, fprime_test_api):
        # the port index must be a compile-time constant; an enum-typed variable
        # has the right type but is not const, so it is rejected
        seq = """
port: Svc.Fpy.SerialPortIndex = Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0
value: U32 = 42
write_to_port(port, value)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_enum_port_accepted(self, fprime_test_api):
        # the port is typed Svc.Fpy.SerialPortIndex; the enum constant resolves
        # to its integer index for the emitted directive
        seq = """
value: U32 = 42
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_2, value)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        assert dirs[0].portIndex == 2

    def test_bare_integer_port_rejected(self, fprime_test_api):
        # a bare integer no longer coerces to the SerialPortIndex enum port type
        seq = """
value: U32 = 42
write_to_port(0, value)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_constant_string_value(self, fprime_test_api):
        # A constant string literal has a compile-time-known length, so it is
        # serializable with a statically-known size and is accepted.
        seq = """
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, "asdf")
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        # FwSizeStoreType (U16, 2 bytes) length prefix + 4 bytes of "asdf"
        assert dirs[0].size == 6

    def test_empty_constant_string_value(self, fprime_test_api):
        seq = """
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, "")
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        assert dirs[0].size == 2  # just the U16 length prefix

    def test_non_constant_size_string_rejected(self, fprime_test_api):
        # A struct containing a runtime-variable-length string is not sized.
        seq = """
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, Ref.DpDemo.StructWithStringMembers("a", Ref.DpDemo.StringArray("x", "y")))
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_string_array_rejected(self, fprime_test_api):
        # an array of runtime-variable-length strings is not sized
        seq = """
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, Ref.DpDemo.StringArray("a", "b"))
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_serialization_roundtrip(self, fprime_test_api):
        # the directive survives a serialize/deserialize round trip
        seq = """
value: U32 = 42
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_1, value)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1

        original = dirs[0]
        serialized = original.serialize()
        _, deserialized = Directive.deserialize(serialized, 0)
        assert isinstance(deserialized, PopSerializableDirective)
        assert deserialized.portIndex == 1
        assert deserialized.size == 4

    def test_struct_serialization_roundtrip(self, fprime_test_api):
        # a struct-value directive survives a serialize/deserialize round trip
        seq = """
v: Ref.SignalPair = Ref.SignalPair(time=1.0, value=2.0)
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_2, v)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1

        original = dirs[0]
        serialized = original.serialize()
        _, deserialized = Directive.deserialize(serialized, 0)
        assert isinstance(deserialized, PopSerializableDirective)
        assert deserialized.portIndex == 2
        assert deserialized.size == 8

    def test_expression_value(self, fprime_test_api):
        # a cast expression is a valid, sized value
        seq = """
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, U32(100 + 200))
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        assert dirs[0].size == 4  # U32

    def test_bare_integer_literal_rejected(self, fprime_test_api):
        # a bare int literal has no serializable (binary) representation; it must
        # be cast to a concrete type first
        seq = """
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, 42)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_bare_float_literal_rejected(self, fprime_test_api):
        seq = """
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, 3.14)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_bare_expression_rejected(self, fprime_test_api):
        seq = """
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_4, 100 + 200)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_float_port_rejected(self, fprime_test_api):
        # the port must be a Svc.Fpy.SerialPortIndex enum constant, not a float
        seq = """
v: U32 = 42
write_to_port(1.5, v)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_struct_value(self, fprime_test_api):
        # Ref.SignalPair: F32 * 2 = 8 bytes
        seq = """
v: Ref.SignalPair = Ref.SignalPair(time=1.0, value=2.0)
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        assert dirs[0].size == 8

    def test_array_value(self, fprime_test_api):
        # Ref.SignalSet: F32[4] = 16 bytes
        seq = """
v: Ref.SignalSet = Ref.SignalSet(1.0, 2.0, 3.0, 4.0)
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        assert dirs[0].size == 16

    def test_enum_i32_value(self, fprime_test_api):
        # Ref.Choice: I32 representation = 4 bytes
        seq = """
v: Ref.Choice = Ref.Choice.RED
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        assert dirs[0].size == 4

    def test_enum_u8_value(self, fprime_test_api):
        # Svc.BlockState: U8 representation = 1 byte
        seq = """
v: Svc.BlockState = Svc.BlockState.NO_BLOCK
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        assert dirs[0].size == 1

    def test_nested_struct_value(self, fprime_test_api):
        # Ref.SignalInfo: SignalType(I32 enum)=4 + SignalSet(F32[4])=16
        #                 + SignalPairSet(SignalPair[4], 8 each)=32 => 52 bytes
        seq = """
v: Ref.SignalInfo = Ref.SignalInfo( \\
    Ref.SignalType.TRIANGLE, \\
    Ref.SignalSet(0.0, 0.0, 0.0, 0.0), \\
    Ref.SignalPairSet( \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0)))
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        assert dirs[0].size == 52

    def test_array_of_struct_value(self, fprime_test_api):
        # Ref.SignalPairSet: Ref.SignalPair[4], 8 bytes each => 32 bytes
        seq = """
v: Ref.SignalPairSet = Ref.SignalPairSet( \\
    Ref.SignalPair(1.0, 2.0), \\
    Ref.SignalPair(3.0, 4.0), \\
    Ref.SignalPair(5.0, 6.0), \\
    Ref.SignalPair(7.0, 8.0))
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        assert dirs[0].size == 32

    def test_nested_array_value(self, fprime_test_api):
        # Ref.TooManyChoices: Ref.ManyChoices[2], each is Ref.Choice[2]
        #                     Ref.Choice is I32 => 2 * (2 * 4) = 16 bytes
        seq = """
v: Ref.TooManyChoices = Ref.TooManyChoices( \\
    Ref.ManyChoices(Ref.Choice.ONE, Ref.Choice.TWO), \\
    Ref.ManyChoices(Ref.Choice.RED, Ref.Choice.BLUE))
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        assert dirs[0].size == 16

    def test_member_array_struct_value(self, fprime_test_api):
        # Ref.ChoiceSlurry: TooManyChoices=16 + Choice(I32)=4
        #                   + ChoicePair(2 * I32)=8 + U8[2]=2 => 30 bytes
        seq = """
v: Ref.ChoiceSlurry = Ref.ChoiceSlurry( \\
    Ref.TooManyChoices( \\
        Ref.ManyChoices(Ref.Choice.ONE, Ref.Choice.TWO), \\
        Ref.ManyChoices(Ref.Choice.RED, Ref.Choice.BLUE)), \\
    Ref.Choice.ONE, \\
    Ref.ChoicePair(Ref.Choice.ONE, Ref.Choice.TWO), \\
    [1, 2])
write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v)
"""
        dirs = pop_dirs(seq)
        assert len(dirs) == 1
        assert dirs[0].size == 30

    def test_port_enum_sourced_from_dictionary(self, fprime_test_api, tmp_path):
        # The SerialPortIndex enum is project-customizable: a dictionary that
        # renames EXAMPLE_PORT_0 must compile using the new name (and reject the
        # old one), proving the port type is not hardcoded in the compiler.
        d = json.loads(Path(default_dictionary).read_text())
        for t in d["typeDefinitions"]:
            if t.get("qualifiedName") == "Svc.Fpy.SerialPortIndex":
                for c in t["enumeratedConstants"]:
                    if c["name"] == "EXAMPLE_PORT_0":
                        c["name"] = "TIME_SYNC_PORT"
                t["default"] = "Svc.Fpy.SerialPortIndex.TIME_SYNC_PORT"
        custom_dict = tmp_path / "RenamedPort.json"
        custom_dict.write_text(json.dumps(d))

        def compile_with(dict_path: str, seq: str):
            fpy.error.file_name = "<custom-dict-test>"
            fpy.error.input_text = seq
            fpy.error.input_lines = seq.splitlines()
            _build_global_scopes.cache_clear()
            state = get_base_compile_state(dict_path)
            body = text_to_ast(seq)
            state = analyze_ast(body, state)
            return analysis_to_fpybc_directives(state)

        try:
            directives, _ = compile_with(
                str(custom_dict),
                "value: U32 = 42\n"
                "write_to_port(Svc.Fpy.SerialPortIndex.TIME_SYNC_PORT, value)\n",
            )
            dirs = [d for d in directives if isinstance(d, PopSerializableDirective)]
            assert len(dirs) == 1 and dirs[0].portIndex == 0

            # The renamed dictionary must reject the old EXAMPLE_PORT_0 name.
            with pytest.raises((fpy.error.CompileError, fpy.error.BackendError)):
                compile_with(
                    str(custom_dict),
                    "value: U32 = 42\n"
                    "write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, value)\n",
                )
        finally:
            # Drop cached scopes built from the custom dict so other tests see
            # the standard dictionary again.
            _build_global_scopes.cache_clear()
