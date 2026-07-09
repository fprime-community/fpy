from fpy.bytecode.directives import Directive, PopSerializableDirective
from fpy.model import DirectiveErrorCode
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_success,
    compile_seq,
)


class TestPop:

    def test_basic_u32(self, fprime_test_api):
        seq = '''
value: U32 = 42
pop(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, value)
'''
        assert_run_success(fprime_test_api, seq)

    def test_basic_u8(self, fprime_test_api):
        seq = '''
value: U8 = 100
pop(0, value)
'''
        assert_run_success(fprime_test_api, seq)

    def test_emits_pop_serializable_directive(self, fprime_test_api):
        seq = '''
value: U32 = 42
pop(0, value)
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopSerializableDirective)]
        assert len(pop_dirs) == 1
        assert pop_dirs[0].portIndex == 0
        assert pop_dirs[0].size == 4  # U32 is 4 bytes

    def test_correct_size_for_different_types(self, fprime_test_api):
        # size should match the byte width of the popped value
        seq = '''
v1: U8 = 1
pop(0, v1)
v2: U32 = 2
pop(1, v2)
v3: F64 = 3.0
pop(2, v3)
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopSerializableDirective)]
        assert len(pop_dirs) == 3
        assert pop_dirs[0].size == 1  # U8
        assert pop_dirs[1].size == 4  # U32
        assert pop_dirs[2].size == 8  # F64

    def test_enum_port_constant(self, fprime_test_api):
        # port index may be given as an enum constant
        seq = '''
value: U32 = 123
pop(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_2, value)
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopSerializableDirective)]
        assert len(pop_dirs) == 1
        assert pop_dirs[0].portIndex == 2

    def test_max_port_index(self, fprime_test_api):
        seq = '''
value: U32 = 42
pop(4, value)  # MAX_SERIAL_PORTS is 5, so 4 is valid
'''
        assert_run_success(fprime_test_api, seq)

    def test_port_out_of_bounds_high(self, fprime_test_api):
        # port index past the last serial port is rejected at compile time
        seq = '''
value: U32 = 42
pop(5, value)  # MAX_SERIAL_PORTS is 5, so 5 is out of range
'''
        assert_compile_failure(fprime_test_api, seq)

    def test_port_out_of_bounds_negative(self, fprime_test_api):
        seq = '''
value: U32 = 42
pop(-1, value)
'''
        assert_compile_failure(fprime_test_api, seq)

    def test_non_constant_port_rejected(self, fprime_test_api):
        # the port index must be a compile-time constant
        seq = '''
port: U32 = 0
value: U32 = 42
pop(port, value)
'''
        assert_compile_failure(fprime_test_api, seq)

    def test_non_constant_size_value_rejected(self, fprime_test_api):
        # strings are not constant-sized, so they can't be popped
        seq = '''
pop(0, "test")
'''
        assert_compile_failure(fprime_test_api, seq, match="constant-sized")

    def test_serialization_roundtrip(self, fprime_test_api):
        # the directive survives a serialize/deserialize round trip
        seq = '''
value: U32 = 42
pop(1, value)
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopSerializableDirective)]
        assert len(pop_dirs) == 1

        original = pop_dirs[0]
        serialized = original.serialize()
        _, deserialized = Directive.deserialize(serialized, 0)
        assert isinstance(deserialized, PopSerializableDirective)
        assert deserialized.portIndex == 1
        assert deserialized.size == 4

    def test_model_stack_underflow(self, fprime_test_api):
        # Popping more bytes than are on the stack is an underflow.  Tested at
        # the directive level because the builtin always pushes the value
        # before popping, so it can't underflow through normal usage.
        from fpy.model import FpySequencerModel

        model = FpySequencerModel()
        model.stack.extend([0x01, 0x02])  # only 2 bytes available

        directive = PopSerializableDirective(portIndex=0, size=4)  # ask for 4
        error_code = model.handle_pop_serializable(directive)
        assert error_code == DirectiveErrorCode.STACK_UNDERFLOW

    def test_model_execution_pops_bytes(self, fprime_test_api):
        seq = '''
value: U32 = 0x12345678
pop(0, value)
'''
        assert_run_success(fprime_test_api, seq)

    def test_expression_value(self, fprime_test_api):
        # a cast expression is a valid, constant-sized value
        seq = '''
pop(0, U32(100 + 200))
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopSerializableDirective)]
        assert len(pop_dirs) == 1
        assert pop_dirs[0].size == 4  # U32

    def test_multiple_pops(self, fprime_test_api):
        seq = '''
v1: U32 = 1
v2: U32 = 2
v3: U32 = 3
pop(0, v1)
pop(1, v2)
pop(2, v3)
'''
        assert_run_success(fprime_test_api, seq)

    def test_bare_integer_literal_rejected(self, fprime_test_api):
        # a bare int literal has no concrete type; it must be cast first
        seq = '''
pop(0, 42)
'''
        assert_compile_failure(fprime_test_api, seq, match="concrete type")

    def test_bare_float_literal_rejected(self, fprime_test_api):
        seq = '''
pop(0, 3.14)
'''
        assert_compile_failure(fprime_test_api, seq, match="concrete type")

    def test_bare_expression_rejected(self, fprime_test_api):
        seq = '''
pop(4, 100 + 200)
'''
        assert_compile_failure(fprime_test_api, seq, match="concrete type")

    def test_float_port_rejected(self, fprime_test_api):
        # the port index must be an integer or enum, not a float
        seq = '''
v: U32 = 42
pop(1.5, v)
'''
        assert_compile_failure(fprime_test_api, seq, match="integer or enum")

    def test_bool_port_rejected(self, fprime_test_api):
        seq = '''
v: U32 = 42
pop(True, v)
'''
        assert_compile_failure(fprime_test_api, seq, match="integer or enum")

    def test_struct_value(self, fprime_test_api):
        # Ref.SignalPair: F32 * 2 = 8 bytes
        seq = '''
v: Ref.SignalPair = Ref.SignalPair(time=1.0, value=2.0)
pop(0, v)
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopSerializableDirective)]
        assert len(pop_dirs) == 1
        assert pop_dirs[0].size == 8
        assert_run_success(fprime_test_api, seq)

    def test_array_value(self, fprime_test_api):
        # Ref.SignalSet: F32[4] = 16 bytes
        seq = '''
v: Ref.SignalSet = Ref.SignalSet(1.0, 2.0, 3.0, 4.0)
pop(0, v)
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopSerializableDirective)]
        assert len(pop_dirs) == 1
        assert pop_dirs[0].size == 16
        assert_run_success(fprime_test_api, seq)

    def test_enum_i32_value(self, fprime_test_api):
        # Ref.Choice: I32 representation = 4 bytes
        seq = '''
v: Ref.Choice = Ref.Choice.RED
pop(0, v)
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopSerializableDirective)]
        assert len(pop_dirs) == 1
        assert pop_dirs[0].size == 4
        assert_run_success(fprime_test_api, seq)

    def test_enum_u8_value(self, fprime_test_api):
        # Svc.BlockState: U8 representation = 1 byte
        seq = '''
v: Svc.BlockState = Svc.BlockState.NO_BLOCK
pop(0, v)
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopSerializableDirective)]
        assert len(pop_dirs) == 1
        assert pop_dirs[0].size == 1

    def test_nested_struct_value(self, fprime_test_api):
        # Ref.SignalInfo: SignalType(I32 enum)=4 + SignalSet(F32[4])=16
        #                 + SignalPairSet(SignalPair[4], 8 each)=32 => 52 bytes
        seq = '''
v: Ref.SignalInfo = Ref.SignalInfo( \\
    Ref.SignalType.TRIANGLE, \\
    Ref.SignalSet(0.0, 0.0, 0.0, 0.0), \\
    Ref.SignalPairSet( \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0), \\
        Ref.SignalPair(0.0, 0.0)))
pop(0, v)
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopSerializableDirective)]
        assert len(pop_dirs) == 1
        assert pop_dirs[0].size == 52
        assert_run_success(fprime_test_api, seq)

    def test_array_of_struct_value(self, fprime_test_api):
        # Ref.SignalPairSet: Ref.SignalPair[4], 8 bytes each => 32 bytes
        seq = '''
v: Ref.SignalPairSet = Ref.SignalPairSet( \\
    Ref.SignalPair(1.0, 2.0), \\
    Ref.SignalPair(3.0, 4.0), \\
    Ref.SignalPair(5.0, 6.0), \\
    Ref.SignalPair(7.0, 8.0))
pop(0, v)
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopSerializableDirective)]
        assert len(pop_dirs) == 1
        assert pop_dirs[0].size == 32
        assert_run_success(fprime_test_api, seq)

    def test_nested_array_value(self, fprime_test_api):
        # Ref.TooManyChoices: Ref.ManyChoices[2], each is Ref.Choice[2]
        #                     Ref.Choice is I32 => 2 * (2 * 4) = 16 bytes
        seq = '''
v: Ref.TooManyChoices = Ref.TooManyChoices( \\
    Ref.ManyChoices(Ref.Choice.ONE, Ref.Choice.TWO), \\
    Ref.ManyChoices(Ref.Choice.RED, Ref.Choice.BLUE))
pop(0, v)
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopSerializableDirective)]
        assert len(pop_dirs) == 1
        assert pop_dirs[0].size == 16

    def test_member_array_struct_value(self, fprime_test_api):
        # Ref.ChoiceSlurry: TooManyChoices=16 + Choice(I32)=4
        #                   + ChoicePair(2 * I32)=8 + U8[2]=2 => 30 bytes
        seq = '''
v: Ref.ChoiceSlurry = Ref.ChoiceSlurry( \\
    Ref.TooManyChoices( \\
        Ref.ManyChoices(Ref.Choice.ONE, Ref.Choice.TWO), \\
        Ref.ManyChoices(Ref.Choice.RED, Ref.Choice.BLUE)), \\
    Ref.Choice.ONE, \\
    Ref.ChoicePair(Ref.Choice.ONE, Ref.Choice.TWO), \\
    [1, 2])
pop(0, v)
'''
        directives, _ = compile_seq(fprime_test_api, seq)
        pop_dirs = [d for d in directives if isinstance(d, PopSerializableDirective)]
        assert len(pop_dirs) == 1
        assert pop_dirs[0].size == 30

    def test_struct_with_string_members_rejected(self, fprime_test_api):
        # a struct containing a string is not constant-sized
        seq = '''
pop(0, Ref.DpDemo.StructWithStringMembers("a", Ref.DpDemo.StringArray("x", "y")))
'''
        assert_compile_failure(fprime_test_api, seq, match="constant-sized")

    def test_string_array_rejected(self, fprime_test_api):
        # an array of strings is not constant-sized
        seq = '''
pop(0, Ref.DpDemo.StringArray("a", "b"))
'''
        assert_compile_failure(fprime_test_api, seq, match="constant-sized")
