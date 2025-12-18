"""
Tests for the assembler and disassembler.

These tests verify that:
1. Directives can be serialized and deserialized correctly (round-trip)
2. The assembler can parse bytecode text and produce correct directives
3. The disassembler can convert directives back to text
4. The full round-trip (text -> directives -> binary -> directives -> text) works
"""

import pytest
from dataclasses import fields

from fpy.bytecode.assembler import (
    parse as fpybc_parse,
    assemble,
    directives_to_fpybc,
)
from fpy.bytecode.directives import (
    Directive,
    # Directives with no args
    NoOpDirective,
    WaitRelDirective,
    WaitAbsDirective,
    OrDirective,
    AndDirective,
    IntEqualDirective,
    IntNotEqualDirective,
    UnsignedLessThanDirective,
    UnsignedLessThanOrEqualDirective,
    UnsignedGreaterThanDirective,
    UnsignedGreaterThanOrEqualDirective,
    SignedLessThanDirective,
    SignedLessThanOrEqualDirective,
    SignedGreaterThanDirective,
    SignedGreaterThanOrEqualDirective,
    FloatGreaterThanOrEqualDirective,
    FloatLessThanOrEqualDirective,
    FloatLessThanDirective,
    FloatGreaterThanDirective,
    FloatEqualDirective,
    FloatNotEqualDirective,
    NotDirective,
    FloatTruncateDirective,
    FloatExtendDirective,
    FloatToSignedIntDirective,
    SignedIntToFloatDirective,
    FloatToUnsignedIntDirective,
    UnsignedIntToFloatDirective,
    ExitDirective,
    PushTimeDirective,
    CallDirective,
    PeekDirective,
    IntAddDirective,
    IntSubtractDirective,
    IntMultiplyDirective,
    UnsignedIntDivideDirective,
    SignedIntDivideDirective,
    FloatAddDirective,
    FloatSubtractDirective,
    FloatMultiplyDirective,
    FloatDivideDirective,
    FloatExponentDirective,
    FloatLogDirective,
    FloatModuloDirective,
    SignedModuloDirective,
    UnsignedModuloDirective,
    IntegerSignedExtend8To64Directive,
    IntegerSignedExtend16To64Directive,
    IntegerSignedExtend32To64Directive,
    IntegerZeroExtend8To64Directive,
    IntegerZeroExtend16To64Directive,
    IntegerZeroExtend32To64Directive,
    IntegerTruncate64To8Directive,
    IntegerTruncate64To16Directive,
    IntegerTruncate64To32Directive,
    # Directives with args
    AllocateDirective,
    StoreRelDirective,
    StoreRelConstOffsetDirective,
    StoreAbsDirective,
    StoreAbsConstOffsetDirective,
    LoadRelDirective,
    LoadAbsDirective,
    DiscardDirective,
    PushValDirective,
    ConstCmdDirective,
    GotoDirective,
    IfDirective,
    PushTlmValDirective,
    PushPrmDirective,
    StackCmdDirective,
    MemCompareDirective,
    GetFieldDirective,
    ReturnDirective,
)
from fpy.types import serialize_directives, deserialize_directives


class TestDirectiveSerializationRoundTrip:
    """Test that each directive can be serialized and deserialized correctly."""

    def test_no_op(self):
        original = NoOpDirective()
        self._test_roundtrip(original)

    def test_wait_rel(self):
        original = WaitRelDirective()
        self._test_roundtrip(original)

    def test_wait_abs(self):
        original = WaitAbsDirective()
        self._test_roundtrip(original)

    def test_exit(self):
        original = ExitDirective()
        self._test_roundtrip(original)

    def test_push_time(self):
        original = PushTimeDirective()
        self._test_roundtrip(original)

    def test_call(self):
        original = CallDirective()
        self._test_roundtrip(original)

    def test_peek(self):
        original = PeekDirective()
        self._test_roundtrip(original)

    # Boolean operators
    def test_or(self):
        original = OrDirective()
        self._test_roundtrip(original)

    def test_and(self):
        original = AndDirective()
        self._test_roundtrip(original)

    def test_not(self):
        original = NotDirective()
        self._test_roundtrip(original)

    # Integer comparison operators
    def test_int_equal(self):
        original = IntEqualDirective()
        self._test_roundtrip(original)

    def test_int_not_equal(self):
        original = IntNotEqualDirective()
        self._test_roundtrip(original)

    def test_unsigned_less_than(self):
        original = UnsignedLessThanDirective()
        self._test_roundtrip(original)

    def test_unsigned_less_than_or_equal(self):
        original = UnsignedLessThanOrEqualDirective()
        self._test_roundtrip(original)

    def test_unsigned_greater_than(self):
        original = UnsignedGreaterThanDirective()
        self._test_roundtrip(original)

    def test_unsigned_greater_than_or_equal(self):
        original = UnsignedGreaterThanOrEqualDirective()
        self._test_roundtrip(original)

    def test_signed_less_than(self):
        original = SignedLessThanDirective()
        self._test_roundtrip(original)

    def test_signed_less_than_or_equal(self):
        original = SignedLessThanOrEqualDirective()
        self._test_roundtrip(original)

    def test_signed_greater_than(self):
        original = SignedGreaterThanDirective()
        self._test_roundtrip(original)

    def test_signed_greater_than_or_equal(self):
        original = SignedGreaterThanOrEqualDirective()
        self._test_roundtrip(original)

    # Float comparison operators
    def test_float_equal(self):
        original = FloatEqualDirective()
        self._test_roundtrip(original)

    def test_float_not_equal(self):
        original = FloatNotEqualDirective()
        self._test_roundtrip(original)

    def test_float_less_than(self):
        original = FloatLessThanDirective()
        self._test_roundtrip(original)

    def test_float_less_than_or_equal(self):
        original = FloatLessThanOrEqualDirective()
        self._test_roundtrip(original)

    def test_float_greater_than(self):
        original = FloatGreaterThanDirective()
        self._test_roundtrip(original)

    def test_float_greater_than_or_equal(self):
        original = FloatGreaterThanOrEqualDirective()
        self._test_roundtrip(original)

    # Integer arithmetic operators
    def test_int_add(self):
        original = IntAddDirective()
        self._test_roundtrip(original)

    def test_int_subtract(self):
        original = IntSubtractDirective()
        self._test_roundtrip(original)

    def test_int_multiply(self):
        original = IntMultiplyDirective()
        self._test_roundtrip(original)

    def test_unsigned_int_divide(self):
        original = UnsignedIntDivideDirective()
        self._test_roundtrip(original)

    def test_signed_int_divide(self):
        original = SignedIntDivideDirective()
        self._test_roundtrip(original)

    def test_signed_modulo(self):
        original = SignedModuloDirective()
        self._test_roundtrip(original)

    def test_unsigned_modulo(self):
        original = UnsignedModuloDirective()
        self._test_roundtrip(original)

    # Float arithmetic operators
    def test_float_add(self):
        original = FloatAddDirective()
        self._test_roundtrip(original)

    def test_float_subtract(self):
        original = FloatSubtractDirective()
        self._test_roundtrip(original)

    def test_float_multiply(self):
        original = FloatMultiplyDirective()
        self._test_roundtrip(original)

    def test_float_divide(self):
        original = FloatDivideDirective()
        self._test_roundtrip(original)

    def test_float_exponent(self):
        original = FloatExponentDirective()
        self._test_roundtrip(original)

    def test_float_log(self):
        original = FloatLogDirective()
        self._test_roundtrip(original)

    def test_float_modulo(self):
        original = FloatModuloDirective()
        self._test_roundtrip(original)

    # Type conversion operators
    def test_float_truncate(self):
        original = FloatTruncateDirective()
        self._test_roundtrip(original)

    def test_float_extend(self):
        original = FloatExtendDirective()
        self._test_roundtrip(original)

    def test_float_to_signed_int(self):
        original = FloatToSignedIntDirective()
        self._test_roundtrip(original)

    def test_signed_int_to_float(self):
        original = SignedIntToFloatDirective()
        self._test_roundtrip(original)

    def test_float_to_unsigned_int(self):
        original = FloatToUnsignedIntDirective()
        self._test_roundtrip(original)

    def test_unsigned_int_to_float(self):
        original = UnsignedIntToFloatDirective()
        self._test_roundtrip(original)

    # Integer extension/truncation
    def test_siext_8_64(self):
        original = IntegerSignedExtend8To64Directive()
        self._test_roundtrip(original)

    def test_siext_16_64(self):
        original = IntegerSignedExtend16To64Directive()
        self._test_roundtrip(original)

    def test_siext_32_64(self):
        original = IntegerSignedExtend32To64Directive()
        self._test_roundtrip(original)

    def test_ziext_8_64(self):
        original = IntegerZeroExtend8To64Directive()
        self._test_roundtrip(original)

    def test_ziext_16_64(self):
        original = IntegerZeroExtend16To64Directive()
        self._test_roundtrip(original)

    def test_ziext_32_64(self):
        original = IntegerZeroExtend32To64Directive()
        self._test_roundtrip(original)

    def test_itrunc_64_8(self):
        original = IntegerTruncate64To8Directive()
        self._test_roundtrip(original)

    def test_itrunc_64_16(self):
        original = IntegerTruncate64To16Directive()
        self._test_roundtrip(original)

    def test_itrunc_64_32(self):
        original = IntegerTruncate64To32Directive()
        self._test_roundtrip(original)

    # Directives with arguments
    def test_allocate(self):
        original = AllocateDirective(size=100)
        self._test_roundtrip(original)

    def test_allocate_zero(self):
        original = AllocateDirective(size=0)
        self._test_roundtrip(original)

    def test_allocate_large(self):
        original = AllocateDirective(size=0xFFFFFFFF)
        self._test_roundtrip(original)

    def test_store_rel(self):
        original = StoreRelDirective(size=8)
        self._test_roundtrip(original)

    def test_store_rel_const_offset(self):
        original = StoreRelConstOffsetDirective(lvar_offset=16, size=4)
        self._test_roundtrip(original)

    def test_store_rel_const_offset_negative(self):
        original = StoreRelConstOffsetDirective(lvar_offset=-8, size=4)
        self._test_roundtrip(original)

    def test_store_abs(self):
        original = StoreAbsDirective(size=8)
        self._test_roundtrip(original)

    def test_store_abs_const_offset(self):
        original = StoreAbsConstOffsetDirective(global_offset=100, size=4)
        self._test_roundtrip(original)

    def test_load_rel(self):
        original = LoadRelDirective(lvar_offset=0, size=8)
        self._test_roundtrip(original)

    def test_load_rel_negative_offset(self):
        original = LoadRelDirective(lvar_offset=-16, size=4)
        self._test_roundtrip(original)

    def test_load_abs(self):
        original = LoadAbsDirective(global_offset=50, size=8)
        self._test_roundtrip(original)

    def test_discard(self):
        original = DiscardDirective(size=4)
        self._test_roundtrip(original)

    def test_push_val_empty(self):
        original = PushValDirective(val=b"")
        self._test_roundtrip(original)

    def test_push_val_single_byte(self):
        original = PushValDirective(val=b"\x42")
        self._test_roundtrip(original)

    def test_push_val_multiple_bytes(self):
        original = PushValDirective(val=b"\x00\x01\x02\x03\x04\x05\x06\x07")
        self._test_roundtrip(original)

    def test_push_val_all_byte_values(self):
        original = PushValDirective(val=bytes(range(256)))
        self._test_roundtrip(original)

    def test_const_cmd(self):
        original = ConstCmdDirective(cmd_opcode=123, args=b"\x01\x02\x03")
        self._test_roundtrip(original)

    def test_const_cmd_empty_args(self):
        original = ConstCmdDirective(cmd_opcode=456, args=b"")
        self._test_roundtrip(original)

    def test_goto(self):
        original = GotoDirective(dir_idx=10)
        self._test_roundtrip(original)

    def test_goto_zero(self):
        original = GotoDirective(dir_idx=0)
        self._test_roundtrip(original)

    def test_if(self):
        original = IfDirective(false_goto_dir_index=5)
        self._test_roundtrip(original)

    def test_push_tlm_val(self):
        original = PushTlmValDirective(chan_id=100)
        self._test_roundtrip(original)

    def test_push_prm(self):
        original = PushPrmDirective(prm_id=200)
        self._test_roundtrip(original)

    def test_stack_cmd(self):
        original = StackCmdDirective(args_size=16)
        self._test_roundtrip(original)

    def test_memcmp(self):
        original = MemCompareDirective(size=32)
        self._test_roundtrip(original)

    def test_get_field(self):
        original = GetFieldDirective(parent_size=64, member_size=8)
        self._test_roundtrip(original)

    def test_return(self):
        original = ReturnDirective(return_val_size=8, call_args_size=16)
        self._test_roundtrip(original)

    def _test_roundtrip(self, original: Directive):
        """Helper to test serialize/deserialize round-trip for a single directive."""
        dirs = [original]
        serialized, _ = serialize_directives(dirs)
        deserialized = deserialize_directives(serialized)
        
        assert len(deserialized) == 1
        result = deserialized[0]
        
        # Check type matches
        assert type(result) == type(original)
        
        # Check all fields match
        for field in fields(original):
            if field.name in ('meta', 'id'):
                continue
            original_val = getattr(original, field.name)
            result_val = getattr(result, field.name)
            assert original_val == result_val, f"Field {field.name}: {original_val} != {result_val}"


class TestAssemblerParsing:
    """Test that the assembler can parse bytecode text correctly."""

    def test_parse_no_op(self):
        text = "no_op\n"
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 1
        assert isinstance(dirs[0], NoOpDirective)

    def test_parse_multiple_no_args(self):
        text = """
no_op
exit
add
sub
"""
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 4
        assert isinstance(dirs[0], NoOpDirective)
        assert isinstance(dirs[1], ExitDirective)
        assert isinstance(dirs[2], IntAddDirective)
        assert isinstance(dirs[3], IntSubtractDirective)

    def test_parse_allocate(self):
        text = "allocate 100\n"
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 1
        assert isinstance(dirs[0], AllocateDirective)
        assert dirs[0].size == 100

    def test_parse_load_rel(self):
        text = "load_rel -8 4\n"
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 1
        assert isinstance(dirs[0], LoadRelDirective)
        assert dirs[0].lvar_offset == -8
        assert dirs[0].size == 4

    def test_parse_store_rel_const_offset(self):
        text = "store_rel_const_offset -16 8\n"
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 1
        assert isinstance(dirs[0], StoreRelConstOffsetDirective)
        assert dirs[0].lvar_offset == -16
        assert dirs[0].size == 8

    def test_parse_push_val_with_bytes(self):
        text = "push_val 1 2 3 4\n"
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 1
        assert isinstance(dirs[0], PushValDirective)
        assert dirs[0].val == b"\x01\x02\x03\x04"

    def test_parse_push_val_with_hex(self):
        text = "push_val 0xFF 0x00 0xAB\n"
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 1
        assert isinstance(dirs[0], PushValDirective)
        assert dirs[0].val == b"\xff\x00\xab"

    def test_parse_push_val_empty(self):
        text = "push_val\n"
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 1
        assert isinstance(dirs[0], PushValDirective)
        assert dirs[0].val == b""

    def test_parse_const_cmd(self):
        text = "const_cmd 123 1 2 3\n"
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 1
        assert isinstance(dirs[0], ConstCmdDirective)
        assert dirs[0].cmd_opcode == 123
        assert dirs[0].args == b"\x01\x02\x03"

    def test_parse_goto_with_index(self):
        text = "goto 5\n"
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 1
        assert isinstance(dirs[0], GotoDirective)
        assert dirs[0].dir_idx == 5

    def test_parse_goto_with_tag(self):
        text = """
goto end
no_op
end:
exit
"""
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 3
        assert isinstance(dirs[0], GotoDirective)
        assert dirs[0].dir_idx == 2  # Points to the 'exit' instruction

    def test_parse_if_with_tag(self):
        text = """
if skip
no_op
skip:
exit
"""
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 3
        assert isinstance(dirs[0], IfDirective)
        assert dirs[0].false_goto_dir_index == 2

    def test_parse_comments(self):
        text = """
# This is a comment
no_op  # inline comment
# another comment
exit
"""
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 2

    def test_parse_all_no_arg_ops(self):
        """Test that all no-arg operations can be parsed."""
        ops = [
            "no_op", "siext_8_64", "siext_16_64", "siext_32_64",
            "ziext_8_64", "ziext_16_64", "ziext_32_64",
            "itrunc_64_8", "itrunc_64_16", "itrunc_64_32",
            "fmod", "smod", "umod", "add", "sub", "mul", "udiv", "sdiv",
            "fadd", "fsub", "fmul", "fpow", "fdiv", "flog",
            "wait_rel", "wait_abs", "or", "and",
            "ieq", "ine", "ult", "ule", "ugt", "uge",
            "slt", "sle", "sgt", "sge",
            "fge", "fle", "flt", "fgt", "feq", "fne",
            "not", "fptrunc", "fpext", "fptosi", "sitofp", "fptoui", "uitofp",
            "exit", "push_time", "call", "peek",
        ]
        for op in ops:
            text = f"{op}\n"
            body = fpybc_parse(text)
            dirs = assemble(body)
            assert len(dirs) == 1, f"Failed for {op}"

    def test_parse_get_field(self):
        text = "get_field 64 8\n"
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 1
        assert isinstance(dirs[0], GetFieldDirective)
        assert dirs[0].parent_size == 64
        assert dirs[0].member_size == 8

    def test_parse_return(self):
        text = "return 8 16\n"
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 1
        assert isinstance(dirs[0], ReturnDirective)
        assert dirs[0].return_val_size == 8
        assert dirs[0].call_args_size == 16


class TestDisassembler:
    """Test that directives_to_fpybc produces correct output."""

    def test_disassemble_no_op(self):
        dirs = [NoOpDirective()]
        text = directives_to_fpybc(dirs)
        assert text.strip() == "no_op"

    def test_disassemble_allocate(self):
        dirs = [AllocateDirective(size=100)]
        text = directives_to_fpybc(dirs)
        assert text.strip() == "allocate 100"

    def test_disassemble_load_rel(self):
        dirs = [LoadRelDirective(lvar_offset=-8, size=4)]
        text = directives_to_fpybc(dirs)
        assert text.strip() == "load_rel -8 4"

    def test_disassemble_push_val(self):
        dirs = [PushValDirective(val=b"\x01\x02\x03")]
        text = directives_to_fpybc(dirs)
        assert text.strip() == "push_val 1 2 3"

    def test_disassemble_goto(self):
        dirs = [GotoDirective(dir_idx=10)]
        text = directives_to_fpybc(dirs)
        assert text.strip() == "goto 10"

    def test_disassemble_const_cmd(self):
        dirs = [ConstCmdDirective(cmd_opcode=123, args=b"\x01\x02")]
        text = directives_to_fpybc(dirs)
        assert text.strip() == "const_cmd 123 1 2"

    def test_disassemble_multiple(self):
        dirs = [
            AllocateDirective(size=16),
            NoOpDirective(),
            ExitDirective(),
        ]
        text = directives_to_fpybc(dirs)
        lines = [line for line in text.strip().split('\n') if line]
        assert len(lines) == 3
        assert lines[0] == "allocate 16"
        assert lines[1] == "no_op"
        assert lines[2] == "exit"


class TestAssemblerDisassemblerRoundTrip:
    """Test that text -> directives -> text produces equivalent results."""

    def test_roundtrip_no_op(self):
        self._test_text_roundtrip("no_op\n")

    def test_roundtrip_allocate(self):
        self._test_text_roundtrip("allocate 100\n")

    def test_roundtrip_load_rel(self):
        self._test_text_roundtrip("load_rel -8 4\n")

    def test_roundtrip_push_val(self):
        self._test_text_roundtrip("push_val 1 2 3\n")

    def test_roundtrip_goto(self):
        self._test_text_roundtrip("goto 5\n")

    def test_roundtrip_if(self):
        self._test_text_roundtrip("if 10\n")

    def test_roundtrip_complex_sequence(self):
        text = """allocate 16
load_rel -8 4
push_val 1 2 3 4
add
store_rel 4
goto 0
exit
"""
        self._test_text_roundtrip(text)

    def test_roundtrip_all_no_arg_ops(self):
        """Test round-trip for all no-arg operations."""
        ops = [
            "no_op", "siext_8_64", "siext_16_64", "siext_32_64",
            "ziext_8_64", "ziext_16_64", "ziext_32_64",
            "itrunc_64_8", "itrunc_64_16", "itrunc_64_32",
            "fmod", "smod", "umod", "add", "sub", "mul", "udiv", "sdiv",
            "fadd", "fsub", "fmul", "fpow", "fdiv", "flog",
            "wait_rel", "wait_abs", "or", "and",
            "ieq", "ine", "ult", "ule", "ugt", "uge",
            "slt", "sle", "sgt", "sge",
            "fge", "fle", "flt", "fgt", "feq", "fne",
            "not", "fptrunc", "fpext", "fptosi", "sitofp", "fptoui", "uitofp",
            "exit", "push_time", "call", "peek",
        ]
        for op in ops:
            self._test_text_roundtrip(f"{op}\n")

    def _test_text_roundtrip(self, original_text: str):
        """Helper to test text -> directives -> text round-trip."""
        # Parse and assemble
        body = fpybc_parse(original_text)
        dirs = assemble(body)
        
        # Disassemble back to text
        result_text = directives_to_fpybc(dirs)
        
        # Parse again and compare directives
        body2 = fpybc_parse(result_text)
        dirs2 = assemble(body2)
        
        assert len(dirs) == len(dirs2)
        for d1, d2 in zip(dirs, dirs2):
            assert type(d1) == type(d2)
            for field in fields(d1):
                if field.name in ('meta', 'id'):
                    continue
                v1 = getattr(d1, field.name)
                v2 = getattr(d2, field.name)
                assert v1 == v2, f"{field.name}: {v1} != {v2}"


class TestFullRoundTrip:
    """Test the complete round-trip: text -> directives -> binary -> directives -> text."""

    def test_full_roundtrip_simple(self):
        text = "no_op\nexit\n"
        self._test_full_roundtrip(text)

    def test_full_roundtrip_with_args(self):
        text = """allocate 32
load_rel -16 8
push_val 0 1 2 3 4 5 6 7
add
store_rel_const_offset -8 8
discard 8
exit
"""
        self._test_full_roundtrip(text)

    def test_full_roundtrip_control_flow(self):
        text = """allocate 8
goto 3
no_op
if 5
no_op
exit
"""
        self._test_full_roundtrip(text)

    def test_full_roundtrip_arithmetic(self):
        text = """add
sub
mul
udiv
sdiv
fadd
fsub
fmul
fdiv
fpow
flog
fmod
smod
umod
"""
        self._test_full_roundtrip(text)

    def test_full_roundtrip_comparisons(self):
        text = """ieq
ine
ult
ule
ugt
uge
slt
sle
sgt
sge
feq
fne
flt
fle
fgt
fge
"""
        self._test_full_roundtrip(text)

    def test_full_roundtrip_type_conversions(self):
        text = """siext_8_64
siext_16_64
siext_32_64
ziext_8_64
ziext_16_64
ziext_32_64
itrunc_64_8
itrunc_64_16
itrunc_64_32
fptrunc
fpext
fptosi
sitofp
fptoui
uitofp
"""
        self._test_full_roundtrip(text)

    def test_full_roundtrip_const_cmd(self):
        text = "const_cmd 12345 0 1 2 3 4 5 6 7 8 9\n"
        self._test_full_roundtrip(text)

    def test_full_roundtrip_empty_push_val(self):
        text = "push_val\n"
        self._test_full_roundtrip(text)

    def test_full_roundtrip_large_push_val(self):
        # Generate a push_val with many bytes
        bytes_str = " ".join(str(i % 256) for i in range(100))
        text = f"push_val {bytes_str}\n"
        self._test_full_roundtrip(text)

    def _test_full_roundtrip(self, original_text: str):
        """Helper to test full round-trip: text -> dirs -> binary -> dirs -> text."""
        # Step 1: Parse and assemble original text to directives
        body = fpybc_parse(original_text)
        dirs = assemble(body)
        
        # Step 2: Serialize directives to binary
        binary, _ = serialize_directives(dirs)
        
        # Step 3: Deserialize binary back to directives
        dirs2 = deserialize_directives(binary)
        
        # Step 4: Convert directives back to text
        result_text = directives_to_fpybc(dirs2)
        
        # Step 5: Parse the result text and compare directives
        body3 = fpybc_parse(result_text)
        dirs3 = assemble(body3)
        
        # Compare original directives with final directives
        assert len(dirs) == len(dirs3), f"Length mismatch: {len(dirs)} != {len(dirs3)}"
        for i, (d1, d3) in enumerate(zip(dirs, dirs3)):
            assert type(d1) == type(d3), f"Type mismatch at {i}: {type(d1)} != {type(d3)}"
            for field in fields(d1):
                if field.name in ('meta', 'id'):
                    continue
                v1 = getattr(d1, field.name)
                v3 = getattr(d3, field.name)
                assert v1 == v3, f"Field {field.name} mismatch at directive {i}: {v1} != {v3}"


class TestEdgeCases:
    """Test edge cases and potential error conditions."""

    def test_unknown_goto_tag_fails(self):
        text = "goto unknown_tag\n"
        body = fpybc_parse(text)
        with pytest.raises(RuntimeError, match="Unknown tag"):
            assemble(body)

    def test_large_directive_index(self):
        # Test with a very large goto index
        dirs = [GotoDirective(dir_idx=0xFFFFFFFE)]
        serialized, _ = serialize_directives(dirs)
        deserialized = deserialize_directives(serialized)
        assert deserialized[0].dir_idx == 0xFFFFFFFE

    def test_negative_offset(self):
        dirs = [LoadRelDirective(lvar_offset=-2147483648, size=4)]  # min I32
        serialized, _ = serialize_directives(dirs)
        deserialized = deserialize_directives(serialized)
        assert deserialized[0].lvar_offset == -2147483648

    def test_max_positive_offset(self):
        dirs = [LoadRelDirective(lvar_offset=2147483647, size=4)]  # max I32
        serialized, _ = serialize_directives(dirs)
        deserialized = deserialize_directives(serialized)
        assert deserialized[0].lvar_offset == 2147483647

    def test_multiple_tags_same_location(self):
        text = """
tag1:
tag2:
no_op
goto tag1
goto tag2
"""
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 3
        assert dirs[1].dir_idx == 0
        assert dirs[2].dir_idx == 0

    def test_tag_at_end(self):
        text = """
no_op
goto end
exit
end:
"""
        body = fpybc_parse(text)
        dirs = assemble(body)
        assert len(dirs) == 3
        assert dirs[1].dir_idx == 3  # Points past the last instruction


class TestMultipleDirectives:
    """Test sequences with multiple directives."""

    def test_serialize_multiple_directives(self):
        dirs = [
            AllocateDirective(size=32),
            PushValDirective(val=b"\x01\x02\x03\x04"),
            StoreRelConstOffsetDirective(lvar_offset=-8, size=4),
            LoadRelDirective(lvar_offset=-8, size=4),
            IntAddDirective(),
            GotoDirective(dir_idx=3),
            ExitDirective(),
        ]
        serialized, _ = serialize_directives(dirs)
        deserialized = deserialize_directives(serialized)
        
        assert len(deserialized) == len(dirs)
        for orig, d in zip(dirs, deserialized):
            assert type(orig) == type(d)

    def test_empty_directive_list(self):
        dirs = []
        serialized, _ = serialize_directives(dirs)
        deserialized = deserialize_directives(serialized)
        assert len(deserialized) == 0

    def test_many_directives(self):
        # Create a sequence with many directives
        dirs = [NoOpDirective() for _ in range(100)]
        serialized, _ = serialize_directives(dirs)
        deserialized = deserialize_directives(serialized)
        assert len(deserialized) == 100
        assert all(isinstance(d, NoOpDirective) for d in deserialized)

