"""
Tests for the assembler and disassembler.

These tests verify that:
1. Directives can be serialized and deserialized correctly (round-trip)
2. The assembler can parse bytecode text and produce correct directives
3. The disassembler can convert directives back to text
4. The full round-trip (text -> directives -> binary -> directives -> text) works
"""

import struct
import pytest
from dataclasses import fields

from fpy.bytecode.assembler import (
    parse as fpybc_parse,
    assemble,
    fpybc_directives_to_fpyasm,
    HEADER_FORMAT,
    HEADER_SIZE,
)
from fpy.bytecode.directives import (
    Directive,
    DirectiveId,
    StackOpDirective,
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
    PushRandDirective,
    PushTimeDirective,
    SetSeedDirective,
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
from fpy.bytecode.assembler import serialize_directives, deserialize_directives

# Every concrete directive, discovered from the class hierarchy so that a newly
# added directive is round-trip tested without editing this file.
ALL_DIRECTIVES = sorted(
    (
        c
        for c in Directive.__subclasses__() + StackOpDirective.__subclasses__()
        if c.opcode is not DirectiveId.INVALID
    ),
    key=lambda c: c.__name__,
)


def _representative(cls) -> Directive:
    """An instance of *cls* carrying a distinct in-range value in every field."""
    kwargs = {}
    for i, f in enumerate(fields(cls)):
        fpy_type = cls._FIELD_TYPES.get(f.name)
        if fpy_type is None:
            kwargs[f.name] = bytes(range(i + 1))
        elif fpy_type.name.startswith("I"):
            kwargs[f.name] = -(i + 1)
        else:
            kwargs[f.name] = i + 1
    return cls(**kwargs)


def assert_roundtrips(original: Directive):
    """Serializing one directive and reading it back yields an equal directive."""
    serialized, _ = serialize_directives([original])
    deserialized, arg_type_names = deserialize_directives(serialized)

    assert len(deserialized) == 1
    assert arg_type_names == []
    result = deserialized[0]
    assert type(result) == type(original)
    for field in fields(original):
        if field.name in ("meta", "id"):
            continue
        original_val = getattr(original, field.name)
        result_val = getattr(result, field.name)
        assert (
            original_val == result_val
        ), f"Field {field.name}: {original_val} != {result_val}"


class TestDirectiveSerializationRoundTrip:
    """Every directive survives serialize -> deserialize unchanged."""

    @pytest.mark.parametrize("cls", ALL_DIRECTIVES, ids=lambda c: c.__name__)
    def test_roundtrip(self, cls):
        assert_roundtrips(_representative(cls))

    def test_opcodes_are_unique(self):
        opcodes = [c.opcode for c in ALL_DIRECTIVES]
        assert len(set(opcodes)) == len(opcodes)

    # Field values _representative does not reach: the ends of each field's
    # range, and the empty and maximal byte strings.
    @pytest.mark.parametrize(
        "directive",
        [
            AllocateDirective(size=0),
            AllocateDirective(size=0xFFFFFFFF),
            StoreRelConstOffsetDirective(lvar_offset=-8, size=4),
            StoreRelConstOffsetDirective(lvar_offset=0x7FFFFFFF, size=4),
            StoreAbsConstOffsetDirective(global_offset=-1, size=4),
            LoadRelDirective(lvar_offset=0, size=8),
            LoadRelDirective(lvar_offset=-0x80000000, size=4),
            LoadAbsDirective(global_offset=50, size=8),
            DiscardDirective(size=0),
            GotoDirective(dir_idx=0),
            IfDirective(false_goto_dir_index=0),
            PushValDirective(val=b""),
            PushValDirective(val=b"\x42"),
            PushValDirective(val=bytes(range(256))),
            ConstCmdDirective(cmd_opcode=456, args=b""),
            ConstCmdDirective(cmd_opcode=0xFFFFFFFF, args=b"\x01\x02\x03"),
            PushTlmValDirective(chan_id=0),
            PushPrmDirective(prm_id=0xFFFFFFFF),
            StackCmdDirective(args_size=0),
            MemCompareDirective(size=0xFFFFFFFF),
            GetFieldDirective(parent_size=64, member_size=8),
            ReturnDirective(return_val_size=0, call_args_size=0),
        ],
        ids=lambda d: type(d).__name__.replace("Directive", ""),
    )
    def test_roundtrip_field_extremes(self, directive):
        assert_roundtrips(directive)


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
        """Every no-arg directive's mnemonic assembles back to its own class."""
        no_arg = [c for c in ALL_DIRECTIVES if not fields(c)]
        text = "".join(c.opcode.name.lower() + "\n" for c in no_arg)
        dirs = assemble(fpybc_parse(text))
        assert [type(d) for d in dirs] == no_arg

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
        text = fpybc_directives_to_fpyasm(dirs)
        assert text.strip() == "no_op"

    def test_disassemble_allocate(self):
        dirs = [AllocateDirective(size=100)]
        text = fpybc_directives_to_fpyasm(dirs)
        assert text.strip() == "allocate 100"

    def test_disassemble_load_rel(self):
        dirs = [LoadRelDirective(lvar_offset=-8, size=4)]
        text = fpybc_directives_to_fpyasm(dirs)
        assert text.strip() == "load_rel -8 4"

    def test_disassemble_push_val(self):
        dirs = [PushValDirective(val=b"\x01\x02\x03")]
        text = fpybc_directives_to_fpyasm(dirs)
        assert text.strip() == "push_val 1 2 3"

    def test_disassemble_goto(self):
        dirs = [GotoDirective(dir_idx=10)]
        text = fpybc_directives_to_fpyasm(dirs)
        assert text.strip() == "goto 10"

    def test_disassemble_const_cmd(self):
        dirs = [ConstCmdDirective(cmd_opcode=123, args=b"\x01\x02")]
        text = fpybc_directives_to_fpyasm(dirs)
        assert text.strip() == "const_cmd 123 1 2"

    def test_disassemble_multiple(self):
        dirs = [
            AllocateDirective(size=16),
            NoOpDirective(),
            ExitDirective(),
        ]
        text = fpybc_directives_to_fpyasm(dirs)
        lines = [line for line in text.strip().split("\n") if line]
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
        """text -> directives -> text -> directives is stable for every no-arg
        directive, and preserves the order and identity of each one."""
        no_arg = [c for c in ALL_DIRECTIVES if not fields(c)]
        text = "".join(c.opcode.name.lower() + "\n" for c in no_arg)

        dirs = assemble(fpybc_parse(text))
        dirs2 = assemble(fpybc_parse(fpybc_directives_to_fpyasm(dirs)))

        assert [type(d) for d in dirs] == no_arg
        assert [type(d) for d in dirs2] == no_arg

    def _test_text_roundtrip(self, original_text: str):
        """Helper to test text -> directives -> text round-trip."""
        dirs = assemble(fpybc_parse(original_text))
        dirs2 = assemble(fpybc_parse(fpybc_directives_to_fpyasm(dirs)))

        assert len(dirs) == len(dirs2)
        for d1, d2 in zip(dirs, dirs2):
            assert type(d1) == type(d2)
            for field in fields(d1):
                if field.name in ("meta", "id"):
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
        dirs2, arg_type_names = deserialize_directives(binary)
        assert arg_type_names == []

        # Step 4: Convert directives back to text
        result_text = fpybc_directives_to_fpyasm(dirs2)

        # Step 5: Parse the result text and compare directives
        body3 = fpybc_parse(result_text)
        dirs3 = assemble(body3)

        # Compare original directives with final directives
        assert len(dirs) == len(dirs3), f"Length mismatch: {len(dirs)} != {len(dirs3)}"
        for i, (d1, d3) in enumerate(zip(dirs, dirs3)):
            assert type(d1) == type(
                d3
            ), f"Type mismatch at {i}: {type(d1)} != {type(d3)}"
            for field in fields(d1):
                if field.name in ("meta", "id"):
                    continue
                v1 = getattr(d1, field.name)
                v3 = getattr(d3, field.name)
                assert (
                    v1 == v3
                ), f"Field {field.name} mismatch at directive {i}: {v1} != {v3}"


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
        deserialized, _ = deserialize_directives(serialized)
        assert deserialized[0].dir_idx == 0xFFFFFFFE

    def test_negative_offset(self):
        dirs = [LoadRelDirective(lvar_offset=-2147483648, size=4)]  # min I32
        serialized, _ = serialize_directives(dirs)
        deserialized, _ = deserialize_directives(serialized)
        assert deserialized[0].lvar_offset == -2147483648

    def test_max_positive_offset(self):
        dirs = [LoadRelDirective(lvar_offset=2147483647, size=4)]  # max I32
        serialized, _ = serialize_directives(dirs)
        deserialized, _ = deserialize_directives(serialized)
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
        deserialized, _ = deserialize_directives(serialized)

        assert len(deserialized) == len(dirs)
        for orig, d in zip(dirs, deserialized):
            assert type(orig) == type(d)

    def test_empty_directive_list(self):
        dirs = []
        serialized, _ = serialize_directives(dirs)
        deserialized, _ = deserialize_directives(serialized)
        assert len(deserialized) == 0

    def test_many_directives(self):
        # Create a sequence with many directives
        dirs = [NoOpDirective() for _ in range(100)]
        serialized, _ = serialize_directives(dirs)
        deserialized, _ = deserialize_directives(serialized)
        assert len(deserialized) == 100
        assert all(isinstance(d, NoOpDirective) for d in deserialized)


class TestArgSpecs:
    """Test that arg specs are correctly serialized and deserialized."""

    def test_no_arg_specs(self):
        dirs = [NoOpDirective()]
        serialized, _ = serialize_directives(dirs, arg_specs=[])
        _, arg_specs = deserialize_directives(serialized)
        assert arg_specs == []

    def test_none_arg_specs(self):
        dirs = [NoOpDirective()]
        serialized, _ = serialize_directives(dirs, arg_specs=None)
        _, arg_specs = deserialize_directives(serialized)
        assert arg_specs == []

    def test_single_primitive_type(self):
        dirs = [NoOpDirective()]
        serialized, _ = serialize_directives(dirs, arg_specs=[("x", "U32", 4)])
        _, arg_specs = deserialize_directives(serialized)
        assert arg_specs == [("x", "U32", 4)]

    def test_multiple_primitive_types(self):
        dirs = [NoOpDirective()]
        specs = [
            ("a", "U8", 1),
            ("b", "U16", 2),
            ("c", "U32", 4),
            ("d", "U64", 8),
            ("e", "I8", 1),
            ("f", "I16", 2),
            ("g", "I32", 4),
            ("h", "I64", 8),
            ("i", "F32", 4),
            ("j", "F64", 8),
            ("k", "bool", 1),
        ]
        serialized, _ = serialize_directives(dirs, arg_specs=specs)
        _, arg_specs = deserialize_directives(serialized)
        assert arg_specs == specs

    def test_qualified_type_names(self):
        dirs = [NoOpDirective()]
        specs = [
            ("x", "U32", 4),
            ("rec", "Svc.DpRecord", 128),
            ("arr", "Ref.DpDemo.U32Array", 256),
            ("en", "Fw.Enabled", 1),
        ]
        serialized, _ = serialize_directives(dirs, arg_specs=specs)
        _, arg_specs = deserialize_directives(serialized)
        assert arg_specs == specs

    def test_arg_specs_with_directives_roundtrip(self):
        dirs = [
            AllocateDirective(size=32),
            PushValDirective(val=b"\x01\x02\x03\x04"),
            ExitDirective(),
        ]
        specs = [("x", "U32", 4), ("y", "F64", 8)]
        serialized, _ = serialize_directives(dirs, arg_specs=specs)
        deserialized, arg_specs = deserialize_directives(serialized)
        assert arg_specs == specs
        assert len(deserialized) == 3
        assert isinstance(deserialized[0], AllocateDirective)
        assert deserialized[0].size == 32


class TestArgSpecsBinaryFormat:
    """Test the binary layout of arg specs in the serialized output."""

    def test_header_argument_count(self):
        """argumentCount in the header should match the number of arg specs."""
        dirs = [NoOpDirective()]
        specs = [("x", "U32", 4), ("y", "U8", 1)]
        serialized, _ = serialize_directives(dirs, arg_specs=specs)
        header = struct.unpack_from(HEADER_FORMAT, serialized)
        argument_count = header[4]  # argumentCount is 5th field
        assert argument_count == 2

    def test_header_argument_count_zero(self):
        dirs = [NoOpDirective()]
        serialized, _ = serialize_directives(dirs, arg_specs=[])
        header = struct.unpack_from(HEADER_FORMAT, serialized)
        argument_count = header[4]
        assert argument_count == 0

    def test_args_section_follows_header(self):
        """The arg specs section should begin immediately after the header."""
        dirs = [NoOpDirective()]
        specs = [("x", "U32", 4)]
        serialized, _ = serialize_directives(dirs, arg_specs=specs)
        # Names are serialized as strings with a FwSizeStoreType length prefix.
        # The default size store type is U16 (16 bits / 2 bytes).
        # After header: 2 byte arg_name_len + "x" (1 byte)
        #             + 2 byte type_name_len + "U32" (3 bytes) + U32 size (4 bytes)
        offset = HEADER_SIZE
        arg_name_len = struct.unpack_from("!H", serialized, offset)[0]
        assert arg_name_len == 1
        arg_name = serialized[offset + 2 : offset + 3].decode("utf-8")
        assert arg_name == "x"
        offset += 3
        type_name_len = struct.unpack_from("!H", serialized, offset)[0]
        assert type_name_len == 3
        type_name = serialized[offset + 2 : offset + 5].decode("utf-8")
        assert type_name == "U32"
        size = struct.unpack_from("!I", serialized, offset + 5)[0]
        assert size == 4

    def test_large_type_size_roundtrip(self):
        """Sizes up to 2^32-1 should survive round-trip."""
        dirs = [NoOpDirective()]
        specs = [("big", "BigStruct", 65535)]
        serialized, _ = serialize_directives(dirs, arg_specs=specs)
        _, arg_specs = deserialize_directives(serialized)
        assert arg_specs == specs

    def test_bad_crc_rejected(self):
        """Corrupting a byte should cause deserialization to fail with a CRC error."""
        dirs = [NoOpDirective()]
        serialized, _ = serialize_directives(dirs)
        corrupted = bytearray(serialized)
        # Corrupt the CRC footer itself (last 4 bytes) so the body parses
        # fine but the CRC check fails.
        corrupted[-1] ^= 0x01
        with pytest.raises(RuntimeError, match="CRC mismatch"):
            deserialize_directives(bytes(corrupted))
