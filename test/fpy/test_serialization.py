"""Unit tests for FpyValue.serialize / FpyValue.deserialize -- the compiler's
reference implementation of the fprime wire format (Fw::SerializeBufferBase).

The correspondence with the C++ serializer must be exact: big-endian, tightly
packed, bools as the FW_SERIALIZE truth bytes (and on deserialize *only*
those), strings as an FwSizeStoreType length prefix plus the utf-8 bytes at
their actual length, and deserialization must reject what C++ rejects (a
truncated buffer, an invalid bool byte, an over-long string) rather than
guessing. Byte expectations are built with struct.pack so serialize is checked
against an independent encoding.
"""

import struct

import pytest

from fpy.dictionary import load_dictionary
from fpy.test_helpers import default_dictionary
from fpy.types import (
    BOOL,
    DeserializeError,
    F32,
    F64,
    FpyType,
    FpyValue,
    I8,
    I16,
    I32,
    I64,
    StructMember,
    TypeKind,
    U8,
    U16,
    U32,
    U64,
)

# String types aren't dictionary definitions; they're built ad hoc from each
# use site's size (see dictionary._resolve_type).
STRING_40 = FpyType(TypeKind.STRING, "String_40", max_length=40)


def _types():
    return load_dictionary(default_dictionary)["type_defs"]


def _byte_cases():
    """(value, expected wire bytes) pairs; asymmetric byte patterns so a
    byte-order bug is visible."""
    types = _types()
    choice = types["Ref.Choice"]
    many_choices = types["Ref.ManyChoices"]
    scalar_struct = types["Ref.ScalarStruct"]

    scalars = FpyValue(
        scalar_struct,
        {
            "i8": FpyValue(I8, -1),
            "i16": FpyValue(I16, -2),
            "i32": FpyValue(I32, -3),
            "i64": FpyValue(I64, -4),
            "u8": FpyValue(U8, 1),
            "u16": FpyValue(U16, 0x0102),
            "u32": FpyValue(U32, 0x01020304),
            "u64": FpyValue(U64, 0x0102030405060708),
            "f32": FpyValue(F32, 1.5),
            "f64": FpyValue(F64, -2.5),
        },
    )
    return [
        pytest.param(FpyValue(U8, 0xAB), struct.pack(">B", 0xAB), id="u8"),
        pytest.param(FpyValue(I8, -1), struct.pack(">b", -1), id="i8-neg"),
        pytest.param(FpyValue(U16, 0x0102), struct.pack(">H", 0x0102), id="u16"),
        pytest.param(FpyValue(I16, -2), struct.pack(">h", -2), id="i16-neg"),
        pytest.param(
            FpyValue(U32, 0x01020304), struct.pack(">I", 0x01020304), id="u32"
        ),
        pytest.param(
            FpyValue(I32, -123456789), struct.pack(">i", -123456789), id="i32-neg"
        ),
        pytest.param(
            FpyValue(U64, 0x0102030405060708),
            struct.pack(">Q", 0x0102030405060708),
            id="u64",
        ),
        pytest.param(
            FpyValue(I64, -3_000_000_000),
            struct.pack(">q", -3_000_000_000),
            id="i64-neg",
        ),
        pytest.param(FpyValue(F32, 1.5), struct.pack(">f", 1.5), id="f32"),
        pytest.param(FpyValue(F64, -2.5), struct.pack(">d", -2.5), id="f64-neg"),
        pytest.param(FpyValue(BOOL, True), b"\xff", id="bool-true"),
        pytest.param(FpyValue(BOOL, False), b"\x00", id="bool-false"),
        pytest.param(
            FpyValue(types["Fw.Enabled"], "ENABLED"),
            struct.pack(">B", 1),
            id="enum-u8-rep",
        ),
        pytest.param(FpyValue(choice, "RED"), struct.pack(">i", 2), id="enum-i32-rep"),
        pytest.param(
            FpyValue(STRING_40, "hi"), struct.pack(">H", 2) + b"hi", id="string"
        ),
        pytest.param(FpyValue(STRING_40, ""), struct.pack(">H", 0), id="string-empty"),
        pytest.param(
            FpyValue(STRING_40, "héllo✓"),
            struct.pack(">H", 9) + "héllo✓".encode("utf-8"),
            id="string-utf8-byte-length",
        ),
        pytest.param(
            FpyValue(many_choices, [FpyValue(choice, "TWO"), FpyValue(choice, "RED")]),
            struct.pack(">ii", 1, 2),
            id="array-of-enums",
        ),
        pytest.param(
            scalars,
            struct.pack(
                ">bhiqBHIQfd",
                -1,
                -2,
                -3,
                -4,
                1,
                0x0102,
                0x01020304,
                0x0102030405060708,
                1.5,
                -2.5,
            ),
            id="struct-every-scalar",
        ),
    ]


class TestSerialize:
    @pytest.mark.parametrize("value, expected", _byte_cases())
    def test_bytes_exact(self, value, expected):
        assert value.serialize() == expected

    def test_string_too_long_rejected(self):
        with pytest.raises(ValueError, match="too long"):
            FpyValue(STRING_40, "a" * 41).serialize()


class TestDeserialize:
    @pytest.mark.parametrize("value, expected", _byte_cases())
    def test_round_trip(self, value, expected):
        got, offset = FpyValue.deserialize(value.type, expected)
        assert offset == len(expected)
        assert got == value

    def test_offset_and_trailing_bytes(self):
        # Deserialization starts at the offset and leaves trailing bytes alone.
        data = b"\xaa" + struct.pack(">H", 0x0102) + b"\xbb"
        got, offset = FpyValue.deserialize(U16, data, 1)
        assert (got.val, offset) == (0x0102, 3)

    @pytest.mark.parametrize("byte", [b"\x01", b"\x7f", b"\xfe"])
    def test_bool_rejects_non_truth_bytes(self, byte):
        # C++ returns FW_DESERIALIZE_FORMAT_ERROR for anything that is not
        # exactly a truth value -- not "nonzero is true".
        with pytest.raises(DeserializeError, match="bool"):
            FpyValue.deserialize(BOOL, byte)

    def test_truncated_primitive_rejected(self):
        with pytest.raises(DeserializeError, match="too short"):
            FpyValue.deserialize(U32, b"\x01\x02\x03")

    def test_truncated_string_payload_rejected(self):
        # Prefix promises 5 bytes but only 2 follow: C++ size-mismatches
        # rather than truncating.
        data = struct.pack(">H", 5) + b"hi"
        with pytest.raises(DeserializeError, match="too short"):
            FpyValue.deserialize(STRING_40, data)

    def test_over_capacity_string_rejected(self):
        data = struct.pack(">H", 41) + b"a" * 41
        with pytest.raises(DeserializeError, match="max length"):
            FpyValue.deserialize(STRING_40, data)

    def test_unknown_enum_value_kept_as_int(self):
        # C++ enum deserialization static_casts without validating membership;
        # an unknown rep value is kept, not rejected.
        got, _ = FpyValue.deserialize(_types()["Ref.Choice"], struct.pack(">i", 99))
        assert got.val == 99

    def test_struct_member_error_propagates(self):
        # A bad bool inside an aggregate surfaces the member's error.
        struct_type = FpyType(
            TypeKind.STRUCT,
            "TestStruct",
            members=(StructMember("a", U16), StructMember("b", BOOL)),
        )
        with pytest.raises(DeserializeError, match="bool"):
            FpyValue.deserialize(struct_type, b"\x01\x02\x01")
