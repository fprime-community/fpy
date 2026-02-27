"""
Tests for the fpy.dictionary module — FPP JSON dictionary parser.

Covers every section of https://fprime.jpl.nasa.gov/latest/docs/reference/fpp-json-dict/:
  - Type Descriptors (integer, float, bool, string, qualifiedIdentifier)
  - Constants
  - Type Definitions (array, enum, struct, alias)
  - Values (primitive, float, bool, string, array, enum, struct, nested)
  - Commands (commandKind variants, formalParams, opcode)
  - Telemetry Channels (type, id, limits, telemetryUpdate)
  - Parameters (type, id, default)
  - Dictionary Metadata & Content structure
  - Serialization round-trips
  - Type max_size and classification
  - Integration tests against RefTopologyDictionary.json
  - Compiler integration (type constructors via _build_global_scopes)
"""

import json
import os
import tempfile

import pytest
from pathlib import Path

from fpy.dictionary import (
    _parse_channels,
    _parse_commands,
    _parse_constants,
    _parse_parameters,
    _parse_type_definitions,
    _resolve_type,
    json_default_to_fpy_value,
    load_dictionary,
    PRIMITIVE_TYPE_MAP,
)
from fpy.types import (
    BOOL,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    FpyType,
    FpyValue,
    StructMember,
    TypeKind,
)

REF_DICT_PATH = str(Path(__file__).parent / "RefTopologyDictionary.json")


# ===================================================================
# Helpers
# ===================================================================


def _write_dict(data: dict) -> str:
    """Write a dictionary to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    return f.name


def _minimal_dict(**overrides) -> dict:
    """Return a minimal valid dictionary structure matching the spec's
    Dictionary Content schema."""
    base = {
        "metadata": {
            "deploymentName": "TestDeployment",
            "frameworkVersion": "3.3.2",
            "projectVersion": "1.0.0",
            "libraryVersions": [],
            "dictionarySpecVersion": "1.0.0",
        },
        "typeDefinitions": [],
        "constants": [],
        "commands": [],
        "parameters": [],
        "events": [],
        "telemetryChannels": [],
        "records": [],
        "containers": [],
        "telemetryPacketSets": [],
    }
    base.update(overrides)
    return base


# ===================================================================
# Section: Type Descriptors
# ===================================================================


class TestTypeDescriptorUnsignedIntegers:
    """Spec §Type Descriptors / Primitive Integer / Unsigned."""

    @pytest.mark.parametrize(
        "name,size",
        [("U8", 8), ("U16", 16), ("U32", 32), ("U64", 64)],
    )
    def test_unsigned_integer_descriptor(self, name, size):
        desc = {"name": name, "kind": "integer", "size": size, "signed": False}
        result = _resolve_type(desc, {})
        assert result.kind.value == name
        assert result.is_unsigned

    @pytest.mark.parametrize(
        "name,expected",
        [("U8", U8), ("U16", U16), ("U32", U32), ("U64", U64)],
    )
    def test_unsigned_maps_to_singleton(self, name, expected):
        desc = {"name": name, "kind": "integer", "size": 64, "signed": False}
        assert _resolve_type(desc, {}) is expected


class TestTypeDescriptorSignedIntegers:
    """Spec §Type Descriptors / Primitive Integer / Signed."""

    @pytest.mark.parametrize(
        "name,size",
        [("I8", 8), ("I16", 16), ("I32", 32), ("I64", 64)],
    )
    def test_signed_integer_descriptor(self, name, size):
        desc = {"name": name, "kind": "integer", "size": size, "signed": True}
        result = _resolve_type(desc, {})
        assert result.kind.value == name
        assert result.is_signed

    @pytest.mark.parametrize(
        "name,expected",
        [("I8", I8), ("I16", I16), ("I32", I32), ("I64", I64)],
    )
    def test_signed_maps_to_singleton(self, name, expected):
        desc = {"name": name, "kind": "integer", "size": 16, "signed": True}
        assert _resolve_type(desc, {}) is expected


class TestTypeDescriptorFloats:
    """Spec §Type Descriptors / Floating-Point."""

    def test_f32_descriptor(self):
        desc = {"name": "F32", "kind": "float", "size": 32}
        assert _resolve_type(desc, {}) is F32

    def test_f64_descriptor(self):
        desc = {"name": "F64", "kind": "float", "size": 64}
        assert _resolve_type(desc, {}) is F64

    def test_unknown_float_size_rejected(self):
        desc = {"name": "F128", "kind": "float", "size": 128}
        with pytest.raises(AssertionError, match="Unknown float size"):
            _resolve_type(desc, {})


class TestTypeDescriptorBool:
    """Spec §Type Descriptors / Boolean."""

    def test_bool_descriptor(self):
        # Spec example: {"name": "bool", "kind": "bool", "size": 8}
        desc = {"name": "bool", "kind": "bool", "size": 8}
        assert _resolve_type(desc, {}) is BOOL

    def test_bool_is_not_integer(self):
        result = _resolve_type({"name": "bool", "kind": "bool", "size": 8}, {})
        assert not result.is_integer


class TestTypeDescriptorString:
    """Spec §Type Descriptors / String."""

    def test_string_descriptor(self):
        # Spec example: {"name": "string", "kind": "string", "size": 80}
        desc = {"name": "string", "kind": "string", "size": 80}
        result = _resolve_type(desc, {})
        assert result.is_string
        assert result.kind == TypeKind.STRING
        assert result.max_length == 80

    def test_string_different_sizes(self):
        for size in [1, 40, 80, 256, 1024]:
            desc = {"name": "string", "kind": "string", "size": size}
            result = _resolve_type(desc, {})
            assert result.max_length == size


class TestTypeDescriptorQualifiedIdentifier:
    """Spec §Type Descriptors / Qualified Identifier."""

    def test_qualified_identifier_resolves(self):
        # Spec example: {"name": "Module1.MyArray", "kind": "qualifiedIdentifier"}
        fake = FpyType(TypeKind.ARRAY, "Module1.MyArray", elem_type=U32, length=3)
        desc = {"name": "Module1.MyArray", "kind": "qualifiedIdentifier"}
        assert _resolve_type(desc, {"Module1.MyArray": fake}) is fake

    def test_qualified_identifier_missing_asserts(self):
        desc = {"name": "No.Such.Type", "kind": "qualifiedIdentifier"}
        with pytest.raises(AssertionError, match="Unknown type reference"):
            _resolve_type(desc, {})

    def test_unknown_kind_asserts(self):
        desc = {"name": "x", "kind": "banana"}
        with pytest.raises(AssertionError, match="Unknown type kind"):
            _resolve_type(desc, {})


class TestPrimitiveTypeMap:
    """Every entry in PRIMITIVE_TYPE_MAP should resolve correctly via _resolve_type."""

    def test_all_primitive_type_map_entries(self):
        for name, expected in PRIMITIVE_TYPE_MAP.items():
            desc = {"kind": "integer", "name": name}
            assert _resolve_type(desc, {}) is expected


# ===================================================================
# Section: Constants
# ===================================================================


class TestConstants:
    """Spec §Constants — qualifiedName, type, value, annotation."""

    def test_positive_integer_constant(self):
        # Spec: positive constants → U64 type
        raw = [
            {
                "qualifiedName": "M1.C",
                "type": {"name": "U64", "kind": "integer", "size": 64, "signed": False},
                "value": 1,
                "annotation": "Constant with value 1",
            }
        ]
        result = _parse_constants(raw, {})
        assert result["M1.C"] == 1

    def test_negative_integer_constant(self):
        # Spec: negative constants → I64 type
        raw = [
            {
                "qualifiedName": "M1.NEG",
                "type": {"name": "I64", "kind": "integer", "size": 64, "signed": True},
                "value": -42,
            }
        ]
        result = _parse_constants(raw, {})
        assert result["M1.NEG"] == -42

    def test_multiple_constants(self):
        raw = [
            {"qualifiedName": "A.B", "value": 100},
            {"qualifiedName": "C.D", "value": 200},
            {"qualifiedName": "E.F", "value": 300},
        ]
        result = _parse_constants(raw, {})
        assert len(result) == 3
        assert result["A.B"] == 100
        assert result["C.D"] == 200
        assert result["E.F"] == 300

    def test_empty_constants(self):
        assert _parse_constants([], {}) == {}


# ===================================================================
# Section: Type Definitions
# ===================================================================


class TestTypeDefArray:
    """Spec §Type Definitions / Array Type Definition."""

    def test_array_basic(self):
        # Spec example: array A = [3] U8
        raw = [
            {
                "kind": "array",
                "qualifiedName": "M1.A",
                "size": 3,
                "elementType": {
                    "name": "U8",
                    "kind": "integer",
                    "signed": False,
                    "size": 8,
                },
                "default": [0, 0, 0],
                "annotation": "My array named A",
            }
        ]
        result = _parse_type_definitions(raw)
        typ = result["M1.A"]
        assert typ.kind == TypeKind.ARRAY
        assert typ.length == 3
        assert typ.elem_type is U8
        assert typ.json_default == [0, 0, 0]

    def test_array_of_strings(self):
        # From spec full example: array StringArray = [2] string size 80
        raw = [
            {
                "kind": "array",
                "qualifiedName": "M.StringArray",
                "size": 2,
                "elementType": {"name": "string", "kind": "string", "size": 80},
                "default": ["A", "B"],
                "annotation": "An array of 2 String values",
            }
        ]
        result = _parse_type_definitions(raw)
        typ = result["M.StringArray"]
        assert typ.kind == TypeKind.ARRAY
        assert typ.length == 2
        assert typ.elem_type.is_string
        assert typ.elem_type.max_length == 80
        assert typ.json_default == ["A", "B"]

    def test_array_of_floats(self):
        raw = [
            {
                "kind": "array",
                "qualifiedName": "M.Vec3",
                "size": 3,
                "elementType": {"name": "F32", "kind": "float", "size": 32},
                "default": [0.0, 0.0, 0.0],
            }
        ]
        result = _parse_type_definitions(raw)
        typ = result["M.Vec3"]
        assert typ.elem_type is F32
        assert typ.length == 3

    def test_array_size_1(self):
        raw = [
            {
                "kind": "array",
                "qualifiedName": "M.Single",
                "size": 1,
                "elementType": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
                "default": [0],
            }
        ]
        result = _parse_type_definitions(raw)
        assert result["M.Single"].length == 1

    def test_array_no_default(self):
        raw = [
            {
                "kind": "array",
                "qualifiedName": "M.NoDef",
                "size": 2,
                "elementType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
            }
        ]
        result = _parse_type_definitions(raw)
        assert result["M.NoDef"].json_default is None


class TestTypeDefEnum:
    """Spec §Type Definitions / Enumeration Type Definition."""

    def test_enum_with_default(self):
        # Spec example: enum Status { YES NO MAYBE } default MAYBE
        raw = [
            {
                "kind": "enum",
                "qualifiedName": "M1.Status",
                "representationType": {
                    "name": "I32",
                    "kind": "integer",
                    "signed": True,
                    "size": 32,
                },
                "enumeratedConstants": [
                    {"name": "YES", "value": 0},
                    {"name": "NO", "value": 1},
                    {"name": "MAYBE", "value": 2, "annotation": "The cat would know"},
                ],
                "default": "M1.Status.MAYBE",
                "annotation": "Schroedinger's status",
            }
        ]
        result = _parse_type_definitions(raw)
        typ = result["M1.Status"]
        assert typ.kind == TypeKind.ENUM
        assert typ.enum_dict == {"YES": 0, "NO": 1, "MAYBE": 2}
        assert typ.json_default == "M1.Status.MAYBE"

    def test_enum_representation_type(self):
        """Enum rep type must be an integer type."""
        raw = [
            {
                "kind": "enum",
                "qualifiedName": "M.E",
                "representationType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                "enumeratedConstants": [{"name": "A", "value": 0}],
            }
        ]
        result = _parse_type_definitions(raw)
        assert result["M.E"].rep_type is U8

    @pytest.mark.parametrize("rep", ["U8", "U16", "U32", "U64", "I8", "I16", "I32", "I64"])
    def test_all_valid_enum_rep_types(self, rep):
        """Spec allows U8-U64, I8-I64 as representation types."""
        signed = rep.startswith("I")
        size = int(rep[1:])
        raw = [
            {
                "kind": "enum",
                "qualifiedName": f"M.E_{rep}",
                "representationType": {"name": rep, "kind": "integer", "size": size, "signed": signed},
                "enumeratedConstants": [{"name": "X", "value": 0}],
            }
        ]
        result = _parse_type_definitions(raw)
        assert result[f"M.E_{rep}"].kind == TypeKind.ENUM

    def test_enum_no_default(self):
        raw = [
            {
                "kind": "enum",
                "qualifiedName": "M.NoDef",
                "representationType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                "enumeratedConstants": [{"name": "A", "value": 0}],
            }
        ]
        result = _parse_type_definitions(raw)
        assert result["M.NoDef"].json_default is None

    def test_enum_many_constants(self):
        consts = [{"name": f"C{i}", "value": i} for i in range(20)]
        raw = [
            {
                "kind": "enum",
                "qualifiedName": "M.Big",
                "representationType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                "enumeratedConstants": consts,
            }
        ]
        result = _parse_type_definitions(raw)
        assert len(result["M.Big"].enum_dict) == 20


class TestTypeDefStruct:
    """Spec §Type Definitions / Struct Type Definition."""

    def test_struct_basic(self):
        # Spec example: struct S { w: [3] U32, x: U32, y: F32 }
        # Note: "w" uses an inline array via the "size" key on the member
        raw = [
            {
                "kind": "struct",
                "qualifiedName": "M1.S",
                "annotation": "Struct for wxy values",
                "members": {
                    "w": {
                        "type": {"name": "U32", "kind": "integer", "signed": False, "size": 32},
                        "index": 0,
                        "size": 3,
                        "annotation": "This is an array",
                    },
                    "x": {
                        "type": {"name": "U32", "kind": "integer", "signed": False, "size": 32},
                        "format": "the count is {}",
                        "index": 1,
                    },
                    "y": {
                        "type": {"name": "F32", "kind": "float", "size": 32},
                        "index": 2,
                    },
                },
                "default": {"w": [0, 0, 0], "x": 0, "y": 0},
            }
        ]
        result = _parse_type_definitions(raw)
        typ = result["M1.S"]
        assert typ.kind == TypeKind.STRUCT
        assert len(typ.members) == 3
        # "w" should be resolved to an array type (inline member array)
        assert typ.members[0].name == "w"
        assert typ.members[0].type.kind == TypeKind.ARRAY
        assert typ.members[0].type.length == 3
        assert typ.members[1].name == "x"
        assert typ.members[1].type is U32
        assert typ.members[2].name == "y"
        assert typ.members[2].type is F32
        assert typ.json_default == {"w": [0, 0, 0], "x": 0, "y": 0}

    def test_struct_member_order_by_index(self):
        """Spec: members are keyed by name; index field determines order."""
        raw = [
            {
                "kind": "struct",
                "qualifiedName": "M.Rev",
                "members": {
                    "z": {"type": {"name": "U8", "kind": "integer", "size": 8, "signed": False}, "index": 2},
                    "a": {"type": {"name": "U8", "kind": "integer", "size": 8, "signed": False}, "index": 0},
                    "m": {"type": {"name": "U8", "kind": "integer", "size": 8, "signed": False}, "index": 1},
                },
            }
        ]
        result = _parse_type_definitions(raw)
        member_names = [m.name for m in result["M.Rev"].members]
        assert member_names == ["a", "m", "z"]

    def test_struct_with_enum_member(self):
        raw = [
            {
                "kind": "enum",
                "qualifiedName": "M.Status",
                "representationType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                "enumeratedConstants": [{"name": "OK", "value": 0}, {"name": "ERR", "value": 1}],
            },
            {
                "kind": "struct",
                "qualifiedName": "M.Result",
                "members": {
                    "code": {"type": {"name": "U32", "kind": "integer", "size": 32, "signed": False}, "index": 0},
                    "status": {"type": {"name": "M.Status", "kind": "qualifiedIdentifier"}, "index": 1},
                },
                "default": {"code": 0, "status": "M.Status.OK"},
            },
        ]
        result = _parse_type_definitions(raw)
        s = result["M.Result"]
        assert s.members[1].type.kind == TypeKind.ENUM

    def test_struct_no_default(self):
        raw = [
            {
                "kind": "struct",
                "qualifiedName": "M.NoDef",
                "members": {
                    "a": {"type": {"name": "U8", "kind": "integer", "size": 8, "signed": False}, "index": 0},
                },
            }
        ]
        result = _parse_type_definitions(raw)
        assert result["M.NoDef"].json_default is None

    def test_struct_single_member(self):
        raw = [
            {
                "kind": "struct",
                "qualifiedName": "M.One",
                "members": {
                    "only": {"type": {"name": "U32", "kind": "integer", "size": 32, "signed": False}, "index": 0},
                },
            }
        ]
        result = _parse_type_definitions(raw)
        assert len(result["M.One"].members) == 1

    def test_struct_referencing_array(self):
        """Struct with a member whose type is a separately-defined array."""
        raw = [
            {
                "kind": "array",
                "qualifiedName": "M.Vec3",
                "size": 3,
                "elementType": {"name": "F32", "kind": "float", "size": 32},
            },
            {
                "kind": "struct",
                "qualifiedName": "M.Pose",
                "members": {
                    "position": {"type": {"name": "M.Vec3", "kind": "qualifiedIdentifier"}, "index": 0},
                    "heading": {"type": {"name": "F32", "kind": "float", "size": 32}, "index": 1},
                },
            },
        ]
        result = _parse_type_definitions(raw)
        pose = result["M.Pose"]
        assert pose.members[0].type.kind == TypeKind.ARRAY
        assert pose.members[0].type.length == 3


class TestTypeDefAlias:
    """Spec §Type Definitions / Type Alias Definition."""

    def test_alias_to_primitive(self):
        # Spec example: type A1 = U32
        raw = [
            {
                "kind": "alias",
                "qualifiedName": "M1.A1",
                "type": {"name": "U32", "kind": "integer", "signed": False, "size": 32},
                "underlyingType": {"name": "U32", "kind": "integer", "signed": False, "size": 32},
                "annotation": "Alias of type U32",
            }
        ]
        result = _parse_type_definitions(raw)
        assert result["M1.A1"] is U32

    def test_alias_chain(self):
        # Spec example: type A2 = A1, where A1 = U32
        raw = [
            {
                "kind": "alias",
                "qualifiedName": "M1.A1",
                "type": {"name": "U32", "kind": "integer", "signed": False, "size": 32},
                "underlyingType": {"name": "U32", "kind": "integer", "signed": False, "size": 32},
            },
            {
                "kind": "alias",
                "qualifiedName": "M1.A2",
                "type": {"name": "M1.A1", "kind": "qualifiedIdentifier"},
                "underlyingType": {"name": "U32", "kind": "integer", "signed": False, "size": 32},
            },
        ]
        result = _parse_type_definitions(raw)
        assert result["M1.A1"] is U32
        assert result["M1.A2"] is U32

    def test_alias_to_enum(self):
        raw = [
            {
                "kind": "enum",
                "qualifiedName": "M.E",
                "representationType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                "enumeratedConstants": [{"name": "A", "value": 0}],
            },
            {
                "kind": "alias",
                "qualifiedName": "M.EA",
                "type": {"name": "M.E", "kind": "qualifiedIdentifier"},
                "underlyingType": {"name": "M.E", "kind": "qualifiedIdentifier"},
            },
        ]
        result = _parse_type_definitions(raw)
        assert result["M.EA"].kind == TypeKind.ENUM

    def test_alias_reverse_order(self):
        """Alias A references B which references U32, but A appears first."""
        raw = [
            {
                "kind": "alias",
                "qualifiedName": "M.A",
                "type": {"name": "M.B", "kind": "qualifiedIdentifier"},
                "underlyingType": {"name": "M.B", "kind": "qualifiedIdentifier"},
            },
            {
                "kind": "alias",
                "qualifiedName": "M.B",
                "type": {"name": "U32", "kind": "integer"},
                "underlyingType": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
            },
        ]
        result = _parse_type_definitions(raw)
        assert result["M.A"] is U32
        assert result["M.B"] is U32


class TestTypeDefUnknownKind:
    """Unknown type definition kinds should assert."""

    def test_unknown_kind_rejected(self):
        raw = [{"kind": "union", "qualifiedName": "M.Bad"}]
        with pytest.raises(AssertionError, match="Unknown type definition kind"):
            _parse_type_definitions(raw)


class TestTypeDefEmpty:
    def test_empty(self):
        assert _parse_type_definitions([]) == {}


class TestTypeDefCrossReferences:
    """Multiple passes should resolve cross-references between arrays and structs."""

    def test_struct_referencing_struct(self):
        raw = [
            {
                "kind": "struct",
                "qualifiedName": "M.Inner",
                "members": {
                    "a": {"type": {"name": "U32", "kind": "integer", "size": 32, "signed": False}, "index": 0},
                },
            },
            {
                "kind": "struct",
                "qualifiedName": "M.Outer",
                "members": {
                    "inner": {"type": {"name": "M.Inner", "kind": "qualifiedIdentifier"}, "index": 0},
                    "flag": {"type": {"name": "bool", "kind": "bool"}, "index": 1},
                },
            },
        ]
        result = _parse_type_definitions(raw)
        assert result["M.Outer"].members[0].type.kind == TypeKind.STRUCT
        assert result["M.Outer"].members[1].type is BOOL

    def test_array_of_enum(self):
        raw = [
            {
                "kind": "enum",
                "qualifiedName": "M.Dir",
                "representationType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                "enumeratedConstants": [
                    {"name": "UP", "value": 0},
                    {"name": "DOWN", "value": 1},
                ],
            },
            {
                "kind": "array",
                "qualifiedName": "M.Dirs",
                "size": 4,
                "elementType": {"name": "M.Dir", "kind": "qualifiedIdentifier"},
            },
        ]
        result = _parse_type_definitions(raw)
        assert result["M.Dirs"].elem_type.kind == TypeKind.ENUM

    def test_array_of_struct(self):
        raw = [
            {
                "kind": "struct",
                "qualifiedName": "M.Point",
                "members": {
                    "x": {"type": {"name": "F32", "kind": "float", "size": 32}, "index": 0},
                    "y": {"type": {"name": "F32", "kind": "float", "size": 32}, "index": 1},
                },
            },
            {
                "kind": "array",
                "qualifiedName": "M.Points",
                "size": 10,
                "elementType": {"name": "M.Point", "kind": "qualifiedIdentifier"},
            },
        ]
        result = _parse_type_definitions(raw)
        assert result["M.Points"].elem_type.kind == TypeKind.STRUCT
        assert result["M.Points"].length == 10


# ===================================================================
# Section: Values — json_default_to_fpy_value
# ===================================================================


class TestValuePrimitiveInteger:
    """Spec §Values / Primitive Integer Values."""

    @pytest.mark.parametrize(
        "typ,val",
        [(U8, 2), (U16, 1000), (U32, 100000), (U64, 2**60), (I8, -2), (I16, -100), (I32, -100000), (I64, -(2**60))],
    )
    def test_integer_values(self, typ, val):
        result = json_default_to_fpy_value(val, typ)
        assert result == FpyValue(typ, val)

    def test_zero(self):
        assert json_default_to_fpy_value(0, U32) == FpyValue(U32, 0)

    def test_wrong_type_asserts(self):
        with pytest.raises(AssertionError, match="Expected int default"):
            json_default_to_fpy_value("not_int", U32)


class TestValueFloat:
    """Spec §Values / Floating-Point Values."""

    def test_f32_value(self):
        result = json_default_to_fpy_value(10.5, F32)
        assert result == FpyValue(F32, 10.5)
        assert isinstance(result.val, float)

    def test_f64_value(self):
        result = json_default_to_fpy_value(3.14159265358979, F64)
        assert result == FpyValue(F64, 3.14159265358979)

    def test_float_from_int(self):
        """JSON may encode 0 as integer, should still work for float types."""
        result = json_default_to_fpy_value(0, F32)
        assert result == FpyValue(F32, 0.0)
        assert isinstance(result.val, float)

    def test_wrong_type_asserts(self):
        with pytest.raises(AssertionError, match="Expected float default"):
            json_default_to_fpy_value("nope", F32)


class TestValueBool:
    """Spec §Values / Boolean Values."""

    def test_true(self):
        assert json_default_to_fpy_value(True, BOOL) == FpyValue(BOOL, True)

    def test_false(self):
        assert json_default_to_fpy_value(False, BOOL) == FpyValue(BOOL, False)

    def test_wrong_type_asserts(self):
        with pytest.raises(AssertionError, match="Expected bool default"):
            json_default_to_fpy_value(1, BOOL)


class TestValueString:
    """Spec §Values / String Values."""

    def test_string_value(self):
        st = FpyType(TypeKind.STRING, "String_80", max_length=80)
        result = json_default_to_fpy_value("Hello World!", st)
        assert result == FpyValue(st, "Hello World!")

    def test_empty_string(self):
        st = FpyType(TypeKind.STRING, "String_80", max_length=80)
        result = json_default_to_fpy_value("", st)
        assert result.val == ""

    def test_wrong_type_asserts(self):
        st = FpyType(TypeKind.STRING, "String_80", max_length=80)
        with pytest.raises(AssertionError, match="Expected str default"):
            json_default_to_fpy_value(42, st)


class TestValueArray:
    """Spec §Values / Array Values."""

    def test_array_of_u32(self):
        # Spec example: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        arr = FpyType(TypeKind.ARRAY, "M.Ten", elem_type=U32, length=10)
        raw = list(range(10))
        result = json_default_to_fpy_value(raw, arr)
        assert len(result.val) == 10
        for i in range(10):
            assert result.val[i] == FpyValue(U32, i)

    def test_wrong_length_asserts(self):
        arr = FpyType(TypeKind.ARRAY, "M.Two", elem_type=U32, length=2)
        with pytest.raises(AssertionError, match="Array default length"):
            json_default_to_fpy_value([1, 2, 3], arr)

    def test_array_of_floats(self):
        arr = FpyType(TypeKind.ARRAY, "M.Fs", elem_type=F32, length=3)
        result = json_default_to_fpy_value([1.0, 2.0, 3.0], arr)
        assert all(isinstance(v.val, float) for v in result.val)

    def test_not_list_asserts(self):
        arr = FpyType(TypeKind.ARRAY, "M.Arr", elem_type=U8, length=2)
        with pytest.raises(AssertionError, match="Expected list default"):
            json_default_to_fpy_value("not_list", arr)


class TestValueEnum:
    """Spec §Values / Enumeration Values — qualified string name."""

    def test_enum_value_qualified(self):
        # Spec example: "Status.YES"
        status = FpyType(
            TypeKind.ENUM,
            "Status",
            enum_dict={"YES": 0, "NO": 1, "MAYBE": 2},
            rep_type=I32,
        )
        result = json_default_to_fpy_value("Status.YES", status)
        assert result == FpyValue(status, "YES")

    def test_enum_value_module_qualified(self):
        status = FpyType(
            TypeKind.ENUM,
            "M1.Status",
            enum_dict={"YES": 0, "NO": 1, "MAYBE": 2},
            rep_type=I32,
        )
        result = json_default_to_fpy_value("M1.Status.MAYBE", status)
        assert result.val == "MAYBE"

    def test_unknown_constant_asserts(self):
        e = FpyType(TypeKind.ENUM, "M.E", enum_dict={"A": 0}, rep_type=U8)
        with pytest.raises(AssertionError, match="Unknown enum constant"):
            json_default_to_fpy_value("M.E.MISSING", e)


class TestValueStruct:
    """Spec §Values / Struct Values — dict of member name -> value."""

    def test_struct_value(self):
        st = FpyType(
            TypeKind.STRUCT,
            "M.Point",
            members=(StructMember("x", I32), StructMember("y", I32)),
        )
        result = json_default_to_fpy_value({"x": 10, "y": 20}, st)
        assert result.val["x"] == FpyValue(I32, 10)
        assert result.val["y"] == FpyValue(I32, 20)

    def test_struct_with_float_members(self):
        st = FpyType(
            TypeKind.STRUCT,
            "M.S",
            members=(StructMember("x", U32), StructMember("y", F32)),
        )
        result = json_default_to_fpy_value({"x": 1, "y": 1.15}, st)
        assert result.val["x"] == FpyValue(U32, 1)
        assert result.val["y"] == FpyValue(F32, 1.15)

    def test_struct_missing_member_asserts(self):
        st = FpyType(
            TypeKind.STRUCT,
            "M.S",
            members=(StructMember("a", U8), StructMember("b", U8)),
        )
        with pytest.raises(AssertionError, match="missing member 'b'"):
            json_default_to_fpy_value({"a": 0}, st)

    def test_not_dict_asserts(self):
        st = FpyType(
            TypeKind.STRUCT,
            "M.S",
            members=(StructMember("a", U8),),
        )
        with pytest.raises(AssertionError, match="Expected dict default"):
            json_default_to_fpy_value([1], st)


class TestValueNested:
    """Deeply nested value conversions."""

    def test_struct_with_array_member(self):
        arr = FpyType(TypeKind.ARRAY, "M.Vec3", elem_type=F32, length=3)
        st = FpyType(
            TypeKind.STRUCT,
            "M.Pose",
            members=(StructMember("pos", arr), StructMember("heading", F32)),
        )
        result = json_default_to_fpy_value(
            {"pos": [1.0, 2.0, 3.0], "heading": 0.5}, st
        )
        assert len(result.val["pos"].val) == 3
        assert result.val["heading"].val == 0.5

    def test_nested_struct(self):
        inner = FpyType(
            TypeKind.STRUCT,
            "M.Inner",
            members=(StructMember("a", U32), StructMember("b", U32)),
        )
        outer = FpyType(
            TypeKind.STRUCT,
            "M.Outer",
            members=(StructMember("inner", inner), StructMember("flag", BOOL)),
        )
        result = json_default_to_fpy_value(
            {"inner": {"a": 1, "b": 2}, "flag": True}, outer
        )
        assert result.val["inner"].val["a"] == FpyValue(U32, 1)
        assert result.val["flag"] == FpyValue(BOOL, True)

    def test_array_of_enums(self):
        color = FpyType(
            TypeKind.ENUM,
            "M.Color",
            enum_dict={"RED": 0, "GREEN": 1, "BLUE": 2},
            rep_type=U8,
        )
        arr = FpyType(TypeKind.ARRAY, "M.Colors", elem_type=color, length=3)
        result = json_default_to_fpy_value(
            ["M.Color.RED", "M.Color.GREEN", "M.Color.BLUE"], arr
        )
        assert [v.val for v in result.val] == ["RED", "GREEN", "BLUE"]

    def test_array_of_structs(self):
        point = FpyType(
            TypeKind.STRUCT,
            "M.Pt",
            members=(StructMember("x", I32), StructMember("y", I32)),
        )
        arr = FpyType(TypeKind.ARRAY, "M.Pts", elem_type=point, length=2)
        result = json_default_to_fpy_value(
            [{"x": 1, "y": 2}, {"x": 3, "y": 4}], arr
        )
        assert len(result.val) == 2
        assert result.val[0].val["x"] == FpyValue(I32, 1)
        assert result.val[1].val["y"] == FpyValue(I32, 4)


# ===================================================================
# Section: Commands
# ===================================================================


class TestCommands:
    """Spec §Commands — name, commandKind, opcode, formalParams, annotation."""

    def test_sync_command_with_params(self):
        """Spec example: sync command SyncParams(param1: U32, param2: string)."""
        raw = [
            {
                "name": "M.c1.SyncParams",
                "commandKind": "sync",
                "opcode": 257,
                "annotation": "A sync command with parameters",
                "formalParams": [
                    {
                        "name": "param1",
                        "annotation": "Param 1",
                        "type": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
                        "ref": False,
                    },
                    {
                        "name": "param2",
                        "annotation": "Param 2",
                        "type": {"name": "string", "kind": "string", "size": 80},
                        "ref": False,
                    },
                ],
            }
        ]
        id_dict, name_dict = _parse_commands(raw, {})
        cmd = id_dict[257]
        assert cmd.name == "M.c1.SyncParams"
        assert cmd.opcode == 257
        assert len(cmd.arguments) == 2
        assert cmd.arguments[0][0] == "param1"
        assert cmd.arguments[0][2] is U32
        assert cmd.arguments[1][0] == "param2"
        assert cmd.arguments[1][2].is_string
        assert "M.c1.SyncParams" in name_dict

    def test_async_command_no_params(self):
        raw = [
            {
                "name": "M.c1.NO_OP",
                "commandKind": "async",
                "opcode": 1,
                "formalParams": [],
                "annotation": "A no-op command",
            }
        ]
        id_dict, name_dict = _parse_commands(raw, {})
        cmd = id_dict[1]
        assert cmd.arguments == []
        assert cmd.description == "A no-op command"

    def test_guarded_command(self):
        raw = [
            {
                "name": "M.c1.Guarded",
                "commandKind": "guarded",
                "opcode": 10,
                "formalParams": [],
            }
        ]
        id_dict, _ = _parse_commands(raw, {})
        assert 10 in id_dict

    def test_set_command(self):
        """Parameter SET commands (commandKind: set)."""
        struct_type = FpyType(
            TypeKind.STRUCT,
            "M.A",
            members=(StructMember("x", U32), StructMember("y", F32)),
        )
        raw = [
            {
                "name": "M.c1.Parameter1_PRM_SET",
                "commandKind": "set",
                "opcode": 259,
                "formalParams": [
                    {
                        "name": "val",
                        "type": {"name": "M.A", "kind": "qualifiedIdentifier"},
                        "ref": False,
                    }
                ],
                "annotation": "Parameter (struct)",
            }
        ]
        id_dict, _ = _parse_commands(raw, {"M.A": struct_type})
        cmd = id_dict[259]
        assert cmd.arguments[0][2].kind == TypeKind.STRUCT

    def test_save_command(self):
        """Parameter SAVE commands (commandKind: save)."""
        raw = [
            {
                "name": "M.c1.Parameter1_PRM_SAVE",
                "commandKind": "save",
                "opcode": 260,
                "formalParams": [],
                "annotation": "Parameter (struct)",
            }
        ]
        id_dict, _ = _parse_commands(raw, {})
        assert id_dict[260].arguments == []

    def test_command_with_enum_arg(self):
        e = FpyType(
            TypeKind.ENUM,
            "M.StatusEnum",
            enum_dict={"YES": 0, "NO": 1, "MAYBE": 2},
            rep_type=U8,
        )
        raw = [
            {
                "name": "M.c1.SetStatus",
                "commandKind": "async",
                "opcode": 55,
                "formalParams": [
                    {
                        "name": "arg1",
                        "type": {"name": "M.StatusEnum", "kind": "qualifiedIdentifier"},
                        "ref": False,
                        "annotation": "The status",
                    }
                ],
            }
        ]
        id_dict, _ = _parse_commands(raw, {"M.StatusEnum": e})
        assert id_dict[55].arguments[0][2].kind == TypeKind.ENUM

    def test_command_with_array_arg(self):
        arr = FpyType(TypeKind.ARRAY, "M.StringArray", elem_type=FpyType(TypeKind.STRING, "String_80", max_length=80), length=2)
        raw = [
            {
                "name": "M.c1.CommandString",
                "commandKind": "sync",
                "opcode": 257,
                "formalParams": [
                    {
                        "name": "arg1",
                        "type": {"name": "M.StringArray", "kind": "qualifiedIdentifier"},
                        "ref": False,
                        "annotation": "description for argument 1",
                    }
                ],
                "annotation": "A command with a single StringArray argument",
            }
        ]
        id_dict, _ = _parse_commands(raw, {"M.StringArray": arr})
        assert id_dict[257].arguments[0][2].kind == TypeKind.ARRAY

    def test_multiple_commands(self):
        raw = [
            {"name": "A.CMD1", "commandKind": "async", "opcode": 1, "formalParams": []},
            {"name": "A.CMD2", "commandKind": "sync", "opcode": 2, "formalParams": []},
            {"name": "B.CMD3", "commandKind": "guarded", "opcode": 3, "formalParams": []},
        ]
        id_dict, name_dict = _parse_commands(raw, {})
        assert len(id_dict) == 3
        assert len(name_dict) == 3

    def test_empty_commands(self):
        id_dict, name_dict = _parse_commands([], {})
        assert id_dict == {}
        assert name_dict == {}

    def test_command_component_and_mnemonic(self):
        """CmdDef.component and .mnemonic properties."""
        raw = [
            {
                "name": "M.c1.SyncParams",
                "commandKind": "sync",
                "opcode": 257,
                "formalParams": [],
            }
        ]
        _, name_dict = _parse_commands(raw, {})
        cmd = name_dict["M.c1.SyncParams"]
        assert cmd.component == "M.c1"
        assert cmd.mnemonic == "SyncParams"


# ===================================================================
# Section: Telemetry Channels
# ===================================================================


class TestTelemetryChannels:
    """Spec §Telemetry Channels — name, type, id, telemetryUpdate, format, limit."""

    def test_channel_with_limits(self):
        """Spec example: Channel1 with low/high limits."""
        raw = [
            {
                "name": "M.c1.Channel1",
                "annotation": "Telemetry channel 1",
                "type": {"name": "F64", "kind": "float", "size": 64},
                "id": 258,
                "telemetryUpdate": "on change",
                "limit": {
                    "low": {"yellow": -1, "orange": -2, "red": -3},
                    "high": {"yellow": 1, "orange": 2, "red": 3},
                },
            }
        ]
        id_dict, name_dict = _parse_channels(raw, {})
        ch = id_dict[258]
        assert ch.name == "M.c1.Channel1"
        assert ch.ch_id == 258
        assert ch.ch_type is F64
        assert "M.c1.Channel1" in name_dict

    def test_channel_always_update(self):
        raw = [
            {
                "name": "M.c1.Ch",
                "type": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
                "id": 100,
                "telemetryUpdate": "always",
            }
        ]
        id_dict, _ = _parse_channels(raw, {})
        assert id_dict[100].ch_type is U32

    def test_channel_on_change_update(self):
        raw = [
            {
                "name": "M.c1.Ch",
                "type": {"name": "I32", "kind": "integer", "size": 32, "signed": True},
                "id": 101,
                "telemetryUpdate": "on change",
            }
        ]
        id_dict, _ = _parse_channels(raw, {})
        assert id_dict[101].ch_type is I32

    def test_channel_bool_type(self):
        raw = [
            {
                "name": "M.c1.Flag",
                "type": {"name": "bool", "kind": "bool"},
                "id": 200,
            }
        ]
        id_dict, _ = _parse_channels(raw, {})
        assert id_dict[200].ch_type is BOOL

    def test_channel_string_type(self):
        raw = [
            {
                "name": "M.c1.Msg",
                "type": {"name": "string", "kind": "string", "size": 256},
                "id": 201,
            }
        ]
        id_dict, _ = _parse_channels(raw, {})
        ch = id_dict[201]
        assert ch.ch_type.is_string
        assert ch.ch_type.max_length == 256

    def test_channel_enum_type(self):
        e = FpyType(TypeKind.ENUM, "M.Status", enum_dict={"OK": 0, "ERR": 1}, rep_type=U8)
        raw = [
            {
                "name": "M.c1.Status",
                "type": {"name": "M.Status", "kind": "qualifiedIdentifier"},
                "id": 300,
            }
        ]
        id_dict, _ = _parse_channels(raw, {"M.Status": e})
        assert id_dict[300].ch_type.kind == TypeKind.ENUM

    def test_multiple_channels(self):
        raw = [
            {"name": "A.Ch1", "type": {"name": "U8", "kind": "integer", "size": 8, "signed": False}, "id": 1},
            {"name": "A.Ch2", "type": {"name": "U16", "kind": "integer", "size": 16, "signed": False}, "id": 2},
            {"name": "B.Ch3", "type": {"name": "U32", "kind": "integer", "size": 32, "signed": False}, "id": 3},
        ]
        id_dict, name_dict = _parse_channels(raw, {})
        assert len(id_dict) == 3
        assert len(name_dict) == 3

    def test_empty_channels(self):
        id_dict, name_dict = _parse_channels([], {})
        assert id_dict == {}
        assert name_dict == {}


# ===================================================================
# Section: Parameters
# ===================================================================


class TestParameters:
    """Spec §Parameters — name, type, id, default, annotation."""

    def test_parameter_u32(self):
        """Spec example: param Parameter1: U32."""
        raw = [
            {
                "name": "M.c1.Parameter1",
                "type": {"name": "U32", "kind": "integer", "signed": False, "size": 32},
                "id": 260,
                "annotation": "This is the annotation for Parameter 1",
                "default": 0,
            }
        ]
        id_dict, name_dict = _parse_parameters(raw, {})
        prm = id_dict[260]
        assert prm.name == "M.c1.Parameter1"
        assert prm.prm_id == 260
        assert prm.prm_type is U32
        assert prm.default == 0
        assert "M.c1.Parameter1" in name_dict

    def test_parameter_struct_type_with_default(self):
        """Spec example: param Parameter1: A with struct default."""
        struct_type = FpyType(
            TypeKind.STRUCT,
            "M.A",
            members=(StructMember("x", U32), StructMember("y", F32)),
        )
        raw = [
            {
                "name": "M.c1.Parameter1",
                "type": {"name": "M.A", "kind": "qualifiedIdentifier"},
                "id": 258,
                "default": {"x": 1, "y": 1.15},
                "annotation": "Parameter (struct)",
            }
        ]
        id_dict, _ = _parse_parameters(raw, {"M.A": struct_type})
        prm = id_dict[258]
        assert prm.prm_type.kind == TypeKind.STRUCT
        assert prm.default == {"x": 1, "y": 1.15}

    def test_parameter_no_default(self):
        raw = [
            {
                "name": "M.c1.P",
                "type": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                "id": 10,
            }
        ]
        id_dict, _ = _parse_parameters(raw, {})
        assert id_dict[10].default is None

    def test_parameter_enum_type(self):
        e = FpyType(TypeKind.ENUM, "M.Choice", enum_dict={"A": 0, "B": 1}, rep_type=U32)
        raw = [
            {
                "name": "M.c1.CHOICE",
                "type": {"name": "M.Choice", "kind": "qualifiedIdentifier"},
                "id": 99,
            }
        ]
        id_dict, _ = _parse_parameters(raw, {"M.Choice": e})
        assert id_dict[99].prm_type.kind == TypeKind.ENUM

    def test_multiple_parameters(self):
        raw = [
            {"name": "A.P1", "type": {"name": "U8", "kind": "integer", "size": 8, "signed": False}, "id": 1},
            {"name": "A.P2", "type": {"name": "U16", "kind": "integer", "size": 16, "signed": False}, "id": 2},
        ]
        id_dict, name_dict = _parse_parameters(raw, {})
        assert len(id_dict) == 2
        assert len(name_dict) == 2

    def test_empty_parameters(self):
        id_dict, name_dict = _parse_parameters([], {})
        assert id_dict == {}
        assert name_dict == {}


# ===================================================================
# Section: Dictionary Metadata
# ===================================================================


class TestDictionaryMetadata:
    """Spec §Dictionaries / Dictionary Metadata."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        load_dictionary.cache_clear()
        yield
        load_dictionary.cache_clear()

    def test_metadata_fields(self):
        data = _minimal_dict()
        path = _write_dict(data)
        try:
            d = load_dictionary(path)
            meta = d["metadata"]
            assert meta["deploymentName"] == "TestDeployment"
            assert meta["frameworkVersion"] == "3.3.2"
            assert meta["projectVersion"] == "1.0.0"
            assert meta["libraryVersions"] == []
            assert meta["dictionarySpecVersion"] == "1.0.0"
        finally:
            os.unlink(path)

    def test_metadata_with_library_versions(self):
        data = _minimal_dict()
        data["metadata"]["libraryVersions"] = ["1.0.0", "2.0.0"]
        path = _write_dict(data)
        try:
            d = load_dictionary(path)
            assert d["metadata"]["libraryVersions"] == ["1.0.0", "2.0.0"]
        finally:
            os.unlink(path)


# ===================================================================
# Section: Dictionary Content — full end-to-end
# ===================================================================


class TestDictionaryContent:
    """Spec §Dictionaries / Dictionary Content — end-to-end loading tests."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        load_dictionary.cache_clear()
        yield
        load_dictionary.cache_clear()

    def test_empty_dictionary(self):
        path = _write_dict(_minimal_dict())
        try:
            d = load_dictionary(path)
            assert d["type_defs"] == {}
            assert d["cmd_id_dict"] == {}
            assert d["ch_id_dict"] == {}
            assert d["prm_id_dict"] == {}
            assert d["constants"] == {}
        finally:
            os.unlink(path)

    def test_full_dictionary_from_spec(self):
        """Build a small but complete dictionary matching the spec's full example."""
        data = _minimal_dict(
            typeDefinitions=[
                {
                    "kind": "array",
                    "qualifiedName": "M.StringArray",
                    "size": 2,
                    "elementType": {"name": "string", "kind": "string", "size": 80},
                    "default": ["A", "B"],
                },
                {
                    "kind": "enum",
                    "qualifiedName": "M.StatusEnum",
                    "representationType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                    "enumeratedConstants": [
                        {"name": "YES", "value": 0},
                        {"name": "NO", "value": 1},
                        {"name": "MAYBE", "value": 2},
                    ],
                    "default": "M.StatusEnum.MAYBE",
                },
                {
                    "kind": "struct",
                    "qualifiedName": "M.A",
                    "members": {
                        "x": {
                            "type": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
                            "index": 0,
                            "format": "The value of x is {}",
                        },
                        "y": {
                            "type": {"name": "F32", "kind": "float", "size": 32},
                            "index": 1,
                            "format": "The value of y is {}",
                        },
                    },
                    "default": {"x": 1, "y": 1.15},
                },
            ],
            commands=[
                {
                    "name": "M.c1.CommandString",
                    "commandKind": "sync",
                    "opcode": 257,
                    "formalParams": [
                        {
                            "name": "arg1",
                            "type": {"name": "M.StringArray", "kind": "qualifiedIdentifier"},
                            "ref": False,
                            "annotation": "description for argument 1",
                        }
                    ],
                    "annotation": "A command with a single StringArray argument",
                },
                {
                    "name": "M.c1.Parameter1_PRM_SET",
                    "commandKind": "set",
                    "opcode": 259,
                    "formalParams": [
                        {
                            "name": "val",
                            "type": {"name": "M.A", "kind": "qualifiedIdentifier"},
                            "ref": False,
                        }
                    ],
                    "annotation": "Parameter (struct)",
                },
                {
                    "name": "M.c1.Parameter1_PRM_SAVE",
                    "commandKind": "save",
                    "opcode": 260,
                    "formalParams": [],
                    "annotation": "Parameter (struct)",
                },
            ],
            parameters=[
                {
                    "name": "M.c1.Parameter1",
                    "type": {"name": "M.A", "kind": "qualifiedIdentifier"},
                    "id": 258,
                    "default": {"x": 1, "y": 1.15},
                    "annotation": "Parameter (struct)",
                }
            ],
            events=[
                {
                    "name": "M.c1.Event1",
                    "severity": "ACTIVITY_HI",
                    "formalParams": [
                        {
                            "name": "arg1",
                            "type": {"name": "M.StatusEnum", "kind": "qualifiedIdentifier"},
                            "ref": False,
                            "annotation": "Description of arg1 formal param",
                        }
                    ],
                    "id": 259,
                    "format": "Event 1 occurred, status {}",
                    "annotation": "Event with one StatusEnum argument",
                }
            ],
            telemetryChannels=[
                {
                    "name": "M.c1.Channel1",
                    "type": {"name": "I32", "kind": "integer", "size": 32, "signed": True},
                    "id": 260,
                    "telemetryUpdate": "on change",
                    "annotation": "Telemetry channel 1 of type I32",
                    "limit": {
                        "low": {"yellow": -1, "orange": -2, "red": -3},
                        "high": {"yellow": 1, "orange": 2, "red": 3},
                    },
                }
            ],
        )
        path = _write_dict(data)
        try:
            d = load_dictionary(path)
            # Type defs
            assert "M.StringArray" in d["type_defs"]
            assert "M.StatusEnum" in d["type_defs"]
            assert "M.A" in d["type_defs"]
            assert d["type_defs"]["M.StringArray"].kind == TypeKind.ARRAY
            assert d["type_defs"]["M.StatusEnum"].kind == TypeKind.ENUM
            assert d["type_defs"]["M.A"].kind == TypeKind.STRUCT
            # Commands
            assert len(d["cmd_id_dict"]) == 3
            assert d["cmd_id_dict"][257].name == "M.c1.CommandString"
            assert d["cmd_id_dict"][259].name == "M.c1.Parameter1_PRM_SET"
            assert d["cmd_id_dict"][260].name == "M.c1.Parameter1_PRM_SAVE"
            # Params
            assert len(d["prm_id_dict"]) == 1
            prm = d["prm_id_dict"][258]
            assert prm.prm_type.kind == TypeKind.STRUCT
            assert prm.default == {"x": 1, "y": 1.15}
            # Channels
            assert len(d["ch_id_dict"]) == 1
            ch = d["ch_id_dict"][260]
            assert ch.ch_type is I32
            # Metadata
            assert d["metadata"]["deploymentName"] == "TestDeployment"
        finally:
            os.unlink(path)

    def test_dictionary_id_and_name_dicts_consistent(self):
        """Every entry in name_dict should appear in id_dict and vice versa."""
        data = _minimal_dict(
            commands=[
                {"name": "A.CMD1", "commandKind": "async", "opcode": 1, "formalParams": []},
                {"name": "A.CMD2", "commandKind": "sync", "opcode": 2, "formalParams": []},
            ],
            telemetryChannels=[
                {"name": "A.Ch1", "type": {"name": "U32", "kind": "integer", "size": 32, "signed": False}, "id": 10},
            ],
            parameters=[
                {"name": "A.P1", "type": {"name": "U8", "kind": "integer", "size": 8, "signed": False}, "id": 20},
            ],
        )
        path = _write_dict(data)
        try:
            d = load_dictionary(path)
            for cmd in d["cmd_name_dict"].values():
                assert d["cmd_id_dict"][cmd.opcode] is cmd
            for ch in d["ch_name_dict"].values():
                assert d["ch_id_dict"][ch.ch_id] is ch
            for prm in d["prm_name_dict"].values():
                assert d["prm_id_dict"][prm.prm_id] is prm
        finally:
            os.unlink(path)


# ===================================================================
# Section: Formal Parameters (used in commands & events)
# ===================================================================


class TestFormalParameters:
    """Spec §Formal Parameters — name, type, ref, annotation."""

    def test_formal_param_fields(self):
        raw = [
            {
                "name": "M.c1.Cmd",
                "commandKind": "sync",
                "opcode": 1,
                "formalParams": [
                    {
                        "name": "param1",
                        "type": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
                        "ref": False,
                        "annotation": "This is param1",
                    }
                ],
            }
        ]
        id_dict, _ = _parse_commands(raw, {})
        arg_name, arg_desc, arg_type = id_dict[1].arguments[0]
        assert arg_name == "param1"
        assert arg_type is U32
        # annotation is stored in description
        assert arg_desc == "This is param1"

    def test_formal_param_no_annotation(self):
        raw = [
            {
                "name": "M.c1.Cmd",
                "commandKind": "sync",
                "opcode": 2,
                "formalParams": [
                    {
                        "name": "p",
                        "type": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                        "ref": False,
                    }
                ],
            }
        ]
        id_dict, _ = _parse_commands(raw, {})
        _, desc, _ = id_dict[2].arguments[0]
        assert desc == ""

    def test_many_formal_params(self):
        params = [
            {
                "name": f"p{i}",
                "type": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
                "ref": False,
            }
            for i in range(5)
        ]
        raw = [{"name": "M.Cmd", "commandKind": "async", "opcode": 10, "formalParams": params}]
        id_dict, _ = _parse_commands(raw, {})
        assert len(id_dict[10].arguments) == 5


# ===================================================================
# Section: Serialization round-trips (Values)
# ===================================================================


class TestSerializationRoundTrips:
    """Verify that values can be serialized and deserialized back correctly,
    exercising the wire format implied by the spec's type definitions."""

    @pytest.mark.parametrize(
        "typ,val",
        [
            (U8, 255),
            (U16, 65535),
            (U32, 2**32 - 1),
            (U64, 2**64 - 1),
            (I8, -128),
            (I16, -32768),
            (I32, -(2**31)),
            (I64, -(2**63)),
            (BOOL, True),
            (BOOL, False),
        ],
    )
    def test_primitive_roundtrip(self, typ, val):
        fv = FpyValue(typ, val)
        data = fv.serialize()
        result, _ = FpyValue.deserialize(typ, data)
        assert result.val == val

    @pytest.mark.parametrize("val", [0.0, 1.5, -42.125])
    def test_f32_roundtrip(self, val):
        fv = FpyValue(F32, val)
        result, _ = FpyValue.deserialize(F32, fv.serialize())
        assert abs(result.val - val) < 1e-6

    @pytest.mark.parametrize("val", [0.0, 3.14159265358979, -1e100])
    def test_f64_roundtrip(self, val):
        fv = FpyValue(F64, val)
        result, _ = FpyValue.deserialize(F64, fv.serialize())
        assert result.val == val

    def test_string_roundtrip(self):
        st = FpyType(TypeKind.STRING, "String_80", max_length=80)
        fv = FpyValue(st, "Hello World!")
        result, _ = FpyValue.deserialize(st, fv.serialize())
        assert result.val == "Hello World!"

    def test_empty_string_roundtrip(self):
        st = FpyType(TypeKind.STRING, "String_80", max_length=80)
        fv = FpyValue(st, "")
        result, _ = FpyValue.deserialize(st, fv.serialize())
        assert result.val == ""

    def test_enum_roundtrip(self):
        e = FpyType(
            TypeKind.ENUM,
            "M.Status",
            enum_dict={"YES": 0, "NO": 1, "MAYBE": 2},
            rep_type=I32,
        )
        fv = FpyValue(e, "MAYBE")
        result, _ = FpyValue.deserialize(e, fv.serialize())
        assert result.val == "MAYBE"

    def test_array_roundtrip(self):
        arr = FpyType(TypeKind.ARRAY, "M.A", elem_type=U32, length=3)
        fv = FpyValue(arr, [FpyValue(U32, 10), FpyValue(U32, 20), FpyValue(U32, 30)])
        result, _ = FpyValue.deserialize(arr, fv.serialize())
        assert [v.val for v in result.val] == [10, 20, 30]

    def test_struct_roundtrip(self):
        st = FpyType(
            TypeKind.STRUCT,
            "M.Point",
            members=(StructMember("x", I32), StructMember("y", I32)),
        )
        fv = FpyValue(st, {"x": FpyValue(I32, -5), "y": FpyValue(I32, 10)})
        result, _ = FpyValue.deserialize(st, fv.serialize())
        assert result.val["x"].val == -5
        assert result.val["y"].val == 10

    def test_nested_roundtrip(self):
        """Struct containing an array — multi-layer serialization."""
        arr = FpyType(TypeKind.ARRAY, "M.Vec", elem_type=F32, length=2)
        st = FpyType(
            TypeKind.STRUCT,
            "M.Pair",
            members=(StructMember("v", arr), StructMember("id", U32)),
        )
        fv = FpyValue(
            st,
            {
                "v": FpyValue(arr, [FpyValue(F32, 1.0), FpyValue(F32, 2.0)]),
                "id": FpyValue(U32, 42),
            },
        )
        result, _ = FpyValue.deserialize(st, fv.serialize())
        assert result.val["id"].val == 42
        assert len(result.val["v"].val) == 2


# ===================================================================
# Section: Inline member arrays in structs
# ===================================================================


class TestStructInlineMemberArrays:
    """Spec: struct members with 'size' key create inline array types."""

    def test_inline_array_member(self):
        raw = [
            {
                "kind": "struct",
                "qualifiedName": "M.WithArr",
                "members": {
                    "data": {
                        "type": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                        "index": 0,
                        "size": 5,
                    },
                    "count": {
                        "type": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
                        "index": 1,
                    },
                },
            }
        ]
        result = _parse_type_definitions(raw)
        st = result["M.WithArr"]
        # The "data" member should be an array type
        assert st.members[0].name == "data"
        assert st.members[0].type.kind == TypeKind.ARRAY
        assert st.members[0].type.length == 5
        assert st.members[0].type.elem_type is U8
        # "count" should be plain U32
        assert st.members[1].type is U32

    def test_inline_array_dedup(self):
        """Two members with same element type and size should share the array type def."""
        raw = [
            {
                "kind": "struct",
                "qualifiedName": "M.TwoArrays",
                "members": {
                    "a": {
                        "type": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                        "index": 0,
                        "size": 3,
                    },
                    "b": {
                        "type": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                        "index": 1,
                        "size": 3,
                    },
                },
            }
        ]
        result = _parse_type_definitions(raw)
        st = result["M.TwoArrays"]
        # Both should reference the same array type object
        assert st.members[0].type is st.members[1].type


# ===================================================================
# Section: Dictionary caching
# ===================================================================


class TestDictionaryCaching:
    """load_dictionary uses lru_cache; test that caching works."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        load_dictionary.cache_clear()
        yield
        load_dictionary.cache_clear()

    def test_same_path_returns_same_object(self):
        path = _write_dict(_minimal_dict())
        try:
            d1 = load_dictionary(path)
            d2 = load_dictionary(path)
            assert d1 is d2
        finally:
            os.unlink(path)

    def test_different_paths_return_different_objects(self):
        path1 = _write_dict(_minimal_dict())
        data2 = _minimal_dict()
        data2["metadata"]["deploymentName"] = "Other"
        path2 = _write_dict(data2)
        try:
            d1 = load_dictionary(path1)
            d2 = load_dictionary(path2)
            assert d1 is not d2
            assert d1["metadata"]["deploymentName"] == "TestDeployment"
            assert d2["metadata"]["deploymentName"] == "Other"
        finally:
            os.unlink(path1)
            os.unlink(path2)


# ===================================================================
# Section: Type max_size computation
# ===================================================================


class TestTypeMaxSize:
    """Spec-implied: types must compute their serialized size correctly."""

    def test_primitive_sizes(self):
        assert U8.max_size == 1
        assert U16.max_size == 2
        assert U32.max_size == 4
        assert U64.max_size == 8
        assert I8.max_size == 1
        assert I16.max_size == 2
        assert I32.max_size == 4
        assert I64.max_size == 8
        assert F32.max_size == 4
        assert F64.max_size == 8
        assert BOOL.max_size == 1

    def test_string_size(self):
        st = FpyType(TypeKind.STRING, "String_80", max_length=80)
        assert st.max_size == 2 + 80  # 2-byte length prefix + data

    def test_enum_size(self):
        e = FpyType(TypeKind.ENUM, "M.E", enum_dict={"A": 0}, rep_type=U32)
        assert e.max_size == 4  # U32 rep_type

    def test_array_size(self):
        arr = FpyType(TypeKind.ARRAY, "M.A", elem_type=U32, length=5)
        assert arr.max_size == 4 * 5

    def test_struct_size(self):
        st = FpyType(
            TypeKind.STRUCT,
            "M.S",
            members=(StructMember("a", U32), StructMember("b", F64)),
        )
        assert st.max_size == 4 + 8

    def test_nested_size(self):
        arr = FpyType(TypeKind.ARRAY, "M.V3", elem_type=F32, length=3)
        st = FpyType(
            TypeKind.STRUCT,
            "M.Pose",
            members=(StructMember("pos", arr), StructMember("heading", F32)),
        )
        assert st.max_size == (4 * 3) + 4


# ===================================================================
# Section: Type classification properties
# ===================================================================


class TestTypeClassification:
    """Verify FpyType classification properties used throughout."""

    def test_integer_classification(self):
        for t in [U8, U16, U32, U64, I8, I16, I32, I64]:
            assert t.is_integer
            assert t.is_primitive
            assert t.is_concrete
            assert not t.is_float
            assert not t.is_string

    def test_float_classification(self):
        for t in [F32, F64]:
            assert t.is_float
            assert t.is_primitive
            assert t.is_concrete
            assert not t.is_integer
            assert not t.is_string

    def test_bool_classification(self):
        assert BOOL.is_primitive
        assert BOOL.is_concrete
        assert not BOOL.is_integer
        assert not BOOL.is_float
        assert not BOOL.is_string

    def test_signed_unsigned(self):
        for t in [U8, U16, U32, U64]:
            assert t.is_unsigned
            assert not t.is_signed
        for t in [I8, I16, I32, I64]:
            assert t.is_signed
            assert not t.is_unsigned

    def test_enum_classification(self):
        e = FpyType(TypeKind.ENUM, "E", enum_dict={"A": 0}, rep_type=U8)
        assert e.is_concrete
        assert not e.is_primitive
        assert not e.is_integer
        assert not e.is_float

    def test_struct_classification(self):
        s = FpyType(TypeKind.STRUCT, "S", members=(StructMember("a", U8),))
        assert s.is_concrete
        assert not s.is_primitive

    def test_array_classification(self):
        a = FpyType(TypeKind.ARRAY, "A", elem_type=U8, length=2)
        assert a.is_concrete
        assert not a.is_primitive

    def test_string_classification(self):
        st = FpyType(TypeKind.STRING, "String_80", max_length=80)
        assert st.is_string
        assert st.is_concrete
        assert not st.is_primitive
        assert not st.is_integer
        assert not st.is_float


# ===================================================================
# Section: Integration tests against RefTopologyDictionary.json
# ===================================================================


class TestLoadDictionary:
    """Load the real Ref dictionary and verify counts, attributes, consistency."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        load_dictionary.cache_clear()
        yield
        load_dictionary.cache_clear()

    def test_loads_ref_dictionary(self):
        d = load_dictionary(REF_DICT_PATH)
        assert "type_defs" in d
        assert "cmd_id_dict" in d
        assert "cmd_name_dict" in d
        assert "ch_id_dict" in d
        assert "ch_name_dict" in d
        assert "prm_id_dict" in d
        assert "prm_name_dict" in d
        assert "constants" in d
        assert "metadata" in d

    def test_type_counts(self):
        d = load_dictionary(REF_DICT_PATH)
        assert len(d["type_defs"]) == 89

    def test_command_counts(self):
        d = load_dictionary(REF_DICT_PATH)
        assert len(d["cmd_id_dict"]) == 109
        assert len(d["cmd_name_dict"]) == 109

    def test_channel_counts(self):
        d = load_dictionary(REF_DICT_PATH)
        assert len(d["ch_id_dict"]) == 178
        assert len(d["ch_name_dict"]) == 178

    def test_parameter_counts(self):
        d = load_dictionary(REF_DICT_PATH)
        assert len(d["prm_id_dict"]) == 11
        assert len(d["prm_name_dict"]) == 11

    def test_constant_counts(self):
        d = load_dictionary(REF_DICT_PATH)
        assert len(d["constants"]) == 15

    def test_command_attributes(self):
        """Verify CmdDef attributes match expected API."""
        d = load_dictionary(REF_DICT_PATH)
        cmd = d["cmd_name_dict"]["Ref.dpDemo.SelectColor"]
        assert cmd.name == "Ref.dpDemo.SelectColor"
        assert isinstance(cmd.opcode, int)
        assert len(cmd.arguments) == 1
        arg_name, arg_desc, arg_type = cmd.arguments[0]
        assert arg_name == "color"
        assert arg_type.kind == TypeKind.ENUM

    def test_channel_attributes(self):
        """Verify ChDef attributes match expected API."""
        d = load_dictionary(REF_DICT_PATH)
        ch = d["ch_name_dict"]["CdhCore.cmdDisp.CommandsDispatched"]
        assert ch.name == "CdhCore.cmdDisp.CommandsDispatched"
        assert isinstance(ch.ch_id, int)
        assert ch.ch_type is U32

    def test_parameter_attributes(self):
        """Verify PrmDef attributes match expected API."""
        d = load_dictionary(REF_DICT_PATH)
        prm = d["prm_name_dict"]["Ref.typeDemo.CHOICE_PRM"]
        assert prm.name == "Ref.typeDemo.CHOICE_PRM"
        assert isinstance(prm.prm_id, int)
        assert prm.prm_type.kind == TypeKind.ENUM

    def test_id_and_name_dicts_consistent(self):
        """Every entry in name_dict should also appear in id_dict."""
        d = load_dictionary(REF_DICT_PATH)
        for cmd in d["cmd_name_dict"].values():
            assert d["cmd_id_dict"][cmd.opcode] is cmd
        for ch in d["ch_name_dict"].values():
            assert d["ch_id_dict"][ch.ch_id] is ch
        for prm in d["prm_name_dict"].values():
            assert d["prm_id_dict"][prm.prm_id] is prm

    def test_enum_type_parsed(self):
        """Enum types should have enum_dict populated."""
        d = load_dictionary(REF_DICT_PATH)
        choice = d["type_defs"]["Ref.Choice"]
        assert choice.kind == TypeKind.ENUM
        assert "ONE" in choice.enum_dict

    def test_struct_type_parsed(self):
        """Struct types should have members populated."""
        d = load_dictionary(REF_DICT_PATH)
        structs = {
            name: typ
            for name, typ in d["type_defs"].items()
            if typ.kind == TypeKind.STRUCT
        }
        assert len(structs) > 0
        for name, typ in structs.items():
            assert typ.members is not None
            assert len(typ.members) > 0

    def test_array_type_parsed(self):
        """Array types should have length and elem_type."""
        d = load_dictionary(REF_DICT_PATH)
        arrays = {
            name: typ
            for name, typ in d["type_defs"].items()
            if typ.kind == TypeKind.ARRAY
        }
        assert len(arrays) > 0
        for name, typ in arrays.items():
            assert typ.length is not None
            assert typ.length > 0
            assert typ.elem_type is not None

    def test_constants_values(self):
        """Spot-check known constants from the Ref dictionary."""
        d = load_dictionary(REF_DICT_PATH)
        assert d["constants"]["Svc.Fpy.MAX_SEQUENCE_STATEMENT_COUNT"] == 2048
        assert d["constants"]["Svc.Fpy.MAX_DIRECTIVE_SIZE"] == 2048

    def test_metadata_present(self):
        d = load_dictionary(REF_DICT_PATH)
        assert "deploymentName" in d["metadata"]
        assert "dictionarySpecVersion" in d["metadata"]


# ===================================================================
# Section: Synthetic end-to-end tests via load_dictionary
# ===================================================================


class TestSyntheticDictionary:
    """End-to-end tests using small hand-crafted dictionaries with load_dictionary."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        load_dictionary.cache_clear()
        yield
        load_dictionary.cache_clear()

    def test_command_no_params(self):
        data = _minimal_dict(
            commands=[
                {
                    "name": "A.b.NO_OP",
                    "commandKind": "async",
                    "opcode": 1,
                    "formalParams": [],
                }
            ]
        )
        path = _write_dict(data)
        try:
            d = load_dictionary(path)
            cmd = d["cmd_id_dict"][1]
            assert cmd.name == "A.b.NO_OP"
            assert cmd.arguments == []
        finally:
            os.unlink(path)

    def test_channel_bool_type(self):
        data = _minimal_dict(
            telemetryChannels=[
                {
                    "name": "A.b.Flag",
                    "type": {"name": "bool", "kind": "bool"},
                    "id": 10,
                }
            ]
        )
        path = _write_dict(data)
        try:
            d = load_dictionary(path)
            assert d["ch_id_dict"][10].ch_type is BOOL
        finally:
            os.unlink(path)

    def test_chained_aliases(self):
        """A -> B -> U32: even if A appears before B, both should resolve."""
        data = _minimal_dict(
            typeDefinitions=[
                {
                    "kind": "alias",
                    "qualifiedName": "Synth.AliasA",
                    "type": {"name": "Synth.AliasB", "kind": "qualifiedIdentifier"},
                    "underlyingType": {"name": "Synth.AliasB", "kind": "qualifiedIdentifier"},
                },
                {
                    "kind": "alias",
                    "qualifiedName": "Synth.AliasB",
                    "type": {"name": "U32", "kind": "integer"},
                    "underlyingType": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
                },
            ]
        )
        path = _write_dict(data)
        try:
            d = load_dictionary(path)
            assert d["type_defs"]["Synth.AliasA"] is U32
            assert d["type_defs"]["Synth.AliasB"] is U32
        finally:
            os.unlink(path)

    def test_struct_with_enum_member(self):
        data = _minimal_dict(
            typeDefinitions=[
                {
                    "kind": "enum",
                    "qualifiedName": "Synth.Compass",
                    "representationType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                    "enumeratedConstants": [{"name": "N", "value": 0}, {"name": "S", "value": 1}],
                },
                {
                    "kind": "struct",
                    "qualifiedName": "Synth.Move",
                    "members": {
                        "dir": {"type": {"name": "Synth.Compass", "kind": "qualifiedIdentifier"}, "index": 0},
                        "dist": {"type": {"name": "U32", "kind": "integer", "size": 32, "signed": False}, "index": 1},
                    },
                },
            ]
        )
        path = _write_dict(data)
        try:
            d = load_dictionary(path)
            move = d["type_defs"]["Synth.Move"]
            assert move.kind == TypeKind.STRUCT
            assert move.members[0].name == "dir"
            assert move.members[0].type.kind == TypeKind.ENUM
            assert move.members[1].name == "dist"
            assert move.members[1].type is U32
        finally:
            os.unlink(path)

    def test_command_with_struct_arg(self):
        """Command whose formal param type is a struct."""
        data = _minimal_dict(
            typeDefinitions=[
                {
                    "kind": "struct",
                    "qualifiedName": "Synth.Pair",
                    "members": {
                        "a": {"type": {"name": "U8", "kind": "integer", "size": 8, "signed": False}, "index": 0},
                        "b": {"type": {"name": "U8", "kind": "integer", "size": 8, "signed": False}, "index": 1},
                    },
                }
            ],
            commands=[
                {
                    "name": "A.b.SEND_PAIR",
                    "commandKind": "async",
                    "opcode": 99,
                    "formalParams": [
                        {
                            "name": "pair",
                            "type": {"name": "Synth.Pair", "kind": "qualifiedIdentifier"},
                            "ref": False,
                        }
                    ],
                }
            ],
        )
        path = _write_dict(data)
        try:
            d = load_dictionary(path)
            cmd = d["cmd_id_dict"][99]
            assert len(cmd.arguments) == 1
            assert cmd.arguments[0][0] == "pair"
            assert cmd.arguments[0][2].kind == TypeKind.STRUCT
        finally:
            os.unlink(path)

    def test_string_type_in_channel(self):
        data = _minimal_dict(
            telemetryChannels=[
                {
                    "name": "A.b.Message",
                    "type": {"name": "string", "kind": "string", "size": 256},
                    "id": 55,
                }
            ]
        )
        path = _write_dict(data)
        try:
            d = load_dictionary(path)
            ch = d["ch_id_dict"][55]
            assert ch.ch_type.is_string
        finally:
            os.unlink(path)

    def test_multiple_commands_unique_opcodes(self):
        data = _minimal_dict(
            commands=[
                {"name": "A.b.CMD1", "commandKind": "async", "opcode": 1, "formalParams": []},
                {"name": "A.b.CMD2", "commandKind": "async", "opcode": 2, "formalParams": []},
                {"name": "A.c.CMD3", "commandKind": "async", "opcode": 3, "formalParams": []},
            ]
        )
        path = _write_dict(data)
        try:
            d = load_dictionary(path)
            assert len(d["cmd_id_dict"]) == 3
            assert len(d["cmd_name_dict"]) == 3
        finally:
            os.unlink(path)


# ===================================================================
# Section: Ref dictionary defaults (integration)
# ===================================================================


class TestRefDictionaryDefaults:
    """Verify defaults are correctly parsed from the real RefTopologyDictionary."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        load_dictionary.cache_clear()
        yield
        load_dictionary.cache_clear()

    def test_enum_default_from_ref(self):
        d = load_dictionary(REF_DICT_PATH)
        choice = d["type_defs"]["Ref.Choice"]
        assert choice.kind == TypeKind.ENUM
        assert choice.json_default == "Ref.Choice.ONE"

    def test_array_default_from_ref(self):
        d = load_dictionary(REF_DICT_PATH)
        arr = d["type_defs"]["Svc.BuffQueueDepth"]
        assert arr.kind == TypeKind.ARRAY
        assert arr.json_default == [0]

    def test_array_of_enums_default_from_ref(self):
        d = load_dictionary(REF_DICT_PATH)
        arr = d["type_defs"]["Ref.ManyChoices"]
        assert arr.kind == TypeKind.ARRAY
        assert arr.json_default == ["Ref.Choice.ONE", "Ref.Choice.ONE"]

    def test_struct_default_from_ref(self):
        d = load_dictionary(REF_DICT_PATH)
        struct = d["type_defs"]["Ref.PacketStat"]
        assert struct.kind == TypeKind.STRUCT
        assert struct.json_default == {
            "BuffRecv": 0,
            "BuffErr": 0,
            "PacketStatus": "Ref.PacketRecvStatus.PACKET_STATE_NO_PACKETS",
        }

    def test_struct_with_float_member_default(self):
        d = load_dictionary(REF_DICT_PATH)
        struct = d["type_defs"]["Ref.SignalPair"]
        assert struct.kind == TypeKind.STRUCT
        assert struct.json_default["time"] == 0.0
        assert struct.json_default["value"] == 0.0

    def test_time_interval_default(self):
        d = load_dictionary(REF_DICT_PATH)
        struct = d["type_defs"]["Fw.TimeIntervalValue"]
        assert struct.kind == TypeKind.STRUCT
        assert struct.json_default == {"seconds": 0, "useconds": 0}


# ===================================================================
# Section: Compiler integration — type constructors via _build_global_scopes
# ===================================================================


class TestTypeCtorDefaults:
    """Tests that types and their constructors have correct defaults."""

    @pytest.fixture(autouse=True)
    def clear_caches(self):
        load_dictionary.cache_clear()
        from fpy.compiler import _build_global_scopes
        _build_global_scopes.cache_clear()
        yield
        load_dictionary.cache_clear()
        _build_global_scopes.cache_clear()

    def _get_callable_scope(self):
        from fpy.compiler import _build_global_scopes
        _, callable_scope, _ = _build_global_scopes(REF_DICT_PATH)
        return callable_scope

    def _get_type_scope(self):
        from fpy.compiler import _build_global_scopes
        type_scope, _, _ = _build_global_scopes(REF_DICT_PATH)
        return type_scope

    def _lookup_callable(self, name: str):
        """Look up a callable by its qualified name in the scope tree."""
        scope = self._get_callable_scope()
        parts = name.split(".")
        current = scope
        for part in parts:
            assert part in current, f"'{part}' not found in scope while looking up '{name}'"
            current = current[part]
        return current

    def _lookup_type(self, name: str) -> FpyType:
        """Look up a type by its qualified name in the type scope tree."""
        scope = self._get_type_scope()
        parts = name.split(".")
        current = scope
        for part in parts:
            assert part in current, f"'{part}' not found in scope while looking up '{name}'"
            current = current[part]
        return current

    def test_struct_member_defaults(self):
        """Struct types should have member_defaults dict on FpyType."""
        typ = self._lookup_type("Ref.SignalPair")
        assert typ.kind == TypeKind.STRUCT
        assert typ.member_defaults is not None
        for m in typ.members:
            assert m.name in typ.member_defaults, f"Struct member '{m.name}' should have a default"
            assert isinstance(typ.member_defaults[m.name], FpyValue), f"Default for '{m.name}' should be FpyValue"

    def test_struct_member_default_values_correct(self):
        typ = self._lookup_type("Ref.SignalPair")
        assert typ.member_defaults["time"].val == 0.0
        assert typ.member_defaults["value"].val == 0.0

    def test_struct_member_with_enum_default(self):
        """Struct with an enum member should have enum FpyValue default."""
        typ = self._lookup_type("Ref.PacketStat")
        defaults = typ.member_defaults
        assert defaults["BuffRecv"] is not None
        assert defaults["BuffRecv"].val == 0
        assert defaults["BuffErr"] is not None
        assert defaults["BuffErr"].val == 0
        assert defaults["PacketStatus"] is not None
        assert defaults["PacketStatus"].val == "PACKET_STATE_NO_PACKETS"

    def test_array_elem_defaults(self):
        """Array types should have elem_defaults directly on FpyType."""
        typ = self._lookup_type("Svc.BuffQueueDepth")
        assert typ.kind == TypeKind.ARRAY
        assert typ.elem_defaults is not None
        assert len(typ.elem_defaults) == 1
        assert typ.elem_defaults[0] is not None
        assert typ.elem_defaults[0].val == 0

    def test_array_multi_element_defaults(self):
        typ = self._lookup_type("Svc.ComQueueDepth")
        assert typ.elem_defaults is not None
        assert len(typ.elem_defaults) == 2
        for d in typ.elem_defaults:
            assert d is not None
            assert d.val == 0

    def test_array_of_enums_elem_defaults(self):
        typ = self._lookup_type("Ref.ManyChoices")
        assert typ.elem_defaults is not None
        assert len(typ.elem_defaults) == 2
        for d in typ.elem_defaults:
            assert d is not None
            assert d.val == "ONE"

    def test_struct_ctor_has_defaults(self):
        """Type ctor args should reflect the member defaults."""
        from fpy.state import TypeCtorSymbol
        ctor = self._lookup_callable("Ref.SignalPair")
        assert isinstance(ctor, TypeCtorSymbol)
        for arg_name, arg_type, arg_default in ctor.args:
            assert arg_default is not None, f"Struct member '{arg_name}' should have a default"
            assert isinstance(arg_default, FpyValue), f"Default for '{arg_name}' should be FpyValue"

    def test_struct_ctor_default_values_correct(self):
        ctor = self._lookup_callable("Ref.SignalPair")
        time_default = ctor.args[0][2]
        value_default = ctor.args[1][2]
        assert time_default.val == 0.0
        assert value_default.val == 0.0

    def test_struct_ctor_with_enum_member_default(self):
        """Struct with an enum member should have enum FpyValue default."""
        ctor = self._lookup_callable("Ref.PacketStat")
        args_dict = {arg[0]: arg for arg in ctor.args}
        assert args_dict["BuffRecv"][2] is not None
        assert args_dict["BuffRecv"][2].val == 0
        assert args_dict["BuffErr"][2] is not None
        assert args_dict["BuffErr"][2].val == 0
        assert args_dict["PacketStatus"][2] is not None
        assert args_dict["PacketStatus"][2].val == "PACKET_STATE_NO_PACKETS"

    def test_array_ctor_has_defaults(self):
        """Array type constructors should have FpyValue defaults for each element."""
        from fpy.state import TypeCtorSymbol
        ctor = self._lookup_callable("Svc.BuffQueueDepth")
        assert isinstance(ctor, TypeCtorSymbol)
        assert len(ctor.args) == 1
        assert ctor.args[0][2] is not None
        assert ctor.args[0][2].val == 0

    def test_array_ctor_multi_element_defaults(self):
        ctor = self._lookup_callable("Svc.ComQueueDepth")
        assert len(ctor.args) == 2
        for _, _, default in ctor.args:
            assert default is not None
            assert default.val == 0

    def test_array_of_enums_ctor_defaults(self):
        ctor = self._lookup_callable("Ref.ManyChoices")
        assert len(ctor.args) == 2
        for _, _, default in ctor.args:
            assert default is not None
            assert default.val == "ONE"

    def test_struct_without_default_has_none_args(self):
        """Structs without defaults should still have None for each member's default."""
        scope = self._get_callable_scope()
        parts = "$CheckState".split(".")
        current = scope
        for part in parts:
            current = current[part]
        ctor = current
        for _, _, default in ctor.args:
            assert default is None
