"""
Tests for the fpy.dictionary module — our own FPP JSON dictionary parser.
"""

import json
import tempfile
import os
import pytest
from pathlib import Path

from fpy.dictionary import (
    _resolve_type,
    _parse_type_definitions,
    _parse_commands,
    _parse_channels,
    _parse_parameters,
    _parse_constants,
    load_dictionary,
    json_default_to_fpy_value,
    PRIMITIVE_TYPE_MAP,
)

from fpy.types import (
    FpyType,
    FpyValue,
    TypeKind,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    BOOL,
    INTERNAL_STRING,
)
from fpy.state import CmdDef, ChDef, PrmDef


REF_DICT_PATH = str(Path(__file__).parent / "RefTopologyDictionary.json")


# ---------------------------------------------------------------------------
# _resolve_type
# ---------------------------------------------------------------------------
class TestResolveType:
    def test_integer_types(self):
        for name, expected in PRIMITIVE_TYPE_MAP.items():
            desc = {"kind": "integer", "name": name}
            assert _resolve_type(desc, {}) is expected

    def test_float_f32(self):
        desc = {"kind": "float", "name": "F32", "size": 32}
        assert _resolve_type(desc, {}) is F32

    def test_float_f64(self):
        desc = {"kind": "float", "name": "F64", "size": 64}
        assert _resolve_type(desc, {}) is F64

    def test_float_unknown_size(self):
        desc = {"kind": "float", "name": "F128", "size": 128}
        with pytest.raises(AssertionError, match="Unknown float size"):
            _resolve_type(desc, {})

    def test_bool(self):
        desc = {"kind": "bool", "name": "bool"}
        assert _resolve_type(desc, {}) is BOOL

    def test_string(self):
        desc = {"kind": "string", "name": "string", "size": 80}
        result = _resolve_type(desc, {})
        assert result.is_string

    def test_qualified_identifier_found(self):
        fake_type = FpyType(TypeKind.U32, "Fake")
        type_defs = {"My.Type": fake_type}
        desc = {"kind": "qualifiedIdentifier", "name": "My.Type"}
        assert _resolve_type(desc, type_defs) is fake_type

    def test_qualified_identifier_missing(self):
        desc = {"kind": "qualifiedIdentifier", "name": "No.Such.Type"}
        with pytest.raises(AssertionError, match="Unknown type reference"):
            _resolve_type(desc, {})

    def test_unknown_kind(self):
        desc = {"kind": "banana", "name": "x"}
        with pytest.raises(AssertionError, match="Unknown type kind"):
            _resolve_type(desc, {})


# ---------------------------------------------------------------------------
# _parse_type_definitions
# ---------------------------------------------------------------------------
class TestParseTypeDefinitions:
    def test_enum(self):
        raw = [
            {
                "kind": "enum",
                "qualifiedName": "My.Color",
                "representationType": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
                "enumeratedConstants": [
                    {"name": "RED", "value": 0},
                    {"name": "GREEN", "value": 1},
                    {"name": "BLUE", "value": 2},
                ],
            }
        ]
        result = _parse_type_definitions(raw)
        assert "My.Color" in result
        typ = result["My.Color"]
        assert typ.kind == TypeKind.ENUM
        assert typ.enum_dict == {"RED": 0, "GREEN": 1, "BLUE": 2}

    def test_alias_to_primitive(self):
        raw = [
            {
                "kind": "alias",
                "qualifiedName": "FwIndexType",
                "type": {"name": "PlatformIndexType", "kind": "qualifiedIdentifier"},
                "underlyingType": {"name": "I16", "kind": "integer", "size": 16, "signed": True},
            }
        ]
        result = _parse_type_definitions(raw)
        assert result["FwIndexType"] is I16

    def test_alias_to_enum(self):
        """Alias whose underlyingType references an enum parsed in the same batch."""
        raw = [
            {
                "kind": "enum",
                "qualifiedName": "My.Status",
                "representationType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                "enumeratedConstants": [{"name": "OK", "value": 0}],
            },
            {
                "kind": "alias",
                "qualifiedName": "My.StatusAlias",
                "type": {"name": "My.Status", "kind": "qualifiedIdentifier"},
                "underlyingType": {"name": "My.Status", "kind": "qualifiedIdentifier"},
            },
        ]
        result = _parse_type_definitions(raw)
        assert result["My.StatusAlias"].kind == TypeKind.ENUM

    def test_array(self):
        raw = [
            {
                "kind": "array",
                "qualifiedName": "My.ThreeU32s",
                "size": 3,
                "elementType": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
                "default": [0, 0, 0],
            }
        ]
        result = _parse_type_definitions(raw)
        typ = result["My.ThreeU32s"]
        assert typ.kind == TypeKind.ARRAY
        assert typ.length == 3
        assert typ.elem_type is U32

    def test_struct(self):
        raw = [
            {
                "kind": "struct",
                "qualifiedName": "My.Point",
                "members": {
                    "x": {"type": {"name": "I32", "kind": "integer", "size": 32, "signed": True}, "index": 0},
                    "y": {"type": {"name": "I32", "kind": "integer", "size": 32, "signed": True}, "index": 1},
                },
            }
        ]
        result = _parse_type_definitions(raw)
        typ = result["My.Point"]
        assert typ.kind == TypeKind.STRUCT
        member_names = [m.name for m in typ.members]

    def test_struct_member_order_by_index(self):
        """Members should be sorted by index, not by dict key order."""
        raw = [
            {
                "kind": "struct",
                "qualifiedName": "My.Reversed",
                "members": {
                    "z": {"type": {"name": "U8", "kind": "integer", "size": 8, "signed": False}, "index": 2},
                    "a": {"type": {"name": "U8", "kind": "integer", "size": 8, "signed": False}, "index": 0},
                    "m": {"type": {"name": "U8", "kind": "integer", "size": 8, "signed": False}, "index": 1},
                },
            }
        ]
        result = _parse_type_definitions(raw)
        member_names = [m.name for m in result["My.Reversed"].members]
        assert member_names == ["a", "m", "z"]

    def test_array_of_enum(self):
        """Array whose element type is an enum defined in the same batch."""
        raw = [
            {
                "kind": "enum",
                "qualifiedName": "My.Dir",
                "representationType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                "enumeratedConstants": [{"name": "UP", "value": 0}, {"name": "DOWN", "value": 1}],
            },
            {
                "kind": "array",
                "qualifiedName": "My.Dirs",
                "size": 4,
                "elementType": {"name": "My.Dir", "kind": "qualifiedIdentifier"},
            },
        ]
        result = _parse_type_definitions(raw)
        assert result["My.Dirs"].kind == TypeKind.ARRAY
        assert result["My.Dirs"].elem_type.kind == TypeKind.ENUM

    def test_struct_referencing_array(self):
        """Struct with a member that is an array — tests cross-reference resolution."""
        raw = [
            {
                "kind": "array",
                "qualifiedName": "My.Vec3",
                "size": 3,
                "elementType": {"name": "F32", "kind": "float", "size": 32},
            },
            {
                "kind": "struct",
                "qualifiedName": "My.Pose",
                "members": {
                    "position": {"type": {"name": "My.Vec3", "kind": "qualifiedIdentifier"}, "index": 0},
                    "heading": {"type": {"name": "F32", "kind": "float", "size": 32}, "index": 1},
                },
            },
        ]
        result = _parse_type_definitions(raw)
        pose = result["My.Pose"]
        assert pose.kind == TypeKind.STRUCT
        # first member should be the array type
        assert pose.members[0].type.kind == TypeKind.ARRAY

    def test_unknown_type_definition_kind(self):
        raw = [{"kind": "union", "qualifiedName": "My.Bad"}]
        with pytest.raises(AssertionError, match="Unknown type definition kind"):
            _parse_type_definitions(raw)

    def test_empty_input(self):
        assert _parse_type_definitions([]) == {}


# ---------------------------------------------------------------------------
# _parse_commands
# ---------------------------------------------------------------------------
class TestParseCommands:
    def test_basic_command(self):
        type_defs = {}
        raw = [
            {
                "name": "Ref.cmdDisp.CMD_NO_OP",
                "commandKind": "async",
                "opcode": 1234,
                "formalParams": [],
                "annotation": "A no-op command",
            }
        ]
        id_dict, name_dict = _parse_commands(raw, type_defs)
        assert 1234 in id_dict
        assert "Ref.cmdDisp.CMD_NO_OP" in name_dict
        cmd = id_dict[1234]
        assert cmd.name == "Ref.cmdDisp.CMD_NO_OP"
        assert cmd.opcode == 1234
        assert cmd.arguments == []

    def test_command_with_args(self):
        enum_type = FpyType(TypeKind.ENUM, "TestCmd.Color", enum_dict={"RED": 0, "GREEN": 1}, rep_type=U32)
        type_defs = {"TestCmd.Color": enum_type}
        raw = [
            {
                "name": "Ref.comp.SET_COLOR",
                "commandKind": "async",
                "opcode": 42,
                "formalParams": [
                    {
                        "name": "color",
                        "type": {"name": "TestCmd.Color", "kind": "qualifiedIdentifier"},
                        "ref": False,
                    },
                    {
                        "name": "brightness",
                        "type": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                        "ref": False,
                    },
                ],
            }
        ]
        id_dict, name_dict = _parse_commands(raw, type_defs)
        cmd = id_dict[42]
        assert len(cmd.arguments) == 2
        assert cmd.arguments[0][0] == "color"
        assert cmd.arguments[0][2].kind == TypeKind.ENUM
        assert cmd.arguments[1][0] == "brightness"
        assert cmd.arguments[1][2] is U8

    def test_empty_commands(self):
        id_dict, name_dict = _parse_commands([], {})
        assert id_dict == {}
        assert name_dict == {}


# ---------------------------------------------------------------------------
# _parse_channels
# ---------------------------------------------------------------------------
class TestParseChannels:
    def test_basic_channel(self):
        raw = [
            {
                "name": "Ref.comp.MyChannel",
                "type": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
                "id": 999,
                "telemetryUpdate": "on change",
                "annotation": "A test channel",
            }
        ]
        id_dict, name_dict = _parse_channels(raw, {})
        assert 999 in id_dict
        assert "Ref.comp.MyChannel" in name_dict
        ch = id_dict[999]
        assert ch.name == "Ref.comp.MyChannel"
        assert ch.ch_id == 999
        assert ch.ch_type is U32

    def test_channel_with_limits(self):
        raw = [
            {
                "name": "Ref.comp.TempSensor",
                "type": {"name": "F32", "kind": "float", "size": 32},
                "id": 500,
                "limit": {
                    "low": {"red": -40.0, "yellow": -10.0},
                    "high": {"yellow": 85.0, "red": 125.0},
                },
            }
        ]
        id_dict, _ = _parse_channels(raw, {})
        ch = id_dict[500]
        assert ch.ch_id == 500

    def test_channel_with_enum_type(self):
        enum_type = FpyType(TypeKind.ENUM, "TestCh.Status", enum_dict={"OK": 0, "ERR": 1}, rep_type=U8)
        type_defs = {"TestCh.Status": enum_type}
        raw = [
            {
                "name": "Ref.comp.Status",
                "type": {"name": "TestCh.Status", "kind": "qualifiedIdentifier"},
                "id": 777,
            }
        ]
        id_dict, _ = _parse_channels(raw, type_defs)
        assert id_dict[777].ch_type.kind == TypeKind.ENUM

    def test_empty_channels(self):
        id_dict, name_dict = _parse_channels([], {})
        assert id_dict == {}
        assert name_dict == {}


# ---------------------------------------------------------------------------
# _parse_parameters
# ---------------------------------------------------------------------------
class TestParseParameters:
    def test_basic_parameter(self):
        raw = [
            {
                "name": "Ref.comp.MY_PARAM",
                "type": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
                "id": 1001,
            }
        ]
        id_dict, name_dict = _parse_parameters(raw, {})
        assert 1001 in id_dict
        assert "Ref.comp.MY_PARAM" in name_dict
        prm = id_dict[1001]
        assert prm.name == "Ref.comp.MY_PARAM"
        assert prm.prm_id == 1001
        assert prm.prm_type is U32

    def test_parameter_with_enum_type(self):
        enum_type = FpyType(TypeKind.ENUM, "TestPrm.Choice", enum_dict={"A": 0, "B": 1, "C": 2}, rep_type=I32)
        type_defs = {"TestPrm.Choice": enum_type}
        raw = [
            {
                "name": "Ref.comp.CHOICE_PRM",
                "type": {"name": "TestPrm.Choice", "kind": "qualifiedIdentifier"},
                "id": 2002,
            }
        ]
        id_dict, _ = _parse_parameters(raw, type_defs)
        assert id_dict[2002].prm_type.kind == TypeKind.ENUM

    def test_empty_parameters(self):
        id_dict, name_dict = _parse_parameters([], {})
        assert id_dict == {}
        assert name_dict == {}


# ---------------------------------------------------------------------------
# _parse_constants
# ---------------------------------------------------------------------------
class TestParseConstants:
    def test_integer_constants(self):
        raw = [
            {"qualifiedName": "Svc.Fpy.MAX_STACK_SIZE", "value": 65535},
            {"qualifiedName": "Svc.Fpy.MAX_DIRECTIVE_SIZE", "value": 2048},
        ]
        result = _parse_constants(raw, {})
        assert result["Svc.Fpy.MAX_STACK_SIZE"] == 65535
        assert result["Svc.Fpy.MAX_DIRECTIVE_SIZE"] == 2048

    def test_empty_constants(self):
        assert _parse_constants([], {}) == {}


# ---------------------------------------------------------------------------
# load_dictionary (integration against real RefTopologyDictionary.json)
# ---------------------------------------------------------------------------
class TestLoadDictionary:
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
        # Find a command with args
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
        """Enum types should have ENUM_DICT populated."""
        d = load_dictionary(REF_DICT_PATH)
        # Ref.Choice is used as a parameter type
        choice = d["type_defs"]["Ref.Choice"]
        assert choice.kind == TypeKind.ENUM
        assert "ONE" in choice.enum_dict

    def test_struct_type_parsed(self):
        """Struct types should have MEMBER_LIST populated."""
        d = load_dictionary(REF_DICT_PATH)
        # Find a struct type
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
        """Array types should have LENGTH and MEMBER_TYPE."""
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


# ---------------------------------------------------------------------------
# Synthetic / edge-case dictionary
# ---------------------------------------------------------------------------
class TestSyntheticDictionary:
    """Tests using small hand-crafted dictionaries to cover edge cases."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        load_dictionary.cache_clear()
        yield
        load_dictionary.cache_clear()

    def _write_dict(self, data: dict) -> str:
        """Write a dictionary to a temp file and return the path."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, f)
        f.close()
        return f.name

    def _minimal_dict(self, **overrides) -> dict:
        """Return a minimal valid dictionary structure with optional overrides."""
        base = {
            "metadata": {"projectName": "Test", "frameworkVersion": "v0", "projectVersion": "v0"},
            "typeDefinitions": [],
            "constants": [],
            "commands": [],
            "parameters": [],
            "events": [],
            "telemetryChannels": [],
        }
        base.update(overrides)
        return base

    def test_empty_dictionary(self):
        path = self._write_dict(self._minimal_dict())
        d = load_dictionary(path)
        assert d["type_defs"] == {}
        assert d["cmd_id_dict"] == {}
        assert d["ch_id_dict"] == {}
        assert d["prm_id_dict"] == {}
        assert d["constants"] == {}
        os.unlink(path)

    def test_command_no_params(self):
        data = self._minimal_dict(
            commands=[
                {
                    "name": "A.b.NO_OP",
                    "commandKind": "async",
                    "opcode": 1,
                    "formalParams": [],
                }
            ]
        )
        path = self._write_dict(data)
        d = load_dictionary(path)
        cmd = d["cmd_id_dict"][1]
        assert cmd.name == "A.b.NO_OP"
        assert cmd.arguments == []
        os.unlink(path)

    def test_channel_bool_type(self):
        data = self._minimal_dict(
            telemetryChannels=[
                {
                    "name": "A.b.Flag",
                    "type": {"name": "bool", "kind": "bool"},
                    "id": 10,
                }
            ]
        )
        path = self._write_dict(data)
        d = load_dictionary(path)
        assert d["ch_id_dict"][10].ch_type is BOOL
        os.unlink(path)

    def test_chained_aliases(self):
        """A -> B -> U32: even if A appears before B, both should resolve."""
        data = self._minimal_dict(
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
        path = self._write_dict(data)
        d = load_dictionary(path)
        assert d["type_defs"]["Synth.AliasA"] is U32
        assert d["type_defs"]["Synth.AliasB"] is U32
        os.unlink(path)

    def test_struct_with_enum_member(self):
        data = self._minimal_dict(
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
        path = self._write_dict(data)
        d = load_dictionary(path)
        move = d["type_defs"]["Synth.Move"]
        assert move.kind == TypeKind.STRUCT
        assert move.members[0].name == "dir"
        assert move.members[0].type.kind == TypeKind.ENUM
        assert move.members[1].name == "dist"
        assert move.members[1].type is U32
        os.unlink(path)

    def test_command_with_struct_arg(self):
        """Command whose formal param type is a struct."""
        data = self._minimal_dict(
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
        path = self._write_dict(data)
        d = load_dictionary(path)
        cmd = d["cmd_id_dict"][99]
        assert len(cmd.arguments) == 1
        assert cmd.arguments[0][0] == "pair"
        assert cmd.arguments[0][2].kind == TypeKind.STRUCT
        os.unlink(path)

    def test_string_type_in_channel(self):
        data = self._minimal_dict(
            telemetryChannels=[
                {
                    "name": "A.b.Message",
                    "type": {"name": "string", "kind": "string", "size": 256},
                    "id": 55,
                }
            ]
        )
        path = self._write_dict(data)
        d = load_dictionary(path)
        ch = d["ch_id_dict"][55]
        assert ch.ch_type.is_string
        os.unlink(path)

    def test_multiple_commands_unique_opcodes(self):
        data = self._minimal_dict(
            commands=[
                {"name": "A.b.CMD1", "commandKind": "async", "opcode": 1, "formalParams": []},
                {"name": "A.b.CMD2", "commandKind": "async", "opcode": 2, "formalParams": []},
                {"name": "A.c.CMD3", "commandKind": "async", "opcode": 3, "formalParams": []},
            ]
        )
        path = self._write_dict(data)
        d = load_dictionary(path)
        assert len(d["cmd_id_dict"]) == 3
        assert len(d["cmd_name_dict"]) == 3
        os.unlink(path)


# ---------------------------------------------------------------------------
# json_default_to_fpy_value
# ---------------------------------------------------------------------------
class TestJsonDefaultToFpyValue:
    """Tests for converting raw JSON default values to FpyValue objects."""

    def test_bool_default(self):
        result = json_default_to_fpy_value(True, BOOL)
        assert result == FpyValue(BOOL, True)

        result = json_default_to_fpy_value(False, BOOL)
        assert result == FpyValue(BOOL, False)

    def test_integer_default(self):
        result = json_default_to_fpy_value(42, U32)
        assert result == FpyValue(U32, 42)

    def test_signed_integer_default(self):
        result = json_default_to_fpy_value(-5, I32)
        assert result == FpyValue(I32, -5)

    def test_float_default(self):
        result = json_default_to_fpy_value(3.14, F32)
        assert result == FpyValue(F32, 3.14)

    def test_float_from_int_default(self):
        """Integers should be accepted as float defaults."""
        result = json_default_to_fpy_value(0, F64)
        assert result == FpyValue(F64, 0.0)

    def test_string_default(self):
        string_type = FpyType(TypeKind.STRING, "String_80", max_length=80)
        result = json_default_to_fpy_value("hello", string_type)
        assert result == FpyValue(string_type, "hello")

    def test_enum_default_qualified_name(self):
        color = FpyType(
            TypeKind.ENUM, "My.Color",
            enum_dict={"RED": 0, "GREEN": 1, "BLUE": 2},
            rep_type=U32,
        )
        result = json_default_to_fpy_value("My.Color.RED", color)
        assert result == FpyValue(color, "RED")

    def test_enum_default_extracts_constant_name(self):
        """The constant name should be extracted from the qualified name."""
        status = FpyType(
            TypeKind.ENUM, "Svc.Status",
            enum_dict={"OK": 0, "ERR": 1},
            rep_type=U8,
        )
        result = json_default_to_fpy_value("Svc.Status.ERR", status)
        assert result.val == "ERR"

    def test_enum_default_unknown_constant(self):
        color = FpyType(
            TypeKind.ENUM, "My.Color",
            enum_dict={"RED": 0},
            rep_type=U32,
        )
        with pytest.raises(AssertionError, match="Unknown enum constant"):
            json_default_to_fpy_value("My.Color.PURPLE", color)

    def test_array_default(self):
        arr_type = FpyType(
            TypeKind.ARRAY, "My.ThreeU32s",
            elem_type=U32, length=3,
        )
        result = json_default_to_fpy_value([10, 20, 30], arr_type)
        assert len(result.val) == 3
        assert result.val[0] == FpyValue(U32, 10)
        assert result.val[1] == FpyValue(U32, 20)
        assert result.val[2] == FpyValue(U32, 30)

    def test_array_default_wrong_length(self):
        arr_type = FpyType(
            TypeKind.ARRAY, "My.TwoU32s",
            elem_type=U32, length=2,
        )
        with pytest.raises(AssertionError, match="Array default length"):
            json_default_to_fpy_value([1, 2, 3], arr_type)

    def test_array_of_floats_default(self):
        arr_type = FpyType(
            TypeKind.ARRAY, "My.F32s",
            elem_type=F32, length=3,
        )
        result = json_default_to_fpy_value([0.0, 1.0, 2.0], arr_type)
        assert len(result.val) == 3
        for elem in result.val:
            assert elem.type == F32

    def test_array_of_enums_default(self):
        color = FpyType(
            TypeKind.ENUM, "My.Color",
            enum_dict={"RED": 0, "GREEN": 1},
            rep_type=U8,
        )
        arr_type = FpyType(
            TypeKind.ARRAY, "My.Colors",
            elem_type=color, length=2,
        )
        result = json_default_to_fpy_value(
            ["My.Color.RED", "My.Color.GREEN"], arr_type
        )
        assert result.val[0] == FpyValue(color, "RED")
        assert result.val[1] == FpyValue(color, "GREEN")

    def test_struct_default(self):
        from fpy.types import StructMember
        struct_type = FpyType(
            TypeKind.STRUCT, "My.Point",
            members=(
                StructMember("x", I32),
                StructMember("y", I32),
            ),
        )
        result = json_default_to_fpy_value({"x": 0, "y": 0}, struct_type)
        assert result.val["x"] == FpyValue(I32, 0)
        assert result.val["y"] == FpyValue(I32, 0)

    def test_struct_default_with_enum_member(self):
        from fpy.types import StructMember
        status = FpyType(
            TypeKind.ENUM, "My.Status",
            enum_dict={"OK": 0, "ERR": 1},
            rep_type=U8,
        )
        struct_type = FpyType(
            TypeKind.STRUCT, "My.Result",
            members=(
                StructMember("code", U32),
                StructMember("status", status),
            ),
        )
        result = json_default_to_fpy_value(
            {"code": 42, "status": "My.Status.OK"}, struct_type
        )
        assert result.val["code"] == FpyValue(U32, 42)
        assert result.val["status"] == FpyValue(status, "OK")

    def test_struct_default_with_array_member(self):
        from fpy.types import StructMember
        arr_type = FpyType(
            TypeKind.ARRAY, "My.Vec3",
            elem_type=F32, length=3,
        )
        struct_type = FpyType(
            TypeKind.STRUCT, "My.Pose",
            members=(
                StructMember("position", arr_type),
                StructMember("heading", F32),
            ),
        )
        result = json_default_to_fpy_value(
            {"position": [0.0, 0.0, 0.0], "heading": 1.0}, struct_type
        )
        assert len(result.val["position"].val) == 3
        assert result.val["heading"] == FpyValue(F32, 1.0)

    def test_nested_struct_default(self):
        from fpy.types import StructMember
        inner = FpyType(
            TypeKind.STRUCT, "My.Inner",
            members=(
                StructMember("a", U32),
                StructMember("b", U32),
            ),
        )
        outer = FpyType(
            TypeKind.STRUCT, "My.Outer",
            members=(
                StructMember("inner", inner),
                StructMember("flag", BOOL),
            ),
        )
        result = json_default_to_fpy_value(
            {"inner": {"a": 1, "b": 2}, "flag": True}, outer
        )
        assert result.val["inner"].val["a"] == FpyValue(U32, 1)
        assert result.val["inner"].val["b"] == FpyValue(U32, 2)
        assert result.val["flag"] == FpyValue(BOOL, True)

    def test_struct_default_missing_member(self):
        from fpy.types import StructMember
        struct_type = FpyType(
            TypeKind.STRUCT, "My.Point",
            members=(
                StructMember("x", I32),
                StructMember("y", I32),
            ),
        )
        with pytest.raises(AssertionError, match="missing member 'y'"):
            json_default_to_fpy_value({"x": 0}, struct_type)


# ---------------------------------------------------------------------------
# Default values stored on FpyType during parsing
# ---------------------------------------------------------------------------
class TestTypeDefinitionDefaults:
    """Tests that _parse_type_definitions captures default values on FpyType."""

    def test_enum_default_parsed(self):
        raw = [
            {
                "kind": "enum",
                "qualifiedName": "My.Color",
                "representationType": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
                "enumeratedConstants": [
                    {"name": "RED", "value": 0},
                    {"name": "GREEN", "value": 1},
                ],
                "default": "My.Color.RED",
            }
        ]
        result = _parse_type_definitions(raw)
        typ = result["My.Color"]
        assert typ.json_default == "My.Color.RED"

    def test_enum_no_default(self):
        raw = [
            {
                "kind": "enum",
                "qualifiedName": "My.NoDefault",
                "representationType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                "enumeratedConstants": [{"name": "A", "value": 0}],
            }
        ]
        result = _parse_type_definitions(raw)
        assert result["My.NoDefault"].json_default is None

    def test_array_default_parsed(self):
        raw = [
            {
                "kind": "array",
                "qualifiedName": "My.ThreeU32s",
                "size": 3,
                "elementType": {"name": "U32", "kind": "integer", "size": 32, "signed": False},
                "default": [0, 0, 0],
            }
        ]
        result = _parse_type_definitions(raw)
        typ = result["My.ThreeU32s"]
        assert typ.json_default == [0, 0, 0]

    def test_array_no_default(self):
        raw = [
            {
                "kind": "array",
                "qualifiedName": "My.NoDef",
                "size": 2,
                "elementType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
            }
        ]
        result = _parse_type_definitions(raw)
        assert result["My.NoDef"].json_default is None

    def test_struct_default_parsed(self):
        raw = [
            {
                "kind": "struct",
                "qualifiedName": "My.Point",
                "members": {
                    "x": {"type": {"name": "I32", "kind": "integer", "size": 32, "signed": True}, "index": 0},
                    "y": {"type": {"name": "I32", "kind": "integer", "size": 32, "signed": True}, "index": 1},
                },
                "default": {"x": 0, "y": 0},
            }
        ]
        result = _parse_type_definitions(raw)
        typ = result["My.Point"]
        assert typ.json_default == {"x": 0, "y": 0}

    def test_struct_no_default(self):
        raw = [
            {
                "kind": "struct",
                "qualifiedName": "My.NoDef",
                "members": {
                    "a": {"type": {"name": "U8", "kind": "integer", "size": 8, "signed": False}, "index": 0},
                },
            }
        ]
        result = _parse_type_definitions(raw)
        assert result["My.NoDef"].json_default is None

    def test_struct_default_with_enum_member(self):
        """Struct with an enum member should store the raw default dict."""
        raw = [
            {
                "kind": "enum",
                "qualifiedName": "My.Status",
                "representationType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                "enumeratedConstants": [{"name": "OK", "value": 0}, {"name": "ERR", "value": 1}],
                "default": "My.Status.OK",
            },
            {
                "kind": "struct",
                "qualifiedName": "My.Result",
                "members": {
                    "code": {"type": {"name": "U32", "kind": "integer", "size": 32, "signed": False}, "index": 0},
                    "status": {"type": {"name": "My.Status", "kind": "qualifiedIdentifier"}, "index": 1},
                },
                "default": {"code": 0, "status": "My.Status.OK"},
            },
        ]
        result = _parse_type_definitions(raw)
        assert result["My.Result"].json_default == {"code": 0, "status": "My.Status.OK"}

    def test_array_of_enums_default(self):
        raw = [
            {
                "kind": "enum",
                "qualifiedName": "My.Dir",
                "representationType": {"name": "U8", "kind": "integer", "size": 8, "signed": False},
                "enumeratedConstants": [{"name": "UP", "value": 0}, {"name": "DOWN", "value": 1}],
                "default": "My.Dir.UP",
            },
            {
                "kind": "array",
                "qualifiedName": "My.Dirs",
                "size": 2,
                "elementType": {"name": "My.Dir", "kind": "qualifiedIdentifier"},
                "default": ["My.Dir.UP", "My.Dir.DOWN"],
            },
        ]
        result = _parse_type_definitions(raw)
        assert result["My.Dirs"].json_default == ["My.Dir.UP", "My.Dir.DOWN"]


# ---------------------------------------------------------------------------
# Defaults from ref dictionary (integration)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Compiler: type defaults (integration with _build_global_scopes)
# ---------------------------------------------------------------------------
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
