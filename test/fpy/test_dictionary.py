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
    PRIMITIVE_TYPE_MAP,
)

from fprime_gds.common.models.serialize.numerical_types import (
    U8Type as U8Value,
    U16Type as U16Value,
    U32Type as U32Value,
    U64Type as U64Value,
    I8Type as I8Value,
    I16Type as I16Value,
    I32Type as I32Value,
    I64Type as I64Value,
    F32Type as F32Value,
    F64Type as F64Value,
)
from fprime_gds.common.models.serialize.bool_type import BoolType as BoolValue
from fprime_gds.common.models.serialize.string_type import StringType as StringValue
from fprime_gds.common.models.serialize.enum_type import EnumType as EnumValue
from fprime_gds.common.models.serialize.serializable_type import (
    SerializableType as StructValue,
)
from fprime_gds.common.models.serialize.array_type import ArrayType as ArrayValue
from fprime_gds.common.templates.ch_template import ChTemplate
from fprime_gds.common.templates.cmd_template import CmdTemplate
from fprime_gds.common.templates.prm_template import PrmTemplate


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
        assert _resolve_type(desc, {}) is F32Value

    def test_float_f64(self):
        desc = {"kind": "float", "name": "F64", "size": 64}
        assert _resolve_type(desc, {}) is F64Value

    def test_float_unknown_size(self):
        desc = {"kind": "float", "name": "F128", "size": 128}
        with pytest.raises(AssertionError, match="Unknown float size"):
            _resolve_type(desc, {})

    def test_bool(self):
        desc = {"kind": "bool", "name": "bool"}
        assert _resolve_type(desc, {}) is BoolValue

    def test_string(self):
        desc = {"kind": "string", "name": "string", "size": 80}
        result = _resolve_type(desc, {})
        assert issubclass(result, StringValue)

    def test_qualified_identifier_found(self):
        fake_type = type("Fake", (U32Value,), {})
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
        assert issubclass(typ, EnumValue)
        assert typ.ENUM_DICT == {"RED": 0, "GREEN": 1, "BLUE": 2}

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
        assert result["FwIndexType"] is I16Value

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
        assert issubclass(result["My.StatusAlias"], EnumValue)

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
        assert issubclass(typ, ArrayValue)
        assert typ.LENGTH == 3
        assert typ.MEMBER_TYPE is U32Value

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
        assert issubclass(typ, StructValue)
        member_names = [m[0] for m in typ.MEMBER_LIST]
        assert member_names == ["x", "y"]

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
        member_names = [m[0] for m in result["My.Reversed"].MEMBER_LIST]
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
        assert issubclass(result["My.Dirs"], ArrayValue)
        assert issubclass(result["My.Dirs"].MEMBER_TYPE, EnumValue)

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
        assert issubclass(pose, StructValue)
        # first member should be the array type
        assert issubclass(pose.MEMBER_LIST[0][1], ArrayValue)

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
        assert cmd.get_full_name() == "Ref.cmdDisp.CMD_NO_OP"
        assert cmd.get_op_code() == 1234
        assert cmd.arguments == []

    def test_command_with_args(self):
        enum_type = EnumValue.construct_type("TestCmd.Color", {"RED": 0, "GREEN": 1}, "U32")
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
        assert issubclass(cmd.arguments[0][2], EnumValue)
        assert cmd.arguments[1][0] == "brightness"
        assert cmd.arguments[1][2] is U8Value

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
        assert ch.get_full_name() == "Ref.comp.MyChannel"
        assert ch.get_id() == 999
        assert ch.ch_type_obj is U32Value

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
        assert ch.get_id() == 500

    def test_channel_with_enum_type(self):
        enum_type = EnumValue.construct_type("TestCh.Status", {"OK": 0, "ERR": 1}, "U8")
        type_defs = {"TestCh.Status": enum_type}
        raw = [
            {
                "name": "Ref.comp.Status",
                "type": {"name": "TestCh.Status", "kind": "qualifiedIdentifier"},
                "id": 777,
            }
        ]
        id_dict, _ = _parse_channels(raw, type_defs)
        assert issubclass(id_dict[777].ch_type_obj, EnumValue)

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
        assert prm.get_full_name() == "Ref.comp.MY_PARAM"
        assert prm.get_id() == 1001
        assert prm.prm_type_obj is U32Value

    def test_parameter_with_enum_type(self):
        enum_type = EnumValue.construct_type("TestPrm.Choice", {"A": 0, "B": 1, "C": 2}, "I32")
        type_defs = {"TestPrm.Choice": enum_type}
        raw = [
            {
                "name": "Ref.comp.CHOICE_PRM",
                "type": {"name": "TestPrm.Choice", "kind": "qualifiedIdentifier"},
                "id": 2002,
            }
        ]
        id_dict, _ = _parse_parameters(raw, type_defs)
        assert issubclass(id_dict[2002].prm_type_obj, EnumValue)

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
        assert len(d["type_defs"]) == 87

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
        """Verify CmdTemplate attributes match expected API."""
        d = load_dictionary(REF_DICT_PATH)
        # Find a command with args
        cmd = d["cmd_name_dict"]["Ref.dpDemo.SelectColor"]
        assert cmd.get_full_name() == "Ref.dpDemo.SelectColor"
        assert isinstance(cmd.get_op_code(), int)
        assert len(cmd.arguments) == 1
        arg_name, arg_desc, arg_type = cmd.arguments[0]
        assert arg_name == "color"
        assert issubclass(arg_type, EnumValue)

    def test_channel_attributes(self):
        """Verify ChTemplate attributes match expected API."""
        d = load_dictionary(REF_DICT_PATH)
        ch = d["ch_name_dict"]["CdhCore.cmdDisp.CommandsDispatched"]
        assert ch.get_full_name() == "CdhCore.cmdDisp.CommandsDispatched"
        assert isinstance(ch.get_id(), int)
        assert ch.ch_type_obj is U32Value

    def test_parameter_attributes(self):
        """Verify PrmTemplate attributes match expected API."""
        d = load_dictionary(REF_DICT_PATH)
        prm = d["prm_name_dict"]["Ref.typeDemo.CHOICE_PRM"]
        assert prm.get_full_name() == "Ref.typeDemo.CHOICE_PRM"
        assert isinstance(prm.get_id(), int)
        assert issubclass(prm.prm_type_obj, EnumValue)

    def test_id_and_name_dicts_consistent(self):
        """Every entry in name_dict should also appear in id_dict."""
        d = load_dictionary(REF_DICT_PATH)

        for cmd in d["cmd_name_dict"].values():
            assert d["cmd_id_dict"][cmd.get_op_code()] is cmd

        for ch in d["ch_name_dict"].values():
            assert d["ch_id_dict"][ch.get_id()] is ch

        for prm in d["prm_name_dict"].values():
            assert d["prm_id_dict"][prm.get_id()] is prm

    def test_enum_type_parsed(self):
        """Enum types should have ENUM_DICT populated."""
        d = load_dictionary(REF_DICT_PATH)
        # Ref.Choice is used as a parameter type
        choice = d["type_defs"]["Ref.Choice"]
        assert issubclass(choice, EnumValue)
        assert "ONE" in choice.ENUM_DICT

    def test_struct_type_parsed(self):
        """Struct types should have MEMBER_LIST populated."""
        d = load_dictionary(REF_DICT_PATH)
        # Find a struct type
        structs = {
            name: typ
            for name, typ in d["type_defs"].items()
            if issubclass(typ, StructValue)
        }
        assert len(structs) > 0
        for name, typ in structs.items():
            assert hasattr(typ, "MEMBER_LIST")
            assert len(typ.MEMBER_LIST) > 0

    def test_array_type_parsed(self):
        """Array types should have LENGTH and MEMBER_TYPE."""
        d = load_dictionary(REF_DICT_PATH)
        arrays = {
            name: typ
            for name, typ in d["type_defs"].items()
            if issubclass(typ, ArrayValue)
        }
        assert len(arrays) > 0
        for name, typ in arrays.items():
            assert hasattr(typ, "LENGTH")
            assert typ.LENGTH > 0
            assert hasattr(typ, "MEMBER_TYPE")

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
        assert cmd.get_full_name() == "A.b.NO_OP"
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
        assert d["ch_id_dict"][10].ch_type_obj is BoolValue
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
        assert d["type_defs"]["Synth.AliasA"] is U32Value
        assert d["type_defs"]["Synth.AliasB"] is U32Value
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
        assert issubclass(move, StructValue)
        assert move.MEMBER_LIST[0][0] == "dir"
        assert issubclass(move.MEMBER_LIST[0][1], EnumValue)
        assert move.MEMBER_LIST[1][0] == "dist"
        assert move.MEMBER_LIST[1][1] is U32Value
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
        assert issubclass(cmd.arguments[0][2], StructValue)
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
        assert issubclass(ch.ch_type_obj, StringValue)
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
