"""
Dictionary parser for FPP JSON dictionaries.

Replaces the fprime-gds JSON loaders (CmdJsonLoader, ChJsonLoader,
PrmJsonLoader, TypeJsonLoader) with a single unified parser that reads
the JSON dictionary and produces the same data structures.
"""

from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

from fprime_gds.common.models.serialize.type_base import BaseType as FppValue
from fprime_gds.common.models.serialize.numerical_types import (
    U8Type as U8Value,
    U16Type as U16Value,
    U32Type as U32Value,
    U64Type as U64Value,
    I8Type as I8Value,
    I16Type as I16Value,
    I32Type as I32Value,
    I64Type as I64Value,
)
from fprime_gds.common.models.serialize.bool_type import BoolType as BoolValue
from fprime_gds.common.models.serialize.string_type import StringType as StringValue
from fprime_gds.common.models.serialize.enum_type import EnumType as EnumValue
from fprime_gds.common.models.serialize.serializable_type import (
    SerializableType as StructValue,
)
from fprime_gds.common.models.serialize.array_type import ArrayType as ArrayValue
from fprime_gds.common.models.serialize.time_type import TimeType as TimeValue
from fprime_gds.common.models.serialize.type_base import DictionaryType
from fprime_gds.common.templates.ch_template import ChTemplate
from fprime_gds.common.templates.cmd_template import CmdTemplate
from fprime_gds.common.templates.prm_template import PrmTemplate


def _cached_or_construct(construct_fn, name: str, *args) -> type[FppValue]:
    """Return an existing type from the construct_type cache if available,
    otherwise call construct_fn(name, *args) to create a new one.

    This avoids AssertionError from construct_type when a type with the same
    name was already created (e.g. at module-import time by types.py) with
    slightly different metadata (format strings, annotations).
    """
    if name in DictionaryType._CONSTRUCTS:
        cached = DictionaryType._CONSTRUCTS[name]
        # _CONSTRUCTS stores (class, props_dict) tuples
        return cached[0] if isinstance(cached, tuple) else cached
    return construct_fn(name, *args)


# Map of primitive type name -> type class
PRIMITIVE_TYPE_MAP: dict[str, type[FppValue]] = {
    "U8": U8Value,
    "U16": U16Value,
    "U32": U32Value,
    "U64": U64Value,
    "I8": I8Value,
    "I16": I16Value,
    "I32": I32Value,
    "I64": I64Value,
    "bool": BoolValue,
}

# Map of representation type name -> string expected by EnumType.construct_type
ENUM_REP_TYPES = {"U8", "U16", "U32", "U64", "I8", "I16", "I32", "I64"}


def _resolve_type(
    type_desc: dict, type_defs: dict[str, type[FppValue]]
) -> type[FppValue]:
    """Resolve a Type Descriptor JSON object to a type class.

    Args:
        type_desc: A Type Descriptor dict with at least 'kind' and 'name'.
        type_defs: Already-parsed type definitions, keyed by qualifiedName.

    Returns:
        A type class (subclass of FppValue).
    """
    kind = type_desc["kind"]
    name = type_desc["name"]

    if kind == "integer":
        assert name in PRIMITIVE_TYPE_MAP, f"Unknown integer type: {name}"
        return PRIMITIVE_TYPE_MAP[name]

    if kind == "float":
        if type_desc["size"] == 32:
            from fprime_gds.common.models.serialize.numerical_types import (
                F32Type as F32Value,
            )

            return F32Value
        elif type_desc["size"] == 64:
            from fprime_gds.common.models.serialize.numerical_types import (
                F64Type as F64Value,
            )

            return F64Value
        else:
            assert False, f"Unknown float size: {type_desc['size']}"

    if kind == "bool":
        return BoolValue

    if kind == "string":
        max_length = type_desc.get("size")
        return _cached_or_construct(StringValue.construct_type, f"String_{max_length}", max_length)

    if kind == "qualifiedIdentifier":
        assert name in type_defs, f"Unknown type reference: {name}"
        return type_defs[name]

    assert False, f"Unknown type kind: {kind}"


def _parse_type_definitions(raw_type_defs: list[dict]) -> dict[str, type[FppValue]]:
    """Parse all type definitions from the dictionary.

    Type definitions can reference each other via qualifiedIdentifier,
    so we do multiple passes until all are resolved.

    Returns:
        dict mapping qualifiedName -> type class
    """
    type_defs: dict[str, type[FppValue]] = {}

    # Separate by kind for ordered processing
    aliases = []
    enums = []
    arrays = []
    structs = []

    for td in raw_type_defs:
        kind = td["kind"]
        if kind == "alias":
            aliases.append(td)
        elif kind == "enum":
            enums.append(td)
        elif kind == "array":
            arrays.append(td)
        elif kind == "struct":
            structs.append(td)
        else:
            assert False, f"Unknown type definition kind: {kind}"

    # Phase 1: Parse enums (they only reference primitive types)
    for td in enums:
        name = td["qualifiedName"]
        rep_type_desc = td["representationType"]
        rep_type_name = rep_type_desc["name"]
        assert rep_type_name in ENUM_REP_TYPES, (
            f"Enum {name} has unsupported representation type: {rep_type_name}"
        )
        enum_dict = {}
        for const in td["enumeratedConstants"]:
            enum_dict[const["name"]] = const["value"]
        type_defs[name] = _cached_or_construct(
            EnumValue.construct_type, name, enum_dict, rep_type_name
        )

    # Phase 2: Parse aliases (may reference primitives or already-parsed types)
    # Aliases can chain (A2 -> A1 -> U32), so we iterate until all resolved.
    remaining_aliases = list(aliases)
    max_iterations = len(remaining_aliases) + 1
    for _ in range(max_iterations):
        still_remaining = []
        for td in remaining_aliases:
            name = td["qualifiedName"]
            # underlyingType is always a concrete primitive type descriptor
            underlying = td["underlyingType"]
            try:
                resolved = _resolve_type(underlying, type_defs)
                type_defs[name] = resolved
            except (AssertionError, KeyError):
                still_remaining.append(td)
        if not still_remaining:
            break
        remaining_aliases = still_remaining
    else:
        unresolved = [td["qualifiedName"] for td in remaining_aliases]
        assert False, f"Could not resolve alias types: {unresolved}"

    # Phase 3: Parse arrays and structs (may reference each other and previous types)
    # Arrays and structs can cross-reference, so we iterate until all resolved.
    remaining = [("array", td) for td in arrays] + [("struct", td) for td in structs]
    max_iterations = len(remaining) + 1
    for _ in range(max_iterations):
        still_remaining = []
        for kind, td in remaining:
            name = td["qualifiedName"]
            try:
                if kind == "array":
                    elem_type = _resolve_type(td["elementType"], type_defs)
                    length = td["size"]
                    type_defs[name] = _cached_or_construct(
                        ArrayValue.construct_type, name, elem_type, length, "{}"
                    )
                else:  # struct
                    members_json = td["members"]
                    # Sort members by index to get declaration order
                    sorted_members = sorted(
                        members_json.items(), key=lambda kv: kv[1]["index"]
                    )
                    member_list = []
                    for member_name, member_desc in sorted_members:
                        member_type = _resolve_type(member_desc["type"], type_defs)
                        # Handle inline member arrays (member has "size" key)
                        if "size" in member_desc:
                            array_size = member_desc["size"]
                            array_name = f"Array_{member_type.__name__}_{array_size}"
                            member_type = _cached_or_construct(
                                ArrayValue.construct_type,
                                array_name,
                                member_type,
                                array_size,
                                "{}",
                            )
                        # Use empty format/description to match GDS loader behavior
                        # and avoid construct_type cache conflicts
                        member_list.append(
                            (member_name, member_type, "{}", "")
                        )
                    type_defs[name] = _cached_or_construct(
                        StructValue.construct_type, name, member_list
                    )
            except (AssertionError, KeyError):
                still_remaining.append((kind, td))
        if not still_remaining:
            break
        remaining = still_remaining
    else:
        unresolved = [td["qualifiedName"] for _, td in remaining]
        assert False, f"Could not resolve types: {unresolved}"

    return type_defs


def _parse_commands(
    raw_commands: list[dict], type_defs: dict[str, type[FppValue]]
) -> tuple[dict[int, CmdTemplate], dict[str, CmdTemplate]]:
    """Parse command definitions from the dictionary.

    Returns:
        Tuple of (id_dict, name_dict) where:
        - id_dict maps opcode (int) -> CmdTemplate
        - name_dict maps qualified name (str) -> CmdTemplate
    """
    id_dict: dict[int, CmdTemplate] = {}
    name_dict: dict[str, CmdTemplate] = {}

    for cmd_json in raw_commands:
        full_name = cmd_json["name"]
        opcode = cmd_json["opcode"]

        # Split "Ref.cmdDisp.CMD_NO_OP" into component="Ref.cmdDisp" and mnemonic="CMD_NO_OP"
        parts = full_name.rsplit(".", 1)
        component = parts[0]
        mnemonic = parts[1]

        arguments = []
        for param in cmd_json.get("formalParams", []):
            param_name = param["name"]
            param_desc = param.get("annotation", "")
            param_type = _resolve_type(param["type"], type_defs)
            arguments.append((param_name, param_desc, param_type))

        description = cmd_json.get("annotation", "")

        cmd = CmdTemplate(opcode, mnemonic, component, arguments, description)
        id_dict[opcode] = cmd
        name_dict[full_name] = cmd

    return id_dict, name_dict


def _parse_channels(
    raw_channels: list[dict], type_defs: dict[str, type[FppValue]]
) -> tuple[dict[int, ChTemplate], dict[str, ChTemplate]]:
    """Parse telemetry channel definitions from the dictionary.

    Returns:
        Tuple of (id_dict, name_dict).
    """
    id_dict: dict[int, ChTemplate] = {}
    name_dict: dict[str, ChTemplate] = {}

    for ch_json in raw_channels:
        full_name = ch_json["name"]
        ch_id = ch_json["id"]

        parts = full_name.rsplit(".", 1)
        component = parts[0]
        ch_name = parts[1]

        ch_type = _resolve_type(ch_json["type"], type_defs)
        fmt_str = ch_json.get("format")
        description = ch_json.get("annotation")

        # Parse limits
        limits = ch_json.get("limit", {})
        low = limits.get("low", {})
        high = limits.get("high", {})

        cmd = ChTemplate(
            ch_id,
            ch_name,
            component,
            ch_type,
            ch_fmt_str=fmt_str,
            ch_desc=description,
            low_red=low.get("red"),
            low_orange=low.get("orange"),
            low_yellow=low.get("yellow"),
            high_yellow=high.get("yellow"),
            high_orange=high.get("orange"),
            high_red=high.get("red"),
        )
        id_dict[ch_id] = cmd
        name_dict[full_name] = cmd

    return id_dict, name_dict


def _parse_parameters(
    raw_params: list[dict], type_defs: dict[str, type[FppValue]]
) -> tuple[dict[int, PrmTemplate], dict[str, PrmTemplate]]:
    """Parse parameter definitions from the dictionary.

    Returns:
        Tuple of (id_dict, name_dict).
    """
    id_dict: dict[int, PrmTemplate] = {}
    name_dict: dict[str, PrmTemplate] = {}

    for prm_json in raw_params:
        full_name = prm_json["name"]
        prm_id = prm_json["id"]

        parts = full_name.rsplit(".", 1)
        component = parts[0]
        prm_name = parts[1]

        prm_type = _resolve_type(prm_json["type"], type_defs)
        default_val = prm_json.get("default")

        prm = PrmTemplate(prm_id, prm_name, component, prm_type, default_val)
        id_dict[prm_id] = prm
        name_dict[full_name] = prm

    return id_dict, name_dict


def _parse_constants(raw_constants: list[dict], type_defs: dict[str, type[FppValue]]) -> dict[str, object]:
    """Parse constants from the dictionary.

    Returns:
        dict mapping qualifiedName -> value (Python primitive).
    """
    constants = {}
    for const in raw_constants:
        name = const["qualifiedName"]
        constants[name] = const["value"]
    return constants


@lru_cache(maxsize=4)
def load_dictionary(dictionary_path: str) -> dict:
    """Load and parse a complete FPP JSON dictionary.

    Returns a dict with keys:
        - 'type_defs': dict[str, type[FppValue]] — all type definitions by qualifiedName
        - 'cmd_id_dict': dict[int, CmdTemplate] — commands by opcode
        - 'cmd_name_dict': dict[str, CmdTemplate] — commands by qualified name
        - 'ch_id_dict': dict[int, ChTemplate] — channels by ID
        - 'ch_name_dict': dict[str, ChTemplate] — channels by qualified name
        - 'prm_id_dict': dict[int, PrmTemplate] — parameters by ID
        - 'prm_name_dict': dict[str, PrmTemplate] — parameters by qualified name
        - 'constants': dict[str, object] — constants by qualifiedName
        - 'metadata': dict — raw metadata
    """
    with open(dictionary_path, "r") as f:
        raw = json.load(f)

    type_defs = _parse_type_definitions(raw.get("typeDefinitions", []))
    cmd_id_dict, cmd_name_dict = _parse_commands(raw.get("commands", []), type_defs)
    ch_id_dict, ch_name_dict = _parse_channels(
        raw.get("telemetryChannels", []), type_defs
    )
    prm_id_dict, prm_name_dict = _parse_parameters(
        raw.get("parameters", []), type_defs
    )
    constants = _parse_constants(raw.get("constants", []), type_defs)

    return {
        "type_defs": type_defs,
        "cmd_id_dict": cmd_id_dict,
        "cmd_name_dict": cmd_name_dict,
        "ch_id_dict": ch_id_dict,
        "ch_name_dict": ch_name_dict,
        "prm_id_dict": prm_id_dict,
        "prm_name_dict": prm_name_dict,
        "constants": constants,
        "metadata": raw.get("metadata", {}),
    }
