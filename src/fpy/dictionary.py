"""
Dictionary parser for FPP JSON dictionaries.

Replaces the fprime-gds JSON loaders with a single unified parser that reads
the JSON dictionary and produces FpyType / CmdDef / ChDef / PrmDef objects.
"""

from __future__ import annotations

import json
from functools import lru_cache

from fpy.types import (
    FpyType,
    TypeKind,
    StructMember,
    CmdDef,
    ChDef,
    PrmDef,
    PRIMITIVE_TYPE_MAP,
    F32,
    F64,
    BOOL,
)


# Map of representation type name -> expected by enum rep_type
ENUM_REP_TYPES = {"U8", "U16", "U32", "U64", "I8", "I16", "I32", "I64"}


def _resolve_type(
    type_desc: dict, type_defs: dict[str, FpyType]
) -> FpyType:
    """Resolve a Type Descriptor JSON object to an FpyType.

    Args:
        type_desc: A Type Descriptor dict with at least 'kind' and 'name'.
        type_defs: Already-parsed type definitions, keyed by qualifiedName.

    Returns:
        An FpyType instance.
    """
    kind = type_desc["kind"]
    name = type_desc["name"]

    if kind == "integer":
        assert name in PRIMITIVE_TYPE_MAP, f"Unknown integer type: {name}"
        return PRIMITIVE_TYPE_MAP[name]

    if kind == "float":
        if type_desc["size"] == 32:
            return F32
        elif type_desc["size"] == 64:
            return F64
        else:
            assert False, f"Unknown float size: {type_desc['size']}"

    if kind == "bool":
        return BOOL

    if kind == "string":
        max_length = type_desc.get("size")
        return FpyType(TypeKind.STRING, f"String_{max_length}", max_length=max_length)

    if kind == "qualifiedIdentifier":
        assert name in type_defs, f"Unknown type reference: {name}"
        return type_defs[name]

    assert False, f"Unknown type kind: {kind}"


def _parse_type_definitions(raw_type_defs: list[dict]) -> dict[str, FpyType]:
    """Parse all type definitions from the dictionary.

    Type definitions can reference each other via qualifiedIdentifier,
    so we do multiple passes until all are resolved.

    Returns:
        dict mapping qualifiedName -> FpyType
    """
    type_defs: dict[str, FpyType] = {}

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
        type_defs[name] = FpyType(
            TypeKind.ENUM,
            name,
            enum_dict=enum_dict,
            rep_type=PRIMITIVE_TYPE_MAP[rep_type_name],
        )

    # Phase 2: Parse aliases (may reference primitives or already-parsed types)
    remaining_aliases = list(aliases)
    max_iterations = len(remaining_aliases) + 1
    for _ in range(max_iterations):
        still_remaining = []
        for td in remaining_aliases:
            name = td["qualifiedName"]
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

    # Phase 3: Parse arrays and structs (may reference each other)
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
                    type_defs[name] = FpyType(
                        TypeKind.ARRAY,
                        name,
                        elem_type=elem_type,
                        length=length,
                    )
                else:  # struct
                    members_json = td["members"]
                    sorted_members = sorted(
                        members_json.items(), key=lambda kv: kv[1]["index"]
                    )
                    member_list = []
                    for member_name, member_desc in sorted_members:
                        member_type = _resolve_type(member_desc["type"], type_defs)
                        # Handle inline member arrays (member has "size" key)
                        if "size" in member_desc:
                            array_size = member_desc["size"]
                            array_name = f"Array_{member_type.name}_{array_size}"
                            # Check if we already have this array type
                            if array_name not in type_defs:
                                type_defs[array_name] = FpyType(
                                    TypeKind.ARRAY,
                                    array_name,
                                    elem_type=member_type,
                                    length=array_size,
                                )
                            member_type = type_defs[array_name]
                        member_list.append(StructMember(member_name, member_type))
                    type_defs[name] = FpyType(
                        TypeKind.STRUCT,
                        name,
                        members=tuple(member_list),
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
    raw_commands: list[dict], type_defs: dict[str, FpyType]
) -> tuple[dict[int, CmdDef], dict[str, CmdDef]]:
    """Parse command definitions from the dictionary.

    Returns:
        Tuple of (id_dict, name_dict) where:
        - id_dict maps opcode (int) -> CmdDef
        - name_dict maps qualified name (str) -> CmdDef
    """
    id_dict: dict[int, CmdDef] = {}
    name_dict: dict[str, CmdDef] = {}

    for cmd_json in raw_commands:
        full_name = cmd_json["name"]
        opcode = cmd_json["opcode"]

        arguments = []
        for param in cmd_json.get("formalParams", []):
            param_name = param["name"]
            param_desc = param.get("annotation", "")
            param_type = _resolve_type(param["type"], type_defs)
            arguments.append((param_name, param_desc, param_type))

        description = cmd_json.get("annotation", "")

        cmd = CmdDef(full_name, opcode, arguments, description)
        id_dict[opcode] = cmd
        name_dict[full_name] = cmd

    return id_dict, name_dict


def _parse_channels(
    raw_channels: list[dict], type_defs: dict[str, FpyType]
) -> tuple[dict[int, ChDef], dict[str, ChDef]]:
    """Parse telemetry channel definitions from the dictionary.

    Returns:
        Tuple of (id_dict, name_dict).
    """
    id_dict: dict[int, ChDef] = {}
    name_dict: dict[str, ChDef] = {}

    for ch_json in raw_channels:
        full_name = ch_json["name"]
        ch_id = ch_json["id"]
        ch_type = _resolve_type(ch_json["type"], type_defs)
        description = ch_json.get("annotation", "")

        ch = ChDef(full_name, ch_id, ch_type, description)
        id_dict[ch_id] = ch
        name_dict[full_name] = ch

    return id_dict, name_dict


def _parse_parameters(
    raw_params: list[dict], type_defs: dict[str, FpyType]
) -> tuple[dict[int, PrmDef], dict[str, PrmDef]]:
    """Parse parameter definitions from the dictionary.

    Returns:
        Tuple of (id_dict, name_dict).
    """
    id_dict: dict[int, PrmDef] = {}
    name_dict: dict[str, PrmDef] = {}

    for prm_json in raw_params:
        full_name = prm_json["name"]
        prm_id = prm_json["id"]
        prm_type = _resolve_type(prm_json["type"], type_defs)
        default_val = prm_json.get("default")

        prm = PrmDef(full_name, prm_id, prm_type, default_val)
        id_dict[prm_id] = prm
        name_dict[full_name] = prm

    return id_dict, name_dict


def _parse_constants(
    raw_constants: list[dict], type_defs: dict[str, FpyType]
) -> dict[str, object]:
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
        - 'type_defs': dict[str, FpyType] — all type definitions by qualifiedName
        - 'cmd_id_dict': dict[int, CmdDef] — commands by opcode
        - 'cmd_name_dict': dict[str, CmdDef] — commands by qualified name
        - 'ch_id_dict': dict[int, ChDef] — channels by ID
        - 'ch_name_dict': dict[str, ChDef] — channels by qualified name
        - 'prm_id_dict': dict[int, PrmDef] — parameters by ID
        - 'prm_name_dict': dict[str, PrmDef] — parameters by qualified name
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
