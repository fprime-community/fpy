"""Serialized value and command transport limits shared by code generators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fpy.bytecode.directives import FwOpcodeType, FwPacketDescriptorType
from fpy.error import BackendError
from fpy.types import FpyValue, FwSizeStoreType, TypeKind

if TYPE_CHECKING:
    from fpy.state import CompileState


def serialize_value(value: FpyValue, node=None) -> bytes:
    """Serialize a value after validating all nested string length prefixes."""
    _check_string_lengths(value, node)
    return value.serialize()


def _check_string_lengths(value: FpyValue, node) -> None:
    kind = value.type.kind
    if kind in (TypeKind.STRING, TypeKind.INTERNAL_STRING):
        encoded = value.val.encode("utf-8") if isinstance(value.val, str) else value.val
        maximum = FwSizeStoreType.value_range()[1]
        if len(encoded) > maximum:
            raise BackendError(
                f"String length ({len(encoded)} UTF-8 bytes) exceeds FwSizeStoreType "
                f"({FwSizeStoreType.name}) range ({maximum})",
                node,
            )
    elif kind == TypeKind.STRUCT:
        for member in value.type.members:
            item = value.val[member.name]
            _check_string_lengths(
                item if isinstance(item, FpyValue) else FpyValue(member.type, item),
                node,
            )
    elif kind == TypeKind.ARRAY:
        for item in value.val:
            _check_string_lengths(
                (
                    item
                    if isinstance(item, FpyValue)
                    else FpyValue(value.type.elem_type, item)
                ),
                node,
            )


def command_size_error(
    state: CompileState, opcode: int, args_size: int, node=None
) -> BackendError | None:
    """Return a diagnostic if a serialized command exceeds its transport buffers."""
    description = f"Command {state.cmd_names_by_opcode.get(opcode, hex(opcode))}"
    if (
        state.cmd_arg_buffer_max_size is not None
        and args_size > state.cmd_arg_buffer_max_size
    ):
        return BackendError(
            f"{description} has {args_size} bytes of arguments, which "
            f"exceeds FW_CMD_ARG_BUFFER_MAX_SIZE ({state.cmd_arg_buffer_max_size})",
            node,
        )
    packet_size = FwPacketDescriptorType.max_size + FwOpcodeType.max_size + args_size
    if (
        state.com_buffer_max_size is not None
        and packet_size > state.com_buffer_max_size
    ):
        return BackendError(
            f"{description} serializes to a {packet_size} byte packet "
            f"(packet descriptor + opcode + {args_size} bytes of arguments), "
            f"which exceeds FW_COM_BUFFER_MAX_SIZE ({state.com_buffer_max_size})",
            node,
        )
    return None
