"""Runtime error outcomes of the fpy bytecode ABI."""

from __future__ import annotations

from enum import Enum

from fpy.bytecode.directives import StackSizeType

# A stack frame header stores the return address and the previous frame's
# offset, each a StackSizeType.
STACK_FRAME_HEADER_SIZE = StackSizeType.max_size * 2


class ValidationError(Exception):
    """The arguments supplied to a sequence do not match its argument spec."""


class DirectiveErrorCode(Enum):
    """The reason a directive terminated its sequence.

    Mirrors `Svc.Fpy.DirectiveErrorCode`, except for
    DESERIALIZE_ERROR_INVALID_BOOL, which only the wasm backend emits.
    """

    NO_ERROR = 0
    STMT_OUT_OF_BOUNDS = 1
    TLM_GET_NOT_CONNECTED = 2
    TLM_CHAN_NOT_FOUND = 3
    PRM_GET_NOT_CONNECTED = 4
    PRM_NOT_FOUND = 5
    CMD_SERIALIZE_FAILURE = 6
    EXIT_WITH_ERROR = 7
    STACK_ACCESS_OUT_OF_BOUNDS = 8
    STACK_OVERFLOW = 9
    DOMAIN_ERROR = 10
    ARRAY_OUT_OF_BOUNDS = 11
    ARITHMETIC_OVERFLOW = 12
    ARITHMETIC_UNDERFLOW = 13
    FRAME_START_OUT_OF_BOUNDS = 14
    STACK_UNDERFLOW = 15
    INVALID_ARG = 16
    CMD_FAIL = 17
    SERIAL_PORT_NOT_CONNECTED = 18
    SERIAL_PORT_INVALID_INDEX = 19
    DESERIALIZE_ERROR_INVALID_BOOL = 20
