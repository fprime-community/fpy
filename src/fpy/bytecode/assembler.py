from __future__ import annotations
import struct
import zlib
from dataclasses import astuple, dataclass, field, fields
from importlib.metadata import version
from numbers import Number
from pathlib import Path
from typing import Union
from lark import Lark, Token, Transformer, v_args
from lark.tree import Meta

from fpy.bytecode.directives import Directive, StackOpDirective, StackSizeType
from fpy.types import FpyType, FpyValue, INTERNAL_STRING

from fpy.error import CompileError

fpybc_grammar_str = (Path(__file__).parent / "grammar.lark").read_text(encoding="utf-8")


def parse(text: str):
    parser = Lark(
        fpybc_grammar_str,
        start="input",
        parser="lalr",
        propagate_positions=True,
        maybe_placeholders=True,
    )

    tree = parser.parse(text)
    transformed = FpyBcTransformer().transform(tree)
    return transformed


@dataclass
class Node:
    meta: Meta = field(repr=False)
    id: int = field(init=False, repr=False, default=None)

    def __hash__(self):
        return hash(self.id)


@dataclass
class NodeOpWithNoArgs(Node):
    op: str


@dataclass
class NodeOpWithArgs(Node):
    op: str
    args: list[Node]


@dataclass
class NodeGotoTag(Node):
    tag: str


@dataclass
class NodeBytes(Node):
    value: int | str


NodeStmt = Union[NodeGotoTag, NodeOpWithNoArgs, NodeOpWithArgs]


@dataclass
class NodeBody(Node):
    stmts: list[NodeStmt]


@v_args(meta=False, inline=False)
def as_list(self, tree):
    return list(tree)


def no_inline_or_meta(type):
    @v_args(meta=False, inline=False)
    def wrapper(self, tree):
        return type(tree)

    return wrapper


def no_inline(type):
    @v_args(meta=True, inline=False)
    def wrapper(self, meta, tree):
        return type(meta, tree)

    return wrapper


def no_meta(type):
    @v_args(meta=False, inline=True)
    def wrapper(self, tree):
        return type(tree)

    return wrapper


def handle_str(meta, s: str):
    return s.strip("'").strip('"')

def handle_bytes(meta, value: list[int|str]|None):
    if value is None:
        return bytes()

    ret = bytes()
    for val in value:
        if isinstance(val, int):
            if val > 255 or val < 0:
                raise RuntimeError("Each byte must be < 256 and >= 0")
            ret += val.to_bytes(1, "little")
        elif isinstance(val, str):
            ret += val.encode("utf-8")

    return ret


def handle_op_with_args(meta, tree: list[Token]):
    return NodeOpWithArgs(meta, tree[0].value, [t for t in tree[1:] if t is not None])


def handle_op_with_no_args(meta, tree: list[Token]):
    return NodeOpWithNoArgs(meta, tree[0].value)


def handle_hex(self, s: str):
    return int(s, 16)


@v_args(meta=True, inline=True)
class FpyBcTransformer(Transformer):
    input = no_inline(NodeBody)

    goto_tag = NodeGotoTag
    op_with_no_args = no_inline(handle_op_with_no_args)
    op_with_args = no_inline(handle_op_with_args)

    NAME = str
    DEC_NUMBER = int
    NEG_DEC_NUMBER = int
    HEX_NUMBER = handle_hex
    bytes = no_inline(handle_bytes)
    FLOAT_NUMBER = float
    STRING = no_inline(handle_str)
    CONST_TRUE = lambda a, b: True
    CONST_FALSE = lambda a, b: False
    OP_WITH_NO_ARGS = str


def assemble(body: NodeBody) -> tuple[bytes, int]:
    line_idx = 0
    # collect all goto tags
    tags: dict[str, int] = {}
    for stmt in body.stmts:
        if isinstance(stmt, NodeGotoTag):
            tags[stmt.tag] = line_idx
        else:
            line_idx += 1

    dirs: list[Directive] = []
    for stmt in body.stmts:
        if isinstance(stmt, NodeGotoTag):
            continue

        # okay, it's a directive. which dir?

        op = stmt.op.upper()
        dir_type = None

        for subcls in Directive.__subclasses__() + StackOpDirective.__subclasses__():
            if subcls.opcode.name == op:
                dir_type = subcls
                break
        assert dir_type is not None, op

        # okay. if it's a no arg dir, just instantiate and add to list
        if isinstance(stmt, NodeOpWithNoArgs):
            dirs.append(dir_type())
        else:
            args = []
            if op == "GOTO" or op == "IF":
                if isinstance(stmt.args[0], str):
                    # lookup tag
                    tag_idx = tags.get(stmt.args[0], None)
                    if tag_idx is None:
                        raise RuntimeError(f"Unknown tag {stmt.args[0]}")
                    args.append(tag_idx)
                else:
                    args = stmt.args
            else:
                args = stmt.args

            dirs.append(dir_type(*args))

    return dirs


def fpybc_directives_to_fpyasm(dirs: list[Directive]) -> str:
    out = ""
    for dir in dirs:
        # write the op name
        out += dir.opcode.name.lower()

        # write the args
        for field in fields(dir):
            field_value = getattr(dir, field.name)
            val = None

            if isinstance(field_value, FpyValue):
                val = field_value.val
            else:
                val = field_value

            if isinstance(val, str):
                out += ' "' + val + '"'
            elif isinstance(val, (Number, bool)):
                out += " " + str(val)
            elif isinstance(val, bytes):
                for byte in val:
                    out += " " + str(byte)
            else:
                assert False, type(val)

        out += "\n"

    return out


def _get_version_tuple() -> tuple[int, int, int]:
    try:
        import re
        v = version("fprime-fpy")
        # Handle versions like "0.0.1a3.dev103+g244fdeadc"
        # Extract just the major.minor.patch part
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", v)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return (0, 0, 0)
    except Exception:
        return (0, 0, 0)


MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION = _get_version_tuple()
SCHEMA_VERSION = 6

HEADER_FORMAT = "!BBBBBHI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


@dataclass
class Header:
    majorVersion: int
    minorVersion: int
    patchVersion: int
    schemaVersion: int
    argumentCount: int
    statementCount: int
    bodySize: int

    def pack(self) -> bytes:
        return struct.pack(
            HEADER_FORMAT,
            self.majorVersion,
            self.minorVersion,
            self.patchVersion,
            self.schemaVersion,
            self.argumentCount,
            self.statementCount,
            self.bodySize,
        )

    @staticmethod
    def unpack(data: bytes) -> Header:
        (major, minor, patch, schema, arg_count, stmt_count, body_size) = struct.unpack_from(HEADER_FORMAT, data)
        return Header(major, minor, patch, schema, arg_count, stmt_count, body_size)


FOOTER_FORMAT = "!I"
FOOTER_SIZE = struct.calcsize(FOOTER_FORMAT)


@dataclass
class Footer:
    crc: int


def _serialize_arg_specs(arg_specs: list[tuple[str, str, int]]) -> bytes:
    """Serialize arg specs as (arg_name, type_name, size) triples.

    Binary format per arg_spec:
        [arg_name as a string]   (FwSizeStoreType-prefixed UTF-8, default 16-bit)
        [type_name as a string]  (FwSizeStoreType-prefixed UTF-8, default 16-bit)
        [StackSizeType bytes: size]
    """
    result = bytes()
    for arg_name, type_name, size in arg_specs:
        result += FpyValue(INTERNAL_STRING, arg_name).serialize()
        result += FpyValue(INTERNAL_STRING, type_name).serialize()
        result += FpyValue(StackSizeType, size).serialize()
    return result


def _deserialize_arg_specs(data: bytes, offset: int, count: int) -> tuple[int, list[tuple[str, str, int]]]:
    """Deserialize arg specs from (arg_name, type_name, size) triples.
    Returns (new_offset, list_of_(arg_name, type_name, size)_tuples)."""
    specs = []
    for _ in range(count):
        arg_name_val, offset = FpyValue.deserialize(INTERNAL_STRING, data, offset)
        type_name_val, offset = FpyValue.deserialize(INTERNAL_STRING, data, offset)
        size_val, offset = FpyValue.deserialize(StackSizeType, data, offset)
        specs.append((arg_name_val.val, type_name_val.val, size_val.val))
    return offset, specs


def deserialize_directives(data: bytes) -> tuple[list[Directive], list[tuple[str, str, int]]]:
    header = _unpack_and_check_header(data)

    # Deserialize arg specs section (immediately after fixed header)
    offset, arg_specs = _deserialize_arg_specs(data, HEADER_SIZE, header.argumentCount)

    dirs = []
    idx = 0
    while idx < header.statementCount:
        offset_and_dir = Directive.deserialize(data, offset)
        if offset_and_dir is None:
            raise RuntimeError("Unable to deserialize sequence")
        offset, dir = offset_and_dir
        dirs.append(dir)
        idx += 1

    if offset != len(data) - FOOTER_SIZE:
        raise RuntimeError(
            f"{len(data) - FOOTER_SIZE - offset} extra bytes at end of sequence"
        )

    # Verify CRC
    expected_crc = struct.unpack_from(FOOTER_FORMAT, data, offset)[0]
    actual_crc = zlib.crc32(data[:offset]) % (1 << 32)
    if expected_crc != actual_crc:
        raise RuntimeError(
            f"CRC mismatch (expected {hex(expected_crc)}, computed {hex(actual_crc)})"
        )

    return dirs, arg_specs


def _unpack_and_check_header(data: bytes) -> Header:
    """Unpack binary header and validate schema version."""
    header = Header.unpack(data)
    if header.schemaVersion != SCHEMA_VERSION:
        raise RuntimeError(
            f"Schema version mismatch: expected {SCHEMA_VERSION}, found {header.schemaVersion}"
        )
    return header


def read_bin_arg_specs(path: Path) -> list[tuple[str, str, int]]:
    """Read the arg specs section of a compiled .bin file.

    This is used at compile time to discover the expected argument types of a
    called sequence without deserializing the full directive body.
    """
    data = path.read_bytes()
    header = _unpack_and_check_header(data)
    _, arg_specs = _deserialize_arg_specs(data, HEADER_SIZE, header.argumentCount)
    return arg_specs


def resolve_arg_specs(
    arg_specs: list[tuple[str, str, int]],
    type_defs: dict[str, FpyType],
) -> list[tuple[str, FpyType]]:
    """Resolve (arg_name, type_name, size) arg_spec triples into (arg_name, FpyType) pairs.

    Looks up each type name in PRIMITIVE_TYPE_MAP first, then in *type_defs*.
    Raises RuntimeError if a type is not found or the size doesn't match.
    """
    from fpy.types import PRIMITIVE_TYPE_MAP

    arg_types = []
    for arg_name, type_name, size in arg_specs:
        if type_name in PRIMITIVE_TYPE_MAP:
            fpy_type = PRIMITIVE_TYPE_MAP[type_name]
        elif type_name in type_defs:
            fpy_type = type_defs[type_name]
        else:
            raise RuntimeError(f"Unknown type '{type_name}' (size {size})")
        if fpy_type.max_size != size:
            raise RuntimeError(
                f"Type '{type_name}' size mismatch: binary says {size}, dictionary says {fpy_type.max_size}"
            )
        arg_types.append((arg_name, fpy_type))
    return arg_types


def serialize_directives(
    dirs: list[Directive],
    arg_specs: list[tuple[str, str, int]] | None = None,
    max_directive_size: int = 2048,
) -> tuple[bytes, int]:
    if arg_specs is None:
        arg_specs = []

    assert len(arg_specs) <= 255, f"Too many sequence arguments ({len(arg_specs)}); should have been caught by CheckSeqRunArgs"

    body_bytes = bytes()

    for dir in dirs:
        dir_bytes = dir.serialize()
        if len(dir_bytes) > max_directive_size:
            print(
                CompileError(
                    f"Directive {dir} in sequence too large (expected less than {max_directive_size}, was {len(dir_bytes)})"
                )
            )
            exit(1)
        body_bytes += dir_bytes

    arg_specs_bytes = _serialize_arg_specs(arg_specs)

    header = Header(
        MAJOR_VERSION,
        MINOR_VERSION,
        PATCH_VERSION,
        SCHEMA_VERSION,
        len(arg_specs),
        len(dirs),
        len(arg_specs_bytes) + len(body_bytes),
    )
    output_bytes = header.pack() + arg_specs_bytes + body_bytes

    crc = zlib.crc32(output_bytes) % (1 << 32)
    footer = Footer(crc)
    output_bytes += struct.pack(FOOTER_FORMAT, *astuple(footer))

    return output_bytes, crc
