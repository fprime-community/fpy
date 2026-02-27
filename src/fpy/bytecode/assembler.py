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

from fpy.bytecode.directives import Directive, StackOpDirective
from fpy.types import FpyValue

from fpy.error import CompileError

fpybc_grammar_str = (Path(__file__).parent / "grammar.lark").read_text()


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


def directives_to_fpybc(dirs: list[Directive]) -> str:
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
SCHEMA_VERSION = 4

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


FOOTER_FORMAT = "!I"
FOOTER_SIZE = struct.calcsize(FOOTER_FORMAT)


@dataclass
class Footer:
    crc: int


def deserialize_directives(bytes: bytes) -> list[Directive]:
    header = Header(*struct.unpack_from(HEADER_FORMAT, bytes))

    if header.schemaVersion != SCHEMA_VERSION:
        raise RuntimeError(
            f"Schema version wrong (expected {SCHEMA_VERSION} found {header.schemaVersion})"
        )

    dirs = []
    idx = 0
    offset = HEADER_SIZE
    while idx < header.statementCount:
        offset_and_dir = Directive.deserialize(bytes, offset)
        if offset_and_dir is None:
            raise RuntimeError("Unable to deserialize sequence")
        offset, dir = offset_and_dir
        dirs.append(dir)
        idx += 1

    if offset != len(bytes) - FOOTER_SIZE:
        raise RuntimeError(
            f"{len(bytes) - FOOTER_SIZE - offset} extra bytes at end of sequence"
        )

    return dirs


def serialize_directives(dirs: list[Directive], max_directive_size: int = 2048) -> tuple[bytes, int]:
    output_bytes = bytes()

    for dir in dirs:
        dir_bytes = dir.serialize()
        if len(dir_bytes) > max_directive_size:
            print(
                CompileError(
                    f"Directive {dir} in sequence too large (expected less than {max_directive_size}, was {len(dir_bytes)})"
                )
            )
            exit(1)
        output_bytes += dir_bytes

    header = Header(
        MAJOR_VERSION,
        MINOR_VERSION,
        PATCH_VERSION,
        SCHEMA_VERSION,
        0,
        len(dirs),
        len(output_bytes),
    )
    output_bytes = struct.pack(HEADER_FORMAT, *astuple(header)) + output_bytes

    crc = zlib.crc32(output_bytes) % (1 << 32)
    footer = Footer(crc)
    output_bytes += struct.pack(FOOTER_FORMAT, *astuple(footer))

    return output_bytes, crc
