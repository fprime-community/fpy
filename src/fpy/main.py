from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import struct
import sys
import time

from fpy.bytecode.assembler import (
    MAJOR_VERSION,
    MINOR_VERSION,
    PATCH_VERSION,
    SCHEMA_VERSION,
    assemble,
    deserialize_directives,
    directives_to_fpybc,
    parse as fpybc_parse,
    resolve_arg_specs,
    serialize_directives,
)
from fpy.bytecode.directives import ConstCmdDirective
import fpy.error
import fpy.model
from fpy.model import DirectiveErrorCode, FpySequencerModel
from fpy.compiler import text_to_ast, ast_to_directives
from fpy.dictionary import load_dictionary
from fpy.types import PRIMITIVE_TYPE_MAP, TypeKind


def human_readable_size(size_bytes):
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_idx = 0
    while size_bytes >= 1024.0 and unit_idx < len(units) - 1:
        size_bytes /= 1024.0
        unit_idx += 1
    size_bytes = int(size_bytes)
    return f"{size_bytes} {units[unit_idx]}"


def get_package_version() -> str:
    try:
        return version("fprime-fpy")
    except Exception:
        return "unknown"


def get_version_str() -> str:
    return f"package {get_package_version()} langauge {MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION} schema {SCHEMA_VERSION}"


def compile_main(args: list[str] = None):
    arg_parser = argparse.ArgumentParser(
        description=f"Fpy compiler {get_version_str()}"
    )
    arg_parser.add_argument(
        "--version", action="version", version=f"%(prog)s {get_version_str()}"
    )
    arg_parser.add_argument("input", type=Path, help="The input .fpy file")
    arg_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default=None,
        help="The output .bin path",
    )
    arg_parser.add_argument(
        "-d",
        "--dictionary",
        type=Path,
        required=True,
        help="The FPrime dictionary .json file",
    )
    arg_parser.add_argument(
        "-b",
        "--bytecode",
        action="store_true",
        default=False,
        help="Whether to output human-readable bytecode to stdout instead of binary",
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Pass this to print out compiler debugging information",
    )
    arg_parser.add_argument(
        "-B",
        "--ground-binary-dir",
        type=Path,
        required=False,
        default=None,
        help="Local directory to resolve .bin file paths for sequence calls (default: input file directory)",
    )
    arg_parser.add_argument(
        "--flight-binary-dir",
        type=str,
        required=False,
        default=None,
        help="Absolute path prefix for .bin files on the spacecraft. "
             "Sequence paths starting with this prefix will have it stripped "
             "and be resolved relative to --ground-binary-dir.",
    )

    if args is not None:
        parsed_args = arg_parser.parse_args(args)
    else:
        parsed_args = arg_parser.parse_args()

    if parsed_args.debug:
        fpy.error.debug = True

    if not parsed_args.input.exists():
        print(f"Input file {parsed_args.input} does not exist")
        sys.exit(1)
    fpy.error.file_name = str(parsed_args.input)

    ground_binary_dir = parsed_args.ground_binary_dir
    if ground_binary_dir is None:
        ground_binary_dir = parsed_args.input.parent
    try:
        body = text_to_ast(parsed_args.input.read_text())
    except RecursionError:
        print("Recursion limit exceeded in parsing")
        sys.exit(1)
    try:
        result = ast_to_directives(body, parsed_args.dictionary, ground_binary_dir=str(ground_binary_dir.resolve()), flight_binary_dir=parsed_args.flight_binary_dir)
    except RecursionError:
        print("Recursion limit exceeded in compiling")
        sys.exit(1)
    if isinstance(
        result,
        (
            fpy.error.CompileError,
            fpy.error.BackendError,
        ),
    ):
        print(result, file=sys.stderr)
        sys.exit(1)

    directives, arg_types = result

    output = parsed_args.output
    if output is None:
        output = parsed_args.input.with_suffix(".bin")
    if parsed_args.bytecode:
        fpybc = directives_to_fpybc(directives)
        print(fpybc)
    else:
        arg_specs = [(name, t.name, t.max_size) for name, t in arg_types]
        output_bytes, crc = serialize_directives(directives, arg_specs)
        output.write_bytes(output_bytes)
        print(f"{output}\nCRC {hex(crc)} size {human_readable_size(len(output_bytes))}")


def model_main(args: list[str] = None):
    arg_parser = argparse.ArgumentParser(description=f"FpySequencer model for testing {get_version_str()}")
    arg_parser.add_argument(
        "--version", action="version", version=f"%(prog)s {get_version_str()}"
    )
    arg_parser.add_argument("input", type=Path, help="The input .bin file")
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether or not to print debug info during sequence execution",
    )
    arg_parser.add_argument(
        "--args",
        type=str,
        default=None,
        help="Hex-encoded sequence arguments (e.g. '0000002a' for U32 value 42)",
    )
    arg_parser.add_argument(
        "--dictionary",
        type=Path,
        default=None,
        help="Path to JSON dictionary (required when sequence has arguments)",
    )

    if args is not None:
        args = arg_parser.parse_args(args)
    else:
        args = arg_parser.parse_args()

    if not args.input.exists():
        print(f"Input file {args.input} does not exist")
        sys.exit(1)

    if args.debug:
        fpy.model.debug = True

    directives, arg_specs = deserialize_directives(args.input.read_bytes())

    # Reconstruct FpyType list from deserialized (name, size) specs
    arg_types = []
    if len(arg_specs) > 0:
        if args.dictionary is None:
            print(f"Must pass --dictionary when sequence has arguments", file=sys.stderr)
            sys.exit(1)
        type_defs = load_dictionary(str(args.dictionary))["type_defs"]
        try:
            arg_types = [t for _, t in resolve_arg_specs(arg_specs, type_defs)]
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

    seq_args = None
    if args.args is not None:
        seq_args = bytes.fromhex(args.args)

    model = FpySequencerModel()
    ret = model.run(directives, arg_types=arg_types, args=seq_args)
    if ret != DirectiveErrorCode.NO_ERROR:
        print("Sequence failed with " + str(ret))
        exit(1)


def assemble_main(args: list[str] = None):
    arg_parser = argparse.ArgumentParser(description=f"Fpy assembler {get_version_str()}")
    arg_parser.add_argument(
        "--version", action="version", version=f"%(prog)s {get_version_str()}"
    )
    arg_parser.add_argument("input", type=Path, help="The input .fpybc file")
    arg_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default=None,
        help="The output .bin path",
    )

    if args is not None:
        args = arg_parser.parse_args(args)
    else:
        args = arg_parser.parse_args()

    if not args.input.exists():
        print(f"Input file {args.input} does not exist")
        exit(1)

    body = fpybc_parse(args.input.read_text())
    directives = assemble(body)
    output = args.output
    if output is None:
        output = args.input.with_suffix(".bin")
    output_bytes, crc = serialize_directives(directives)
    output.write_bytes(output_bytes)
    print(f"{output}\nCRC {hex(crc)} size {human_readable_size(len(output_bytes))}")


def disassemble_main(args: list[str] = None):
    arg_parser = argparse.ArgumentParser(description=f"Fpy disassembler {get_version_str()}")
    arg_parser.add_argument(
        "--version", action="version", version=f"%(prog)s {get_version_str()}"
    )
    arg_parser.add_argument("input", type=Path, help="The input .bin file")
    arg_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default=None,
        help="The output .fpybc path",
    )

    if args is not None:
        args = arg_parser.parse_args(args)
    else:
        args = arg_parser.parse_args()

    if not args.input.exists():
        print(f"Input file {args.input} does not exist")
        exit(1)

    dirs, _ = deserialize_directives(args.input.read_bytes())
    fpybc = directives_to_fpybc(dirs)
    output = args.output
    if output is None:
        output = args.input.with_suffix(".fpybc")
    output.write_text(fpybc)
    print("Done")


FW_PACKET_COMMAND = 0


def build_command_packet(cmd_opcode: int, args: bytes) -> bytes:
    """Build an F Prime command packet for ZMQ transport.

    Format: size(4B) + descriptor_type(2B) + opcode(4B) + args
    The size field covers descriptor_type + opcode + args and is stripped
    by the GDS ZmqGround receiver before forwarding to the framing protocol.
    The descriptor_type is a ComCfg.Apid (U16), not a U32.
    """
    descriptor_type = struct.pack(">H", FW_PACKET_COMMAND)
    opcode = struct.pack(">I", cmd_opcode)
    payload = descriptor_type + opcode + args
    size = struct.pack(">I", len(payload))
    return size + payload


def send_command_zmq(cmd_opcode: int, args: bytes, zmq_addr: str):
    """Send a pre-serialized command to the GDS via ZMQ.

    The ZMQ message format is: b"FSW" + command_packet
    """
    import zmq

    packet = build_command_packet(cmd_opcode, args)

    context = zmq.Context()
    sock = context.socket(zmq.PUB)
    try:
        sock.connect(zmq_addr)
        # PUB/SUB requires a brief delay for connection establishment
        time.sleep(0.1)
        sock.send(b"FSW" + packet)
    finally:
        sock.close()
        context.term()


def run_main(args: list[str] = None):
    arg_parser = argparse.ArgumentParser(
        description=f"Run an Fpy sequence via the GDS {get_version_str()}"
    )
    arg_parser.add_argument(
        "--version", action="version", version=f"%(prog)s {get_version_str()}"
    )
    arg_parser.add_argument(
        "-c",
        "--command",
        type=str,
        required=True,
        help="The sequence run command name (e.g., Ref.cmdSeq.RUN_ARGS)",
    )
    arg_parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="The sequence binary path to pass to the run command",
    )
    arg_parser.add_argument(
        "-d",
        "--dictionary",
        type=Path,
        required=True,
        help="The FPrime dictionary .json file",
    )
    arg_parser.add_argument(
        "-B",
        "--ground-binary-dir",
        type=Path,
        required=False,
        default=None,
        help="Local directory to resolve .bin file paths for sequence calls",
    )
    arg_parser.add_argument(
        "--flight-binary-dir",
        type=str,
        required=False,
        default=None,
        help="Absolute path prefix for .bin files on the spacecraft",
    )
    arg_parser.add_argument(
        "--zmq-addr",
        type=str,
        default="ipc:///tmp/fprime-server-in",
        help="ZMQ address for the GDS uplink (default: ipc:///tmp/fprime-server-in)",
    )
    arg_parser.add_argument(
        "seq_args",
        nargs="*",
        help="Sequence argument values as Fpy literal expressions",
    )

    if args is not None:
        parsed_args = arg_parser.parse_args(args)
    else:
        parsed_args = arg_parser.parse_args()

    # Load dictionary and look up the command
    dictionary = load_dictionary(str(parsed_args.dictionary))
    cmd_name_dict = dictionary["cmd_name_dict"]

    cmd_name = parsed_args.command
    if cmd_name not in cmd_name_dict:
        print(f"Unknown command '{cmd_name}'", file=sys.stderr)
        sys.exit(1)

    cmd_def = cmd_name_dict[cmd_name]
    cmd_args = cmd_def.args  # list of (name, description, FpyType)

    # Validate this is a sequence-run command: (string, enum, Svc.SeqArgs)
    if (
        len(cmd_args) != 3
        or not cmd_args[0][2].is_string
        or cmd_args[1][2].kind != TypeKind.ENUM
        or cmd_args[2][2].name != "Svc.SeqArgs"
    ):
        print(
            f"Command '{cmd_name}' is not a sequence run command. "
            f"Expected signature (string, enum, Svc.SeqArgs), got: "
            f"({', '.join(a[2].name for a in cmd_args)})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get the default value for the block state enum (2nd arg)
    block_type = cmd_args[1][2]
    block_default = block_type.json_default
    assert block_default is not None, (
        f"Block state enum '{block_type.name}' has no default value in the dictionary"
    )

    # Build synthetic Fpy source
    seq_args_str = ", ".join(parsed_args.seq_args)
    if seq_args_str:
        source = f'{cmd_name}("{parsed_args.input}", {block_default}, {seq_args_str})\n'
    else:
        source = f'{cmd_name}("{parsed_args.input}", {block_default})\n'

    # Compile it
    fpy.error.file_name = "<fprime-fpy-run>"
    fpy.error.input_text = source
    fpy.error.input_lines = source.splitlines()

    try:
        body = text_to_ast(source)
    except RecursionError:
        print("Recursion limit exceeded in parsing", file=sys.stderr)
        sys.exit(1)

    ground_binary_dir = parsed_args.ground_binary_dir
    if ground_binary_dir is None:
        ground_binary_dir = Path(".")

    try:
        result = ast_to_directives(
            body,
            parsed_args.dictionary,
            ground_binary_dir=str(ground_binary_dir.resolve()),
            flight_binary_dir=parsed_args.flight_binary_dir,
        )
    except RecursionError:
        print("Recursion limit exceeded in compiling", file=sys.stderr)
        sys.exit(1)

    if isinstance(result, (fpy.error.CompileError, fpy.error.BackendError)):
        print(result, file=sys.stderr)
        sys.exit(1)

    directives, _ = result

    cmd_directives = [d for d in directives if isinstance(d, ConstCmdDirective)]
    assert len(cmd_directives) == 1, (
        f"Expected exactly 1 ConstCmdDirective, got {len(cmd_directives)}"
    )

    directive = cmd_directives[0]

    print(f"Sending {cmd_name}(\"{parsed_args.input}\", {block_default}"
          + (f", {seq_args_str}" if seq_args_str else "")
          + f") via {parsed_args.zmq_addr}")

    try:
        send_command_zmq(directive.cmd_opcode, directive.args, parsed_args.zmq_addr)
    except Exception as e:
        print(f"Failed to send command: {e}", file=sys.stderr)
        sys.exit(1)