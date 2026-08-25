from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import socket
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
    fpybc_directives_to_fpyasm,
    parse as fpybc_parse,
    serialize_directives,
)
from fpy.bytecode.directives import ConstCmdDirective, StackCmdDirective
import fpy.error
from fpy.compiler import (
    analysis_to_llvm_module,
    analysis_to_wasm,
    analysis_to_wat,
    analyze_ast,
    text_to_ast,
    analysis_to_fpybc_directives,
)
from fpy.state import get_base_compile_state
from fpy.error import parse_warning_set


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
    return f"package {get_package_version()}, langauge {MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}, schema {SCHEMA_VERSION}"


def get_backend_version_str() -> str | None:
    """The LLVM backend's toolchain versions, or None if it isn't installed."""
    try:
        from fpy.codegen_llvm import backend_version_str
    except fpy.error.BackendError:
        return None
    return backend_version_str()


def parse_seq_maps(specs: list[str]) -> list[tuple[str, str]]:
    """Parse repeated --seq-map values ('BIN_PREFIX=FPY_PREFIX') into ordered
    (bin_prefix, fpy_prefix) pairs. Raises ValueError on a value with no '='."""
    maps = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --seq-map value {spec!r} (expected BIN_PREFIX=FPY_PREFIX)"
            )
        bin_prefix, fpy_prefix = spec.split("=", 1)
        maps.append((bin_prefix, fpy_prefix))
    return maps


def _add_seq_map_argument(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "--seq-map",
        action="append",
        default=[],
        metavar="BIN_PREFIX=FPY_PREFIX",
        dest="seq_map",
        help=(
            "Map a called sequence's onboard binary path to its .fpy source "
            "(repeatable): a path starting with BIN_PREFIX has that prefix "
            "replaced with FPY_PREFIX and its extension replaced with .fpy; "
            "the first mapping that yields an existing file wins. An empty "
            "BIN_PREFIX matches every path."
        ),
    )


def compile_main(args: list[str] = None):
    backend_version = get_backend_version_str()
    compiler_version = get_version_str()
    if backend_version is not None:
        compiler_version += f"\nbackend:\n{backend_version}"
    arg_parser = argparse.ArgumentParser(description=f"Fpy compiler {compiler_version}")
    arg_parser.add_argument(
        "--version", action="version", version=f"%(prog)s {compiler_version}"
    )
    arg_parser.add_argument("input", type=Path, help="The input .fpy file")
    arg_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default=None,
        help="The output path",
    )
    arg_parser.add_argument(
        "-d",
        "--dictionary",
        type=Path,
        required=True,
        help="The FPrime dictionary .json file",
    )
    arg_parser.add_argument(
        "--emit",
        choices=["fpybin", "fpyasm", "llvm-ir", "wasm", "wat"],
        default="fpybin",
        help=(
            "Codegen backend / output format: 'fpybin' (binary fpy bytecode, the "
            "default), 'fpyasm' (human-readable fpy bytecode assembly), "
            "'llvm-ir' (LLVM IR), 'wasm' (WebAssembly binary), 'wat' "
            "(WebAssembly text)"
        ),
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Pass this to print out compiler debugging information",
    )
    _add_seq_map_argument(arg_parser)
    arg_parser.add_argument(
        "-i",
        "--imports",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        dest="imports",
        help=(
            "Directory to search when resolving absolute `import` statements "
            "(repeatable). An import that resolves in more than one --imports "
            "directory is ambiguous and an error."
        ),
    )
    arg_parser.add_argument(
        "--ignore",
        type=str,
        default="",
        help="Comma-separated warning types to silence (e.g. 'import-underscore,empty-range'), or 'all'",
    )
    arg_parser.add_argument(
        "--error",
        type=str,
        default="",
        help="Comma-separated warning types to promote to hard errors, or 'all'",
    )
    if args is not None:
        parsed_args = arg_parser.parse_args(args)
    else:
        parsed_args = arg_parser.parse_args()

    if parsed_args.debug:
        fpy.error.debug = True

    try:
        ignored_warnings = parse_warning_set(parsed_args.ignore)
        error_warnings = parse_warning_set(parsed_args.error)
        seq_maps = parse_seq_maps(parsed_args.seq_map)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    overlap = ignored_warnings & error_warnings
    if overlap:
        names = ", ".join(sorted(w.value for w in overlap))
        print(
            f"Warning type(s) cannot be both ignored and promoted to errors: {names}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not parsed_args.input.exists():
        print(f"Input file {parsed_args.input} does not exist")
        sys.exit(1)
    fpy.error.file_name = str(parsed_args.input)

    # The import directories are the -i/--imports directories, in order; the
    # input file's own directory anchors its relative imports but is not
    # among them. Exact duplicate directories are dropped: a repeated -i
    # flag carries no information.
    import_directories = list(
        dict.fromkeys(str(d.resolve()) for d in parsed_args.imports)
    )

    # reading dictionary
    try:
        state = get_base_compile_state(
            str(parsed_args.dictionary.resolve()),
            seq_maps,
            ignored_warnings=ignored_warnings,
            error_warnings=error_warnings,
            import_directories=import_directories,
            main_file_dir=str(parsed_args.input.parent.resolve()),
            main_file_path=str(parsed_args.input.resolve()),
        )
    except fpy.error.DictionaryError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    # syntax
    try:
        body = text_to_ast(parsed_args.input.read_text(encoding="utf-8"))
    except RecursionError:
        print("Recursion limit exceeded in parsing", file=sys.stderr)
        sys.exit(1)
    except fpy.error.CompileError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    # semantics
    try:
        state = analyze_ast(body, state)
    except RecursionError:
        print("Recursion limit exceeded in semantics passes", file=sys.stderr)
        sys.exit(1)
    except fpy.error.CompileError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    # codegen
    try:
        if parsed_args.emit == "llvm-ir":
            output, seq_arg_types = analysis_to_llvm_module(state)
        elif parsed_args.emit == "wasm":
            output, seq_arg_types = analysis_to_wasm(state)
        elif parsed_args.emit == "wat":
            output, seq_arg_types = analysis_to_wat(state)
        elif parsed_args.emit in ["fpybin", "fpyasm"]:
            output, seq_arg_types = analysis_to_fpybc_directives(state)
        else:
            assert False, parsed_args.emit
    except fpy.error.BackendError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    output_path = parsed_args.output

    if parsed_args.emit == "fpyasm":
        if output_path is None:
            output_path = parsed_args.input.with_suffix(".fpyasm")
        fpyasm = fpybc_directives_to_fpyasm(output)
        output_path.write_text(fpyasm)
    elif parsed_args.emit == "llvm-ir":
        if output_path is None:
            output_path = parsed_args.input.with_suffix(".ll")
        # output is an llvmlite ir.Module; str() yields the textual LLVM IR.
        output_path.write_text(str(output))
        print(f"{output_path}")
    elif parsed_args.emit == "wasm":
        if output_path is None:
            output_path = parsed_args.input.with_suffix(".wasm")
        # output is the runnable wasm binary.
        output_path.write_bytes(output)
        print(f"{output_path}\nsize {human_readable_size(len(output))}")
    elif parsed_args.emit == "wat":
        if output_path is None:
            output_path = parsed_args.input.with_suffix(".wat")
        # output is the WebAssembly text (LLVM textual assembly).
        output_path.write_text(output)
        print(f"{output_path}")
    elif parsed_args.emit == "fpybin":
        output_path = parsed_args.output
        if output_path is None:
            output_path = parsed_args.input.with_suffix(".bin")
        arg_specs = [(name, t.name, t.max_size) for name, t in seq_arg_types]
        try:
            output_bytes, crc = serialize_directives(
                output, arg_specs, max_directive_size=state.max_directive_size
            )
        except fpy.error.BackendError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        output_path.write_bytes(output_bytes)
        print(
            f"{output_path}\nCRC {hex(crc)} size {human_readable_size(len(output_bytes))}"
        )
    else:
        assert False, parsed_args.emit


def assemble_main(args: list[str] = None):
    arg_parser = argparse.ArgumentParser(
        description=f"Fpy assembler {get_version_str()}"
    )
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

    body = fpybc_parse(args.input.read_text(encoding="utf-8"))
    directives = assemble(body)
    output = args.output
    if output is None:
        output = args.input.with_suffix(".bin")
    try:
        output_bytes, crc = serialize_directives(directives)
    except fpy.error.BackendError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    output.write_bytes(output_bytes)
    print(f"{output}\nCRC {hex(crc)} size {human_readable_size(len(output_bytes))}")


def disassemble_main(args: list[str] = None):
    arg_parser = argparse.ArgumentParser(
        description=f"Fpy disassembler {get_version_str()}"
    )
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
    fpybc = fpybc_directives_to_fpyasm(dirs)
    output = args.output
    if output is None:
        output = args.input.with_suffix(".fpybc")
    output.write_text(fpybc, encoding="utf-8")
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


def send_command_tcp(cmd_opcode: int, args: bytes, tcp_addr: str, tcp_port: int):
    """Send a pre-serialized command to the GDS via the TCP server.

    Protocol:
      1. Connect and register as a GUI client: b"Register GUI\\n"
      2. Send command: b"A5A5 FSW " + b"ZZZZ" + size(4B) + payload
         where the ZZZZ frame is what TcpServerFramerDeframer expects.
    """
    packet = build_command_packet(cmd_opcode, args)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((tcp_addr, tcp_port))
        sock.sendall(b"Register GUI\n")
        sock.sendall(b"A5A5 FSW " + b"ZZZZ" + packet)
        # Brief delay to let the server process before we close
        time.sleep(0.1)
    finally:
        sock.close()


def cmd_main(args: list[str] = None):
    arg_parser = argparse.ArgumentParser(
        description=f"Run an Fpy command via the GDS {get_version_str()}",
        epilog="Example: %(prog)s 'Ref.seqDisp.RUN_ARGS(\"seq.bin\", NO_WAIT)' -d dict.json",
    )
    arg_parser.add_argument(
        "--version", action="version", version=f"%(prog)s {get_version_str()}"
    )
    arg_parser.add_argument(
        "source",
        type=str,
        help="A single line of valid Fpy source, containing a command with constant arguments (e.g., 'Ref.seqDisp.RUN_ARGS(\"seq.bin\", NO_WAIT)')",
    )
    arg_parser.add_argument(
        "-d",
        "--dictionary",
        type=Path,
        required=True,
        help="The FPrime dictionary .json file",
    )
    _add_seq_map_argument(arg_parser)
    arg_parser.add_argument(
        "--zmq-addr",
        type=str,
        default="ipc:///tmp/fprime-server-in",
        help="ZMQ address for the GDS uplink (default: ipc:///tmp/fprime-server-in)",
    )
    arg_parser.add_argument(
        "--tcp-addr",
        type=str,
        default=None,
        help="TCP server address as host:port (e.g. 127.0.0.1:50050). If provided, use TCP instead of ZMQ.",
    )

    if args is not None:
        parsed_args = arg_parser.parse_args(args)
    else:
        parsed_args = arg_parser.parse_args()

    try:
        seq_maps = parse_seq_maps(parsed_args.seq_map)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    source = parsed_args.source
    if not source.endswith("\n"):
        source += "\n"

    # Compile it
    fpy.error.file_name = "<fprime-fpy-cmd>"
    fpy.error.input_text = source
    fpy.error.input_lines = source.splitlines()

    try:
        body = text_to_ast(source)
    except RecursionError:
        print("Recursion limit exceeded in parsing", file=sys.stderr)
        sys.exit(1)
    except fpy.error.CompileError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    try:
        state = get_base_compile_state(
            str(parsed_args.dictionary.resolve()),
            seq_maps,
        )
    except fpy.error.DictionaryError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    try:
        state = analyze_ast(body, state)
        directives, _ = analysis_to_fpybc_directives(state)
    except RecursionError:
        print("Recursion limit exceeded in compiling", file=sys.stderr)
        sys.exit(1)
    except (fpy.error.CompileError, fpy.error.BackendError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    stack_cmds = [d for d in directives if isinstance(d, StackCmdDirective)]
    if stack_cmds:
        print(
            "Command arguments must be constant expressions",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd_directives = [d for d in directives if isinstance(d, ConstCmdDirective)]
    if len(cmd_directives) != 1:
        print(
            f"Expected 1 command with constant arguments, "
            f"but got {len(cmd_directives)}",
            file=sys.stderr,
        )
        sys.exit(1)

    directive = cmd_directives[0]

    if parsed_args.tcp_addr is not None:
        parts = parsed_args.tcp_addr.rsplit(":", 1)
        if len(parts) != 2:
            print(
                f"Invalid --tcp-addr format: {parsed_args.tcp_addr!r} (expected host:port)",
                file=sys.stderr,
            )
            sys.exit(1)
        tcp_host = parts[0]
        try:
            tcp_port = int(parts[1])
        except ValueError:
            print(
                f"Invalid port in --tcp-addr: {parts[1]!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Sending {source.strip()} via TCP {parsed_args.tcp_addr}")
        try:
            send_command_tcp(directive.cmd_opcode, directive.args, tcp_host, tcp_port)
        except Exception as e:
            print(f"Failed to send command: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Sending {source.strip()} via {parsed_args.zmq_addr}")
        try:
            send_command_zmq(directive.cmd_opcode, directive.args, parsed_args.zmq_addr)
        except Exception as e:
            print(f"Failed to send command: {e}", file=sys.stderr)
            sys.exit(1)
