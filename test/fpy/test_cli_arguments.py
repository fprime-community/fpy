"""Every command-line argument of every fpy command.

Each test declares the argument or arguments it exercises with @covers. The
`test_every_argument_is_covered` case reads those declarations back and fails
when a command grows an argument that no test drives, so the matrix cannot
silently fall behind the parsers.

The commands run end to end against the real dictionary, so these also cover
the reporting and exit paths each argument reaches.
"""

import argparse

import pytest

import fpy.error
from fpy import main as fpy_main
from fpy.test_helpers import default_dictionary

NO_OP = "CdhCore.cmdDisp.CMD_NO_OP()\n"


def covers(*dests: str):
    """Declare which parser destinations a test exercises."""

    def decorate(fn):
        fn.covers = frozenset(dests)
        return fn

    return decorate


class _ParserBuilt(Exception):
    def __init__(self, parser):
        self.parser = parser


def parser_of(main_fn) -> argparse.ArgumentParser:
    """The parser a command builds, captured without running the command.

    Every add_argument call happens before the command parses, so intercepting
    the parse yields the fully built parser.
    """
    original = argparse.ArgumentParser.parse_args

    def intercept(self, *args, **kwargs):
        raise _ParserBuilt(self)

    argparse.ArgumentParser.parse_args = intercept
    try:
        main_fn([])
    except _ParserBuilt as built:
        return built.parser
    finally:
        argparse.ArgumentParser.parse_args = original
    raise AssertionError(f"{main_fn.__name__} never built a parser")


@pytest.fixture
def source(tmp_path):
    """A compilable sequence file."""
    path = tmp_path / "seq.fpy"
    path.write_text(NO_OP)
    return path


@pytest.fixture(autouse=True)
def _restore_debug_flag():
    """--debug sets a module-level flag; keep it out of other tests."""
    before = fpy.error.debug
    yield
    fpy.error.debug = before


def compile_argv(source, *extra):
    return [str(source), "-d", default_dictionary, *extra]


class TestCompile:
    @covers("help")
    def test_help_lists_usage(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.compile_main(["--help"])
        assert exit_info.value.code == 0
        assert "usage:" in capsys.readouterr().out

    @covers("version")
    def test_version_reports_language_version(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.compile_main(["--version"])
        assert exit_info.value.code == 0
        assert "langauge" in capsys.readouterr().out

    @covers("input")
    def test_input_is_compiled(self, source):
        fpy_main.compile_main(compile_argv(source))
        assert source.with_suffix(".bin").exists()

    @covers("input")
    def test_missing_input_exits(self, tmp_path, capsys):
        missing = tmp_path / "absent.fpy"
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.compile_main(compile_argv(missing))
        assert exit_info.value.code == 1
        assert "does not exist" in capsys.readouterr().out

    @covers("output")
    @pytest.mark.parametrize("flag", ["-o", "--output"])
    def test_output_selects_destination(self, source, tmp_path, flag):
        destination = tmp_path / "elsewhere.bin"
        fpy_main.compile_main(compile_argv(source, flag, str(destination)))
        assert destination.exists()
        assert not source.with_suffix(".bin").exists()

    @covers("dictionary")
    @pytest.mark.parametrize("flag", ["-d", "--dictionary"])
    def test_dictionary_is_loaded(self, source, flag):
        fpy_main.compile_main([str(source), flag, default_dictionary])
        assert source.with_suffix(".bin").exists()

    @covers("dictionary")
    def test_dictionary_is_required(self, source):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.compile_main([str(source)])
        assert exit_info.value.code == 2

    @covers("dictionary")
    def test_incompatible_dictionary_exits(self, source, tmp_path, capsys):
        incompatible = tmp_path / "incompatible.json"
        incompatible.write_text("{}")
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.compile_main([str(source), "-d", str(incompatible)])
        assert exit_info.value.code == 1
        assert "incompatible with this version" in capsys.readouterr().err

    @covers("dictionary")
    @pytest.mark.xfail(
        strict=True,
        reason="a dictionary that is not valid JSON raises JSONDecodeError out of "
        "load_dictionary, so the command dies with a traceback instead of the "
        "DictionaryError message its handler is written for",
    )
    def test_malformed_dictionary_exits(self, source, tmp_path, capsys):
        malformed = tmp_path / "malformed.json"
        malformed.write_text("{ not json")
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.compile_main([str(source), "-d", str(malformed)])
        assert exit_info.value.code == 1

    @covers("emit")
    @pytest.mark.parametrize(
        "emit, suffix",
        [
            ("fpybin", ".bin"),
            ("fpyasm", ".fpyasm"),
            ("llvm-ir", ".ll"),
            ("wasm", ".wasm"),
            ("wat", ".wat"),
        ],
    )
    def test_emit_selects_output_format(self, source, emit, suffix):
        fpy_main.compile_main(compile_argv(source, "--emit", emit))
        assert source.with_suffix(suffix).exists()

    @covers("emit")
    def test_unknown_emit_is_rejected(self, source):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.compile_main(compile_argv(source, "--emit", "nonsense"))
        assert exit_info.value.code == 2

    @covers("debug")
    def test_debug_sets_the_debug_flag(self, source):
        fpy.error.debug = False
        fpy_main.compile_main(compile_argv(source, "--debug"))
        assert fpy.error.debug is True

    @covers("ground_binary_dir")
    @pytest.mark.parametrize("flag", ["-g", "--ground-binary-dir"])
    def test_ground_binary_dir_is_accepted(self, source, tmp_path, flag):
        binaries = tmp_path / "binaries"
        binaries.mkdir()
        fpy_main.compile_main(compile_argv(source, flag, str(binaries)))
        assert source.with_suffix(".bin").exists()

    @covers("imports")
    @pytest.mark.parametrize("flag", ["-i", "--imports"])
    def test_imports_directory_resolves_an_absolute_import(
        self, tmp_path, source, flag
    ):
        library = tmp_path / "lib"
        library.mkdir()
        (library / "helper.fpy").write_text("def noop():\n    return\n")
        source.write_text("import helper\nhelper.noop()\n")
        fpy_main.compile_main(compile_argv(source, flag, str(library)))
        assert source.with_suffix(".bin").exists()

    @covers("ignore")
    def test_ignore_silences_a_warning(self, source, capsys):
        source.write_text("for i in 5 .. 0:\n    " + NO_OP)
        fpy_main.compile_main(compile_argv(source, "--ignore", "empty-range"))
        assert "Range is empty" not in capsys.readouterr().out

    @covers("ignore")
    def test_unknown_ignored_warning_exits(self, source, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.compile_main(compile_argv(source, "--ignore", "not-a-warning"))
        assert exit_info.value.code == 1
        assert capsys.readouterr().err

    @covers("error")
    def test_error_promotes_a_warning(self, source, capsys):
        source.write_text("for i in 5 .. 0:\n    " + NO_OP)
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.compile_main(compile_argv(source, "--error", "empty-range"))
        assert exit_info.value.code == 1
        assert "Range is empty" in capsys.readouterr().err

    @covers("error", "ignore")
    def test_warning_cannot_be_both_ignored_and_promoted(self, source, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.compile_main(
                compile_argv(
                    source, "--ignore", "empty-range", "--error", "empty-range"
                )
            )
        assert exit_info.value.code == 1
        assert "cannot be both" in capsys.readouterr().err


@pytest.fixture
def fpybc_source(tmp_path):
    """An assemblable fpy bytecode file."""
    path = tmp_path / "prog.fpybc"
    path.write_text("no_op\nexit\n")
    return path


class TestAssemble:
    @covers("help")
    def test_help_lists_usage(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.assemble_main(["--help"])
        assert exit_info.value.code == 0
        assert "usage:" in capsys.readouterr().out

    @covers("version")
    def test_version_reports_language_version(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.assemble_main(["--version"])
        assert exit_info.value.code == 0
        assert "langauge" in capsys.readouterr().out

    @covers("input")
    def test_input_is_assembled(self, fpybc_source):
        fpy_main.assemble_main([str(fpybc_source)])
        assert fpybc_source.with_suffix(".bin").exists()

    @covers("input")
    def test_missing_input_exits(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.assemble_main([str(tmp_path / "absent.fpybc")])
        assert exit_info.value.code == 1
        assert "does not exist" in capsys.readouterr().out

    @covers("output")
    @pytest.mark.parametrize("flag", ["-o", "--output"])
    def test_output_selects_destination(self, fpybc_source, tmp_path, flag):
        destination = tmp_path / "elsewhere.bin"
        fpy_main.assemble_main([str(fpybc_source), flag, str(destination)])
        assert destination.exists()


@pytest.fixture
def binary_source(tmp_path, fpybc_source):
    """An assembled .bin file, the disassembler's input."""
    fpy_main.assemble_main([str(fpybc_source)])
    return fpybc_source.with_suffix(".bin")


class TestDisassemble:
    @covers("help")
    def test_help_lists_usage(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.disassemble_main(["--help"])
        assert exit_info.value.code == 0
        assert "usage:" in capsys.readouterr().out

    @covers("version")
    def test_version_reports_language_version(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.disassemble_main(["--version"])
        assert exit_info.value.code == 0
        assert "langauge" in capsys.readouterr().out

    @covers("input")
    def test_input_is_disassembled(self, binary_source):
        fpy_main.disassemble_main([str(binary_source)])
        assert binary_source.with_suffix(".fpybc").read_text().strip()

    @covers("input")
    def test_missing_input_exits(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.disassemble_main([str(tmp_path / "absent.bin")])
        assert exit_info.value.code == 1
        assert "does not exist" in capsys.readouterr().out

    @covers("output")
    @pytest.mark.parametrize("flag", ["-o", "--output"])
    def test_output_selects_destination(self, binary_source, tmp_path, flag):
        destination = tmp_path / "elsewhere.fpybc"
        fpy_main.disassemble_main([str(binary_source), flag, str(destination)])
        assert destination.read_text().strip()


class TestCmd:
    @covers("help")
    def test_help_lists_usage(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.cmd_main(["--help"])
        assert exit_info.value.code == 0
        assert "usage:" in capsys.readouterr().out

    @covers("version")
    def test_version_reports_language_version(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.cmd_main(["--version"])
        assert exit_info.value.code == 0
        assert "langauge" in capsys.readouterr().out

    @covers("source", "dictionary", "zmq_addr")
    def test_source_is_compiled_and_sent(self, monkeypatch, capsys):
        sent = {}
        monkeypatch.setattr(
            fpy_main,
            "send_command_zmq",
            lambda opcode, args, addr: sent.update(opcode=opcode, addr=addr),
        )
        fpy_main.cmd_main([NO_OP.strip(), "-d", default_dictionary])
        assert "opcode" in sent
        assert "Sending" in capsys.readouterr().out

    @covers("source")
    def test_source_that_does_not_compile_exits(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.cmd_main(["this is not fpy(", "-d", default_dictionary])
        assert exit_info.value.code == 1
        assert capsys.readouterr().err

    @covers("source")
    def test_source_without_a_command_exits(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.cmd_main(["x: U32 = 1", "-d", default_dictionary])
        assert exit_info.value.code == 1
        assert "Expected 1 command" in capsys.readouterr().err

    @covers("dictionary")
    def test_dictionary_is_required(self):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.cmd_main([NO_OP.strip()])
        assert exit_info.value.code == 2

    @covers("ground_binary_dir")
    @pytest.mark.parametrize("flag", ["-g", "--ground-binary-dir"])
    def test_ground_binary_dir_is_accepted(self, monkeypatch, tmp_path, flag):
        monkeypatch.setattr(fpy_main, "send_command_zmq", lambda *a: None)
        fpy_main.cmd_main(
            [NO_OP.strip(), "-d", default_dictionary, flag, str(tmp_path)]
        )

    @covers("zmq_addr")
    def test_zmq_addr_is_passed_through(self, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(
            fpy_main,
            "send_command_zmq",
            lambda opcode, args, addr: seen.update(addr=addr),
        )
        fpy_main.cmd_main(
            [NO_OP.strip(), "-d", default_dictionary, "--zmq-addr", "tcp://127.0.0.1:1"]
        )
        assert seen["addr"] == "tcp://127.0.0.1:1"

    @covers("zmq_addr")
    def test_zmq_send_failure_exits(self, monkeypatch, capsys):
        def explode(*args):
            raise RuntimeError("no uplink")

        monkeypatch.setattr(fpy_main, "send_command_zmq", explode)
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.cmd_main([NO_OP.strip(), "-d", default_dictionary])
        assert exit_info.value.code == 1
        assert "Failed to send command" in capsys.readouterr().err

    @covers("tcp_addr")
    def test_tcp_addr_selects_tcp(self, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(
            fpy_main,
            "send_command_tcp",
            lambda opcode, args, host, port: seen.update(host=host, port=port),
        )
        fpy_main.cmd_main(
            [
                NO_OP.strip(),
                "-d",
                default_dictionary,
                "--tcp-addr",
                "127.0.0.1:50050",
            ]
        )
        assert seen == {"host": "127.0.0.1", "port": 50050}
        assert "via TCP" in capsys.readouterr().out

    @covers("tcp_addr")
    def test_tcp_addr_without_port_exits(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.cmd_main(
                [NO_OP.strip(), "-d", default_dictionary, "--tcp-addr", "localhost"]
            )
        assert exit_info.value.code == 1
        assert "expected host:port" in capsys.readouterr().err

    @covers("tcp_addr")
    def test_tcp_addr_with_bad_port_exits(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.cmd_main(
                [NO_OP.strip(), "-d", default_dictionary, "--tcp-addr", "host:nope"]
            )
        assert exit_info.value.code == 1
        assert "Invalid port" in capsys.readouterr().err

    @covers("dictionary")
    def test_incompatible_dictionary_exits(self, tmp_path, capsys):
        incompatible = tmp_path / "incompatible.json"
        incompatible.write_text("{}")
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.cmd_main([NO_OP.strip(), "-d", str(incompatible)])
        assert exit_info.value.code == 1
        assert "incompatible with this version" in capsys.readouterr().err

    @covers("zmq_addr")
    def test_zmq_addr_reaches_a_real_socket(self, capsys):
        """The default transport, exercised through zmq itself rather than a
        stand-in, so the packet is really built and published."""
        fpy_main.cmd_main(
            [
                NO_OP.strip(),
                "-d",
                default_dictionary,
                "--zmq-addr",
                "tcp://127.0.0.1:56789",
            ]
        )
        assert "Sending" in capsys.readouterr().out

    @covers("tcp_addr")
    def test_tcp_addr_reaches_a_real_socket(self, capsys):
        """The TCP transport against a real listener, which receives the
        registration line and the framed command packet."""
        import socket
        import threading

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        received = []

        def accept_one():
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(5)
                # The registration line and the packet are sent separately, so
                # read until the sender closes.
                chunks = []
                while True:
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                received.append(b"".join(chunks))

        server = threading.Thread(target=accept_one)
        server.start()
        try:
            fpy_main.cmd_main(
                [
                    NO_OP.strip(),
                    "-d",
                    default_dictionary,
                    "--tcp-addr",
                    f"127.0.0.1:{port}",
                ]
            )
        finally:
            server.join(timeout=5)
            listener.close()

        assert received and received[0].startswith(b"Register GUI\n")
        assert b"A5A5 FSW ZZZZ" in received[0]

    @covers("tcp_addr")
    def test_tcp_send_failure_exits(self, monkeypatch, capsys):
        def explode(*args):
            raise OSError("connection refused")

        monkeypatch.setattr(fpy_main, "send_command_tcp", explode)
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.cmd_main(
                [NO_OP.strip(), "-d", default_dictionary, "--tcp-addr", "127.0.0.1:1"]
            )
        assert exit_info.value.code == 1
        assert "Failed to send command" in capsys.readouterr().err


class TestDepend:
    @covers("help")
    def test_help_lists_usage(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.depend_main(["--help"])
        assert exit_info.value.code == 0
        assert "usage:" in capsys.readouterr().out

    @covers("version")
    def test_version_reports_language_version(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.depend_main(["--version"])
        assert exit_info.value.code == 0
        assert "langauge" in capsys.readouterr().out

    @covers("input", "dictionary")
    def test_input_with_no_dependencies_prints_nothing(self, source, capsys):
        fpy_main.depend_main([str(source), "-d", default_dictionary])
        assert capsys.readouterr().out == ""

    @covers("input")
    def test_missing_input_exits(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.depend_main(
                [str(tmp_path / "absent.fpy"), "-d", default_dictionary]
            )
        assert exit_info.value.code == 1
        assert "does not exist" in capsys.readouterr().err

    @covers("input")
    def test_input_that_does_not_parse_exits(self, tmp_path, capsys):
        bad = tmp_path / "bad.fpy"
        bad.write_text("this is not fpy(\n")
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.depend_main([str(bad), "-d", default_dictionary])
        assert exit_info.value.code == 1
        assert capsys.readouterr().err

    @covers("dictionary")
    def test_dictionary_is_required(self, source):
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.depend_main([str(source)])
        assert exit_info.value.code == 2

    @covers("dictionary")
    def test_incompatible_dictionary_exits(self, source, tmp_path, capsys):
        incompatible = tmp_path / "incompatible.json"
        incompatible.write_text("{}")
        with pytest.raises(SystemExit) as exit_info:
            fpy_main.depend_main([str(source), "-d", str(incompatible)])
        assert exit_info.value.code == 1
        assert "incompatible with this version" in capsys.readouterr().err

    @covers("input", "ground_binary_dir")
    def test_sequence_dependency_is_reported(self, source, tmp_path, capsys):
        binaries = tmp_path / "binaries"
        binaries.mkdir()
        source.write_text('Ref.seqDisp.RUN_ARGS("child.bin", Svc.BlockState.BLOCK)\n')
        fpy_main.depend_main(
            [str(source), "-d", default_dictionary, "-g", str(binaries)]
        )
        assert "child.bin" in capsys.readouterr().out

    @covers("ground_binary_dir")
    @pytest.mark.parametrize("flag", ["-g", "--ground-binary-dir"])
    def test_ground_binary_dir_is_accepted(self, source, tmp_path, flag):
        binaries = tmp_path / "binaries"
        binaries.mkdir()
        fpy_main.depend_main(
            [str(source), "-d", default_dictionary, flag, str(binaries)]
        )

    @covers("imports")
    @pytest.mark.parametrize("flag", ["-i", "--imports"])
    def test_imports_directory_resolves_an_absolute_import(
        self, tmp_path, source, flag
    ):
        library = tmp_path / "lib"
        library.mkdir()
        (library / "helper.fpy").write_text("def noop():\n    return\n")
        source.write_text("import helper\nhelper.noop()\n")
        fpy_main.depend_main(
            [str(source), "-d", default_dictionary, flag, str(library)]
        )


COMMANDS = {
    "compile": (fpy_main.compile_main, TestCompile),
    "assemble": (fpy_main.assemble_main, TestAssemble),
    "disassemble": (fpy_main.disassemble_main, TestDisassemble),
    "cmd": (fpy_main.cmd_main, TestCmd),
    "depend": (fpy_main.depend_main, TestDepend),
}


def declared_coverage(test_class) -> set[str]:
    """Every destination the class's tests declare with @covers."""
    declared = set()
    for name in dir(test_class):
        declared |= getattr(getattr(test_class, name), "covers", frozenset())
    return declared


@pytest.mark.parametrize("command", sorted(COMMANDS), ids=str)
def test_every_argument_is_covered(command):
    """A command's every argument is exercised by some test above."""
    main_fn, test_class = COMMANDS[command]
    arguments = {action.dest for action in parser_of(main_fn)._actions}
    untested = arguments - declared_coverage(test_class)
    assert not untested, (
        f"{command} has untested argument(s): {sorted(untested)}. "
        f"Add a test to {test_class.__name__} and mark it @covers(...)."
    )


def test_covers_declarations_name_real_arguments():
    """No @covers names an argument its command does not have."""
    for command, (main_fn, test_class) in sorted(COMMANDS.items()):
        arguments = {action.dest for action in parser_of(main_fn)._actions}
        unknown = declared_coverage(test_class) - arguments
        assert not unknown, f"{command} has no argument(s) {sorted(unknown)}"
