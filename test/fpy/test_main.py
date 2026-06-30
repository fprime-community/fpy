import pytest

from fpy import main as fpy_main
from fpy.bytecode.directives import ConstCmdDirective
import fpy.error as fpy_error
import fpy.model as fpy_model


@pytest.mark.parametrize(
    "size,expected",
    [
        (0, "0 B"),
        (512, "512 B"),
        (1024, "1 KB"),
        (1536, "1 KB"),
        (5 * 1024 * 1024, "5 MB"),
    ],
)
def test_human_readable_size(size, expected):
    assert fpy_main.human_readable_size(size) == expected


def test_compile_main_ground_binary_dir(monkeypatch, tmp_path, capsys):
    """--ground-binary-dir is resolved and passed to get_base_compile_state."""
    input_path = tmp_path / "seq.fpy"
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")
    bin_dir = tmp_path / "binaries"
    bin_dir.mkdir()

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")

    captured_kwargs = {}

    def fake_get_base_compile_state(dictionary, ground_binary_dir=None):
        captured_kwargs["ground_binary_dir"] = ground_binary_dir
        return "STATE"

    monkeypatch.setattr(fpy_main, "get_base_compile_state", fake_get_base_compile_state)
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)
    monkeypatch.setattr(
        fpy_main, "analysis_to_fpybc_directives", lambda body, state: (["directive"], [])
    )
    monkeypatch.setattr(
        fpy_main, "serialize_directives", lambda directives, arg_specs: (b"\x01", 0x1)
    )

    fpy_main.compile_main(
        [
            str(input_path),
            "--dictionary",
            str(dict_path),
            "--ground-binary-dir",
            str(bin_dir),
        ]
    )

    assert captured_kwargs["ground_binary_dir"] == str(bin_dir.resolve())


def test_compile_main_ground_binary_dir_defaults_to_input_parent(monkeypatch, tmp_path, capsys):
    """When --ground-binary-dir is not passed, it defaults to the input file's parent."""
    input_path = tmp_path / "seq.fpy"
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")

    captured_kwargs = {}

    def fake_get_base_compile_state(dictionary, ground_binary_dir=None):
        captured_kwargs["ground_binary_dir"] = ground_binary_dir
        return "STATE"

    monkeypatch.setattr(fpy_main, "get_base_compile_state", fake_get_base_compile_state)
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)
    monkeypatch.setattr(
        fpy_main, "analysis_to_fpybc_directives", lambda body, state: (["directive"], [])
    )
    monkeypatch.setattr(
        fpy_main, "serialize_directives", lambda directives, arg_specs: (b"\x01", 0x1)
    )

    fpy_main.compile_main(
        [
            str(input_path),
            "--dictionary",
            str(dict_path),
        ]
    )

    assert captured_kwargs["ground_binary_dir"] == str(input_path.parent.resolve())


def test_compile_main_missing_input(tmp_path, capsys):
    missing = tmp_path / "missing.fpy"
    dict_path = tmp_path / "dict.json"
    with pytest.raises(SystemExit) as exc:
        fpy_main.compile_main(
            [
                str(missing),
                "--dictionary",
                str(dict_path),
            ]
        )
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.out


def test_compile_main_fpyasm_output(monkeypatch, tmp_path, capsys):
    """--emit fpyasm writes the assembly text and does not serialize a binary."""
    input_path = tmp_path / "seq.fpy"
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")

    monkeypatch.setattr(fpy_error, "debug", False, raising=False)
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")
    monkeypatch.setattr(
        fpy_main, "get_base_compile_state", lambda dictionary, ground_binary_dir=None: "STATE"
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)

    def fake_analysis_to_fpybc_directives(body, state):
        assert body == "AST"
        assert state == "STATE"
        return ["directive"], []

    monkeypatch.setattr(
        fpy_main, "analysis_to_fpybc_directives", fake_analysis_to_fpybc_directives
    )
    monkeypatch.setattr(fpy_main, "fpybc_directives_to_fpyasm", lambda directives: "FPYASM")

    def fail_serialize(*args):
        raise AssertionError("serialize_directives should not be called")

    monkeypatch.setattr(fpy_main, "serialize_directives", fail_serialize)

    fpy_main.compile_main(
        [
            str(input_path),
            "--dictionary",
            str(dict_path),
            "--emit",
            "fpyasm",
            "--debug",
        ]
    )

    output_path = input_path.with_suffix(".fpyasm")
    assert output_path.read_text() == "FPYASM"
    assert fpy_error.debug is True


def test_compile_main_wat_output(monkeypatch, tmp_path, capsys):
    """--emit wat writes the WebAssembly text to a .wat file."""
    input_path = tmp_path / "seq.fpy"
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")
    monkeypatch.setattr(
        fpy_main, "get_base_compile_state", lambda dictionary, ground_binary_dir=None: "STATE"
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)

    def fake_analysis_to_wat(body, state):
        assert body == "AST"
        assert state == "STATE"
        return "WAT_TEXT", []

    monkeypatch.setattr(fpy_main, "analysis_to_wat", fake_analysis_to_wat)

    def fail_serialize(*args):
        raise AssertionError("serialize_directives should not be called")

    monkeypatch.setattr(fpy_main, "serialize_directives", fail_serialize)

    fpy_main.compile_main(
        [str(input_path), "--dictionary", str(dict_path), "--emit", "wat"]
    )

    output_path = input_path.with_suffix(".wat")
    assert output_path.read_text() == "WAT_TEXT"
    assert str(output_path) in capsys.readouterr().out


def test_compile_main_binary_output(monkeypatch, tmp_path, capsys):
    input_path = tmp_path / "seq.fpy"
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")
    monkeypatch.setattr(
        fpy_main, "get_base_compile_state", lambda dictionary, ground_binary_dir=None: "STATE"
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)
    monkeypatch.setattr(
        fpy_main,
        "analysis_to_fpybc_directives",
        lambda body, state: (["directive"], []),
    )
    monkeypatch.setattr(
        fpy_main,
        "serialize_directives",
        lambda directives, arg_specs: (b"\x01\x02", 0xABCD),
    )

    fpy_main.compile_main(
        [
            str(input_path),
            "--dictionary",
            str(dict_path),
        ]
    )

    output_path = input_path.with_suffix(".bin")
    assert output_path.read_bytes() == b"\x01\x02"
    captured = capsys.readouterr()
    assert "CRC 0xabcd" in captured.out
    assert "2 B" in captured.out


def test_model_main_success(monkeypatch, tmp_path):
    binary = tmp_path / "seq.bin"
    binary.write_bytes(b"data")

    monkeypatch.setattr(fpy_model, "debug", False, raising=False)
    monkeypatch.setattr(fpy_main, "deserialize_directives", lambda data: (["dir"], []))

    instances = []

    class DummyModel:
        def __init__(self):
            instances.append(self)
            self.ran_with = None

        def run(self, directives, tlm_db=None, args=None, arg_types=None):
            self.ran_with = directives
            return 0, fpy_main.DirectiveErrorCode.NO_ERROR

    monkeypatch.setattr(fpy_main, "FpySequencerModel", DummyModel)

    fpy_main.model_main([str(binary), "--debug"])

    assert fpy_model.debug is True
    assert instances[0].ran_with == ["dir"]


def test_model_main_failure(monkeypatch, tmp_path, capsys):
    binary = tmp_path / "seq.bin"
    binary.write_bytes(b"data")

    monkeypatch.setattr(fpy_main, "deserialize_directives", lambda data: (["dir"], []))

    class DummyModel:
        def run(self, directives, tlm_db=None, args=None, arg_types=None):
            return 0, fpy_main.DirectiveErrorCode.STACK_OVERFLOW

    monkeypatch.setattr(fpy_main, "FpySequencerModel", DummyModel)

    with pytest.raises(SystemExit) as exc:
        fpy_main.model_main([str(binary)])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Sequence trapped" in captured.out


def test_assemble_main_missing_input(tmp_path, capsys):
    source = tmp_path / "seq.fpybc"
    with pytest.raises(SystemExit) as exc:
        fpy_main.assemble_main([str(source)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.out


def test_assemble_main_writes_binary(monkeypatch, tmp_path, capsys):
    source = tmp_path / "seq.fpybc"
    source.write_text("bc")

    monkeypatch.setattr(fpy_main, "fpybc_parse", lambda text: ["body"])
    monkeypatch.setattr(fpy_main, "assemble", lambda body: ["dirs"])
    monkeypatch.setattr(
        fpy_main,
        "serialize_directives",
        lambda directives: (b"\x03\x04\x05", 0x1234),
    )

    fpy_main.assemble_main([str(source)])

    output_path = source.with_suffix(".bin")
    assert output_path.read_bytes() == b"\x03\x04\x05"
    captured = capsys.readouterr()
    assert "CRC 0x1234" in captured.out


def test_disassemble_main_missing_input(tmp_path, capsys):
    source = tmp_path / "seq.bin"
    with pytest.raises(SystemExit) as exc:
        fpy_main.disassemble_main([str(source)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.out


def test_disassemble_main_writes_text(monkeypatch, tmp_path, capsys):
    source = tmp_path / "seq.bin"
    source.write_bytes(b"data")

    monkeypatch.setattr(fpy_main, "deserialize_directives", lambda data: (["dirs"], []))
    monkeypatch.setattr(fpy_main, "fpybc_directives_to_fpyasm", lambda dirs: "FPYBC")

    fpy_main.disassemble_main([str(source)])

    output_path = source.with_suffix(".fpybc")
    assert output_path.read_text() == "FPYBC"
    captured = capsys.readouterr()
    assert captured.out.strip() == "Done"


# ---------------------------------------------------------------------------
# cmd_main tests
# ---------------------------------------------------------------------------


def test_cmd_main_compiles_and_sends(monkeypatch, capsys):
    """Happy path: compiles the provided source and sends via ZMQ."""
    captured_source = {}

    def fake_text_to_ast(text):
        captured_source["text"] = text
        return "AST"

    monkeypatch.setattr(fpy_main, "text_to_ast", fake_text_to_ast)
    monkeypatch.setattr(
        fpy_main, "get_base_compile_state", lambda dictionary, ground_binary_dir=None: "STATE"
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)

    directive = ConstCmdDirective(cmd_opcode=0x10006001, args=b"\xAB\xCD")

    monkeypatch.setattr(
        fpy_main, "analysis_to_fpybc_directives", lambda body, state: ([directive], [])
    )

    sent = {}

    def fake_send(cmd_opcode, args, zmq_addr):
        sent["cmd_opcode"] = cmd_opcode
        sent["args"] = args
        sent["zmq_addr"] = zmq_addr

    monkeypatch.setattr(fpy_main, "send_command_zmq", fake_send)

    fpy_main.cmd_main([
        'Ref.cmdSeq0.RUN_ARGS("seq.bin", NO_WAIT, 42)',
        "-d", "dict.json",
    ])

    assert captured_source["text"] == 'Ref.cmdSeq0.RUN_ARGS("seq.bin", NO_WAIT, 42)\n'
    assert sent["cmd_opcode"] == 0x10006001
    assert sent["args"] == b"\xAB\xCD"
    assert "Sending" in capsys.readouterr().out


def test_cmd_main_compile_error(monkeypatch, capsys):
    """Exit 1 when the compiler raises an error."""
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")
    monkeypatch.setattr(
        fpy_main, "get_base_compile_state", lambda dictionary, ground_binary_dir=None: "STATE"
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)

    def raise_compile_error(body, state):
        raise fpy_error.CompileError("bad arg", None)

    monkeypatch.setattr(fpy_main, "analysis_to_fpybc_directives", raise_compile_error)

    with pytest.raises(SystemExit) as exc:
        fpy_main.cmd_main([
            'Ref.cmdSeq0.RUN_ARGS("seq.bin", NO_WAIT, bad_value)',
            "-d", "dict.json",
        ])

    assert exc.value.code == 1


def test_cmd_main_non_const_arg(monkeypatch, capsys):
    """Exit 1 when compilation produces a non-const (stack) command."""
    from fpy.bytecode.directives import StackCmdDirective

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")
    monkeypatch.setattr(
        fpy_main, "get_base_compile_state", lambda dictionary, ground_binary_dir=None: "STATE"
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)
    monkeypatch.setattr(
        fpy_main, "analysis_to_fpybc_directives",
        lambda body, state: ([StackCmdDirective(args_size=10)], []),
    )

    with pytest.raises(SystemExit) as exc:
        fpy_main.cmd_main([
            'Ref.cmdSeq0.RUN_ARGS("seq.bin", NO_WAIT, some_tlm)',
            "-d", "dict.json",
        ])

    assert exc.value.code == 1
    assert "Command arguments must be constant expressions" in capsys.readouterr().err


def test_cmd_main_send_failure(monkeypatch, capsys):
    """Exit 1 when the ZMQ send raises an exception."""
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")
    monkeypatch.setattr(
        fpy_main, "get_base_compile_state", lambda dictionary, ground_binary_dir=None: "STATE"
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)

    directive = ConstCmdDirective(cmd_opcode=0x10006001, args=b"")
    monkeypatch.setattr(
        fpy_main, "analysis_to_fpybc_directives",
        lambda body, state: ([directive], []),
    )

    def fail_send(*a):
        raise ConnectionError("ZMQ not reachable")

    monkeypatch.setattr(fpy_main, "send_command_zmq", fail_send)

    with pytest.raises(SystemExit) as exc:
        fpy_main.cmd_main([
            'Ref.cmdSeq0.RUN_ARGS("seq.bin", NO_WAIT)',
            "-d", "dict.json",
        ])

    assert exc.value.code == 1
    assert "Failed to send command" in capsys.readouterr().err


def test_cmd_main_ground_binary_dir(monkeypatch, tmp_path, capsys):
    """--ground-binary-dir is resolved and passed to get_base_compile_state."""
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")

    captured_kwargs = {}

    def fake_get_base_compile_state(dictionary, ground_binary_dir=None):
        captured_kwargs["ground_binary_dir"] = ground_binary_dir
        return "STATE"

    monkeypatch.setattr(fpy_main, "get_base_compile_state", fake_get_base_compile_state)
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)
    monkeypatch.setattr(
        fpy_main, "analysis_to_fpybc_directives",
        lambda body, state: ([ConstCmdDirective(cmd_opcode=0x10006001, args=b"")], []),
    )
    monkeypatch.setattr(fpy_main, "send_command_zmq", lambda *a: None)

    bin_dir = tmp_path / "bins"
    bin_dir.mkdir()

    fpy_main.cmd_main([
        'Ref.cmdSeq0.RUN_ARGS("seq.bin", NO_WAIT)',
        "-d", "dict.json",
        "-g", str(bin_dir),
    ])

    assert captured_kwargs["ground_binary_dir"] == str(bin_dir.resolve())


def test_cmd_main_zmq_addr(monkeypatch, capsys):
    """--zmq-addr is passed through to send_command_zmq."""
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")
    monkeypatch.setattr(
        fpy_main, "get_base_compile_state", lambda dictionary, ground_binary_dir=None: "STATE"
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)

    directive = ConstCmdDirective(cmd_opcode=0x10006001, args=b"")
    monkeypatch.setattr(
        fpy_main, "analysis_to_fpybc_directives",
        lambda body, state: ([directive], []),
    )

    sent = {}
    monkeypatch.setattr(fpy_main, "send_command_zmq", lambda o, a, addr: sent.update(addr=addr))

    fpy_main.cmd_main([
        'Ref.cmdSeq0.RUN_ARGS("seq.bin", NO_WAIT)',
        "-d", "dict.json",
        "--zmq-addr", "tcp://192.168.1.1:50050",
    ])

    assert sent["addr"] == "tcp://192.168.1.1:50050"


# ---------------------------------------------------------------------------
# depend_main tests
# ---------------------------------------------------------------------------


def test_depend_main_missing_input(tmp_path, capsys):
    missing = tmp_path / "missing.fpy"
    dict_path = tmp_path / "dict.json"
    with pytest.raises(SystemExit) as exc:
        fpy_main.depend_main([str(missing), "--dictionary", str(dict_path)])
    assert exc.value.code == 1
    assert "does not exist" in capsys.readouterr().err


def test_depend_main_ground_binary_dir_resolved(monkeypatch, tmp_path):
    """-g is resolved to an absolute path before being passed to get_base_compile_state."""
    fpy_path = tmp_path / "seq.fpy"
    fpy_path.write_text("content")
    bin_dir = tmp_path / "bins"
    bin_dir.mkdir()

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda _text: "AST")

    captured = {}

    def fake_get_base_compile_state(dictionary, ground_binary_dir=None):
        captured["ground_binary_dir"] = ground_binary_dir
        return "STATE"

    monkeypatch.setattr(fpy_main, "get_base_compile_state", fake_get_base_compile_state)
    monkeypatch.setattr(fpy_main, "ast_to_dependencies", lambda _body, _state: [])

    fpy_main.depend_main([str(fpy_path), "-d", "dict.json", "-g", str(bin_dir)])

    assert captured["ground_binary_dir"] == str(bin_dir.resolve())


def test_depend_main_default_ground_binary_dir(monkeypatch, tmp_path):
    """When -g is omitted, ground_binary_dir defaults to the input file's parent."""
    fpy_path = tmp_path / "seq.fpy"
    fpy_path.write_text("content")

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda _text: "AST")

    captured = {}

    def fake_get_base_compile_state(dictionary, ground_binary_dir=None):
        captured["ground_binary_dir"] = ground_binary_dir
        return "STATE"

    monkeypatch.setattr(fpy_main, "get_base_compile_state", fake_get_base_compile_state)
    monkeypatch.setattr(fpy_main, "ast_to_dependencies", lambda _body, _state: [])

    fpy_main.depend_main([str(fpy_path), "-d", "dict.json"])

    assert captured["ground_binary_dir"] == str(tmp_path.resolve())


def test_depend_main_compile_error_exits(monkeypatch, tmp_path, capsys):
    """A compile error raised by ast_to_dependencies is printed to stderr and exits 1."""
    fpy_path = tmp_path / "seq.fpy"
    fpy_path.write_text("content")

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda _text: "AST")
    monkeypatch.setattr(
        fpy_main, "get_base_compile_state", lambda dictionary, ground_binary_dir=None: "STATE"
    )

    def raise_compile_error(_body, _state):
        raise fpy_error.CompileError("bad syntax", None)

    monkeypatch.setattr(fpy_main, "ast_to_dependencies", raise_compile_error)

    with pytest.raises(SystemExit) as exc:
        fpy_main.depend_main([str(fpy_path), "-d", "dict.json"])

    assert exc.value.code == 1
    assert "bad syntax" in capsys.readouterr().err


def test_depend_main_outputs_deps(monkeypatch, tmp_path, capsys):
    """Each dependency path is printed to stdout on its own line."""
    fpy_path = tmp_path / "seq.fpy"
    fpy_path.write_text("content")

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda _text: "AST")
    monkeypatch.setattr(
        fpy_main, "get_base_compile_state", lambda dictionary, ground_binary_dir=None: "STATE"
    )
    monkeypatch.setattr(
        fpy_main,
        "ast_to_dependencies",
        lambda _body, _state: ["/tmp/a.bin", "/tmp/b.bin"],
    )

    fpy_main.depend_main([str(fpy_path), "-d", "dict.json"])

    assert capsys.readouterr().out == "/tmp/a.bin\n/tmp/b.bin\n"


def test_build_command_packet():
    """Command packet has correct wire format: size(4B) + descriptor(2B) + opcode(4B) + args."""
    import struct

    packet = fpy_main.build_command_packet(0x10006001, b"\x01\x02\x03")

    # size = 2 (descriptor) + 4 (opcode) + 3 (args) = 9
    expected_size = struct.pack(">I", 9)
    expected_descriptor = struct.pack(">H", 0)  # FW_PACKET_COMMAND = 0
    expected_opcode = struct.pack(">I", 0x10006001)
    expected_args = b"\x01\x02\x03"

    assert packet == expected_size + expected_descriptor + expected_opcode + expected_args
