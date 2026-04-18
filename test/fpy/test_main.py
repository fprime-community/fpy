from pathlib import Path

import pytest

from fpy import main as fpy_main
from fpy.bytecode.directives import ConstCmdDirective
import fpy.error as fpy_error
import fpy.model as fpy_model
from fpy.types import CmdDef, FpyType, TypeKind


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
    """--ground-binary-dir is resolved and passed to ast_to_directives."""
    input_path = tmp_path / "seq.fpy"
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")
    bin_dir = tmp_path / "binaries"
    bin_dir.mkdir()

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")

    captured_kwargs = {}

    def fake_ast_to_directives(body, dictionary, ground_binary_dir=None, flight_binary_dir=None):
        captured_kwargs["ground_binary_dir"] = ground_binary_dir
        captured_kwargs["flight_binary_dir"] = flight_binary_dir
        return ["directive"], []

    monkeypatch.setattr(fpy_main, "ast_to_directives", fake_ast_to_directives)
    monkeypatch.setattr(fpy_main, "directives_to_fpybc", lambda directives: "FPYBC")

    fpy_main.compile_main(
        [
            str(input_path),
            "--dictionary",
            str(dict_path),
            "--bytecode",
            "--ground-binary-dir",
            str(bin_dir),
        ]
    )

    assert captured_kwargs["ground_binary_dir"] == str(bin_dir.resolve())
    assert captured_kwargs["flight_binary_dir"] is None


def test_compile_main_ground_binary_dir_defaults_to_input_parent(monkeypatch, tmp_path, capsys):
    """When --ground-binary-dir is not passed, it defaults to the input file's parent."""
    input_path = tmp_path / "seq.fpy"
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")

    captured_kwargs = {}

    def fake_ast_to_directives(body, dictionary, ground_binary_dir=None, flight_binary_dir=None):
        captured_kwargs["ground_binary_dir"] = ground_binary_dir
        return ["directive"], []

    monkeypatch.setattr(fpy_main, "ast_to_directives", fake_ast_to_directives)
    monkeypatch.setattr(fpy_main, "directives_to_fpybc", lambda directives: "FPYBC")

    fpy_main.compile_main(
        [
            str(input_path),
            "--dictionary",
            str(dict_path),
            "--bytecode",
        ]
    )

    assert captured_kwargs["ground_binary_dir"] == str(input_path.parent.resolve())


def test_compile_main_flight_binary_dir(monkeypatch, tmp_path, capsys):
    """--flight-binary-dir is passed through to ast_to_directives."""
    input_path = tmp_path / "seq.fpy"
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")

    captured_kwargs = {}

    def fake_ast_to_directives(body, dictionary, ground_binary_dir=None, flight_binary_dir=None):
        captured_kwargs["ground_binary_dir"] = ground_binary_dir
        captured_kwargs["flight_binary_dir"] = flight_binary_dir
        return ["directive"], []

    monkeypatch.setattr(fpy_main, "ast_to_directives", fake_ast_to_directives)
    monkeypatch.setattr(fpy_main, "directives_to_fpybc", lambda directives: "FPYBC")

    fpy_main.compile_main(
        [
            str(input_path),
            "--dictionary",
            str(dict_path),
            "--bytecode",
            "--ground-binary-dir",
            str(tmp_path),
            "--flight-binary-dir",
            "/seq/bin",
        ]
    )

    assert captured_kwargs["ground_binary_dir"] == str(tmp_path.resolve())
    assert captured_kwargs["flight_binary_dir"] == "/seq/bin"


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


def test_compile_main_bytecode_output(monkeypatch, tmp_path, capsys):
    input_path = tmp_path / "seq.fpy"
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")

    monkeypatch.setattr(fpy_error, "debug", False, raising=False)
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")

    def fake_ast_to_directives(body, dictionary, ground_binary_dir=None, flight_binary_dir=None):
        assert body == "AST"
        assert Path(dictionary) == dict_path
        return ["directive"], []

    monkeypatch.setattr(fpy_main, "ast_to_directives", fake_ast_to_directives)
    monkeypatch.setattr(fpy_main, "directives_to_fpybc", lambda directives: "FPYBC")

    def fail_serialize(*args):
        raise AssertionError("serialize_directives should not be called")

    monkeypatch.setattr(fpy_main, "serialize_directives", fail_serialize)

    fpy_main.compile_main(
        [
            str(input_path),
            "--dictionary",
            str(dict_path),
            "--bytecode",
            "--debug",
        ]
    )

    captured = capsys.readouterr()
    assert captured.out.strip() == "FPYBC"
    assert fpy_error.debug is True


def test_compile_main_binary_output(monkeypatch, tmp_path, capsys):
    input_path = tmp_path / "seq.fpy"
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")
    monkeypatch.setattr(
        fpy_main,
        "ast_to_directives",
        lambda body, dictionary, ground_binary_dir=None, flight_binary_dir=None: (["directive"], []),
    )
    monkeypatch.setattr(fpy_main, "directives_to_fpybc", lambda directives: "FPYBC")
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
            return fpy_main.DirectiveErrorCode.NO_ERROR

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
            return fpy_main.DirectiveErrorCode.EXIT_WITH_ERROR

    monkeypatch.setattr(fpy_main, "FpySequencerModel", DummyModel)

    with pytest.raises(SystemExit) as exc:
        fpy_main.model_main([str(binary)])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Sequence failed" in captured.out


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
    monkeypatch.setattr(fpy_main, "directives_to_fpybc", lambda dirs: "FPYBC")

    fpy_main.disassemble_main([str(source)])

    output_path = source.with_suffix(".fpybc")
    assert output_path.read_text() == "FPYBC"
    captured = capsys.readouterr()
    assert captured.out.strip() == "Done"


# ---------------------------------------------------------------------------
# Helpers for run_main tests
# ---------------------------------------------------------------------------


def _make_string_type():
    return FpyType(TypeKind.STRING, "string", max_length=240)


def _make_block_enum_type():
    return FpyType(
        TypeKind.ENUM,
        "Svc.FpySequencer.BlockState",
        enum_dict={"BLOCK": 0, "NO_BLOCK": 1},
        json_default="Svc.FpySequencer.BlockState.BLOCK",
    )


def _make_seq_args_type():
    return FpyType(TypeKind.STRUCT, "Svc.SeqArgs")


def _make_seq_run_cmd():
    return CmdDef(
        name="Ref.cmdSeq.RUN_ARGS",
        opcode=0x10006001,
        args=[
            ("fileName", "The sequence file name", _make_string_type()),
            ("block", "Block state", _make_block_enum_type()),
            ("buffer", "Sequence arguments", _make_seq_args_type()),
        ],
    )


def _make_non_seq_cmd():
    return CmdDef(
        name="Ref.health.HLTH_PING_ENABLE",
        opcode=0x1234,
        args=[
            ("enable", "Enable flag", FpyType(TypeKind.BOOL, "bool")),
        ],
    )


# ---------------------------------------------------------------------------
# run_main tests
# ---------------------------------------------------------------------------


def test_run_main_unknown_command(monkeypatch, capsys):
    """Exit 1 when the command name isn't in the dictionary."""
    monkeypatch.setattr(
        fpy_main, "load_dictionary", lambda _: {"cmd_name_dict": {}},
    )

    with pytest.raises(SystemExit) as exc:
        fpy_main.run_main([
            "-c", "Ref.cmdSeq.DOES_NOT_EXIST",
            "-d", "dict.json",
            "-i", "seq.bin",
        ])

    assert exc.value.code == 1
    assert "Unknown command" in capsys.readouterr().err


def test_run_main_not_seq_run_command(monkeypatch, capsys):
    """Exit 1 when the command doesn't have the seq-run signature."""
    monkeypatch.setattr(
        fpy_main,
        "load_dictionary",
        lambda _: {"cmd_name_dict": {"Ref.health.HLTH_PING_ENABLE": _make_non_seq_cmd()}},
    )

    with pytest.raises(SystemExit) as exc:
        fpy_main.run_main([
            "-c", "Ref.health.HLTH_PING_ENABLE",
            "-d", "dict.json",
            "-i", "seq.bin",
        ])

    assert exc.value.code == 1
    assert "not a sequence run command" in capsys.readouterr().err


def test_run_main_compiles_and_sends(monkeypatch, capsys):
    """Happy path: compiles the synthetic source and sends via ZMQ."""
    monkeypatch.setattr(
        fpy_main,
        "load_dictionary",
        lambda _: {"cmd_name_dict": {"Ref.cmdSeq.RUN_ARGS": _make_seq_run_cmd()}},
    )

    captured_source = {}

    def fake_text_to_ast(text):
        captured_source["text"] = text
        return "AST"

    monkeypatch.setattr(fpy_main, "text_to_ast", fake_text_to_ast)

    directive = ConstCmdDirective(cmd_opcode=0x10006001, args=b"\xAB\xCD")

    def fake_ast_to_directives(body, dictionary, ground_binary_dir=None, flight_binary_dir=None):
        return [directive], []

    monkeypatch.setattr(fpy_main, "ast_to_directives", fake_ast_to_directives)

    sent = {}

    def fake_send(cmd_opcode, args, zmq_addr):
        sent["cmd_opcode"] = cmd_opcode
        sent["args"] = args
        sent["zmq_addr"] = zmq_addr

    monkeypatch.setattr(fpy_main, "send_command_zmq", fake_send)

    fpy_main.run_main([
        "-c", "Ref.cmdSeq.RUN_ARGS",
        "-d", "dict.json",
        "-i", "seq.bin",
        "42",
    ])

    assert 'Ref.cmdSeq.RUN_ARGS("seq.bin"' in captured_source["text"]
    assert "42" in captured_source["text"]
    assert sent["cmd_opcode"] == 0x10006001
    assert sent["args"] == b"\xAB\xCD"
    assert "Sending" in capsys.readouterr().out


def test_run_main_no_seq_args(monkeypatch, capsys):
    """When no sequence args are given, the source omits the trailing args."""
    monkeypatch.setattr(
        fpy_main,
        "load_dictionary",
        lambda _: {"cmd_name_dict": {"Ref.cmdSeq.RUN_ARGS": _make_seq_run_cmd()}},
    )

    captured_source = {}

    def fake_text_to_ast(text):
        captured_source["text"] = text
        return "AST"

    monkeypatch.setattr(fpy_main, "text_to_ast", fake_text_to_ast)

    directive = ConstCmdDirective(cmd_opcode=0x10006001, args=b"")
    monkeypatch.setattr(
        fpy_main, "ast_to_directives",
        lambda body, dictionary, ground_binary_dir=None, flight_binary_dir=None: ([directive], []),
    )
    monkeypatch.setattr(fpy_main, "send_command_zmq", lambda *a: None)

    fpy_main.run_main([
        "-c", "Ref.cmdSeq.RUN_ARGS",
        "-d", "dict.json",
        "-i", "seq.bin",
    ])

    source = captured_source["text"]
    assert source == 'Ref.cmdSeq.RUN_ARGS("seq.bin", Svc.FpySequencer.BlockState.BLOCK)\n'


def test_run_main_compile_error(monkeypatch, capsys):
    """Exit 1 when the compiler returns an error."""
    monkeypatch.setattr(
        fpy_main,
        "load_dictionary",
        lambda _: {"cmd_name_dict": {"Ref.cmdSeq.RUN_ARGS": _make_seq_run_cmd()}},
    )
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")

    error = fpy_error.CompileError("bad arg", None)
    monkeypatch.setattr(
        fpy_main, "ast_to_directives",
        lambda body, dictionary, ground_binary_dir=None, flight_binary_dir=None: error,
    )

    with pytest.raises(SystemExit) as exc:
        fpy_main.run_main([
            "-c", "Ref.cmdSeq.RUN_ARGS",
            "-d", "dict.json",
            "-i", "seq.bin",
            "bad_value",
        ])

    assert exc.value.code == 1


def test_run_main_send_failure(monkeypatch, capsys):
    """Exit 1 when the ZMQ send raises an exception."""
    monkeypatch.setattr(
        fpy_main,
        "load_dictionary",
        lambda _: {"cmd_name_dict": {"Ref.cmdSeq.RUN_ARGS": _make_seq_run_cmd()}},
    )
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")

    directive = ConstCmdDirective(cmd_opcode=0x10006001, args=b"")
    monkeypatch.setattr(
        fpy_main, "ast_to_directives",
        lambda body, dictionary, ground_binary_dir=None, flight_binary_dir=None: ([directive], []),
    )

    def fail_send(*a):
        raise ConnectionError("ZMQ not reachable")

    monkeypatch.setattr(fpy_main, "send_command_zmq", fail_send)

    with pytest.raises(SystemExit) as exc:
        fpy_main.run_main([
            "-c", "Ref.cmdSeq.RUN_ARGS",
            "-d", "dict.json",
            "-i", "seq.bin",
        ])

    assert exc.value.code == 1
    assert "Failed to send command" in capsys.readouterr().err


def test_run_main_ground_binary_dir(monkeypatch, tmp_path, capsys):
    """--ground-binary-dir is resolved and passed to ast_to_directives."""
    monkeypatch.setattr(
        fpy_main,
        "load_dictionary",
        lambda _: {"cmd_name_dict": {"Ref.cmdSeq.RUN_ARGS": _make_seq_run_cmd()}},
    )
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")

    captured_kwargs = {}

    def fake_ast_to_directives(body, dictionary, ground_binary_dir=None, flight_binary_dir=None):
        captured_kwargs["ground_binary_dir"] = ground_binary_dir
        captured_kwargs["flight_binary_dir"] = flight_binary_dir
        return [ConstCmdDirective(cmd_opcode=0x10006001, args=b"")], []

    monkeypatch.setattr(fpy_main, "ast_to_directives", fake_ast_to_directives)
    monkeypatch.setattr(fpy_main, "send_command_zmq", lambda *a: None)

    bin_dir = tmp_path / "bins"
    bin_dir.mkdir()

    fpy_main.run_main([
        "-c", "Ref.cmdSeq.RUN_ARGS",
        "-d", "dict.json",
        "-B", str(bin_dir),
        "--flight-binary-dir", "/seq/",
        "-i", "seq.bin",
    ])

    assert captured_kwargs["ground_binary_dir"] == str(bin_dir.resolve())
    assert captured_kwargs["flight_binary_dir"] == "/seq/"


def test_run_main_zmq_addr(monkeypatch, capsys):
    """--zmq-addr is passed through to send_command_zmq."""
    monkeypatch.setattr(
        fpy_main,
        "load_dictionary",
        lambda _: {"cmd_name_dict": {"Ref.cmdSeq.RUN_ARGS": _make_seq_run_cmd()}},
    )
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")

    directive = ConstCmdDirective(cmd_opcode=0x10006001, args=b"")
    monkeypatch.setattr(
        fpy_main, "ast_to_directives",
        lambda body, dictionary, ground_binary_dir=None, flight_binary_dir=None: ([directive], []),
    )

    sent = {}
    monkeypatch.setattr(fpy_main, "send_command_zmq", lambda o, a, addr: sent.update(addr=addr))

    fpy_main.run_main([
        "-c", "Ref.cmdSeq.RUN_ARGS",
        "-d", "dict.json",
        "--zmq-addr", "tcp://192.168.1.1:50050",
        "-i", "seq.bin",
    ])

    assert sent["addr"] == "tcp://192.168.1.1:50050"


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
