from types import SimpleNamespace

import pytest

from fpy import main as fpy_main
from fpy.bytecode.directives import ConstCmdDirective
import fpy.error as fpy_error


def fake_compile_state(**kwargs):
    """A stand-in CompileState with the attributes compile_main touches."""
    return SimpleNamespace(max_directive_size=2048)


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


def test_compile_main_seq_maps(monkeypatch, tmp_path, capsys):
    """Repeated --seq-map values are parsed in order and passed to
    get_base_compile_state."""
    input_path = tmp_path / "seq.fpy"
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")

    captured = {}

    def fake_get_base_compile_state(dictionary, seq_maps=None, **kwargs):
        captured["seq_maps"] = seq_maps
        return fake_compile_state()

    monkeypatch.setattr(fpy_main, "get_base_compile_state", fake_get_base_compile_state)
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)
    monkeypatch.setattr(
        fpy_main,
        "analysis_to_fpybc_directives",
        lambda state: (["directive"], []),
    )
    monkeypatch.setattr(
        fpy_main,
        "serialize_directives",
        lambda directives, arg_specs, **kwargs: (b"\x01", 0x1),
    )

    fpy_main.compile_main(
        [
            str(input_path),
            "--dictionary",
            str(dict_path),
            "--seq-map",
            "a=b",
            "--seq-map",
            "c=d",
        ]
    )

    assert captured["seq_maps"] == [("a", "b"), ("c", "d")]


def test_compile_main_invalid_seq_map(monkeypatch, tmp_path, capsys):
    """A --seq-map value with no '=' is reported to stderr and exits 1."""
    input_path = tmp_path / "seq.fpy"
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")

    with pytest.raises(SystemExit) as exc:
        fpy_main.compile_main(
            [
                str(input_path),
                "--dictionary",
                str(dict_path),
                "--seq-map",
                "nomapping",
            ]
        )

    assert exc.value.code == 1
    assert "Invalid --seq-map value" in capsys.readouterr().err


def _run_compile_capturing_kwargs(monkeypatch, argv):
    """Invoke compile_main with the codegen chain stubbed out, returning the
    kwargs get_base_compile_state was called with."""
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")

    captured_kwargs = {}

    def fake_get_base_compile_state(dictionary, seq_maps=None, **kwargs):
        captured_kwargs.update(kwargs)
        return fake_compile_state()

    monkeypatch.setattr(fpy_main, "get_base_compile_state", fake_get_base_compile_state)
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)
    monkeypatch.setattr(
        fpy_main,
        "analysis_to_fpybc_directives",
        lambda state: (["directive"], []),
    )
    monkeypatch.setattr(
        fpy_main,
        "serialize_directives",
        lambda directives, arg_specs, **kwargs: (b"\x01", 0x1),
    )

    fpy_main.compile_main(argv)
    return captured_kwargs


def test_compile_main_include_dirs_are_import_directories(monkeypatch, tmp_path):
    """The import directories are exactly the resolved -i/--imports dirs, in
    order; the input file's own dir is NOT among them (it anchors relative
    imports instead, via main_file_dir)."""
    input_path = tmp_path / "sub" / "seq.fpy"
    input_path.parent.mkdir()
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")
    inc_a = tmp_path / "inc_a"
    inc_b = tmp_path / "inc_b"
    inc_a.mkdir()
    inc_b.mkdir()

    captured_kwargs = _run_compile_capturing_kwargs(
        monkeypatch,
        [
            str(input_path),
            "--dictionary",
            str(dict_path),
            "-i",
            str(inc_a),
            "--imports",
            str(inc_b),
        ],
    )

    assert captured_kwargs["import_directories"] == [
        str(inc_a.resolve()),
        str(inc_b.resolve()),
    ]
    assert captured_kwargs["main_file_dir"] == str(input_path.parent.resolve())


def test_compile_main_duplicate_includes_are_deduped(monkeypatch, tmp_path):
    """A repeated -i directory (even spelled differently) collapses to one
    import-directory entry after resolution: it carries no information."""
    input_path = tmp_path / "seq.fpy"
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")
    inc = tmp_path / "inc"
    inc.mkdir()

    captured_kwargs = _run_compile_capturing_kwargs(
        monkeypatch,
        [
            str(input_path),
            "--dictionary",
            str(dict_path),
            "-i",
            str(inc),
            "-i",
            str(tmp_path / "inc" / ".." / "inc"),
        ],
    )

    assert captured_kwargs["import_directories"] == [str(inc.resolve())]


def test_compile_main_no_includes_means_no_import_directories(monkeypatch, tmp_path):
    """With no -i flags, there are no import directories; the input file's
    own dir still anchors its relative imports."""
    input_path = tmp_path / "seq.fpy"
    input_path.write_text("content")
    dict_path = tmp_path / "dict.json"
    dict_path.write_text("{}")

    captured_kwargs = _run_compile_capturing_kwargs(
        monkeypatch,
        [
            str(input_path),
            "--dictionary",
            str(dict_path),
        ],
    )

    assert captured_kwargs["import_directories"] == []
    assert captured_kwargs["main_file_dir"] == str(input_path.parent.resolve())


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
        fpy_main,
        "get_base_compile_state",
        lambda dictionary, seq_maps=None, **kwargs: "STATE",
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)

    def fake_analysis_to_fpybc_directives(state):
        assert state == "STATE"
        return ["directive"], []

    monkeypatch.setattr(
        fpy_main, "analysis_to_fpybc_directives", fake_analysis_to_fpybc_directives
    )
    monkeypatch.setattr(
        fpy_main, "fpybc_directives_to_fpyasm", lambda directives: "FPYASM"
    )

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
        fpy_main,
        "get_base_compile_state",
        lambda dictionary, seq_maps=None, **kwargs: "STATE",
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)

    def fake_analysis_to_wat(state):
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
        fpy_main,
        "get_base_compile_state",
        lambda dictionary, seq_maps=None, **kwargs: fake_compile_state(),
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)
    monkeypatch.setattr(
        fpy_main,
        "analysis_to_fpybc_directives",
        lambda state: (["directive"], []),
    )
    monkeypatch.setattr(
        fpy_main,
        "serialize_directives",
        lambda directives, arg_specs, **kwargs: (b"\x01\x02", 0xABCD),
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
        fpy_main,
        "get_base_compile_state",
        lambda dictionary, seq_maps=None, **kwargs: "STATE",
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)

    directive = ConstCmdDirective(cmd_opcode=0x10006001, args=b"\xab\xcd")

    monkeypatch.setattr(
        fpy_main, "analysis_to_fpybc_directives", lambda state: ([directive], [])
    )

    sent = {}

    def fake_send(cmd_opcode, args, zmq_addr):
        sent["cmd_opcode"] = cmd_opcode
        sent["args"] = args
        sent["zmq_addr"] = zmq_addr

    monkeypatch.setattr(fpy_main, "send_command_zmq", fake_send)

    fpy_main.cmd_main(
        [
            'Ref.cmdSeq0.RUN_ARGS("seq.bin", NO_WAIT, 42)',
            "-d",
            "dict.json",
        ]
    )

    assert captured_source["text"] == 'Ref.cmdSeq0.RUN_ARGS("seq.bin", NO_WAIT, 42)\n'
    assert sent["cmd_opcode"] == 0x10006001
    assert sent["args"] == b"\xab\xcd"
    assert "Sending" in capsys.readouterr().out


def test_cmd_main_compile_error(monkeypatch, capsys):
    """Exit 1 when the compiler raises an error."""
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")
    monkeypatch.setattr(
        fpy_main,
        "get_base_compile_state",
        lambda dictionary, seq_maps=None, **kwargs: "STATE",
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)

    def raise_compile_error(state):
        raise fpy_error.CompileError("bad arg", None)

    monkeypatch.setattr(fpy_main, "analysis_to_fpybc_directives", raise_compile_error)

    with pytest.raises(SystemExit) as exc:
        fpy_main.cmd_main(
            [
                'Ref.cmdSeq0.RUN_ARGS("seq.bin", NO_WAIT, bad_value)',
                "-d",
                "dict.json",
            ]
        )

    assert exc.value.code == 1


def test_cmd_main_non_const_arg(monkeypatch, capsys):
    """Exit 1 when compilation produces a non-const (stack) command."""
    from fpy.bytecode.directives import StackCmdDirective

    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")
    monkeypatch.setattr(
        fpy_main,
        "get_base_compile_state",
        lambda dictionary, seq_maps=None, **kwargs: "STATE",
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)
    monkeypatch.setattr(
        fpy_main,
        "analysis_to_fpybc_directives",
        lambda state: ([StackCmdDirective(args_size=10)], []),
    )

    with pytest.raises(SystemExit) as exc:
        fpy_main.cmd_main(
            [
                'Ref.cmdSeq0.RUN_ARGS("seq.bin", NO_WAIT, some_tlm)',
                "-d",
                "dict.json",
            ]
        )

    assert exc.value.code == 1
    assert "Command arguments must be constant expressions" in capsys.readouterr().err


def test_cmd_main_send_failure(monkeypatch, capsys):
    """Exit 1 when the ZMQ send raises an exception."""
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")
    monkeypatch.setattr(
        fpy_main,
        "get_base_compile_state",
        lambda dictionary, seq_maps=None, **kwargs: "STATE",
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)

    directive = ConstCmdDirective(cmd_opcode=0x10006001, args=b"")
    monkeypatch.setattr(
        fpy_main,
        "analysis_to_fpybc_directives",
        lambda state: ([directive], []),
    )

    def fail_send(*a):
        raise ConnectionError("ZMQ not reachable")

    monkeypatch.setattr(fpy_main, "send_command_zmq", fail_send)

    with pytest.raises(SystemExit) as exc:
        fpy_main.cmd_main(
            [
                'Ref.cmdSeq0.RUN_ARGS("seq.bin", NO_WAIT)',
                "-d",
                "dict.json",
            ]
        )

    assert exc.value.code == 1
    assert "Failed to send command" in capsys.readouterr().err


def test_cmd_main_seq_maps(monkeypatch, tmp_path, capsys):
    """Repeated --seq-map values are parsed in order and passed to
    get_base_compile_state."""
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")

    captured = {}

    def fake_get_base_compile_state(dictionary, seq_maps=None, **kwargs):
        captured["seq_maps"] = seq_maps
        return "STATE"

    monkeypatch.setattr(fpy_main, "get_base_compile_state", fake_get_base_compile_state)
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)
    monkeypatch.setattr(
        fpy_main,
        "analysis_to_fpybc_directives",
        lambda state: ([ConstCmdDirective(cmd_opcode=0x10006001, args=b"")], []),
    )
    monkeypatch.setattr(fpy_main, "send_command_zmq", lambda *a: None)

    fpy_main.cmd_main(
        [
            'Ref.cmdSeq0.RUN_ARGS("seq.bin", NO_WAIT)',
            "-d",
            "dict.json",
            "--seq-map",
            "a=b",
            "--seq-map",
            "c=d",
        ]
    )

    assert captured["seq_maps"] == [("a", "b"), ("c", "d")]


def test_cmd_main_zmq_addr(monkeypatch, capsys):
    """--zmq-addr is passed through to send_command_zmq."""
    monkeypatch.setattr(fpy_main, "text_to_ast", lambda text: "AST")
    monkeypatch.setattr(
        fpy_main,
        "get_base_compile_state",
        lambda dictionary, seq_maps=None, **kwargs: "STATE",
    )
    monkeypatch.setattr(fpy_main, "analyze_ast", lambda body, state: state)

    directive = ConstCmdDirective(cmd_opcode=0x10006001, args=b"")
    monkeypatch.setattr(
        fpy_main,
        "analysis_to_fpybc_directives",
        lambda state: ([directive], []),
    )

    sent = {}
    monkeypatch.setattr(
        fpy_main, "send_command_zmq", lambda o, a, addr: sent.update(addr=addr)
    )

    fpy_main.cmd_main(
        [
            'Ref.cmdSeq0.RUN_ARGS("seq.bin", NO_WAIT)',
            "-d",
            "dict.json",
            "--zmq-addr",
            "tcp://192.168.1.1:50050",
        ]
    )

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

    assert (
        packet == expected_size + expected_descriptor + expected_opcode + expected_args
    )
