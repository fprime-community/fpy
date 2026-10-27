"""String literal values and their exact output on both real sequencers."""

import pytest

from fpy.compiler import text_to_ast
from fpy.dictionary import load_dictionary
from fpy.error import CompileError
from fpy.harness import SequencerHarness
from fpy.syntax import AstString
from fpy.test_helpers import (
    FPYBC,
    CompilationFailed,
    assert_run_success,
    compile_seq,
    compile_seq_wasm,
    default_dictionary,
)

STRING_CASES = [
    pytest.param('""', "", id="empty-double"),
    pytest.param("''", "", id="empty-single"),
    pytest.param("'\"hi\"'", '"hi"', id="preserve-double-quotes"),
    pytest.param("\"'hi'\"", "'hi'", id="preserve-single-quotes"),
    pytest.param(r'''"say \"hi\""''', 'say "hi"', id="escaped-double-quotes"),
    pytest.param(r"'it\'s'", "it's", id="escaped-single-quote"),
    pytest.param(r'"a\nb\tc\rd"', "a\nb\tc\rd", id="control-escapes"),
    pytest.param(r'"C:\\temp\\"', "C:\\temp\\", id="escaped-backslashes"),
    pytest.param(r'"\\n"', r"\n", id="decode-once"),
    pytest.param('"héllo✓"', "héllo✓", id="literal-utf8"),
    pytest.param(r'"h\u00e9llo\u2713"', "héllo✓", id="unicode-escapes"),
    pytest.param(r'"\x41\101"', "AA", id="hex-and-octal"),
    pytest.param(r'"\U0001f680"', "🚀", id="long-unicode-escape"),
    pytest.param(r'"\N{ROCKET}"', "🚀", id="named-unicode-escape"),
]

SURROGATE_LITERALS = [
    pytest.param(r'"\ud800"', id="high-surrogate"),
    pytest.param(r'"\udfff"', id="low-surrogate"),
    pytest.param(r'"\ud83d\ude80"', id="surrogate-pair"),
    pytest.param(r'"a\ud800b"', id="embedded-surrogate"),
    pytest.param(r"'\ud800'", id="single-quoted-high-surrogate"),
    pytest.param(r"'\udfff'", id="single-quoted-low-surrogate"),
]


@pytest.mark.parametrize("literal,expected", STRING_CASES)
def test_string_literal_value(literal, expected):
    node = text_to_ast(literal).stmts[0]
    assert isinstance(node, AstString)
    assert node.value == expected


@pytest.mark.parametrize("quote", ['"', "'"])
def test_null_byte_has_source_diagnostic(quote):
    """Report raw NUL bytes at their source location, not as internal errors."""
    literal = f"{quote}a\0b{quote}"
    with pytest.raises(
        CompileError, match="Invalid string literal:.*null bytes"
    ) as exc:
        text_to_ast("# preceding line\n" f"log({literal})\n")
    assert exc.value.node.line == 2
    assert exc.value.node.column == 5
    assert "Internal error" not in str(exc.value)


@pytest.mark.parametrize(
    "literal",
    [r'"\x"', r'"\x0g"', r'"\u123"', r'"\U00110000"', r'"\N{NO SUCH NAME}"'],
)
def test_invalid_string_escape_has_source_diagnostic(literal):
    with pytest.raises(CompileError, match="Invalid string literal") as exc:
        text_to_ast("# preceding line\n" f"log({literal})\n")
    assert exc.value.node.line == 2
    assert exc.value.node.column == 5
    assert "Internal error" not in str(exc.value)


@pytest.mark.parametrize("literal", SURROGATE_LITERALS)
def test_surrogate_escape_has_source_diagnostic(literal):
    with pytest.raises(CompileError, match="Invalid string literal:.*UTF-8") as exc:
        text_to_ast("# preceding line\n" f"log({literal})\n")
    assert exc.value.node.line == 2
    assert exc.value.node.column == 5
    assert "Internal error" not in str(exc.value)


@pytest.mark.parametrize("literal", SURROGATE_LITERALS)
def test_surrogate_escape_is_compile_error(literal, single_backend):
    compile_fn = compile_seq if single_backend == FPYBC else compile_seq_wasm
    with pytest.raises(
        CompilationFailed, match="Invalid string literal:.*UTF-8"
    ) as exc:
        compile_fn("# preceding line\n" f"log({literal})\n")
    assert "Internal error" not in str(exc.value)


@pytest.mark.skipif(
    "config.getoption('--use-gds')", reason="inspects local sequencer harness output"
)
@pytest.mark.parametrize("literal,expected", STRING_CASES)
def test_string_literal_event_and_command(
    literal, expected, single_backend, fprime_test_api, monkeypatch
):
    # Observe the actual C++ harness reply; do not replace sequencer execution.
    replies = []
    original_run = SequencerHarness.run

    def record_run(harness, request):
        result = original_run(harness, request)
        replies.append((request, result))
        return result

    monkeypatch.setattr(SequencerHarness, "run", record_run)
    assert_run_success(
        fprime_test_api,
        f"log({literal})\nCdhCore.cmdDisp.CMD_NO_OP_STRING({literal})\n",
    )
    assert len(replies) == 1
    request, result = replies[0]

    message = expected
    if single_backend == FPYBC:
        message = (
            f"(FpySequencer) LogActivityHi : Sequence {request['seqFile']}: {expected}"
        )
    assert (5, message) in [
        (event["severity"], event["text"]) for event in result["events"]
    ]

    dictionary = load_dictionary(default_dictionary)
    types = dictionary["type_defs"]
    payload = expected.encode("utf-8")
    opcode = dictionary["cmd_name_dict"]["CdhCore.cmdDisp.CMD_NO_OP_STRING"].opcode
    expected_command = (
        opcode.to_bytes(types["FwOpcodeType"].max_size, "big")
        + len(payload).to_bytes(types["FwSizeStoreType"].max_size, "big")
        + payload
    )
    assert result["cmds"] == [expected_command.hex()]
