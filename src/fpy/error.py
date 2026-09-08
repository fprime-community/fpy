# compiler debug flag
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import sys
import traceback
from typing import Any, NoReturn

from lark import LarkError, Token, UnexpectedToken
from lark.indenter import DedentError


# ANSI color codes (only used if outputting to a terminal)
class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def enabled(cls) -> bool:
        return sys.stderr.isatty()

    @classmethod
    def stdout_enabled(cls) -> bool:
        return sys.stdout.isatty()

    @classmethod
    def red(cls, s: str) -> str:
        return f"{cls.RED}{s}{cls.RESET}" if cls.enabled() else s

    @classmethod
    def green(cls, s: str) -> str:
        return f"{cls.GREEN}{s}{cls.RESET}" if cls.stdout_enabled() else s

    @classmethod
    def yellow(cls, s: str) -> str:
        return f"{cls.YELLOW}{s}{cls.RESET}" if cls.enabled() else s

    @classmethod
    def cyan(cls, s: str) -> str:
        return f"{cls.CYAN}{s}{cls.RESET}" if cls.enabled() else s

    @classmethod
    def bold(cls, s: str) -> str:
        return f"{cls.BOLD}{s}{cls.RESET}" if cls.stdout_enabled() else s

    @classmethod
    def dim(cls, s: str) -> str:
        return f"{cls.DIM}{s}{cls.RESET}" if cls.enabled() else s


# assigned in compiler_main
file_name = None
# assigned in compiler_main
debug = False
# assigned in text_to_ast
input_text = None
# assigned in text_to_ast
input_lines = None


@contextmanager
def diagnostic_context(new_file_name: str):
    """Point the diagnostics at another file for the duration of the block, so
    that errors in it point into that file rather than into the current one.
    Restores the previous context (file_name, input_text, input_lines) on exit.

    The block is expected to set input_text/input_lines itself (text_to_ast
    does this)."""
    global file_name, input_text, input_lines
    old = (file_name, input_text, input_lines)
    file_name = new_file_name
    try:
        yield
    finally:
        file_name, input_text, input_lines = old


# the number of lines to show around a compiler error
COMPILER_ERROR_CONTEXT_LINE_COUNT = 1


class SyntaxErrorDuringTransform(Exception):
    """Raised during AST transformation for user-facing syntax errors."""

    def __init__(self, msg: str, node=None):
        self.msg = msg
        self.node = node
        super().__init__(msg)


def format_diagnostic(msg: str, node, stack_trace: str = "", color=Colors.red) -> str:
    """Render a source-located diagnostic (error or warning).

    *color* is the Colors.* function used to highlight the message and the
    caret -- red for errors, yellow for warnings. *node* may be an AST node, a
    Token, or None (file-level diagnostic with no source snippet)."""

    stack_trace_optional = f"{stack_trace}\n" if debug and stack_trace else ""
    file_name_str = file_name if file_name is not None else "<unknown file>"

    if node is None:
        return f"{stack_trace_optional}{Colors.cyan(file_name_str)}: {Colors.bold(color(msg))}"

    meta = node if isinstance(node, Token) else node.meta

    # line can be None for the end-of-input token: there is no source to point at
    if meta.line is None:
        return f"{stack_trace_optional}{Colors.cyan(file_name_str)}: {Colors.bold(color(msg))}"

    source_start_line = meta.line - 1 - COMPILER_ERROR_CONTEXT_LINE_COUNT
    source_start_line = max(0, source_start_line)
    # end_line can be None for $END token
    end_line = meta.end_line if meta.end_line is not None else meta.line
    source_end_line = end_line - 1 + COMPILER_ERROR_CONTEXT_LINE_COUNT
    source_end_line = min(len(input_lines), source_end_line)

    # this is the list of all the src lines we will display
    source_to_display: list[str] = input_lines[source_start_line : source_end_line + 1]

    # reserve this much space for the line numbers
    # add two extra spaces for the caret to display multiline errors on lhs
    line_number_space = 6 if source_end_line < 998 else 10

    node_lines = end_line - meta.line

    # prefix all the lines with the prefix and line number
    # right justified line number, then a |, then the line
    # also if this is a multiline error, highlight the lines that errored with a >
    source_to_display = [
        (
            ("> " if line_idx in range(meta.line - 1, end_line - 1) else "")
            + str(source_start_line + line_idx + 1)
        ).rjust(line_number_space)
        + " | "
        + line
        for line_idx, line in enumerate(source_to_display)
    ]

    if node_lines > 1:
        source_to_display_str = "\n".join(source_to_display)
        # it's a multiline node. don't try to highlight the whole thing
        # just print the err and the offending text
        location = f"{file_name_str}:{meta.line}-{end_line}"
        return f"{stack_trace_optional}{Colors.cyan(location)} {Colors.bold(color(msg))}\n{source_to_display_str}"

    node_start_line_in_ctx = meta.line - 1 - source_start_line
    # end_column can be None for $END token
    end_column = meta.end_column if meta.end_column is not None else meta.column + 1
    caret_str = "^" * (end_column - meta.column)
    error_highlight = " " * (meta.column - 1 + line_number_space + 3) + color(caret_str)
    source_to_display.insert(node_start_line_in_ctx + 1, error_highlight)
    location = f"{file_name_str}:{meta.line}"
    result = (
        f"{stack_trace_optional}{Colors.cyan(location)} {Colors.bold(color(msg))}\n"
    )
    result += "\n".join(source_to_display)

    return result


@dataclass
class CompileError(Exception):
    msg: str
    node: Any = None

    def __post_init__(self):
        self.stack_trace = "\n".join(traceback.format_stack(limit=8)[:-1])

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return format_diagnostic(self.msg, self.node, self.stack_trace, Colors.red)


@dataclass
class BackendError(Exception):
    msg: str
    node: Any = None

    def __repr__(self):
        if self.node is not None:
            return format_diagnostic(self.msg, self.node, color=Colors.red)
        file_name_str = file_name if file_name is not None else "<unknown file>"
        return f"{Colors.cyan(file_name_str)}: {Colors.bold(Colors.red(self.msg))}"

    def __str__(self):
        return self.__repr__()


class DictionaryError(Exception):
    """Raised when the F´ dictionary is missing a type fpy requires, or defines
    one in a way that is incompatible with fpy's built-in canonical version.

    This is not a problem with the user's .fpy source -- it means the dictionary
    itself can't be used by this version of fpy.  Carries a human-readable
    explanation so the failure surfaces as a clear message instead of a
    traceback.
    """

    def __init__(self, type_name: str, detail: str):
        self.type_name = type_name
        self.detail = detail
        super().__init__(detail)

    def __str__(self) -> str:
        header = Colors.bold(
            Colors.red(
                f"Dictionary type {self.type_name!r} is incompatible with this version of fpy."
            )
        )
        return f"{header}\n  {self.detail}"


class WarningType(str, Enum):
    """The set of diagnostics the compiler may warn about."""

    EMPTY_RANGE = "empty-range"
    IMPORT_DUPLICATE = "import-duplicate"
    IMPORT_UNDERSCORE = "import-underscore"
    SHADOW_VALUE = "shadow-value"
    SHADOW_CALLABLE = "shadow-callable"
    UNREACHABLE_TIMEOUT_BODY = "unreachable-timeout-body"

    @classmethod
    def from_value(cls, value: str) -> "WarningType":
        """Look up a WarningType by its CLI string value.

        Raises ValueError (listing the valid values) if unknown."""
        for member in cls:
            if member.value == value:
                return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(f"Unknown warning type {value!r}. Valid types: {valid}")


# Sentinel accepted by --ignore / --error to mean "every warning type".
WARNING_ALL = "all"


def parse_warning_set(spec: str) -> set[WarningType]:
    """Parse a comma-separated `--ignore`/`--error` spec into a set of types.

    Empty/whitespace-only entries are skipped.  The special value "all"
    expands to every WarningType.  Raises ValueError on an unknown type."""
    result: set[WarningType] = set()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token == WARNING_ALL:
            result.update(WarningType)
            continue
        result.add(WarningType.from_value(token))
    return result


@dataclass
class CompileWarning:
    """A non-fatal diagnostic."""

    type: WarningType
    msg: str
    node: Any = None

    def __str__(self) -> str:
        label = f"warning [{self.type.value}]: {self.msg}"
        return format_diagnostic(label, self.node, color=Colors.yellow)

    __repr__ = __str__


def handle_lark_error(err) -> NoReturn:
    """Turn a lark parse error into a CompileError and raise it."""
    assert isinstance(err, LarkError), err
    if isinstance(err, UnexpectedToken):
        raise CompileError("Invalid syntax", err.token)
    if isinstance(err, DedentError):
        raise CompileError(err.args[0])
    raise CompileError(str(err))
