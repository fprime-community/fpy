# compiler debug flag
from dataclasses import dataclass
import sys
import traceback
from typing import Any

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


# the number of lines to show around a compiler error
COMPILER_ERROR_CONTEXT_LINE_COUNT = 1


class SyntaxErrorDuringTransform(Exception):
    """Raised during AST transformation for user-facing syntax errors."""
    def __init__(self, msg: str, node=None):
        self.msg = msg
        self.node = node
        super().__init__(msg)


@dataclass
class CompileError:
    msg: str
    node: Any = None

    def __post_init__(self):
        self.stack_trace = "\n".join(traceback.format_stack(limit=8)[:-1])

    def __repr__(self):

        stack_trace_optional = f"{self.stack_trace}\n" if debug else ""
        file_name_str = file_name if file_name is not None else "<unknown file>"

        if self.node is None:
            return f"{stack_trace_optional}{Colors.cyan(file_name_str)}: {Colors.bold(Colors.red(self.msg))}"

        meta = self.node if isinstance(self.node, Token) else self.node.meta

        source_start_line = meta.line - 1 - COMPILER_ERROR_CONTEXT_LINE_COUNT
        source_start_line = max(0, source_start_line)
        # end_line can be None for $END token
        end_line = meta.end_line if meta.end_line is not None else meta.line
        source_end_line = end_line - 1 + COMPILER_ERROR_CONTEXT_LINE_COUNT
        source_end_line = min(len(input_lines), source_end_line)

        # this is the list of all the src lines we will display
        source_to_display: list[str] = input_lines[
            source_start_line : source_end_line + 1
        ]

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
            return f"{stack_trace_optional}{Colors.cyan(location)} {Colors.bold(Colors.red(self.msg))}\n{source_to_display_str}"

        node_start_line_in_ctx = meta.line - 1 - source_start_line
        # end_column can be None for $END token
        end_column = meta.end_column if meta.end_column is not None else meta.column + 1
        caret_str = "^" * (end_column - meta.column)
        error_highlight = " " * (meta.column - 1 + line_number_space + 3) + Colors.red(caret_str)
        source_to_display.insert(node_start_line_in_ctx + 1, error_highlight)
        location = f"{file_name_str}:{meta.line}"
        result = f"{stack_trace_optional}{Colors.cyan(location)} {Colors.bold(Colors.red(self.msg))}\n"
        result += "\n".join(source_to_display)

        return result


@dataclass
class BackendError:
    msg: str

    def __repr__(self):
        file_name_str = file_name if file_name is not None else "<unknown file>"
        return f"{Colors.cyan(file_name_str)}: {Colors.bold(Colors.red(self.msg))}"


def handle_lark_error(err):
    import sys
    assert isinstance(err, LarkError), err
    if isinstance(err, UnexpectedToken):
        print(str(CompileError("Invalid syntax", err.token)), file=sys.stderr)
    elif isinstance(err, DedentError):
        print(str(CompileError(err.args[0])), file=sys.stderr)
    exit(1)
