from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator, List, Literal as TypingLiteral, Union
from lark import Token, Transformer, v_args
from lark.tree import Meta
from lark.lark import PostLex
from lark.indenter import DedentError
from decimal import Decimal


class PythonIndenter(PostLex):
    # from lark, but slightly modified to fix a bug
    """This is a postlexer that "injects" indent/dedent tokens based on indentation.

    It keeps track of the current indentation, as well as the current level of parentheses.
    Inside parentheses, the indentation is ignored, and no indent/dedent tokens get generated.
    See also: the ``postlex`` option in `Lark`.
    """
    paren_level: int
    indent_level: List[int]
    NL_type = "_NEWLINE"
    OPEN_PAREN_types = ["LPAR", "LSQB", "LBRACE"]
    CLOSE_PAREN_types = ["RPAR", "RSQB", "RBRACE"]
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    tab_len = 8

    def __init__(self) -> None:
        self.paren_level = 0
        self.indent_level = [0]
        assert self.tab_len > 0

    def handle_NL(self, token: Token) -> Iterator[Token]:
        if self.paren_level > 0:
            return

        yield token

        if not "\n" in token:
            return

        indent_str = token.rsplit("\n", 1)[1]  # Tabs and spaces
        # Only count leading whitespace; a trailing comment may appear
        # at end-of-file when there is no final newline.
        indent = 0
        for ch in indent_str:
            if ch == " ":
                indent += 1
            elif ch == "\t":
                indent += self.tab_len
            else:
                break

        if indent > self.indent_level[-1]:
            self.indent_level.append(indent)
            yield Token.new_borrow_pos(self.INDENT_type, indent_str, token)
        else:
            while indent < self.indent_level[-1]:
                self.indent_level.pop()
                yield Token.new_borrow_pos(self.DEDENT_type, indent_str, token)

            if indent != self.indent_level[-1]:
                raise DedentError(
                    "Unexpected dedent to column %s. Expected dedent to %s"
                    % (indent, self.indent_level[-1])
                )

    def _process(self, stream):
        for token in stream:
            if token.type == self.NL_type:
                yield from self.handle_NL(token)
            else:
                yield token

            if token.type in self.OPEN_PAREN_types:
                self.paren_level += 1
            elif token.type in self.CLOSE_PAREN_types:
                self.paren_level -= 1
                assert self.paren_level >= 0

        # At EOF, always inject a NEWLINE so the grammar can require
        # NEWLINE after small statements without requiring trailing newlines in input
        yield Token(self.NL_type, "\n")
        while len(self.indent_level) > 1:
            self.indent_level.pop()
            yield Token(self.DEDENT_type, "")

        assert self.indent_level == [0], self.indent_level

    def process(self, stream):
        self.paren_level = 0
        self.indent_level = [0]
        return self._process(stream)


@dataclass
class Ast:
    meta: Meta = field(repr=False)
    id: int = field(init=False, repr=False, default=None)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, value):
        if not isinstance(value, Ast):
            return False
        assert self.id is not None
        return self.id == value.id


@dataclass
class AstIdent(Ast):
    name: str


@dataclass()
class AstString(Ast):
    value: str


@dataclass
class AstNumber(Ast):
    value: int | Decimal


@dataclass
class AstBoolean(Ast):
    value: TypingLiteral[True] | TypingLiteral[False]


AstLiteral = Union[AstString, AstNumber, AstBoolean]


@dataclass
class AstGetAttr(Ast):
    parent: "AstExpr"
    attr: str


@dataclass
class AstIndexExpr(Ast):
    parent: "AstExpr"
    item: AstExpr


@dataclass
class AstNamedArgument(Ast):
    name: str
    value: "AstExpr"


@dataclass
class AstFuncCall(Ast):
    func: "AstExpr"
    # args can contain both positional (AstExpr) and named arguments (AstNamedArgument)
    args: list[Union["AstExpr", AstNamedArgument]] | None


@dataclass
class AstPass(Ast):
    pass  # ha ha


@dataclass
class AstBinaryOp(Ast):
    lhs: AstExpr
    op: str
    rhs: AstExpr


@dataclass
class AstUnaryOp(Ast):
    op: str
    val: AstExpr


@dataclass
class AstRange(Ast):
    lower_bound: AstExpr
    op: str
    upper_bound: AstExpr


AstOp = Union[AstBinaryOp, AstUnaryOp]

AstReference = Union[AstGetAttr, AstIndexExpr, AstIdent]
AstExpr = Union[AstFuncCall, AstLiteral, AstReference, AstOp, AstRange]


@dataclass
class AstAssign(Ast):
    lhs: AstExpr
    type_ann: AstExpr | None
    rhs: AstExpr


@dataclass
class AstElif(Ast):
    condition: AstExpr
    body: "AstBlock"


@dataclass()
class AstIf(Ast):
    condition: AstExpr
    body: "AstBlock"
    elifs: list[AstElif]
    els: Union["AstBlock", None]


@dataclass
class AstFor(Ast):
    loop_var: AstIdent
    range: AstExpr
    body: AstBlock


@dataclass
class AstWhile(Ast):
    condition: AstExpr
    body: AstBlock


@dataclass
class AstCheck(Ast):
    condition: AstExpr
    timeout: Union[AstExpr, None]  # Default: no timeout
    persist: Union[AstExpr, None]  # Default: 0 second interval
    freq: Union[AstExpr, None]    # Default: 1 second interval
    body: "AstBlock"
    timeout_body: Union["AstBlock", None] = None


@dataclass
class AstAssert(Ast):
    condition: AstExpr
    exit_code: Union[AstExpr, None]


@dataclass
class AstBreak(Ast):
    pass


@dataclass
class AstContinue(Ast):
    pass


@dataclass
class AstReturn(Ast):
    value: Union[AstExpr, None]


@dataclass
class AstDef(Ast):
    name: AstIdent
    # parameters is a list of (ident, type, default_value) tuples
    # default_value is None if no default is provided
    parameters: Union[list[tuple[AstIdent, AstExpr, AstExpr | None]], None]
    return_type: Union[AstExpr, None]
    body: AstBlock


AstStmt = Union[
    AstExpr,
    AstAssign,
    AstPass,
    AstIf,
    AstElif,
    AstFor,
    AstBreak,
    AstContinue,
    AstWhile,
    AstCheck,
    AstAssert,
    AstDef,
    AstReturn
]
AstStmtWithExpr = Union[
    AstExpr, AstAssign, AstIf, AstElif, AstFor, AstWhile, AstCheck, AstAssert, AstDef, AstReturn
]
AstNodeWithSideEffects = Union[
    AstFuncCall,
    AstAssign,
    AstIf,
    AstElif,
    AstFor,
    AstWhile,
    AstCheck,
    AstAssert,
    AstBreak,
    AstContinue,
    AstDef,
    AstReturn
]


@dataclass
class AstBlock(Ast):
    stmts: list[AstStmt]


for cls in Ast.__subclasses__():
    cls.__hash__ = Ast.__hash__
    # cls.__repr__ = Ast.__repr__


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


# Check statement clause handlers.
# The check_stmt grammar has multiple optional clauses (timeout, persist, freq).
# We use separate grammar rules for each clause so we can tag them and identify
# which optional clauses were provided, regardless of how many are present.
def handle_check_clause(tag):
    """Create a handler that tags an expression with the given clause name."""
    @v_args(meta=True, inline=True)
    def wrapper(self, meta, expr):
        return (tag, expr)
    return wrapper


def handle_check_clauses(meta, children):
    """Parse multi-line check clauses and body statements.
    
    Returns a tuple of (clause_list, body_stmts) where clause_list is a list of
    (clause_type, expr) tuples and body_stmts is an AstBlock.
    """
    clauses = []
    stmts = []
    
    for child in children:
        if isinstance(child, tuple) and len(child) == 2:
            # This is a clause: (clause_type, expr)
            clauses.append(child)
        else:
            # This is a statement AST node
            stmts.append(child)
    
    # Return as a special tuple that handle_check_stmt can recognize
    return ("check_clauses_result", clauses, AstBlock(meta, stmts))


def handle_check_stmt(meta, children):
    """Parse check statement with optional timeout/persist/freq clauses."""
    from fpy.error import SyntaxErrorDuringTransform
    
    condition = children[0]
    timeout = None
    persist = None
    freq = None
    body = None
    timeout_body = None
    
    def set_clause(clause_type, expr):
        nonlocal timeout, persist, freq
        if clause_type == "timeout":
            if timeout is not None:
                raise SyntaxErrorDuringTransform(f"Duplicate 'timeout' clause in check statement", expr)
            timeout = expr
        elif clause_type == "persist":
            if persist is not None:
                raise SyntaxErrorDuringTransform(f"Duplicate 'persist' clause in check statement", expr)
            persist = expr
        elif clause_type == "freq":
            if freq is not None:
                raise SyntaxErrorDuringTransform(f"Duplicate 'freq' clause in check statement", expr)
            freq = expr
    
    for child in children[1:]:
        # Handle check_clauses which returns ("check_clauses_result", clauses, body)
        if isinstance(child, tuple) and len(child) == 3 and child[0] == "check_clauses_result":
            _, clauses, stmts = child
            for clause_type, expr in clauses:
                set_clause(clause_type, expr)
            body = stmts
        elif isinstance(child, tuple) and len(child) == 2:
            clause_type, expr = child
            set_clause(clause_type, expr)
        elif isinstance(child, AstBlock):
            if body is None:
                body = child
            else:
                timeout_body = child
    
    assert body is not None, "check statement must have a body"
    return AstCheck(meta, condition, timeout, persist, freq, body, timeout_body)


def handle_parameter(meta, args):
    """Parse a single parameter: (name, type, default_value or None)"""
    assert len(args) in (2, 3), f"Expected 2 or 3 args, got {len(args)}: {args}"
    name, type_expr = args[0], args[1]
    default_value = args[2] if len(args) == 3 else None
    return (name, type_expr, default_value)


@v_args(meta=True, inline=True)
class FpyTransformer(Transformer):
    input = no_inline(AstBlock)
    pass_stmt = AstPass

    assign_stmt = AstAssign

    for_stmt = AstFor
    while_stmt = AstWhile
    block = no_inline(AstBlock)
    break_stmt = AstBreak
    continue_stmt = AstContinue

    assert_stmt = AstAssert

    if_stmt = AstIf

    check_timeout = handle_check_clause("timeout")
    check_persist = handle_check_clause("persist")
    check_freq = handle_check_clause("freq")
    check_timeout_final = handle_check_clause("timeout")
    check_persist_final = handle_check_clause("persist")
    check_freq_final = handle_check_clause("freq")

    @v_args(meta=True, inline=True)
    def check_clause(self, meta, x):
        return x  # pass through

    @v_args(meta=True, inline=True)
    def check_clause_final(self, meta, x):
        return x  # pass through

    check_clauses = no_inline(handle_check_clauses)
    check_stmt = no_inline(handle_check_stmt)

    elifs = no_inline_or_meta(list)
    elif_ = AstElif
    block = no_inline(AstBlock)
    binary_op = AstBinaryOp
    unary_op = AstUnaryOp

    func_call = AstFuncCall
    arguments = no_inline_or_meta(list)
    named_argument = AstNamedArgument

    @v_args(meta=True, inline=True)
    def positional_argument(self, meta, value):
        # Just return the expression directly for positional arguments
        return value

    string = AstString
    number = AstNumber
    boolean = AstBoolean
    name = AstIdent
    get_attr = AstGetAttr
    index_expr = AstIndexExpr
    range = AstRange

    def_stmt = AstDef
    parameter = no_inline(handle_parameter)
    parameters = no_inline_or_meta(list)  # Just convert to list
    return_stmt = AstReturn

    NAME = lambda self, token: token[1:] if token.startswith('$') else token
    DEC_NUMBER = int
    FLOAT_NUMBER = Decimal
    HEX_NUMBER = lambda self, token: int(token, 16)
    COMPARISON_OP = str
    RANGE_OP = str
    STRING = handle_str
    CONST_TRUE = lambda a, b: True
    CONST_FALSE = lambda a, b: False
    ADD_OP: str
    SUB_OP: str
    DIV_OP: str
    MUL_OP: str
    FLOOR_DIV_OP: str
    MOD_OP: str
    POW_OP: str
