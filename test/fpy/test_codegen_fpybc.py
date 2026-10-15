from pathlib import Path

from fpy.bytecode.directives import (
    IntegerZeroExtend32To64Directive,
    PushValDirective,
    UnsignedIntToFloatDirective,
)
from fpy.compiler import analyze_ast, analysis_to_fpybc_directives, text_to_ast
from fpy.state import get_base_compile_state


def test_expression_directives_keep_their_source_nodes():
    dictionary = Path(__file__).with_name("RefTopologyDictionary.json")
    state = analyze_ast(
        text_to_ast("x: U32 = 1\ny: F64 = x + 1\nz: F64 = 2 + 2\n"),
        get_base_compile_state(str(dictionary)),
    )
    expr = state.main_block.stmts[1].rhs
    folded_expr = state.main_block.stmts[2].rhs
    directives, _ = analysis_to_fpybc_directives(state)

    # Nested operand coercion and outer result coercion must each retain
    # the expression responsible for the instruction's diagnostic location.
    extensions = [
        d for d in directives if isinstance(d, IntegerZeroExtend32To64Directive)
    ]
    conversions = [d for d in directives if isinstance(d, UnsignedIntToFloatDirective)]
    assert len(extensions) == len(conversions) == 1
    assert extensions[0].source_node is expr.lhs
    assert conversions[0].source_node is expr

    # A folded expression emits one push at its already-coerced type.
    folded = [d for d in directives if d.source_node is folded_expr]
    assert len(folded) == 1
    assert isinstance(folded[0], PushValDirective)
    assert folded[0].val == state.const_expr_values[folded_expr].serialize()
