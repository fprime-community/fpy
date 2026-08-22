from fpy.syntax import (
    AstAssign,
    AstBinaryOp,
    AstExpr,
    AstFuncCall,
    AstGetAttr,
    AstUnaryOp,
)
from fpy.types import is_instance_compat
from fpy.test_helpers import assert_run_success, compile_seq


def _main_stmts(seq: str):
    state, _, _ = compile_seq(seq)
    return state.main_block.stmts


def _expr_stmts(stmts):
    return [s for s in stmts if is_instance_compat(s, AstExpr)]


class TestElided:
    def test_pure_exprs_are_deleted(self):
        seq = """
x: U32 = 1
arr: Svc.ComQueueDepth = Svc.ComQueueDepth(456, 123)
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 1, 2)
f: F64 = 2.5
1
x
t.seconds
arr[1]
f + 1.0
f * 2.0 - f
-f
x / 4
x / 0
x ** 2
x // 4
x % 3
f // 4.0
f % 3.0
x < 2 and x != 0
not x == 1
U8(x)
F64(x)
Fw.Time(TimeBase.TB_NONE, 0, x, x)
fabs(f)
fabs(f * 2.0)
1 + 2 * 3
"""
        stmts = _main_stmts(seq)
        assert _expr_stmts(stmts) == []
        assert all(isinstance(s, AstAssign) for s in stmts)

    def test_nested_blocks(self):
        seq = """
x: U32 = 1
def f():
    x
    x == 1
if x == 1:
    x
    x < 2
else:
    x
while x < 1:
    x
for i in 0..3:
    i
"""
        state, _, _ = compile_seq(seq)
        for stmt in state.main_block.stmts:
            # every nested block ends up empty
            for field in ("body", "else_body", "else_block"):
                block = getattr(stmt, field, None)
                if block is not None and hasattr(block, "stmts"):
                    assert _expr_stmts(block.stmts) == []

    def test_runs(self, fprime_test_api):
        seq = """
x: U32 = 1
x == 1
def f():
    x
f()
if x == 1:
    x
else:
    x
while x < 3:
    x
    x = U32(x + 1)
assert x == 3
"""
        assert_run_success(fprime_test_api, seq)


class TestKept:
    def _kept(self, seq: str, expr_type=None):
        exprs = _expr_stmts(_main_stmts(seq))
        assert len(exprs) == 1, exprs
        if expr_type is not None:
            assert isinstance(exprs[0], expr_type)

    def test_command(self):
        self._kept("CdhCore.cmdDisp.CMD_NO_OP()", AstFuncCall)

    def test_script_function(self):
        self._kept("def f():\n    pass\nf()", AstFuncCall)

    def test_impure_builtin(self):
        self._kept("sleep(1)", AstFuncCall)
        self._kept("now()", AstFuncCall)
        self._kept("rand()", AstFuncCall)

    def test_faulting_builtin(self):
        # ln faults on a non-positive operand, iabs on I64 min
        self._kept("f: F64 = 2.5\nln(f)", AstFuncCall)
        self._kept("x: I64 = 2\niabs(x)", AstFuncCall)

    def test_integer_overflow(self):
        # integer + - * and negation fault on I64 overflow
        self._kept("x: U32 = 2\nx + 1", AstBinaryOp)
        self._kept("x: U32 = 2\n1 - x", AstBinaryOp)
        self._kept("x: U32 = 2\nx * x", AstBinaryOp)
        self._kept("x: I8 = 2\n-x", AstUnaryOp)

    def test_pure_builtin_with_impure_arg(self):
        self._kept("fabs(F64(rand()))", AstFuncCall)

    def test_ctor_with_impure_arg(self):
        self._kept("Fw.Time(TimeBase.TB_NONE, 0, rand(), 0)", AstFuncCall)

    def test_telemetry_read(self):
        self._kept("Ref.typeDemo.Float1Ch", AstGetAttr)
        self._kept("Ref.typeDemo.ChoicePairCh.firstChoice", AstGetAttr)

    def test_parameter_read(self):
        self._kept("Ref.typeDemo.CHOICE_PRM", AstGetAttr)

    def test_division_by_non_constant(self):
        self._kept("x: U32 = 2\n1 // x", AstBinaryOp)
        self._kept("x: U32 = 2\n1 % x", AstBinaryOp)
        self._kept("f: F64 = 2.0\n1.0 // f", AstBinaryOp)
        self._kept("f: F64 = 2.0\n1.0 % f", AstBinaryOp)

    def test_division_by_constant_zero(self):
        self._kept("x: U32 = 2\nx // 0", AstBinaryOp)
        self._kept("x: U32 = 2\nx % 0", AstBinaryOp)

    def test_division_by_constant_minus_one(self):
        # I64 min // -1 overflows
        self._kept("x: I64 = 2\nx // -1", AstBinaryOp)

    def test_non_constant_array_index(self):
        self._kept(
            "arr: Svc.ComQueueDepth = Svc.ComQueueDepth(456, 123)\n"
            "i: I64 = 1\narr[i]"
        )

    def test_time_comparison(self):
        # desugars into a comparison of a script function call's result; the
        # function faults on incomparable time bases
        self._kept(
            "a: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 1, 2)\n"
            "b: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 1, 2)\n"
            "a < b",
            AstBinaryOp,
        )

    def test_embedded_effect(self):
        # the effect is nested inside an otherwise pure expression
        self._kept("def f() -> U32:\n    return 1\nf() + 1 == 2", AstBinaryOp)
