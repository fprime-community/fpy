from fpy.model import DirectiveErrorCode
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_failure,
    assert_run_success,
)


class TestCommandCalls:

    def test_call_cmd(self, fprime_test_api):
        seq = """
CdhCore.cmdDisp.CMD_NO_OP()
"""
        assert_run_success(fprime_test_api, seq)

    def test_call_namespace_fails(self, fprime_test_api):
        seq = """
CdhCore.cmdDisp()
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_call_cmd_with_str_arg(self, fprime_test_api):
        seq = """
CdhCore.cmdDisp.CMD_NO_OP_STRING("hello world")
"""
        assert_run_success(fprime_test_api, seq)

    def test_call_cmd_with_int_arg(self, fprime_test_api):
        seq = """
Ref.sendBuffComp.PARAMETER3_PRM_SET(4)
"""
        assert_run_success(fprime_test_api, seq)

    def test_cmd_with_enum(self, fprime_test_api):
        seq = """
Ref.SG5.Settings(123, 0.5, 0.5, Ref.SignalType.TRIANGLE)
"""
        assert_run_success(fprime_test_api, seq)

    def test_instantiate_type_for_cmd(self, fprime_test_api):
        seq = """
Ref.typeDemo.CHOICE_PAIR(Ref.ChoicePair(Ref.Choice.ONE, Ref.Choice.TWO))
"""
        assert_run_success(fprime_test_api, seq)

    def test_cmd_return_val(self, fprime_test_api):
        seq = """
ret: Fw.CmdResponse = CdhCore.cmdDisp.CMD_NO_OP()
if ret == Fw.CmdResponse.OK:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_too_many_dirs(self, fprime_test_api):
        from fpy.types import MAX_DIRECTIVES_COUNT

        seq = "CdhCore.cmdDisp.CMD_NO_OP()\n" * (MAX_DIRECTIVES_COUNT + 1)
        assert_compile_failure(fprime_test_api, seq)

    def test_dir_too_large(self, fprime_test_api):
        # TODO this doesn't actually crash cuz the dir is too large... not sure at the moment how to trigger this
        from fpy.types import MAX_DIRECTIVE_SIZE

        seq = 'CdhCore.cmdDisp.CMD_NO_OP_STRING("' + "a" * MAX_DIRECTIVE_SIZE + '")'
        assert_compile_failure(fprime_test_api, seq)

    def test_multi_arg_variable_arg_cmd(self, fprime_test_api):
        seq = """
var1: I32 = 1
var2: F32 = 1.0
var3: U8 = 8
CdhCore.cmdDisp.CMD_TEST_CMD_1(var1, var2, var3)
"""
        assert_run_success(fprime_test_api, seq)

    def test_weird_arg_type(self, fprime_test_api):
        seq = """
CdhCore.cmdDisp.CMD_NO_OP_STRING(CdhCore.cmdDisp.CMD_NO_OP_STRING)
"""
        assert_compile_failure(fprime_test_api, seq)

class TestCommandArguments:

    def test_non_const_str_arg(self, fprime_test_api):
        seq = """
CdhCore.cmdDisp.CMD_NO_OP_STRING(Ref.cmdSeq.SeqPath)
"""
        # currently can't do non const string args
        assert_compile_failure(fprime_test_api, seq)

    def test_non_const_int_arg(self, fprime_test_api):
        seq = """
var: U8 = 255
Ref.sendBuffComp.PARAMETER3_PRM_SET(var)
"""
        assert_run_success(fprime_test_api, seq)

    def test_non_const_float_arg(self, fprime_test_api):
        seq = """
var: F32 = 1.2
Ref.sendBuffComp.PARAMETER4_PRM_SET(var)
"""
        assert_run_success(fprime_test_api, seq)

    def test_non_const_builtin_arg(self, fprime_test_api):
        seq = """
var: U32 = 1
var2: U32 = 123123
sleep(var, var2)
"""
        assert_run_success(fprime_test_api, seq)

    def test_math_after_cmd(self, fprime_test_api):
        seq = """
var: I32 = 1
CdhCore.cmdDisp.CMD_NO_OP()
# making sure that the cmd doesn't mess with the stack
if var + 1 == 2:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_named_arg_cmd_call(self, fprime_test_api):
        """Named arguments work with command calls."""
        seq = """
CdhCore.cmdDisp.CMD_TEST_CMD_1(arg1=1, arg2=1.0, arg3=1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_named_arg_cmd_call_reorder(self, fprime_test_api):
        """Named arguments can reorder command arguments."""
        seq = """
CdhCore.cmdDisp.CMD_TEST_CMD_1(arg3=1, arg1=1, arg2=1.0)
"""

        assert_run_success(fprime_test_api, seq)

    def test_named_arg_coercion_in_cmd(self, fprime_test_api):
        """Named arguments with coercion in command calls."""
        seq = """
# arg1 is I32, arg2 is F32, arg3 is U8
# Test coercion from narrower to wider types
val1: I16 = 1  # I16 -> I32
val2: I16 = 2  # I16 -> F32
val3: U8 = 3   # U8 -> U8 (exact match)
CdhCore.cmdDisp.CMD_TEST_CMD_1(arg1=val1, arg2=val2, arg3=val3)
"""

        assert_run_success(fprime_test_api, seq)

class TestNamespaces:

    def test_get_item_of_namespace(self, fprime_test_api):
        seq = """
value: U32 = CdhCore.cmdDisp[0]
"""
        assert_compile_failure(fprime_test_api, seq)

class TestFlags:

    def test_set_flag_basic(self, fprime_test_api):
        """Setting flags.EXIT_ON_CMD_FAIL to True should succeed."""
        seq = """
flags.EXIT_ON_CMD_FAIL = True
"""
        assert_run_success(fprime_test_api, seq)

    def test_set_flag_false(self, fprime_test_api):
        """Setting flags.EXIT_ON_CMD_FAIL to False should succeed."""
        seq = """
flags.EXIT_ON_CMD_FAIL = False
"""
        assert_run_success(fprime_test_api, seq)

    def test_set_and_get_flag(self, fprime_test_api):
        """Setting and reading flags.EXIT_ON_CMD_FAIL should work."""
        seq = """
flags.EXIT_ON_CMD_FAIL = True
assert flags.EXIT_ON_CMD_FAIL == True
"""
        assert_run_success(fprime_test_api, seq)

    def test_set_flag_toggle(self, fprime_test_api):
        """Setting a flag to True then False should work."""
        seq = """
flags.EXIT_ON_CMD_FAIL = True
assert flags.EXIT_ON_CMD_FAIL == True
flags.EXIT_ON_CMD_FAIL = False
assert flags.EXIT_ON_CMD_FAIL == False
"""
        assert_run_success(fprime_test_api, seq)

    def test_set_flag_dynamic_value(self, fprime_test_api):
        """flags.EXIT_ON_CMD_FAIL should accept a runtime bool expression."""
        seq = """
x: bool = True
flags.EXIT_ON_CMD_FAIL = x
assert flags.EXIT_ON_CMD_FAIL == True
"""
        assert_run_success(fprime_test_api, seq)

    def test_get_flag_in_expression(self, fprime_test_api):
        """flags.EXIT_ON_CMD_FAIL should be usable in boolean expressions."""
        seq = """
flags.EXIT_ON_CMD_FAIL = True
x: bool = flags.EXIT_ON_CMD_FAIL and True
assert x == True
"""
        assert_run_success(fprime_test_api, seq)

    def test_get_flag_assign_to_var(self, fprime_test_api):
        """flags.EXIT_ON_CMD_FAIL can be assigned to a bool variable."""
        seq = """
flags.EXIT_ON_CMD_FAIL = False
flag_val: bool = flags.EXIT_ON_CMD_FAIL
assert flag_val == False
flags.EXIT_ON_CMD_FAIL = True
flag_val = flags.EXIT_ON_CMD_FAIL
assert flag_val == True
"""
        assert_run_success(fprime_test_api, seq)

    def test_flag_default_is_true(self, fprime_test_api):
        """flags.EXIT_ON_CMD_FAIL defaults to True."""
        seq = """
assert flags.EXIT_ON_CMD_FAIL == True
"""
        assert_run_success(fprime_test_api, seq)

class TestExitOnCmdFail:
    """Tests the EXIT_ON_CMD_FAIL flag across all 4 cases:
    (unhandled/handled) x (flag=True/flag=False), for both failing and successful commands.
    """

    # -- Unhandled (bare) command + failing --

    def test_unhandled_fail_flag_true_exits(self, fprime_test_api):
        """Bare failing command with flag=True should exit with error."""
        seq = """
flags.EXIT_ON_CMD_FAIL = True
Ref.cmdSeq.RUN("", Svc.FpySequencer.BlockState.NO_BLOCK)
"""
        assert_run_failure(
            fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR,
        )

    def test_unhandled_fail_flag_false_continues(self, fprime_test_api):
        """Bare failing command with flag=False should not halt."""
        seq = """
flags.EXIT_ON_CMD_FAIL = False
Ref.cmdSeq.RUN("", Svc.FpySequencer.BlockState.NO_BLOCK)
"""
        assert_run_success(fprime_test_api, seq)

    # -- Handled (captured) command + failing --

    def test_handled_fail_flag_true_continues(self, fprime_test_api):
        """Captured failing command with flag=True should NOT auto-assert."""
        seq = """
flags.EXIT_ON_CMD_FAIL = True
resp: Fw.CmdResponse = Ref.cmdSeq.RUN("test", Svc.FpySequencer.BlockState.NO_BLOCK)
assert resp == Fw.CmdResponse.EXECUTION_ERROR
"""
        assert_run_success(fprime_test_api, seq)

    def test_handled_fail_flag_false_continues(self, fprime_test_api):
        """Captured failing command with flag=False should not halt."""
        seq = """
flags.EXIT_ON_CMD_FAIL = False
resp: Fw.CmdResponse = Ref.cmdSeq.RUN("test", Svc.FpySequencer.BlockState.NO_BLOCK)
assert resp == Fw.CmdResponse.EXECUTION_ERROR
"""
        assert_run_success(fprime_test_api, seq)

    # -- Successful commands (should always pass regardless of flag/handling) --

    def test_unhandled_success_flag_true(self, fprime_test_api):
        """Bare successful command with flag=True should pass."""
        seq = """
flags.EXIT_ON_CMD_FAIL = True
CdhCore.cmdDisp.CMD_NO_OP()
"""
        assert_run_success(fprime_test_api, seq)

    def test_unhandled_success_flag_false(self, fprime_test_api):
        """Bare successful command with flag=False should pass."""
        seq = """
flags.EXIT_ON_CMD_FAIL = False
CdhCore.cmdDisp.CMD_NO_OP()
"""
        assert_run_success(fprime_test_api, seq)

    # -- Edge cases --

    def test_toggle_off_before_cmd(self, fprime_test_api):
        """Setting EXIT_ON_CMD_FAIL then unsetting it before a failing cmd should not exit."""
        seq = """
flags.EXIT_ON_CMD_FAIL = True
flags.EXIT_ON_CMD_FAIL = False
Ref.cmdSeq.RUN("", Svc.FpySequencer.BlockState.NO_BLOCK)
"""
        assert_run_success(fprime_test_api, seq)

    # -- Bare commands in nested scopes (flag=True) --

    def test_bare_cmd_in_if_block(self, fprime_test_api):
        """Auto-assert fires for bare commands inside if blocks."""
        seq = """
flags.EXIT_ON_CMD_FAIL = True
if True:
    Ref.cmdSeq.RUN("", Svc.FpySequencer.BlockState.NO_BLOCK)
"""
        assert_run_failure(
            fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR,
        )

    def test_bare_cmd_in_while_block(self, fprime_test_api):
        """Auto-assert fires for bare commands inside while loops."""
        seq = """
flags.EXIT_ON_CMD_FAIL = True
x: bool = True
while x:
    Ref.cmdSeq.RUN("", Svc.FpySequencer.BlockState.NO_BLOCK)
    x = False
"""
        assert_run_failure(
            fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR,
        )

    def test_bare_cmd_in_function(self, fprime_test_api):
        """Auto-assert fires for bare commands inside functions."""
        seq = """
flags.EXIT_ON_CMD_FAIL = True
def do_cmd():
    Ref.cmdSeq.RUN("", Svc.FpySequencer.BlockState.NO_BLOCK)
do_cmd()
"""
        assert_run_failure(
            fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR,
        )

    def test_bare_cmd_in_for_loop(self, fprime_test_api):
        """Auto-assert fires for bare commands inside for loops."""
        seq = """
flags.EXIT_ON_CMD_FAIL = True
for i in 0 .. 1:
    Ref.cmdSeq.RUN("", Svc.FpySequencer.BlockState.NO_BLOCK)
"""
        assert_run_failure(
            fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR,
        )
