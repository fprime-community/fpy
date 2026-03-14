from fpy.types import FpyValue, U32

from fpy.model import DirectiveErrorCode
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_failure,
    assert_run_success,
    lookup_type,
)

def test_call_cmd(fprime_test_api):
    seq = """
CdhCore.cmdDisp.CMD_NO_OP()
"""
    assert_run_success(fprime_test_api, seq)


def test_call_namespace_fails(fprime_test_api):
    seq = """
CdhCore.cmdDisp()
"""

    assert_compile_failure(fprime_test_api, seq)


def test_call_cmd_with_str_arg(fprime_test_api):
    seq = """
CdhCore.cmdDisp.CMD_NO_OP_STRING("hello world")
"""
    assert_run_success(fprime_test_api, seq)


def test_call_cmd_with_int_arg(fprime_test_api):
    seq = """
Ref.sendBuffComp.PARAMETER3_PRM_SET(4)
"""
    assert_run_success(fprime_test_api, seq)


def test_cmd_with_enum(fprime_test_api):
    seq = """
Ref.SG5.Settings(123, 0.5, 0.5, Ref.SignalType.TRIANGLE)
"""
    assert_run_success(fprime_test_api, seq)


def test_instantiate_type_for_cmd(fprime_test_api):
    seq = """
Ref.typeDemo.CHOICE_PAIR(Ref.ChoicePair(Ref.Choice.ONE, Ref.Choice.TWO))
"""
    assert_run_success(fprime_test_api, seq)


def test_geq_tlm(fprime_test_api):
    seq = """
CdhCore.cmdDisp.CMD_NO_OP()
# NOTE! this is not guaranteed to work, if the tlm gets written
# too slowly to the DB then this will fail
if CdhCore.cmdDisp.CommandsDispatched >= 1:
    exit(0)
exit(1)
"""

    assert_run_success(
        fprime_test_api,
        seq,
        {"CdhCore.cmdDisp.CommandsDispatched": FpyValue(U32, 1).serialize()},
    )


def test_get_struct_member_of_tlm(fprime_test_api):
    seq = """
Ref.typeDemo.CHOICE_PAIR(Ref.ChoicePair(Ref.Choice.ONE, Ref.Choice.ONE))
if Ref.typeDemo.ChoicePairCh.firstChoice == Ref.Choice.ONE:
    exit(0)
exit(1)
"""

    assert_run_success(
        fprime_test_api,
        seq,
        {
            "Ref.typeDemo.ChoicePairCh": FpyValue(lookup_type(fprime_test_api, "Ref.ChoicePair"),
                {"firstChoice": "ONE", "secondChoice": "ONE"}
            ).serialize()
        },
    )


def test_non_const_str_arg(fprime_test_api):
    seq = """
CdhCore.cmdDisp.CMD_NO_OP_STRING(Ref.cmdSeq.SeqPath)
"""
    # currently can't do non const string args
    assert_compile_failure(fprime_test_api, seq)


def test_non_const_int_arg(fprime_test_api):
    seq = """
var: U8 = 255
Ref.sendBuffComp.PARAMETER3_PRM_SET(var)
"""
    assert_run_success(fprime_test_api, seq)


def test_non_const_float_arg(fprime_test_api):
    seq = """
var: F32 = 1.2
Ref.sendBuffComp.PARAMETER4_PRM_SET(var)
"""
    assert_run_success(fprime_test_api, seq)


def test_non_const_builtin_arg(fprime_test_api):
    seq = """
var: U32 = 1
var2: U32 = 123123
sleep(var, var2)
"""
    assert_run_success(fprime_test_api, seq)


def test_math_after_cmd(fprime_test_api):
    seq = """
var: I32 = 1
CdhCore.cmdDisp.CMD_NO_OP()
# making sure that the cmd doesn't mess with the stack
if var + 1 == 2:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_cmd_return_val(fprime_test_api):
    seq = """
ret: Fw.CmdResponse = CdhCore.cmdDisp.CMD_NO_OP()
if ret == Fw.CmdResponse.OK:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_too_many_dirs(fprime_test_api):
    from fpy.types import MAX_DIRECTIVES_COUNT

    seq = "CdhCore.cmdDisp.CMD_NO_OP()\n" * (MAX_DIRECTIVES_COUNT + 1)
    assert_compile_failure(fprime_test_api, seq)


def test_dir_too_large(fprime_test_api):
    # TODO this doesn't actually crash cuz the dir is too large... not sure at the moment how to trigger this
    from fpy.types import MAX_DIRECTIVE_SIZE

    seq = 'CdhCore.cmdDisp.CMD_NO_OP_STRING("' + "a" * MAX_DIRECTIVE_SIZE + '")'
    assert_compile_failure(fprime_test_api, seq)


def test_multi_arg_variable_arg_cmd(fprime_test_api):
    seq = """
var1: I32 = 1
var2: F32 = 1.0
var3: U8 = 8
CdhCore.cmdDisp.CMD_TEST_CMD_1(var1, var2, var3)
"""
    assert_run_success(fprime_test_api, seq)


def test_weird_arg_type(fprime_test_api):
    seq = """
CdhCore.cmdDisp.CMD_NO_OP_STRING(CdhCore.cmdDisp.CMD_NO_OP_STRING)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_assign_tlm_struct_member_bad(fprime_test_api):
    seq = """
Ref.cmdSeq.Debug.nextStatementOpcode = 0
"""

    assert_compile_failure(fprime_test_api, seq)


def test_calling_namespace_should_fail_gracefully(fprime_test_api):
    seq = """
Ref.typeDemo()
"""

    assert_compile_failure(fprime_test_api, seq)


def test_func_call_a_namespace(fprime_test_api):
    seq = """
CdhCore.cmdDisp()
"""
    assert_compile_failure(fprime_test_api, seq)


def test_get_item_of_namespace(fprime_test_api):
    seq = """
value: U32 = CdhCore.cmdDisp[0]
"""
    assert_compile_failure(fprime_test_api, seq)


def test_named_arg_cmd_call(fprime_test_api):
    """Named arguments work with command calls."""
    seq = """
CdhCore.cmdDisp.CMD_TEST_CMD_1(arg1=1, arg2=1.0, arg3=1)
"""

    assert_run_success(fprime_test_api, seq)


def test_named_arg_cmd_call_reorder(fprime_test_api):
    """Named arguments can reorder command arguments."""
    seq = """
CdhCore.cmdDisp.CMD_TEST_CMD_1(arg3=1, arg1=1, arg2=1.0)
"""

    assert_run_success(fprime_test_api, seq)


def test_named_arg_coercion_in_cmd(fprime_test_api):
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


def test_set_flag_basic(fprime_test_api):
    """set_flag with a FlagId enum constant and True value should succeed."""
    seq = """
set_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL, True)
"""
    assert_run_success(fprime_test_api, seq)


def test_set_flag_false(fprime_test_api):
    """set_flag with False value should succeed."""
    seq = """
set_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL, False)
"""
    assert_run_success(fprime_test_api, seq)


def test_set_and_get_flag(fprime_test_api):
    """set_flag followed by get_flag should return the set value."""
    seq = """
set_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL, True)
assert get_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL) == True
"""
    assert_run_success(fprime_test_api, seq)


def test_set_flag_toggle(fprime_test_api):
    """Setting a flag to True then False should work."""
    seq = """
set_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL, True)
assert get_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL) == True
set_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL, False)
assert get_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL) == False
"""
    assert_run_success(fprime_test_api, seq)


def test_set_flag_dynamic_value(fprime_test_api):
    """set_flag should accept a runtime bool expression for value."""
    seq = """
x: bool = True
set_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL, x)
assert get_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL) == True
"""
    assert_run_success(fprime_test_api, seq)


def test_get_flag_in_expression(fprime_test_api):
    """get_flag result should be usable in boolean expressions."""
    seq = """
set_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL, True)
x: bool = get_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL) and True
assert x == True
"""
    assert_run_success(fprime_test_api, seq)


def test_set_flag_wrong_type(fprime_test_api):
    """set_flag with an integer instead of FlagId should fail compilation."""
    seq = """
set_flag(0, True)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_get_flag_wrong_type(fprime_test_api):
    """get_flag with an integer instead of FlagId should fail compilation."""
    seq = """
get_flag(0)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_set_flag_non_const_index(fprime_test_api):
    """set_flag with a non-constant flag index should fail compilation."""
    seq = """
x: Svc.Fpy.FlagId = Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL
set_flag(x, True)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_get_flag_non_const_index(fprime_test_api):
    """get_flag with a non-constant flag index should fail compilation."""
    seq = """
x: Svc.Fpy.FlagId = Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL
get_flag(x)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_get_flag_assign_to_var(fprime_test_api):
    """get_flag result can be assigned to a bool variable."""
    seq = """
set_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL, False)
flag_val: bool = get_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL)
assert flag_val == False
set_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL, True)
flag_val = get_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL)
assert flag_val == True
"""
    assert_run_success(fprime_test_api, seq)


def test_exit_on_cmd_fail_flag_causes_exit(fprime_test_api):
    """When EXIT_ON_CMD_FAIL is set and a command fails, the sequence should exit with error."""
    seq = """
set_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL, True)
Ref.cmdSeq.RUN("", Svc.FpySequencer.BlockState.NO_BLOCK)
"""
    assert_run_failure(
        fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR,
    )


def test_no_exit_on_cmd_fail_flag_allows_failure(fprime_test_api):
    """When EXIT_ON_CMD_FAIL is explicitly off, a failing command should not halt the sequence."""
    seq = """
set_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL, False)
resp: Fw.CmdResponse = Ref.cmdSeq.RUN("test", Svc.FpySequencer.BlockState.NO_BLOCK)
assert resp == Fw.CmdResponse.EXECUTION_ERROR
"""
    assert_run_success(fprime_test_api, seq)


def test_exit_on_cmd_fail_with_successful_cmd(fprime_test_api):
    """When EXIT_ON_CMD_FAIL is set but the command succeeds, no exit should occur."""
    seq = """
set_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL, True)
CdhCore.cmdDisp.CMD_NO_OP()
"""
    assert_run_success(fprime_test_api, seq)


def test_exit_on_cmd_fail_toggle_off_before_cmd(fprime_test_api):
    """Setting EXIT_ON_CMD_FAIL then unsetting it before a failing cmd should not exit."""
    seq = """
set_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL, True)
set_flag(Svc.Fpy.FlagId.EXIT_ON_CMD_FAIL, False)
Ref.cmdSeq.RUN("", Svc.FpySequencer.BlockState.NO_BLOCK)
"""
    assert_run_success(fprime_test_api, seq)
