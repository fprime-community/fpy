from fpy.types import U32

from fpy.test_helpers import assert_compile_failure, assert_run_success

def test_simple_var(fprime_test_api):
    seq = """
var: U32 = 1
"""

    assert_run_success(fprime_test_api, seq)


def test_var_escaped_reserved_word(fprime_test_api):
    # $ prefix can be used to escape reserved words like 'def', 'while', etc.
    seq = """
$def: U32 = 1
$while: U32 = 2
$if: U32 = 3
"""

    assert_run_success(fprime_test_api, seq)


def test_large_var(fprime_test_api):
    seq = """
var: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
"""

    assert_run_success(fprime_test_api, seq)


def test_var_assign_to_var(fprime_test_api):
    seq = """
x: U32 = 1
var: U32 = x
"""

    assert_run_success(fprime_test_api, seq)


def test_nonexistent_var(fprime_test_api):
    seq = """
var = 1
"""

    assert_compile_failure(fprime_test_api, seq)


def test_namespace_type_annotation_fails(fprime_test_api):
    seq = """
var: Svc = 1
"""

    assert_compile_failure(fprime_test_api, seq)


def test_create_after_assign_var(fprime_test_api):
    seq = """
var = 1
var: U32 = 2
"""

    assert_compile_failure(fprime_test_api, seq)


def test_bad_assign_type(fprime_test_api):
    seq = """
var: failure = 1
"""

    assert_compile_failure(fprime_test_api, seq)


def test_weird_assign_type(fprime_test_api):
    seq = """
var: CdhCore.cmdDisp.CMD_NO_OP = 1
"""

    assert_compile_failure(fprime_test_api, seq)


def test_reassign(fprime_test_api):
    seq = """
var: U32 = 1
var = 2
"""

    assert_run_success(fprime_test_api, seq)


def test_reassign_ann(fprime_test_api):
    seq = """
var: U32 = 1
var: U32 = 2
"""
    assert_compile_failure(fprime_test_api, seq)


def test_assign_inconsistent_type(fprime_test_api):
    seq = """
var: U32 = 1
var: U16 = 2
"""

    assert_compile_failure(fprime_test_api, seq)


def test_assign_function_value(fprime_test_api):
    seq = """
var: U32 = CdhCore.cmdDisp.CMD_NO_OP
"""

    assert_compile_failure(fprime_test_api, seq)


def test_assign_float_to_int(fprime_test_api):
    seq = """
val: I64 = 1.0
"""

    assert_compile_failure(fprime_test_api, seq)


def test_negative_val_unsigned_type(fprime_test_api):
    seq = """
val1: U32 = -1
"""
    assert_compile_failure(fprime_test_api, seq)


def test_assign_complex(fprime_test_api):
    seq = """
var: I64 = 1 + 1
var = var + 3
if var == 5:
    exit(0)
exit(1)
"""
    assert_run_success(fprime_test_api, seq)


def test_assign_cycle(fprime_test_api):
    seq = """
var: I64 = var
"""
    assert_compile_failure(fprime_test_api, seq)


def test_assign_cycle_2(fprime_test_api):
    seq = """
var: I64 = (var + 1)
"""
    assert_compile_failure(fprime_test_api, seq)


def test_use_before_declare(fprime_test_api):
    seq = """
var: I64 = var2
var2: I64 = 0
"""
    assert_compile_failure(fprime_test_api, seq)


def test_assign_bad_lhs_1(fprime_test_api):
    seq = """
Svc.ComQueueDepth = 55
"""
    assert_compile_failure(fprime_test_api, seq)


def test_assign_bad_lhs_2(fprime_test_api):
    seq = """
CdhCore.cmdDisp.CMD_NO_OP = 55
"""
    assert_compile_failure(fprime_test_api, seq)


def test_scope_override_name(fprime_test_api):
    # With block scoping, each indentation block creates a new scope.
    # So while/if bodies can shadow variables from outer scopes.
    seq = """
i: U8 = 0
while True:
    i: U8 = 1
    if i == 1:
        exit(0)
    exit(1)
"""

    assert_run_success(fprime_test_api, seq)


def test_override_global_name(fprime_test_api):
    # Can't shadow dictionary namespaces with user variables
    seq = """
CdhCore: U8 = 1
if CdhCore == 1:
    exit(0)
exit(1)
"""

    assert_compile_failure(fprime_test_api, seq)


def test_redeclare_after_scope(fprime_test_api):
    seq = """
for i in 0 .. 7:
    pass
i: U16 = 0
assert i == 0
"""

    assert_run_success(fprime_test_api, seq)


def test_redeclare_after_for_in_if(fprime_test_api):
    # For-loop var is scoped to the for body. After the if block, the
    # name is free to be re-declared.
    seq = """
if True:
    for i in 0 .. 7:
        pass
i: I64 = 0
assert i == 0
"""

    assert_run_success(fprime_test_api, seq)


def test_redeclare_in_nested_scopes(fprime_test_api):
    seq = """
z: U8 = 123
for i in 0 .. 7:
    for z in 0 .. 7:
        assert z < 8
assert z == 123
"""

    assert_run_success(fprime_test_api, seq)


def test_var_type_ann_bad(fprime_test_api):
    seq = """
var: Fw.Time.asdf = 0
"""

    assert_compile_failure(fprime_test_api, seq)


def test_var_type_ann_bad_2(fprime_test_api):
    seq = """
var: Svc = 0
"""

    assert_compile_failure(fprime_test_api, seq)
