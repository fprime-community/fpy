from fpy.types import U32

from fpy.test_helpers import assert_compile_failure, assert_run_success, compile_seq
from fpy.error import WarningType


class TestDeclaration:

    def test_simple_var(self, fprime_test_api):
        seq = """
var: U32 = 1
"""

        assert_run_success(fprime_test_api, seq)

    def test_var_escaped_reserved_word(self, fprime_test_api):
        # $ prefix can be used to escape reserved words like 'def', 'while', etc.
        seq = """
$def: U32 = 1
$while: U32 = 2
$if: U32 = 3
"""

        assert_run_success(fprime_test_api, seq)

    def test_large_var(self, fprime_test_api):
        seq = """
var: Svc.DpRecord = Svc.DpRecord(0, 1, 2, 3, 4, 5, Fw.DpState.UNTRANSMITTED)
"""

        assert_run_success(fprime_test_api, seq)

    def test_var_assign_to_var(self, fprime_test_api):
        seq = """
x: U32 = 1
var: U32 = x
"""

        assert_run_success(fprime_test_api, seq)

    def test_nonexistent_var(self, fprime_test_api):
        seq = """
var = 1
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_module_type_annotation_fails(self, fprime_test_api):
        seq = """
var: Svc = 1
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_var_type_ann_bad(self, fprime_test_api):
        seq = """
var: Fw.Time.asdf = 0
"""

        assert_compile_failure(fprime_test_api, seq)


class TestAssignment:

    def test_create_after_assign_var(self, fprime_test_api):
        seq = """
var = 1
var: U32 = 2
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_bad_assign_type(self, fprime_test_api):
        seq = """
var: failure = 1
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_weird_assign_type(self, fprime_test_api):
        seq = """
var: CdhCore.cmdDisp.CMD_NO_OP = 1
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_reassign(self, fprime_test_api):
        seq = """
var: U32 = 1
var = 2
"""

        assert_run_success(fprime_test_api, seq)

    def test_reassign_ann(self, fprime_test_api):
        seq = """
var: U32 = 1
var: U32 = 2
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_assign_function_value(self, fprime_test_api):
        seq = """
var: U32 = CdhCore.cmdDisp.CMD_NO_OP
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_assign_float_to_int(self, fprime_test_api):
        seq = """
val: I64 = 1.0
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_negative_val_unsigned_type(self, fprime_test_api):
        seq = """
val1: U32 = -1
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_assign_complex(self, fprime_test_api):
        seq = """
var: I64 = 1 + 1
var = var + 3
if var == 5:
    exit(0)
exit(1)
"""
        assert_run_success(fprime_test_api, seq)

    def test_assign_cycle(self, fprime_test_api):
        seq = """
var: I64 = var
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_assign_cycle_2(self, fprime_test_api):
        seq = """
var: I64 = (var + 1)
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_assign_bad_lhs_1(self, fprime_test_api):
        seq = """
Svc.ComQueueDepth = 55
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_assign_bad_lhs_2(self, fprime_test_api):
        seq = """
CdhCore.cmdDisp.CMD_NO_OP = 55
"""
        assert_compile_failure(fprime_test_api, seq)


class TestScoping:

    def test_use_before_declare(self, fprime_test_api):
        seq = """
var: I64 = var2
var2: I64 = 0
"""
        assert_compile_failure(fprime_test_api, seq)

    def test_scope_override_name(self, fprime_test_api):
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

    def test_override_global_name(self, fprime_test_api):
        # Shadowing a dictionary name with a user variable is allowed, but warns
        # (shadow-value). See TestShadowWarnings below.
        seq = """
CdhCore: U8 = 1
if CdhCore == 1:
    exit(0)
exit(1)
"""

        assert_run_success(fprime_test_api, seq)

    def test_redeclare_after_scope(self, fprime_test_api):
        seq = """
for i in 0 .. 7:
    pass
i: U16 = 0
assert i == 0
"""

        assert_run_success(fprime_test_api, seq)

    def test_redeclare_after_for_in_if(self, fprime_test_api):
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

    def test_redeclare_in_nested_scopes(self, fprime_test_api):
        seq = """
z: U8 = 123
for i in 0 .. 7:
    for z in 0 .. 7:
        assert z < 8
assert z == 123
"""

        assert_run_success(fprime_test_api, seq)


class TestShadowWarnings:
    """Declaring a variable whose name already resolves in an ENCLOSING scope
    (an outer block, the global scope, or the builtin/dictionary base scope) is
    allowed but emits the `shadow-value` warning (never `shadow-callable`), and
    still compiles and runs. Redefining a name already in the SAME scope stays a
    hard error."""

    def test_inner_block_shadows_outer_variable_warns(self, fprime_test_api):
        # `i` in the while body shadows the global `i`.
        seq = """
i: U8 = 0
while True:
    i: U8 = 1
    exit(0)
"""
        state, _, _ = compile_seq(seq)
        types = {w.type for w in state.warnings}
        assert WarningType.SHADOW_VALUE in types, f"expected shadow-value: {types}"
        assert WarningType.SHADOW_CALLABLE not in types, f"unexpected: {types}"
        # The warning is non-fatal: the sequence still compiles and runs.
        assert_run_success(fprime_test_api, seq)

    def test_root_variable_shadows_dictionary_name_warns(self, fprime_test_api):
        # A top-level variable named after a dictionary namespace shadows the
        # base (dictionary) name -- a warning, not an error.
        seq = """
CdhCore: U8 = 1
exit(0)
"""
        state, _, _ = compile_seq(seq)
        types = {w.type for w in state.warnings}
        assert WarningType.SHADOW_VALUE in types, f"expected shadow-value: {types}"
        assert_run_success(fprime_test_api, seq)

    def test_same_scope_variable_redeclare_is_error(self, fprime_test_api):
        # Redefining a variable in the SAME scope is not shadowing -- it stays a
        # hard error, with no warning fallback.
        seq = """
x: U8 = 0
x: U8 = 1
"""
        assert_compile_failure(fprime_test_api, seq, match="already been defined")

    def test_no_shadow_warning_for_unique_names(self, fprime_test_api):
        seq = """
a: U8 = 0
while True:
    b: U8 = 1
    exit(0)
"""
        state, _, _ = compile_seq(seq)
        assert not any(
            w.type in (WarningType.SHADOW_VALUE, WarningType.SHADOW_CALLABLE)
            for w in state.warnings
        ), f"unique names should not warn, got {state.warnings}"
        assert_run_success(fprime_test_api, seq)
