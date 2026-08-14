"""Semantic analysis, one class per pass.

The passes run in the order compiler.analyze_ast lists them, so a sequence
reaching a later pass already satisfies every earlier one. Each class covers the
conditions under which its pass reports an error, alongside the shape of program
it accepts.
"""

import pytest

from fpy.test_helpers import (
    assert_compile_failure,
    assert_compile_success,
    assert_run_success,
)


class TestPickTypesAndResolveFields:
    """Resolves attribute and item accesses and gives every expression a type."""

    def test_struct_member_access_resolves(self, fprime_test_api):
        seq = """
x: Ref.PacketStat = Ref.PacketStat(1, 2, Ref.PacketRecvStatus.PACKET_STATE_NO_PACKETS)
assert x.BuffRecv == 1
"""
        assert_run_success(fprime_test_api, seq)

    def test_array_item_access_resolves(self, fprime_test_api):
        seq = """
x: Svc.ComQueueDepth = Svc.ComQueueDepth(10, 20)
assert x[1] == 20
"""
        assert_run_success(fprime_test_api, seq)

    def test_member_access_on_non_struct(self, fprime_test_api):
        assert_compile_failure(
            fprime_test_api,
            "x: U32 = 1\ny: U32 = x.member\n",
            match="is not a struct, cannot access members",
        )

    def test_member_access_on_string_bearing_struct(self, fprime_test_api):
        # A channel is the only way to name a value of a string-bearing type,
        # since such a type cannot be declared.
        assert_compile_failure(
            fprime_test_api,
            "x: U32 = CdhCore.version.CustomVersion01.versionString\n",
            match="is not constant-sized .contains strings., cannot access members",
        )

    def test_index_into_empty_anonymous_array(self, fprime_test_api):
        assert_compile_failure(
            fprime_test_api,
            "x: U32 = [][0]\n",
            match="Cannot index into an empty anonymous array",
        )

    @pytest.mark.parametrize(
        "expression, expected",
        [("-x", "Op - undefined for bool")],
        ids=["negate"],
    )
    def test_unary_operator_undefined_for_type(
        self, fprime_test_api, expression, expected
    ):
        assert_compile_failure(
            fprime_test_api, f"x: bool = True\ny: bool = {expression}\n", match=expected
        )

    def test_assignment_to_non_target(self, fprime_test_api):
        assert_compile_failure(fprime_test_api, "1 = 2\n", match="Invalid assignment")


class TestCalculateConstExprValues:
    """Folds constant expressions, reporting the ones that have no value."""

    def test_constant_arithmetic_folds(self, fprime_test_api):
        assert_run_success(fprime_test_api, "assert 2 ** 10 == 1024\n")

    @pytest.mark.parametrize(
        "literal",
        ["1e39", "3.5e38", "-1e39"],
        ids=str,
    )
    def test_float_literal_out_of_range(self, fprime_test_api, literal):
        assert_compile_failure(
            fprime_test_api,
            f"x: F32 = {literal}\n",
            match="is out of range for type F32",
        )

    def test_domain_error(self, fprime_test_api):
        assert_compile_failure(
            fprime_test_api, "x: F64 = (-1.0) ** 0.5\n", match="Domain error"
        )

    @pytest.mark.parametrize(
        "type_name, expression",
        [("U32", "1 // 0"), ("U32", "1 % 0"), ("F64", "1.0 / 0.0")],
        ids=["floordiv", "mod", "truediv"],
    )
    def test_divide_by_zero(self, fprime_test_api, type_name, expression):
        assert_compile_failure(
            fprime_test_api,
            f"x: {type_name} = {expression}\n",
            match="[Dd]ivide by zero",
        )


class TestCheckSequenceArgs:
    """Checks the sequence() parameter list against the binary format's limits."""

    def test_sequence_arguments_are_accepted(self, fprime_test_api):
        assert_compile_success(fprime_test_api, "sequence(a: U8, b: U32)\n")

    def test_too_many_sequence_arguments(self, fprime_test_api):
        params = ", ".join(f"a{i}: U8" for i in range(256))
        assert_compile_failure(
            fprime_test_api,
            f"sequence({params})\n",
            match="Too many sequence arguments",
        )
