from fpy.model import DirectiveErrorCode
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_failure,
    assert_run_success,
)


class TestAssert:

    def test_assert(self, fprime_test_api):
        seq = """
assert True
assert not False
"""

        assert_run_success(fprime_test_api, seq)

    def test_assert_failure(self, fprime_test_api):
        seq = """
assert False
"""

        assert_run_failure(fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR)

    def test_assert_failure_with_exit_code(self, fprime_test_api):
        seq = """
assert False, 123
"""

        assert_run_failure(fprime_test_api, seq, 123)

    def test_assert_wrong_bool_type(self, fprime_test_api):
        seq = """
assert 123
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_assert_wrong_exit_code_type(self, fprime_test_api):
        seq = """
assert True, True
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_exit_code_does_not_impersonate_fault(self, fprime_test_api):
        """User exits and runtime faults are separate channels: exit(10)
        matches the raw code 10, but must NOT satisfy an expected
        DOMAIN_ERROR fault even though DOMAIN_ERROR's value is also 10."""
        if fprime_test_api is not None:
            return  # GDS mode reports failures via events, not channels
        assert_run_failure(fprime_test_api, "exit(10)", 10)
        try:
            assert_run_failure(
                fprime_test_api, "exit(10)", DirectiveErrorCode.DOMAIN_ERROR
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("exit(10) was accepted as a DOMAIN_ERROR fault")
