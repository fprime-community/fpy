from fpy.model import DirectiveErrorCode
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_failure,
    assert_run_success,
)

def test_assert(fprime_test_api):
    seq = """
assert True
assert not False
"""

    assert_run_success(fprime_test_api, seq)


def test_assert_failure(fprime_test_api):
    seq = """
assert False
"""

    assert_run_failure(fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR)


def test_assert_failure_with_exit_code(fprime_test_api):
    seq = """
assert False, 123
"""

    assert_run_failure(fprime_test_api, seq, DirectiveErrorCode.EXIT_WITH_ERROR)


def test_assert_wrong_bool_type(fprime_test_api):
    seq = """
assert 123
"""

    assert_compile_failure(fprime_test_api, seq)


def test_assert_wrong_exit_code_type(fprime_test_api):
    seq = """
assert True, True
"""

    assert_compile_failure(fprime_test_api, seq)
