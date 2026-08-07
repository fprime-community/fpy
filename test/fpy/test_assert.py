from fpy.bytecode.errors import DirectiveErrorCode
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_failure,
    assert_run_success,
)


class TestAssert:

    def test_assert(self):
        seq = """
assert True
assert not False
"""

        assert_run_success(seq)

    def test_assert_failure(self):
        seq = """
assert False
"""

        assert_run_failure(seq, DirectiveErrorCode.EXIT_WITH_ERROR)

    def test_assert_failure_with_exit_code(self):
        seq = """
assert False, 123
"""

        assert_run_failure(seq, 123)

    def test_assert_wrong_bool_type(self):
        seq = """
assert 123
"""

        assert_compile_failure(seq)

    def test_assert_wrong_exit_code_type(self):
        seq = """
assert True, True
"""

        assert_compile_failure(seq)
