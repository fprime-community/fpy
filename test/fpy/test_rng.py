import pytest

from fpy.test_helpers import assert_run_success


class TestRng:

    @pytest.mark.skipif("config.getoption('--use-gds')", reason="local stub expectation only")
    def test_rng(self, fprime_test_api):
        seq = """
value: U32 = rng()
assert value == 1
"""
        assert_run_success(fprime_test_api, seq)
