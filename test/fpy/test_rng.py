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

    @pytest.mark.skipif("config.getoption('--use-gds')", reason="local stub expectation only")
    def test_set_seed(self, fprime_test_api):
        seq = """
set_seed(123)
assert rng() == 1
"""
        assert_run_success(fprime_test_api, seq)

    @pytest.mark.skipif("not config.getoption('--use-gds')", reason="requires live GDS RNG implementation")
    def test_rng_seeded_sequence_gds(self, fprime_test_api):
        seq = """
set_seed(123456789)
assert rng() == 184
assert rng() == 156
set_seed(123456789)
assert rng() == 184
"""
        assert_run_success(fprime_test_api, seq)
