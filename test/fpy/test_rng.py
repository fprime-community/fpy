import pytest

from fpy.test_helpers import assert_run_success


class TestRng:

    def test_rand(self, fprime_test_api):
        seq = """
value: U32 = rand()
assert value >= 0
"""
        assert_run_success(fprime_test_api, seq)

    def test_randf(self, fprime_test_api):
        seq = """
value: F64 = randf()
assert value >= 0.0
assert value < 1.0
"""
        assert_run_success(fprime_test_api, seq)

    @pytest.mark.skipif("config.getoption('--use-gds')", reason="Python model PRNG expectation only")
    def test_rand_seeded_sequence(self, fprime_test_api):
        seq = """
set_seed(123456789)
assert rand() == 2754794679
assert rand() == 1899526012
set_seed(123456789)
assert rand() == 2754794679
"""
        assert_run_success(fprime_test_api, seq)

    @pytest.mark.skipif("not config.getoption('--use-gds')", reason="requires live GDS C++ PRNG implementation")
    def test_rand_seeded_sequence_gds(self, fprime_test_api):
        seq = """
set_seed(123456789)
assert rand() == 2288500408
assert rand() == 4254805660
set_seed(123456789)
assert rand() == 2288500408
"""
        assert_run_success(fprime_test_api, seq)

    @pytest.mark.skipif("config.getoption('--use-gds')", reason="Python model PRNG expectation only")
    def test_randf_seeded_sequence(self, fprime_test_api):
        seq = """
set_seed(123456789)
assert randf() == 0.6414006182458252
assert randf() == 0.4422678640112281
set_seed(123456789)
assert randf() == 0.6414006182458252
"""
        assert_run_success(fprime_test_api, seq)

    @pytest.mark.skipif("not config.getoption('--use-gds')", reason="requires live GDS C++ PRNG implementation")
    def test_randf_seeded_sequence_gds(self, fprime_test_api):
        seq = """
set_seed(123456789)
assert randf() == 0.5328330229967833
assert randf() == 0.9906491404399276
set_seed(123456789)
assert randf() == 0.5328330229967833
"""
        assert_run_success(fprime_test_api, seq)

    @pytest.mark.skipif("config.getoption('--use-gds')", reason="Python model initial_time_us expectation only")
    def test_rand_uses_time_as_initial_seed(self, fprime_test_api):
        seq = """
assert rand() == 1309080412
"""
        assert_run_success(fprime_test_api, seq, initial_time_us=5_000_000)

    @pytest.mark.skipif("config.getoption('--use-gds')", reason="Python model initial_time_us expectation only")
    def test_set_seed_overrides_time_initialized_rng(self, fprime_test_api):
        seq = """
ignored: U32 = rand()
set_seed(123456789)
assert rand() == 2754794679
"""
        assert_run_success(fprime_test_api, seq, initial_time_us=5_000_000)

    @pytest.mark.skipif("not config.getoption('--use-gds')", reason="requires live GDS C++ PRNG implementation")
    def test_set_seed_overrides_time_initialized_rng_gds(self, fprime_test_api):
        seq = """
ignored: U32 = rand()
set_seed(123456789)
assert rand() == 2288500408
"""
        assert_run_success(fprime_test_api, seq)
