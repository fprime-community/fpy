import pytest

from fpy.test_helpers import assert_run_success

# The sequencer's PRNG is std::mt19937. The expected values below are what it
# produces for the given seeds; they hold on the harness and on a live GDS
# deployment alike, since both run the same flight code.


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

    def test_rand_seeded_sequence(self, fprime_test_api):
        seq = """
set_seed(123456789)
assert rand() == 2288500408
assert rand() == 4254805660
set_seed(123456789)
assert rand() == 2288500408
"""
        assert_run_success(fprime_test_api, seq)

    def test_randf_seeded_sequence(self, fprime_test_api):
        seq = """
set_seed(123456789)
assert randf() == 0.5328330229967833
assert randf() == 0.9906491404399276
set_seed(123456789)
assert randf() == 0.5328330229967833
"""
        assert_run_success(fprime_test_api, seq)

    @pytest.mark.skipif(
        "config.getoption('--use-gds')",
        reason="needs a controlled clock, which a live GDS does not have",
    )
    def test_rand_uses_time_as_initial_seed(self, fprime_test_api):
        # An unseeded PRNG seeds itself from the current time (std::seed_seq
        # over time base, context, seconds, useconds).
        seq = """
assert rand() == 3962384540
"""
        assert_run_success(fprime_test_api, seq, initial_time_us=5_000_000)

    def test_set_seed_overrides_time_initialized_rng(self, fprime_test_api):
        seq = """
ignored: U32 = rand()
set_seed(123456789)
assert rand() == 2288500408
"""
        assert_run_success(fprime_test_api, seq)
