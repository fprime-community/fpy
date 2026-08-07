"""Tests for the random number directives.

The seeded expectations are the sequencer's std::mt19937 output. They are
exact: the standard specifies the generator, so the same seed gives the same
values on every standard library.
"""

from fpy.test_helpers import assert_run_success


class TestRng:

    def test_rand(self):
        seq = """
value: U32 = rand()
assert value >= 0
"""
        assert_run_success(seq)

    def test_randf(self):
        seq = """
value: F64 = randf()
assert value >= 0.0
assert value < 1.0
"""
        assert_run_success(seq)

    def test_rand_seeded_sequence(self):
        seq = """
set_seed(123456789)
assert rand() == 2288500408
assert rand() == 4254805660
set_seed(123456789)
assert rand() == 2288500408
"""
        assert_run_success(seq)

    def test_randf_seeded_sequence(self):
        seq = """
set_seed(123456789)
assert randf() == 0.5328330229967833
assert randf() == 0.9906491404399276
set_seed(123456789)
assert randf() == 0.5328330229967833
"""
        assert_run_success(seq)

    def test_rand_uses_time_as_initial_seed(self):
        # Unseeded, the generator is seeded from the current time, so the value
        # follows from the clock the sequence starts with.
        seq = """
assert rand() == 3962384540
"""
        assert_run_success(seq, initial_time_us=5_000_000)

    def test_set_seed_overrides_time_initialized_rng(self):
        seq = """
ignored: U32 = rand()
set_seed(123456789)
assert rand() == 2288500408
"""
        assert_run_success(seq, initial_time_us=5_000_000)
