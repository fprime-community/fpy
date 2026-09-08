from fpy.bytecode.directives import DirectiveErrorCode
from fpy.types import F32, FpyValue
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_failure,
    assert_run_success,
    lookup_type,
)


class TestParameters:

    def test_read_enum_prm(self, fprime_test_api):
        seq = """
if Ref.typeDemo.CHOICE_PRM == Ref.Choice.TWO:
    exit(0)
exit(1)
"""

        assert_run_success(
            fprime_test_api,
            seq,
            prms={
                "Ref.typeDemo.CHOICE_PRM": FpyValue(
                    lookup_type(fprime_test_api, "Ref.Choice"), "TWO"
                ).serialize()
            },
        )

    def test_read_numeric_prm(self, fprime_test_api):
        seq = """
timeout: F32 = Ref.cmdSeq0.STATEMENT_TIMEOUT_SECS
if timeout * 2.0 == 5.0:
    exit(0)
exit(1)
"""

        assert_run_success(
            fprime_test_api,
            seq,
            prms={"Ref.cmdSeq0.STATEMENT_TIMEOUT_SECS": FpyValue(F32, 2.5).serialize()},
        )

    def test_get_struct_member_of_prm(self, fprime_test_api):
        seq = """
if Ref.typeDemo.CHOICE_PAIR_PRM.secondChoice == Ref.Choice.RED:
    exit(0)
exit(1)
"""

        assert_run_success(
            fprime_test_api,
            seq,
            prms={
                "Ref.typeDemo.CHOICE_PAIR_PRM": FpyValue(
                    lookup_type(fprime_test_api, "Ref.ChoicePair"),
                    {"firstChoice": "BLUE", "secondChoice": "RED"},
                ).serialize()
            },
        )

    def test_get_array_element_of_prm(self, fprime_test_api):
        seq = """
if Ref.typeDemo.CHOICES_PRM[1] == Ref.Choice.BLUE:
    exit(0)
exit(1)
"""

        assert_run_success(
            fprime_test_api,
            seq,
            prms={
                "Ref.typeDemo.CHOICES_PRM": FpyValue(
                    lookup_type(fprime_test_api, "Ref.ManyChoices"), ["ONE", "BLUE"]
                ).serialize()
            },
        )

    def test_assign_prm_member_bad(self, fprime_test_api):
        seq = """
Ref.typeDemo.CHOICE_PAIR_PRM.firstChoice = Ref.Choice.ONE
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_nonexistent_prm(self, fprime_test_api):
        seq = """
x: U32 = Ref.typeDemo.NonExistentParam
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_prm_not_found(self, fprime_test_api):
        # The spacecraft's parameter database has no value for the parameter,
        # so the read fails the sequence.
        seq = """
choice: Ref.Choice = Ref.typeDemo.CHOICE_PRM
"""

        assert_run_failure(
            fprime_test_api, seq, error_code=DirectiveErrorCode.PRM_NOT_FOUND
        )

    def test_string_prm_read_rejected(self, fprime_test_api):
        # A string's serialized size varies at runtime, so a string parameter
        # cannot be read (no backend can lay out or compare the value).
        seq = """
if Ref.cmdSeq0.SEQ_BASE_DIR == "seq":
    exit(0)
exit(1)
"""

        assert_compile_failure(fprime_test_api, seq, match="not constant-sized")
