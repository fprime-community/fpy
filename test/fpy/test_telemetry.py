from fpy.types import FpyValue, U32
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_success,
    lookup_type,
)


class TestTelemetry:

    def test_geq_tlm(self, fprime_test_api):
        seq = """
CdhCore.cmdDisp.CMD_NO_OP()
# NOTE! this is not guaranteed to work, if the tlm gets written
# too slowly to the DB then this will fail
if CdhCore.cmdDisp.CommandsDispatched >= 1:
    exit(0)
exit(1)
"""

        assert_run_success(
            fprime_test_api,
            seq,
            {"CdhCore.cmdDisp.CommandsDispatched": FpyValue(U32, 1).serialize()},
        )

    def test_get_struct_member_of_tlm(self, fprime_test_api):
        seq = """
Ref.typeDemo.CHOICE_PAIR(Ref.ChoicePair(Ref.Choice.ONE, Ref.Choice.ONE))
if Ref.typeDemo.ChoicePairCh.firstChoice == Ref.Choice.ONE:
    exit(0)
exit(1)
"""

        assert_run_success(
            fprime_test_api,
            seq,
            {
                "Ref.typeDemo.ChoicePairCh": FpyValue(lookup_type(fprime_test_api, "Ref.ChoicePair"),
                    {"firstChoice": "ONE", "secondChoice": "ONE"}
                ).serialize()
            },
        )

    def test_assign_tlm_struct_member_bad(self, fprime_test_api):
        seq = """
Ref.cmdSeq.Debug.nextStatementOpcode = 0
"""

        assert_compile_failure(fprime_test_api, seq)
