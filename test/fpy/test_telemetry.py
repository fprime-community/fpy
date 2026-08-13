from fpy.bytecode.directives import DirectiveErrorCode
from fpy.types import BOOL, F32, FpyValue, U32
from fpy.test_helpers import (
    assert_compile_failure,
    assert_run_failure,
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
                "Ref.typeDemo.ChoicePairCh": FpyValue(
                    lookup_type(fprime_test_api, "Ref.ChoicePair"),
                    {"firstChoice": "ONE", "secondChoice": "ONE"},
                ).serialize()
            },
        )

    def test_read_f32_tlm(self, fprime_test_api):
        seq = """
assert Ref.typeDemo.Float1Ch == 1.5
"""

        assert_run_success(
            fprime_test_api,
            seq,
            {"Ref.typeDemo.Float1Ch": FpyValue(F32, 1.5).serialize()},
        )

    def test_read_bool_tlm(self, fprime_test_api):
        seq = """
assert Ref.cmdSeq0.BreakpointInUse
"""

        assert_run_success(
            fprime_test_api,
            seq,
            {"Ref.cmdSeq0.BreakpointInUse": FpyValue(BOOL, True).serialize()},
        )

    def test_tlm_read_twice_sees_each_value(self, fprime_test_api):
        # Each mention of a channel is its own read; both reads see the
        # (same) injected value.
        seq = """
x: U32 = CdhCore.cmdDisp.CommandsDispatched
y: U32 = CdhCore.cmdDisp.CommandsDispatched
assert x == y
"""

        assert_run_success(
            fprime_test_api,
            seq,
            {"CdhCore.cmdDisp.CommandsDispatched": FpyValue(U32, 3).serialize()},
        )

    def test_tlm_as_cmd_arg(self, fprime_test_api):
        # A telemetry read evaluated at runtime as a command argument.
        seq = """
var1: I32 = -2
CdhCore.cmdDisp.CMD_TEST_CMD_1(var1, Ref.typeDemo.Float1Ch, 8)
"""

        assert_run_success(
            fprime_test_api,
            seq,
            {"Ref.typeDemo.Float1Ch": FpyValue(F32, 1.5).serialize()},
        )

    def test_assign_tlm_struct_member_bad(self, fprime_test_api):
        seq = """
Ref.cmdSeq0.Debug.nextStatementOpcode = 0
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_nonexistent_tlm_channel(self, fprime_test_api):
        seq = """
x: U32 = CdhCore.cmdDisp.NonExistentChannel
"""

        assert_compile_failure(fprime_test_api, seq)

    def test_tlm_chan_not_found(self, fprime_test_api):
        # The spacecraft has no value for the channel, so the read fails the
        # sequence.
        seq = """
x: U32 = CdhCore.cmdDisp.CommandsDispatched
"""

        assert_run_failure(
            fprime_test_api, seq, error_code=DirectiveErrorCode.TLM_CHAN_NOT_FOUND
        )

    def test_string_tlm_read_rejected(self, fprime_test_api):
        # A string's serialized size varies at runtime, so a string channel
        # cannot be read (no backend can lay out or compare the value).
        seq = """
if CdhCore.version.FrameworkVersion == "abc":
    exit(0)
exit(1)
"""

        assert_compile_failure(fprime_test_api, seq, match="not constant-sized")

    def test_struct_with_strings_tlm_read_rejected(self, fprime_test_api):
        seq = """
if CdhCore.version.CustomVersion01 == CdhCore.version.CustomVersion01:
    exit(0)
exit(1)
"""

        assert_compile_failure(fprime_test_api, seq, match="not constant-sized")

    def test_string_tlm_as_cmd_arg_rejected(self, fprime_test_api):
        seq = """
CdhCore.cmdDisp.CMD_NO_OP_STRING(Ref.cmdSeq0.SeqPath)
"""

        assert_compile_failure(fprime_test_api, seq, match="not constant-sized")
