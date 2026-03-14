"""
Reproduction tests for compiler bugs.
"""
import pytest
from fpy.test_helpers import (
    assert_run_success,
    assert_run_failure,
    assert_compile_failure,
    assert_compile_success,
)
from fpy.model import DirectiveErrorCode


# When --use-gds is NOT passed (the default), override fprime_test_api with None
# so tests run against the Python model instead of a live GDS.
@pytest.fixture(name="fprime_test_api", scope="module")
def fprime_test_api_override(request):
    if request.config.getoption("--use-gds"):
        return request.getfixturevalue("fprime_test_api_session")
    return None


# ── Bug 1: Floor division const-fold disagrees with runtime ──────────────

class TestFloorDivConstFoldVsRuntime:
    """The const folder uses Python's // (floor toward -∞), but the VM
    uses truncation toward zero. This means constant expressions and
    runtime expressions give different results for negative operands."""

    def test_int_floor_div_negative_const_vs_runtime(self, fprime_test_api):
        """Constant -7 // 2 should give the same result as runtime -7 // 2.

        If the const folder uses Python // (floor), it gets -4.
        If the VM uses truncation, it gets -3.
        This test exposes the discrepancy.
        """
        # Runtime path (operands are variables, not const-folded)
        seq = """
a: I64 = -7
b: I64 = 2
result: I64 = a // b
assert result == -3
"""
        assert_run_success(fprime_test_api, seq)

    def test_int_floor_div_negative_const_folded(self, fprime_test_api):
        """Same operation, but with constant operands that get const-folded.

        The const folder computes (-7) // 2 using Python's //, getting -4.
        But the intended runtime semantics (per the VM) would be -3.
        """
        seq = """
result: I64 = (-7) // 2
assert result == -3
"""
        # This will FAIL if the const folder produces -4 instead of -3
        assert_run_success(fprime_test_api, seq)

    def test_float_floor_div_negative_const_vs_runtime(self, fprime_test_api):
        """Float floor division: runtime uses trunc-toward-zero."""
        seq = """
a: F64 = -5.5
b: F64 = 2.0
result: F64 = a // b
assert result == -2.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_float_floor_div_negative_const_folded(self, fprime_test_api):
        """Float floor division with constants: const folder uses Python //.

        Python: (-5.5) // 2.0 = -3.0 (floor)
        VM: trunc(-5.5 / 2.0) = trunc(-2.75) = -2.0
        """
        seq = """
result: F64 = (-5.5) // 2.0
assert result == -2.0
"""
        # This will FAIL if the const folder produces -3.0 instead of -2.0
        assert_run_success(fprime_test_api, seq)


# ── Bug 2: Wrong offset when assigning to struct member's array element ──

class TestStructMemberArrayAssign:
    """When assigning to my_struct.array_member[idx], the codegen
    computes frame_offset = base_sym.frame_offset + idx * elem_size,
    but forgets to add the array member's offset within the struct.
    This writes to the wrong memory location."""

    def test_write_struct_array_member_const_idx(self, fprime_test_api):
        """Write to a struct's array member at a constant index.

        Ref.SignalInfo has:
          type: Ref.SignalType (4 bytes)
          history: F32[4] (16 bytes)  -- offset 4 within struct
          pairHistory: Ref.SignalPairSet (32 bytes)

        Writing to info.history[1] should write at offset 4+4=8 within the struct,
        NOT at offset 4 (which would be within the 'type' member).
        """
        seq = """
info: Ref.SignalInfo = Ref.SignalInfo( \
    Ref.SignalType.TRIANGLE, \
    Ref.SignalSet(0.0, 0.0, 0.0, 0.0), \
    Ref.SignalPairSet( \
        Ref.SignalPair(0.0, 0.0), \
        Ref.SignalPair(0.0, 0.0), \
        Ref.SignalPair(0.0, 0.0), \
        Ref.SignalPair(0.0, 0.0)))
info.history[1] = 42.0
assert info.history[1] == 42.0
assert info.history[0] == 0.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_write_struct_array_member_var_idx(self, fprime_test_api):
        """Same but with a variable index -- also missing the struct offset."""
        seq = """
info: Ref.SignalInfo = Ref.SignalInfo( \
    Ref.SignalType.TRIANGLE, \
    Ref.SignalSet(0.0, 0.0, 0.0, 0.0), \
    Ref.SignalPairSet( \
        Ref.SignalPair(0.0, 0.0), \
        Ref.SignalPair(0.0, 0.0), \
        Ref.SignalPair(0.0, 0.0), \
        Ref.SignalPair(0.0, 0.0)))
idx: I64 = 1
info.history[idx] = 42.0
assert info.history[1] == 42.0
assert info.history[0] == 0.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_write_struct_array_member_nonzero_history(self, fprime_test_api):
        """Write to history[0] — the first element of an array member at a
        non-zero offset within the struct.  Even index 0 is wrong when the
        struct-member offset is omitted."""
        seq = """
info: Ref.SignalInfo = Ref.SignalInfo( \
    Ref.SignalType.TRIANGLE, \
    Ref.SignalSet(0.0, 0.0, 0.0, 0.0), \
    Ref.SignalPairSet( \
        Ref.SignalPair(0.0, 0.0), \
        Ref.SignalPair(0.0, 0.0), \
        Ref.SignalPair(0.0, 0.0), \
        Ref.SignalPair(0.0, 0.0)))
info.history[0] = 77.0
assert info.history[0] == 77.0
"""
        assert_run_success(fprime_test_api, seq)


# ── Bug 3: Crash when assigning to struct member of array element ────────

class TestArrayElemStructMemberAssign:
    """Assigning to arr[idx].member crashes the compiler with TypeError
    because visit_AstIndexExpr never sets base_offset on FieldAccess,
    so the struct member's emit_AstAssign path does None + int."""

    def test_write_array_elem_struct_member(self, fprime_test_api):
        """Ref.SignalPairSet is Ref.SignalPair[4].
        Ref.SignalPair has {time: F32, value: F32}.
        Writing to pairs[0].value should work."""
        seq = """
pairs: Ref.SignalPairSet = Ref.SignalPairSet( \
    Ref.SignalPair(1.0, 2.0), \
    Ref.SignalPair(3.0, 4.0), \
    Ref.SignalPair(5.0, 6.0), \
    Ref.SignalPair(7.0, 8.0))
pairs[0].value = 99.0
assert pairs[0].value == 99.0
assert pairs[0].time == 1.0
"""
        assert_run_success(fprime_test_api, seq)

    def test_write_array_elem_struct_member_var_idx(self, fprime_test_api):
        """Same but with variable index."""
        seq = """
pairs: Ref.SignalPairSet = Ref.SignalPairSet( \
    Ref.SignalPair(1.0, 2.0), \
    Ref.SignalPair(3.0, 4.0), \
    Ref.SignalPair(5.0, 6.0), \
    Ref.SignalPair(7.0, 8.0))
idx: I64 = 1
pairs[idx].value = 99.0
assert pairs[1].value == 99.0
assert pairs[1].time == 3.0
"""
        assert_run_success(fprime_test_api, seq)
