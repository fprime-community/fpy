"""Tests for the typed compiler warning system.

Warnings are non-fatal diagnostics with a stable `WarningType` string that the
user can silence (`--ignore`) or promote to a hard error (`--error`).  These
tests drive the machinery through the `empty-range` warning, which is the one
warning currently emitted by semantic analysis.
"""
import pytest

from fpy.test_helpers import CompilationFailed, compile_seq
from fpy.error import (
    WARNING_ALL,
    CompileWarning,
    WarningType,
    parse_warning_set,
)


# A sequence whose for-loop range is statically empty (5 >= 3), which triggers
# the EMPTY_RANGE warning during semantic analysis.
EMPTY_RANGE_SEQ = """\
for i in 5 .. 3:
    pass
"""


class TestWarningTypeParsing:
    """Parsing of the comma-separated --ignore / --error CLI specs."""

    def test_single_type(self):
        assert parse_warning_set("empty-range") == {WarningType.EMPTY_RANGE}

    def test_comma_separated(self):
        assert parse_warning_set("empty-range,import-side-effects") == {
            WarningType.EMPTY_RANGE,
            WarningType.IMPORT_SIDE_EFFECTS,
        }

    def test_whitespace_tolerated(self):
        assert parse_warning_set("  empty-range ,  import-side-effects  ") == {
            WarningType.EMPTY_RANGE,
            WarningType.IMPORT_SIDE_EFFECTS,
        }

    def test_empty_spec_is_empty_set(self):
        assert parse_warning_set("") == set()
        assert parse_warning_set("  ") == set()

    def test_all_expands_to_every_type(self):
        assert parse_warning_set(WARNING_ALL) == set(WarningType)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown warning type"):
            parse_warning_set("not-a-real-warning")

    def test_from_value_roundtrip(self):
        for member in WarningType:
            assert WarningType.from_value(member.value) is member


class TestWarningEmission:
    """A warning is collected on the state and does not fail compilation."""

    def test_empty_range_warns(self):
        state, _, _ = compile_seq(EMPTY_RANGE_SEQ)
        assert any(w.type == WarningType.EMPTY_RANGE for w in state.warnings), (
            f"expected an empty-range warning, got {state.warnings}"
        )

    def test_warning_does_not_fail_compile(self):
        # Should not raise -- warnings are non-fatal by default.
        compile_seq(EMPTY_RANGE_SEQ)


class TestIgnoreWarnings:
    """--ignore silently drops the warning."""

    def test_ignore_suppresses_warning(self):
        state, _, _ = compile_seq(
            EMPTY_RANGE_SEQ, ignored_warnings={WarningType.EMPTY_RANGE}
        )
        assert state.warnings == []

    def test_ignore_all_suppresses_warning(self):
        state, _, _ = compile_seq(
            EMPTY_RANGE_SEQ, ignored_warnings=set(WarningType)
        )
        assert state.warnings == []

    def test_ignore_unrelated_type_keeps_warning(self):
        state, _, _ = compile_seq(
            EMPTY_RANGE_SEQ, ignored_warnings={WarningType.IMPORT_SIDE_EFFECTS}
        )
        assert any(w.type == WarningType.EMPTY_RANGE for w in state.warnings)


class TestEscalateWarnings:
    """--error promotes the warning to a hard compile error."""

    def test_error_escalates_to_failure(self):
        with pytest.raises(CompilationFailed):
            compile_seq(
                EMPTY_RANGE_SEQ, error_warnings={WarningType.EMPTY_RANGE}
            )

    def test_escalated_message_mentions_type(self):
        with pytest.raises(CompilationFailed, match=r"empty-range"):
            compile_seq(
                EMPTY_RANGE_SEQ, error_warnings={WarningType.EMPTY_RANGE}
            )

    def test_unrelated_error_type_does_not_fail(self):
        # Escalating a different warning type must not affect the empty-range warning.
        state, _, _ = compile_seq(
            EMPTY_RANGE_SEQ, error_warnings={WarningType.IMPORT_SIDE_EFFECTS}
        )
        assert any(w.type == WarningType.EMPTY_RANGE for w in state.warnings)


class TestWarningRendering:
    """The rendered warning is clearly labelled and carries its type."""

    def test_str_contains_type_and_message(self):
        w = CompileWarning(WarningType.EMPTY_RANGE, "Range is empty", None)
        rendered = str(w)
        assert "warning" in rendered
        assert "empty-range" in rendered
        assert "Range is empty" in rendered
