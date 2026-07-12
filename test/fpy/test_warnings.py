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
        assert parse_warning_set("empty-range,import-underscore") == {
            WarningType.EMPTY_RANGE,
            WarningType.IMPORT_UNDERSCORE,
        }

    def test_whitespace_tolerated(self):
        assert parse_warning_set("  empty-range ,  import-underscore  ") == {
            WarningType.EMPTY_RANGE,
            WarningType.IMPORT_UNDERSCORE,
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
    """An expected warning is collected and does not fail compilation.
    `expected_warnings` both permits the warning and requires it to be emitted,
    so a separate presence assertion is unnecessary."""

    def test_empty_range_warns(self):
        # If EMPTY_RANGE were not emitted, the helper would fail the test.
        compile_seq(EMPTY_RANGE_SEQ, expected_warnings={WarningType.EMPTY_RANGE})


class TestIgnoreWarnings:
    """--ignore silently drops the warning."""

    def test_ignore_suppresses_warning(self):
        state, _, _ = compile_seq(
            EMPTY_RANGE_SEQ, ignored_warnings={WarningType.EMPTY_RANGE}
        )
        assert state.warnings == []

    def test_ignore_all_suppresses_warning(self):
        state, _, _ = compile_seq(EMPTY_RANGE_SEQ, ignored_warnings=set(WarningType))
        assert state.warnings == []

    def test_ignore_unrelated_type_keeps_warning(self):
        state, _, _ = compile_seq(
            EMPTY_RANGE_SEQ,
            ignored_warnings={WarningType.IMPORT_UNDERSCORE},
            expected_warnings={WarningType.EMPTY_RANGE},
        )
        assert any(w.type == WarningType.EMPTY_RANGE for w in state.warnings)


class TestEscalateWarnings:
    """--error promotes the warning to a hard compile error."""

    def test_error_escalates_to_failure(self):
        with pytest.raises(CompilationFailed):
            compile_seq(EMPTY_RANGE_SEQ, error_warnings={WarningType.EMPTY_RANGE})

    def test_escalated_message_mentions_type(self):
        with pytest.raises(CompilationFailed, match=r"empty-range"):
            compile_seq(EMPTY_RANGE_SEQ, error_warnings={WarningType.EMPTY_RANGE})

    def test_unrelated_error_type_does_not_fail(self):
        # Escalating a different warning type must not affect the empty-range warning.
        state, _, _ = compile_seq(
            EMPTY_RANGE_SEQ, error_warnings={WarningType.IMPORT_UNDERSCORE}
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
