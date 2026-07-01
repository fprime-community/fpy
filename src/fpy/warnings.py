from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any

from fpy.error import Colors, format_diagnostic


class WarningType(str, Enum):
    """The set of diagnostics the compiler may warn about."""

    EMPTY_RANGE = "empty-range"
    IMPORT_SIDE_EFFECTS = "import-side-effects"

    @classmethod
    def from_value(cls, value: str) -> "WarningType":
        """Look up a WarningType by its CLI string value.

        Raises ValueError (listing the valid values) if unknown."""
        for member in cls:
            if member.value == value:
                return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(f"Unknown warning type {value!r}. Valid types: {valid}")


# Sentinel accepted by --ignore / --error to mean "every warning type".
WARNING_ALL = "all"


def parse_warning_set(spec: str) -> set[WarningType]:
    """Parse a comma-separated `--ignore`/`--error` spec into a set of types.

    Empty/whitespace-only entries are skipped.  The special value "all"
    expands to every WarningType.  Raises ValueError on an unknown type."""
    result: set[WarningType] = set()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token == WARNING_ALL:
            result.update(WarningType)
            continue
        result.add(WarningType.from_value(token))
    return result


@dataclass
class CompileWarning:
    """A non-fatal diagnostic."""

    type: WarningType
    msg: str
    node: Any = None

    def __str__(self) -> str:
        label = f"warning [{self.type.value}]: {self.msg}"
        return format_diagnostic(label, self.node, color=Colors.yellow)

    __repr__ = __str__
