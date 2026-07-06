#!/usr/bin/env python3
"""Verify spec-to-test links in the AsciiDoc spec sources.

The spec references the tests that pin down a statement with compact numbered
AsciiDoc links. The link's title (shown on hover) names the test, and the URL
points at its line:

    _Tests:_ link:test/fpy/test_imports.py#L123[1,title="test/fpy/test_imports.py::TestFoo::test_bar"], link:...[2,title="..."]

Any link macro whose title contains `.py::` is treated as a test link.
The title must be `<path>::<Class>::<function>` (or `<path>::<function>` for
a module-level test), with <path> relative to the repo root. The URL must be
`<path>#L<line>` where <line> is the line of the test's `def`. The link text
must be the link's 1-based position among the test links on its line.

Usage:

    python3 verify/spec_links.py [--fix] [FILE.adoc ...]

Without --fix, exits nonzero if any link names a missing file or test, has a
stale line number, or is mislabeled. With --fix, stale line numbers and
labels are rewritten in place; missing files and tests are still errors.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TEST_LINK_RE = re.compile(
    r'link:(?P<url>[^\[\s]+)\[(?P<label>[^\],]*),title="(?P<text>[^"\s]+\.py::[^"\s]+)"\]'
)


def collect_defs(py_path: Path) -> dict[str, int]:
    """Map 'func' and 'Class::func' to the line number of their `def`."""
    tree = ast.parse(py_path.read_text(), filename=str(py_path))
    defs: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[node.name] = node.lineno
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defs[f"{node.name}::{sub.name}"] = sub.lineno
    return defs


def process(md_path: Path, fix: bool) -> tuple[int, int, list[str]]:
    """Check (and with fix=True, rewrite) one markdown file.

    Returns (links checked, links fixed, errors)."""
    errors: list[str] = []
    checked = 0
    fixed = 0
    def_cache: dict[Path, dict[str, int]] = {}
    lines = md_path.read_text().splitlines(keepends=True)

    new_lines = []
    for lineno, line in enumerate(lines, start=1):
        position = 0

        def handle(match: re.Match) -> str:
            nonlocal checked, fixed, position
            checked += 1
            position += 1
            label, url, text = match["label"], match["url"], match["text"]
            where = f"{md_path}:{lineno}"
            path_str, _, qualname = text.partition("::")
            py_path = REPO / path_str
            if not py_path.is_file():
                errors.append(f"{where}: no such file: {path_str}")
                return match[0]
            if py_path not in def_cache:
                def_cache[py_path] = collect_defs(py_path)
            def_line = def_cache[py_path].get(qualname)
            if def_line is None:
                errors.append(f"{where}: no test '{qualname}' in {path_str}")
                return match[0]
            expected = f'link:{path_str}#L{def_line}[{position},title="{text}"]'
            if match[0] != expected:
                if fix:
                    fixed += 1
                    return expected
                errors.append(
                    f"{where}: stale link for {text}: '{match[0]}' should be '{expected}' (run with --fix)"
                )
            return match[0]

        new_lines.append(TEST_LINK_RE.sub(handle, line))

    if fix and new_lines != lines:
        md_path.write_text("".join(new_lines))
    return checked, fixed, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=sorted((REPO / "docs" / "spec").glob("*.adoc")),
        help="AsciiDoc files to check (default: docs/spec/*.adoc)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="rewrite stale line numbers and labels in place",
    )
    args = parser.parse_args()

    ok = True
    for md_path in args.files:
        if not md_path.is_file():
            print(f"error: no such file: {md_path}", file=sys.stderr)
            ok = False
            continue
        checked, fixed, errors = process(md_path, args.fix)
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        summary = f"{md_path}: {checked} test links"
        if args.fix:
            summary += f", {fixed} fixed"
        print(summary)
        ok = ok and not errors
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
