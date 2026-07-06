#!/usr/bin/env bash
# Build the Fpy specification into a single self-contained HTML page.
#
# Renders docs/spec.adoc (which include::s the per-section files under
# docs/spec/) with Asciidoctor.js, run through npx so no Ruby toolchain is
# needed -- only Node, which the repo already uses.
#
# Usage: docs/build.sh [output.html]   (default: docs/spec.html)
set -euo pipefail

cd "$(dirname "$0")"
out="${1:-spec.html}"

npx --yes asciidoctor -o "$out" spec.adoc

echo "Built docs/$out"
