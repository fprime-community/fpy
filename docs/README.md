# Fpy specification docs

The specification is written in [AsciiDoc](https://asciidoc.org/), one file per
section, and rendered to a single self-contained HTML page with
[Asciidoctor.js](https://github.com/asciidoctor/asciidoctor.js), run via `npx`
so no Ruby toolchain is needed -- only Node, which the repo already uses.

The `.adoc` files under `spec/` are the **source of truth** -- edit them
directly.

## Layout

- `spec.adoc` -- master document: header, introduction, and `include::` lines
  pulling in each section in order.
- `spec/NN-*.adoc` -- one file per top-level section.
- `build.sh` -- renders `spec.adoc` to `spec.html`.

## Build

```sh
docs/build.sh
```

This writes `docs/spec.html`, a standalone page with styles and a left-hand
table of contents inlined. `spec.html` is a build artifact.

## Preview / serve

Open `docs/spec.html` directly, or serve the directory:

```sh
python3 -m http.server -d docs 8000   # then visit http://localhost:8000/spec.html
```

## Editing

Edit the `spec/*.adoc` files. To add a section, create `spec/NN-name.adoc` and
add an `include::` line to `spec.adoc` -- keep a blank line after each
`include::` so Asciidoctor keeps the sections at the top level.

Cross references use `<<slug,link text>>`, where `slug` is a heading's anchor
(lowercase, spaces to hyphens); the `:idprefix:`/`:idseparator:` settings in
`spec.adoc` make Asciidoctor generate those IDs.

Each statement may be followed by a `_Tests:_` line linking to the tests that
verify it. Those links are checked by `verify/spec_links.py` (a pre-commit
hook); run `uv run python verify/spec_links.py --fix` to refresh stale line
numbers.
