# Versioning

Fpy uses versioning with the following semantics.

| Component         | Meaning |
|-------------------|---------|
| `MAJOR`           | Breaking changes to front- or backend, and/or significant new features |
| `MINOR`           | Breaking changes to front- or backend, possibly new features |
| `PATCH`           | New features or bug fixes |

`PATCH` guarantees front- and backend backward compatibility, but may or may not introduce new features.

In addition, the `SCHEMA` version is updated on any breaking changes to the backend. This would always require an update to the `FpySequencer` component.
