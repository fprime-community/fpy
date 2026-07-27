Guidelines:
* This project uses `uv`. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`.
* Use `uv sync` to create/update the `.venv` and install dependencies.
* Run commands in the project environment with `uv run` (e.g. run Python with `uv run python3`).
* Run tests with `uv run pytest`.
* If no tests are found, that usually means that there is an import error in the test files
* If something should never happen, assert. Don't silently return or use defensive `if x is None: return` guards for cases that represent bugs or invariant violations—crash instead so bugs surface immediately.
* Compile errors in semantics passes should always return ASAP. No continuing on if an error has been discovered.