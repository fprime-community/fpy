Guidelines:
* If you're going to run Python, use `python3`
* Use the venv in the folder named `venv`
* Use pytest to run tests.
* If no tests are found, that usually means that there is an import error in the test files
* If something should never happen, assert. Don't silently return or use defensive `if x is None: return` guards for cases that represent bugs or invariant violations—crash instead so bugs surface immediately.