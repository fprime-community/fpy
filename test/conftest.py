import subprocess
from pathlib import Path

import pytest
import fpy.model

# Repo layout: this file lives in test/.
_TEST_DIR = Path(__file__).parent
_SPACEWASM_DIR = _TEST_DIR / "spacewasm"
_RUNNER_DIR = _TEST_DIR / "spacewasm_runner"
_RUNNER_MANIFEST = _RUNNER_DIR / "Cargo.toml"
_RUNNER_BIN = _RUNNER_DIR / "target" / "release" / "fpy-spacewasm-runner"


def pytest_addoption(parser):
    parser.addoption(
        "--fpy-debug",
        action="store_true",
        default=False,
        help="Enable debug output from the FPY sequencer model",
    )
    parser.addoption(
        "--use-gds",
        action="store_true",
        default=False,
        help="Run sequences against a live F Prime GDS instead of the Python model",
    )
    parser.addoption(
        "--wasm",
        action="store_true",
        default=False,
        help="Compile and run sequences through the LLVM/wasm backend "
        "(NASA spacewasm) instead of the fpy bytecode VM",
    )


def _build_spacewasm_runner():
    """Build the spacewasm runner harness once and return the binary path.

    Surfaces the two common setup gaps (submodule not checked out, toolchain too
    old) with an actionable message rather than a cryptic cargo error.
    """
    if not (_SPACEWASM_DIR / "Cargo.toml").exists():
        pytest.exit(
            "spacewasm submodule is not checked out. Run:\n"
            "  git submodule update --init test/spacewasm",
            returncode=1,
        )
    try:
        subprocess.run(
            ["cargo", "build", "--release", "--manifest-path", str(_RUNNER_MANIFEST)],
            check=True,
        )
    except FileNotFoundError:
        pytest.exit(
            "cargo not found. Install Rust (>=1.85, spacewasm is edition 2024):\n"
            "  https://rustup.rs",
            returncode=1,
        )
    except subprocess.CalledProcessError as e:
        pytest.exit(
            "Failed to build the spacewasm runner harness "
            f"({_RUNNER_MANIFEST}). If this is a toolchain version error, "
            "spacewasm needs Rust >=1.85; run `rustup update`.\n"
            f"cargo exited with {e.returncode}.",
            returncode=1,
        )
    return str(_RUNNER_BIN)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "wasm: end-to-end LLVM/wasm tests; only run when --wasm is passed",
    )

    # Flip the test helpers over to the LLVM/wasm backend for the whole run.
    import fpy.test_helpers as test_helpers

    test_helpers.USE_WASM = config.getoption("--wasm")
    if test_helpers.USE_WASM:
        test_helpers.SPACEWASM_RUNNER = _build_spacewasm_runner()


@pytest.fixture(autouse=True)
def _ensure_wasm_runner(request):
    # wasm-marked tests always run on the wasm backend, regardless of --wasm, so
    # make sure the spacewasm runner is built before any of them run. The build
    # result is cached on the module global, so this only builds once per session.
    if "wasm" not in request.keywords:
        return
    import fpy.test_helpers as test_helpers

    if test_helpers.SPACEWASM_RUNNER is None:
        test_helpers.SPACEWASM_RUNNER = _build_spacewasm_runner()


@pytest.fixture(autouse=True)
def configure_fpy_debug(request):
    """Automatically configure fpy.model.debug based on --fpy-debug flag."""
    original_debug = fpy.model.debug
    fpy.model.debug = request.config.getoption("--fpy-debug")
    yield
    fpy.model.debug = original_debug


# When --use-gds is NOT passed (the default), override fprime_test_api with None
# so tests run against the Python model instead of a live GDS.
# When --use-gds IS passed, delegate to the fprime-gds plugin's session fixture
# so tests run against the real deployment.
@pytest.fixture(name="fprime_test_api", scope="module")
def fprime_test_api_override(request):
    if request.config.getoption("--use-gds"):
        return request.getfixturevalue("fprime_test_api_session")
    return None