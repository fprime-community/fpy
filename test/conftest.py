import subprocess
import tempfile
from pathlib import Path

import pytest
import fpy.model

# Repo layout: this file lives in test/.
_TEST_DIR = Path(__file__).parent
_REPO_ROOT = _TEST_DIR.parent
_SPACEWASM_DIR = _TEST_DIR / "spacewasm"
_RUNNER_DIR = _TEST_DIR / "spacewasm_runner"
_RUNNER_MANIFEST = _RUNNER_DIR / "Cargo.toml"
_RUNNER_BIN = _RUNNER_DIR / "target" / "release" / "fpy-spacewasm-runner"

_FPRIME_DIR = _TEST_DIR / "fprime"
_HARNESS_DIR = _TEST_DIR / "harness"
_HARNESS_BUILD = _HARNESS_DIR / "build"
_HARNESS_BIN = _HARNESS_BUILD / "bin" / "Linux" / "FpyHarness"


def pytest_addoption(parser):
    parser.addoption(
        "--fpy-debug",
        action="store_true",
        default=False,
        help="Enable debug output from the FPY sequencer model",
    )
    parser.addoption(
        "--wasm",
        action="store_true",
        default=False,
        help="Compile and run sequences through the LLVM/wasm backend "
        "(NASA spacewasm) instead of the fpy bytecode VM",
    )
    parser.addoption(
        "--harness",
        action="store_true",
        default=False,
        help="Run sequences on the real Svc::FpySequencer (built from the "
        "test/fprime submodule) instead of the Python model",
    )


def _build_harness():
    """Build the sequencer harness once and return the binary path.

    Surfaces the setup gaps -- submodule not checked out, no C++ compiler --
    with an actionable message rather than a wall of CMake output."""
    if not (_FPRIME_DIR / "CMakeLists.txt").exists():
        pytest.exit(
            "fprime submodule is not checked out. Run:\n"
            "  git submodule update --init --depth 1 test/fprime",
            returncode=1,
        )
    configure = [
        "cmake",
        "-S",
        str(_REPO_ROOT),
        "-B",
        str(_HARNESS_BUILD),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_STANDARD=17",
        "-DBUILD_TESTING=OFF",
    ]
    build = ["cmake", "--build", str(_HARNESS_BUILD), "--target", "FpyHarness"]
    for command in (configure, build):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            pytest.exit(
                "Failed to build the sequencer harness.\n"
                f"  {' '.join(command)}\n"
                "The build needs a C++17 compiler; everything else (cmake, "
                "ninja, the fpp tools) comes from the harness dependency "
                "group, so `uv sync --group harness` covers the rest.\n\n"
                f"{result.stdout[-3000:]}\n{result.stderr[-3000:]}",
                returncode=1,
            )
    return str(_HARNESS_BIN)


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


def _use_short_temp_root():
    """Put temp files somewhere short enough for the sequencer to name.

    A sequence path reaches the sequencer as an Fw::CmdStringArg, which holds
    FW_CMD_STRING_MAX_SIZE (40) characters; a longer path is truncated and the
    command fails to deserialize. Tests build child sequences under
    tempfile.TemporaryDirectory(), whose root follows TMPDIR and can be long
    enough on its own to blow the budget -- so the root is pinned here rather
    than left to the environment."""
    root = Path("/tmp/fpy")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    tempfile.tempdir = str(root)


def pytest_configure(config):
    _use_short_temp_root()
    config.addinivalue_line(
        "markers",
        "wasm: end-to-end LLVM/wasm tests; always run on the wasm backend, "
        "even without --wasm (requires the spacewasm submodule and Rust)",
    )

    # Flip the test helpers over to the LLVM/wasm backend for the whole run.
    import fpy.test_helpers as test_helpers

    test_helpers.USE_WASM = config.getoption("--wasm")
    if test_helpers.USE_WASM:
        test_helpers.SPACEWASM_RUNNER = _build_spacewasm_runner()

    if not config.getoption("--harness"):
        return

    from fpy.harness import Harness

    harness = Harness(_build_harness())
    info = harness.info()
    _assert_harness_matches_dictionary(info)
    test_helpers.HARNESS = harness


def _assert_harness_matches_dictionary(info):
    """The harness and the compiler must agree on the sequencer's limits.

    The compiler reads them from the dictionary and the harness reports what it
    was built with; a mismatch means the dictionary describes a different
    deployment than the one running the sequences, which would show up later as
    a pile of unrelated-looking failures."""
    from fpy.dictionary import load_dictionary
    from fpy.test_helpers import default_dictionary

    constants = load_dictionary(default_dictionary)["constants"]
    mismatches = []
    for name, reported in info["constants"].items():
        for qualified in (f"Svc.Fpy.{name}", name):
            if qualified in constants:
                declared = constants[qualified].val
                if declared != reported:
                    mismatches.append(
                        f"{qualified}: dictionary {declared}, harness {reported}"
                    )
                break
    if mismatches:
        pytest.exit(
            "The harness was built against a different configuration than the "
            "dictionary the compiler uses:\n  " + "\n  ".join(mismatches),
            returncode=1,
        )


@pytest.fixture(scope="session", autouse=True)
def _close_harness():
    yield
    import fpy.test_helpers as test_helpers

    if test_helpers.HARNESS is not None:
        test_helpers.HARNESS.close()


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
