import os
import subprocess
import tempfile
from pathlib import Path

import pytest

# Repo layout: this file lives in test/.
_TEST_DIR = Path(__file__).parent
_REPO_ROOT = _TEST_DIR.parent
_FPRIME_DIR = _TEST_DIR / "fprime"
_HARNESS_DIR = _TEST_DIR / "harness"
_HARNESS_BUILD = _HARNESS_DIR / "build"
_HARNESS_BIN = _HARNESS_BUILD / "bin" / "Linux" / "FpyHarness"
_WASM_HARNESS_BIN = _HARNESS_BUILD / "bin" / "Linux" / "WasmSeqHarness"
_WASM_FPRIME_DIR = _TEST_DIR / "fprime-wasm"
_WASM_PATCH = _HARNESS_DIR / "patches" / "wasm-sequencer-local-fixes.patch"


def pytest_addoption(parser):
    parser.addoption(
        "--wasm",
        action="store_true",
        default=False,
        help="Compile the whole suite to wasm and run it on the real "
        "Svc::WasmSequencer instead of the fpy bytecode VM",
    )


def _build_harness():
    """Build the sequencer harness once and return the binary path.

    FPY_HARNESS_BIN names an already-built harness and skips the build, which
    is how CI builds it once and shares it across the Python matrix.

    Surfaces the setup gaps -- submodule not checked out, no C++ compiler --
    with an actionable message rather than a wall of CMake output."""
    prebuilt = os.environ.get("FPY_HARNESS_BIN")
    if prebuilt:
        if not Path(prebuilt).exists():
            pytest.exit(
                f"FPY_HARNESS_BIN points at {prebuilt}, which does not exist",
                returncode=1,
            )
        return prebuilt

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


def _assert_wasm_patch_applied():
    """The WasmSequencer checkout carries local fixes without which nothing
    runs; a submodule bump silently drops them, so say so plainly."""
    host = _WASM_FPRIME_DIR / "Svc" / "WasmSequencer" / "WasmSequencerHost.cpp"
    if host.exists() and "FPY-LOCAL" in host.read_text():
        return
    pytest.exit(
        "The WasmSequencer checkout is missing its local fixes, without which "
        "a sequence cannot report an exit code. Reapply them:\n"
        f"  git -C {_WASM_FPRIME_DIR} apply {_WASM_PATCH.resolve()}\n"
        f"See {_WASM_PATCH.parent / 'README.md'} for what they are.",
        returncode=1,
    )


def _build_wasm_harness():
    """Build the WasmSequencer harness and return its binary path.

    Shares the build the bytecode harness already configures; the component
    only builds when its checkout and cargo are both present."""
    _assert_wasm_patch_applied()
    _build_harness()
    result = subprocess.run(
        ["cmake", "--build", str(_HARNESS_BUILD), "--target", "WasmSeqHarness"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not _WASM_HARNESS_BIN.exists():
        pytest.exit(
            "Failed to build the WasmSequencer harness. It needs cargo and the "
            "component checkout:\n"
            "  git submodule update --init --depth 1 test/fprime-wasm\n\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}",
            returncode=1,
        )
    return str(_WASM_HARNESS_BIN)


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
        "even without --wasm (requires the fprime-wasm submodule and cargo)",
    )

    # Flip the test helpers over to the LLVM/wasm backend for the whole run.
    import fpy.test_helpers as test_helpers

    test_helpers.USE_WASM = config.getoption("--wasm")

    from fpy.harness import Harness

    harness = Harness(_build_harness())
    info = harness.info()
    _assert_harness_matches_dictionary(info)
    test_helpers.HARNESS = harness

    # The wasm-marked tests always run on the wasm backend, so its harness is
    # needed whether or not --wasm puts the whole suite there.
    test_helpers.WASM_HARNESS = Harness(_build_wasm_harness())


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
    if test_helpers.WASM_HARNESS is not None:
        test_helpers.WASM_HARNESS.close()
