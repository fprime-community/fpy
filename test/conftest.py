import pytest
import fpy.harness


def pytest_addoption(parser):
    parser.addoption(
        "--use-gds",
        action="store_true",
        default=False,
        help="Run sequences against a live F Prime GDS instead of the "
        "harness running a local Svc::FpySequencer",
    )
    parser.addoption(
        "--wasm",
        action="store_true",
        default=False,
        help="Compile and run sequences through the LLVM/wasm backend "
        "(NASA spacewasm) instead of the fpy bytecode VM",
    )
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Rewrite the golden files under test/fpy/golden with the "
        "current outputs instead of comparing against them",
    )


_wasm_harness_built = False


# FIXME why is this only here for the wasm harness? shouldn't we do this for the fpybc harness too?
def _build_wasm_harness_once():
    """Build the wasm harness once per session, exiting with an actionable
    message on setup gaps (submodule missing, tools missing)."""
    global _wasm_harness_built
    if _wasm_harness_built:
        return
    try:
        fpy.harness.build_wasm_harness()
    except fpy.harness.HarnessError as e:
        pytest.exit(str(e), returncode=1)
    _wasm_harness_built = True


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "wasm: end-to-end LLVM/wasm tests; always run on the wasm backend, "
        "even without --wasm (requires the fprime-wasm submodule and Rust)",
    )

    # Flip the test helpers over to the LLVM/wasm backend for the whole run.
    import fpy.test_helpers as test_helpers

    test_helpers.USE_WASM = config.getoption("--wasm")
    if test_helpers.USE_WASM and not config.getoption("--use-gds"):
        _build_wasm_harness_once()

    # The FpySequencer harness builds itself lazily, on the first test that
    # runs a sequence through it (fpy.harness.fpybc_harness), so runs that
    # never touch it -- compiler unit tests, --collect-only -- skip the
    # build entirely.
    # FIXME I think we should do the same thing for both harnesses probably.
    # just build them lazily. why wouldn't that work for the test_wasm files?
    # i think for the test_wasm tests you should just pass a backend="wasm" kw to the assert_xyz funcs, then you could remove the marker system


def pytest_unconfigure(config):
    fpy.harness.close_all()


@pytest.fixture
def update_goldens(request):
    return request.config.getoption("--update-goldens")


@pytest.fixture(autouse=True)
def _ensure_wasm_harness(request):
    # wasm-marked tests always run on the wasm backend, regardless of --wasm,
    # so make sure the wasm harness is built before any of them run.
    if "wasm" in request.keywords:
        _build_wasm_harness_once()


# When --use-gds is NOT passed (the default), override fprime_test_api with None
# so tests run against the harness instead of a live GDS.
# When --use-gds IS passed, delegate to the fprime-gds plugin's session fixture
# so tests run against the real deployment.
@pytest.fixture(name="fprime_test_api", scope="module")
def fprime_test_api_override(request):
    if request.config.getoption("--use-gds"):
        return request.getfixturevalue("fprime_test_api_session")
    return None
