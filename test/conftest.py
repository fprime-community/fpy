import pytest
import fpy.harness
import fpy.test_helpers as test_helpers


def pytest_addoption(parser):
    parser.addoption(
        "--use-gds",
        action="store_true",
        default=False,
        help="Run sequences against a live F Prime GDS instead of the "
        "harnesses running a local sequencer",
    )
    parser.addoption(
        "--backend",
        choices=("both", test_helpers.FPYBC, test_helpers.WASM),
        default="both",
        help="Which backends the sequence tests drive (default: both). Each "
        "sequence is analyzed once, then compiled and run through every "
        "selected backend: fpybc on a Svc::FpySequencer, wasm (the LLVM "
        "backend) on a Svc::WasmSequencer.",
    )


def _selected_backends(config) -> tuple[str, ...]:
    choice = config.getoption("--backend")
    if choice == "both":
        return test_helpers.ALL_BACKENDS
    return (choice,)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "wasm: tests of the LLVM/wasm backend itself (IR shape, host "
        "imports); they run its toolchain directly and are skipped only by "
        "--backend fpybc",
    )
    config.addinivalue_line(
        "markers",
        "fpybc_only(reason): drive the assert_* helpers on the fpybc backend "
        "only, for behavior the wasm backend does not (or deliberately will "
        "not) share",
    )
    config.addinivalue_line(
        "markers",
        "wasm_only(reason): drive the assert_* helpers on the wasm backend only",
    )
    config.addinivalue_line(
        "markers",
        "harness_only(reason): skip under --use-gds, for behavior that depends "
        "on state only the harnesses can stage",
    )

    # Both harnesses build themselves lazily, on the first test that runs a
    # sequence through them (fpy.harness.fpy_harness / wasm_harness), so runs
    # that never touch one -- compiler unit tests, --collect-only, --backend
    # fpybc -- skip its build entirely.


def pytest_unconfigure(config):
    fpy.harness.close_all()


@pytest.fixture(autouse=True)
def _narrow_backends(request):
    """Points test_helpers.active_backends at the backends this test drives:
    the run's --backend selection, narrowed by the fpybc_only/wasm_only
    markers. Skips the test when nothing is left."""
    backends = list(_selected_backends(request.config))
    if request.node.get_closest_marker("fpybc_only"):
        backends = [b for b in backends if b == test_helpers.FPYBC]
    if request.node.get_closest_marker("wasm_only") or ("wasm" in request.keywords):
        backends = [b for b in backends if b == test_helpers.WASM]
    if request.config.getoption("--use-gds"):
        harness_only = request.node.get_closest_marker("harness_only")
        if harness_only:
            pytest.skip(harness_only.args[0])
        # A live deployment drives only the backends whose sequencer it has.
        commands = request.getfixturevalue(
            "fprime_test_api"
        ).pipeline.dictionaries.command_name
        backends = [b for b in backends if test_helpers.GDS_RUN_CMD[b] in commands]
    if not backends:
        pytest.skip("none of this test's backends are selected (--backend)")
    saved = test_helpers.active_backends
    test_helpers.active_backends = tuple(backends)
    yield
    test_helpers.active_backends = saved


@pytest.fixture(params=test_helpers.ALL_BACKENDS)
def single_backend(request, _narrow_backends):
    """Runs the test once per backend, with the assert_* helpers narrowed to
    just that backend; the fixture's value names it. For tests whose expected
    results differ by backend."""
    backend = request.param
    if backend not in test_helpers.active_backends:
        pytest.skip(f"the {backend} backend is not selected")
    saved = test_helpers.active_backends
    test_helpers.active_backends = (backend,)
    yield backend
    test_helpers.active_backends = saved


# When --use-gds is NOT passed (the default), override fprime_test_api with None
# so tests run against the harness instead of a live GDS.
# When --use-gds IS passed, delegate to the fprime-gds plugin's session fixture
# so tests run against the real deployment.
@pytest.fixture(name="fprime_test_api", scope="module")
def fprime_test_api_override(request):
    if request.config.getoption("--use-gds"):
        return request.getfixturevalue("fprime_test_api_session")
    return None


@pytest.fixture(scope="session", autouse=True)
def _gds_leave_clean(request):
    """Under --use-gds, points the deployment's sequencers back at their
    default base directory once the session's runs are done."""
    if not request.config.getoption("--use-gds"):
        yield
        return
    api = request.getfixturevalue("fprime_test_api_session")
    yield
    test_helpers._gds_set_base_dir(api, "")
