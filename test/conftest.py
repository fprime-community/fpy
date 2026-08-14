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


def pytest_configure(config):
    # Flip the test helpers over to the LLVM/wasm backend for the whole run.
    import fpy.test_helpers as test_helpers

    test_helpers.BACKEND = "wasm" if config.getoption("--wasm") else "fpybc"

    # Both harnesses build themselves lazily, on the first test that runs a
    # sequence through them (fpy.harness.fpybc_harness / wasm_harness), so
    # runs that never touch one -- compiler unit tests, --collect-only --
    # skip its build entirely.


def pytest_unconfigure(config):
    fpy.harness.close_all()


@pytest.fixture
def update_goldens(request):
    return request.config.getoption("--update-goldens")


# When --use-gds is NOT passed (the default), override fprime_test_api with None
# so tests run against the harness instead of a live GDS.
# When --use-gds IS passed, delegate to the fprime-gds plugin's session fixture
# so tests run against the real deployment.
@pytest.fixture(name="fprime_test_api", scope="module")
def fprime_test_api_override(request):
    if request.config.getoption("--use-gds"):
        return request.getfixturevalue("fprime_test_api_session")
    return None
