import pytest
import fpy.model


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
        help="Compile and run sequences through the LLVM/wasm backend (wasmtime) "
        "instead of the fpy bytecode VM",
    )


def pytest_configure(config):
    # Flip the test helpers over to the LLVM/wasm backend for the whole run.
    import fpy.test_helpers as test_helpers

    test_helpers.USE_WASM = config.getoption("--wasm")


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