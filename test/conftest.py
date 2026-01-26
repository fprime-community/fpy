import pytest
import fpy.model


def pytest_addoption(parser):
    parser.addoption(
        "--fpy-debug",
        action="store_true",
        default=False,
        help="Enable debug output from the FPY sequencer model",
    )


@pytest.fixture(autouse=True)
def configure_fpy_debug(request):
    """Automatically configure fpy.model.debug based on --fpy-debug flag."""
    original_debug = fpy.model.debug
    fpy.model.debug = request.config.getoption("--fpy-debug")
    yield
    fpy.model.debug = original_debug
