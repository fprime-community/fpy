def pytest_addoption(parser):
    parser.addoption(
        "--use-gds",
        action="store_true",
        default=False,
        help="Run sequences against a live F-Prime GDS instead of the Python model",
    )
