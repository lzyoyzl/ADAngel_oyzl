import pytest


def pytest_addoption(parser):
    parser.addoption("--run-sm120", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-sm120"):
        return
    skip = pytest.mark.skip(reason="requires --run-sm120 on RTX 5090")
    for item in items:
        if "sm120" in item.keywords:
            item.add_marker(skip)
