import pytest

from app.guardrails.pii import build_analyzer


@pytest.fixture(scope="session")
def analyzer():
    return build_analyzer()
