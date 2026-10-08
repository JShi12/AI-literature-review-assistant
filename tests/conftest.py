import pytest


@pytest.fixture(autouse=True)
def no_live_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must never reach the real OpenAI API.

    Code under test calls load_dotenv(), which would pick up a developer's real key from .env; an
    existing (empty) variable isn't overridden by python-dotenv, so any accidental live call fails fast
    with an auth error instead of spending money.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "")
