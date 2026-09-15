from __future__ import annotations

from lit_review_assistant.app import is_read_only_demo


def test_is_read_only_demo_false_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("READ_ONLY_DEMO", raising=False)

    assert is_read_only_demo() is False


def test_is_read_only_demo_true_for_truthy_values(monkeypatch) -> None:
    for value in ["1", "true", "True", "TRUE", "yes", "Yes"]:
        monkeypatch.setenv("READ_ONLY_DEMO", value)
        assert is_read_only_demo() is True, f"expected True for {value!r}"


def test_is_read_only_demo_false_for_other_values(monkeypatch) -> None:
    for value in ["0", "false", "no", "", "  "]:
        monkeypatch.setenv("READ_ONLY_DEMO", value)
        assert is_read_only_demo() is False, f"expected False for {value!r}"
