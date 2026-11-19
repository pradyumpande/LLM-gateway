from types import SimpleNamespace

import pytest

from app.guardrails.pii import InputTooLongError, mask_messages


class FakeAnalyzer:
    def analyze(self, *, text, language, entities, score_threshold):
        assert language == "en"
        assert "IN_PHONE" in entities
        start = text.find("98765 43210")
        if start < 0:
            return []
        return [
            SimpleNamespace(
                entity_type="IN_PHONE",
                start=start,
                end=start + len("98765 43210"),
                score=0.6,
            )
        ]


def test_mask_messages_replaces_text_and_preserves_input():
    messages = [
        {"role": "user", "content": "Call 98765 43210"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Again: 98765 43210"},
                {"type": "image_url", "image_url": {"url": "https://example.test/x"}},
            ],
        },
    ]

    masked, results = mask_messages(messages, FakeAnalyzer())

    assert masked[0]["content"] == "Call <PHONE>"
    assert masked[1]["content"][0]["text"] == "Again: <PHONE>"
    assert masked[1]["content"][1] == messages[1]["content"][1]
    assert messages[0]["content"] == "Call 98765 43210"
    assert len(results) == 2


def test_mask_messages_enforces_aggregate_input_limit():
    with pytest.raises(InputTooLongError, match="maximum is 5"):
        mask_messages(
            [
                {"role": "user", "content": "123"},
                {"role": "user", "content": "456"},
            ],
            FakeAnalyzer(),
            max_input_chars=5,
        )
