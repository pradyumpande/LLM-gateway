from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.guardrails.pii import (
    InputTooLongError,
    MaskState,
    mask_messages,
    mask_text,
    resolve_overlaps,
)


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


def test_mask_messages_payload_shapes_and_preserves_input(analyzer):
    messages = [
        {"role": "system", "content": "System uses 98765 43210"},
        {"role": "user", "content": "Call 98765 43210"},
        {"role": "assistant", "content": "Call 98765 43210"},
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "Again: 98765 43210"},
                {"type": "image_url", "image_url": {"url": "https://example.test/x"}},
                {"text": "Do not scan this 98765 43210"},
            ],
        },
        {"role": "assistant", "content": None, "tool_calls": [{"id": "safe"}]},
    ]
    before_call = deepcopy(messages)

    masked, report = mask_messages(
        analyzer,
        messages,
        ["IN_PHONE", "EMAIL_ADDRESS", "AADHAAR", "CREDIT_CARD"],
        0.5,
    )

    assert masked[0]["content"] == "System uses <PHONE_1>"
    assert masked[1]["content"] == "Call <PHONE_1>"
    assert masked[2]["content"] == "Call <PHONE_1>"
    assert masked[3]["content"][0]["text"] == "Again: <PHONE_1>"
    assert masked[3]["content"][1] == messages[3]["content"][1]
    assert masked[3]["content"][2] == messages[3]["content"][2]
    assert masked[4]["content"] is None
    assert messages == before_call
    assert report == {"counts": {"IN_PHONE": 4}, "total": 4}
    assert "98765 43210" not in repr(report)
    assert masked is not messages
    assert masked[3]["content"] is not messages[3]["content"]


def test_masking_handles_multiple_entities_repeats_and_distinct_values(analyzer):
    text = (
        "Call 98765 43210 or 91234 56789; "
        "mail asha@example.com and asha@example.com; "
        "Aadhaar 2345 6789 0124."
    )
    masked, report = mask_messages(
        analyzer,
        [{"role": "user", "content": text}],
        ["IN_PHONE", "EMAIL_ADDRESS", "AADHAAR"],
        0.5,
    )

    assert masked[0]["content"] == (
        "Call <PHONE_1> or <PHONE_2>; "
        "mail <EMAIL_1> and <EMAIL_1>; Aadhaar <AADHAAR_1>."
    )
    assert report == {
        "counts": {"IN_PHONE": 2, "EMAIL_ADDRESS": 2, "AADHAAR": 1},
        "total": 5,
    }


def test_masking_returns_text_without_pii_unchanged(analyzer):
    text = "This message contains no recognizable personal data."
    masked, report = mask_messages(
        analyzer,
        [{"role": "user", "content": text}],
        ["IN_PHONE", "EMAIL_ADDRESS", "AADHAAR", "CREDIT_CARD"],
        0.5,
    )

    assert masked[0]["content"] == text
    assert report == {"counts": {}, "total": 0}


def test_masking_preserves_offsets_at_text_start_and_end(analyzer):
    text = "98765 43210 is my phone; email: asha@example.com"
    masked, report = mask_messages(
        analyzer,
        [{"role": "user", "content": text}],
        ["IN_PHONE", "EMAIL_ADDRESS"],
        0.5,
    )

    assert masked[0]["content"] == "<PHONE_1> is my phone; email: <EMAIL_1>"
    assert report == {"counts": {"IN_PHONE": 1, "EMAIL_ADDRESS": 1}, "total": 2}


def test_overlap_resolution_prefers_score_then_longer_span():
    card = SimpleNamespace(
        entity_type="CREDIT_CARD",
        start=0,
        end=19,
        score=1.0,
    )
    aadhaar_overlap = SimpleNamespace(
        entity_type="AADHAAR",
        start=5,
        end=17,
        score=1.0,
    )
    lower_score = SimpleNamespace(
        entity_type="IN_PHONE",
        start=20,
        end=31,
        score=0.6,
    )

    assert resolve_overlaps([aadhaar_overlap, lower_score, card]) == [
        card,
        lower_score,
    ]


def test_credit_card_overlap_returns_only_credit_card(analyzer):
    text = "4111 1111 1111 1111"
    raw_results = analyzer.analyze(
        text=text,
        language="en",
        entities=["CREDIT_CARD", "AADHAAR", "IN_PHONE"],
        score_threshold=0.5,
    )

    results = resolve_overlaps(raw_results)

    assert len(results) == 1
    assert results[0].entity_type == "CREDIT_CARD"
    assert text[results[0].start : results[0].end] == text


def test_mask_text_reuses_entity_value_placeholder():
    text = "Call 98765 43210; call 98765 43210."
    first = text.index("98765 43210")
    second = text.index("98765 43210", first + 1)
    results = [
        SimpleNamespace(
            entity_type="IN_PHONE",
            start=first,
            end=first + len("98765 43210"),
            score=0.6,
        ),
        SimpleNamespace(
            entity_type="IN_PHONE",
            start=second,
            end=second + len("98765 43210"),
            score=0.6,
        ),
    ]

    assert mask_text(text, results, MaskState()) == (
        "Call <PHONE_1>; call <PHONE_1>."
    )


def test_mask_messages_enforces_aggregate_input_limit():
    with pytest.raises(InputTooLongError, match="maximum is 5"):
        mask_messages(
            FakeAnalyzer(),
            [
                {"role": "user", "content": "123"},
                {"role": "user", "content": "456"},
            ],
            ["IN_PHONE"],
            0.5,
            5,
        )
