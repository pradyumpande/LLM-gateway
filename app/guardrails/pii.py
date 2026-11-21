from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from app.guardrails.recognizers import (
    AadhaarRecognizer,
    IndianPanRecognizer,
    IndianPhoneRecognizer,
    Recognition,
    recognize_pii,
)


DEFAULT_ENTITIES = [
    "EMAIL_ADDRESS",
    "IN_PHONE",
    "AADHAAR",
    "IN_PAN",
    "CREDIT_CARD",
]
SUPPORTED_ENTITIES = [*DEFAULT_ENTITIES, "PERSON"]
LABELS = {
    "EMAIL_ADDRESS": "EMAIL",
    "IN_PHONE": "PHONE",
    "AADHAAR": "AADHAAR",
    "IN_PAN": "PAN",
    "CREDIT_CARD": "CARD",
    "PERSON": "PERSON",
}

@dataclass
class MaskState:
    mapping: dict[tuple[str, str], str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


_MASKS = {
    "AADHAAR": "<AADHAAR>",
    "IN_PAN": "<PAN>",
    "IN_PHONE": "<PHONE>",
    "credit_card": "<CARD>",
    "email": "<EMAIL>",
    "national_id": "<NATIONAL_ID>",
    "phone": "<PHONE>",
    "ssn": "<SSN>",
}


class InputTooLongError(ValueError):
    pass


def total_text_chars(messages: Sequence[dict[str, Any]]) -> int:
    text_length = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text_length += len(content)
        elif isinstance(content, list):
            text_length += sum(
                len(part["text"])
                for part in content
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                )
            )
    return text_length


def build_analyzer() -> AnalyzerEngine:
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    ).create_engine()
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en"],
    )
    analyzer.registry.add_recognizer(AadhaarRecognizer())
    analyzer.registry.add_recognizer(IndianPhoneRecognizer())
    analyzer.registry.add_recognizer(IndianPanRecognizer())
    return analyzer


def resolve_overlaps(results: Sequence[RecognizerResult]) -> list[RecognizerResult]:
    ordered = sorted(
        results,
        key=lambda item: (-item.score, -(item.end - item.start), item.start),
    )
    kept: list[RecognizerResult] = []
    for result in ordered:
        if all(
            result.end <= existing.start or result.start >= existing.end
            for existing in kept
        ):
            kept.append(result)
    return sorted(kept, key=lambda item: item.start)


def mask_text(
    text: str,
    results: Sequence[RecognizerResult],
    state: MaskState,
) -> str:
    output: list[str] = []
    last = 0
    for result in results:
        if not 0 <= result.start < result.end <= len(text):
            raise ValueError("Analyzer returned a span outside the input text.")
        value = text[result.start : result.end]
        key = (result.entity_type, value)
        if key not in state.mapping:
            count = state.counts.get(result.entity_type, 0) + 1
            state.counts[result.entity_type] = count
            label = LABELS.get(result.entity_type, result.entity_type)
            state.mapping[key] = f"<{label}_{count}>"
        output.append(text[last : result.start])
        output.append(state.mapping[key])
        last = result.end
    output.append(text[last:])
    return "".join(output)


def mask_messages(
    analyzer: AnalyzerEngine,
    messages: Sequence[dict[str, Any]],
    entities: Sequence[str] = DEFAULT_ENTITIES,
    threshold: float = 0.5,
    max_input_chars: int = 200_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    masked_messages = deepcopy(messages)
    text_length = total_text_chars(masked_messages)
    if text_length > max_input_chars:
        raise InputTooLongError(
            f"Input is {text_length} characters; maximum is {max_input_chars}."
        )

    state = MaskState()
    report: dict[str, Any] = {"counts": {}, "total": 0}
    for masked_message in masked_messages:
        content = masked_message.get("content")
        if isinstance(content, str):
            results = resolve_overlaps(
                analyzer.analyze(
                    text=content,
                    language="en",
                    entities=list(entities),
                    score_threshold=threshold,
                )
            )
            masked_message["content"] = mask_text(content, results, state)
            _update_report(report, results)
        elif isinstance(content, list):
            masked_parts: list[Any] = []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    results = resolve_overlaps(
                        analyzer.analyze(
                            text=part["text"],
                            language="en",
                            entities=list(entities),
                            score_threshold=threshold,
                        )
                    )
                    masked_part = dict(part)
                    masked_part["text"] = mask_text(part["text"], results, state)
                    masked_parts.append(masked_part)
                    _update_report(report, results)
                else:
                    masked_parts.append(part)
            masked_message["content"] = masked_parts

    return masked_messages, report


def _update_report(
    report: dict[str, Any],
    results: Sequence[RecognizerResult],
) -> None:
    counts = report["counts"]
    for result in results:
        counts[result.entity_type] = counts.get(result.entity_type, 0) + 1
        report["total"] += 1


def mask_pii(
    text: str,
    recognitions: Sequence[Recognition] | None = None,
    score_threshold: float = 0.5,
) -> tuple[str, list[Recognition]]:
    matches = (
        list(recognitions)
        if recognitions is not None
        else recognize_pii(text, score_threshold)
    )
    masked_text = text
    for match in reversed(matches):
        if match.entity not in _MASKS:
            raise ValueError(f"Unsupported PII entity {match.entity!r}.")
        if not 0 <= match.start < match.end <= len(text):
            raise ValueError("PII recognition span is outside the input text.")
        masked_text = (
            masked_text[:match.start]
            + _MASKS[match.entity]
            + masked_text[match.end:]
        )
    return masked_text, matches
