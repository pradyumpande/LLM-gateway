from collections.abc import Sequence
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
LABELS = {
    "EMAIL_ADDRESS": "EMAIL",
    "IN_PHONE": "PHONE",
    "AADHAAR": "AADHAAR",
    "IN_PAN": "PAN",
    "CREDIT_CARD": "CARD",
    "PERSON": "PERSON",
}

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


def _mask_text(
    text: str,
    analyzer: AnalyzerEngine,
    score_threshold: float,
) -> tuple[str, list[RecognizerResult]]:
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=DEFAULT_ENTITIES,
        score_threshold=score_threshold,
    )
    selected: list[RecognizerResult] = []
    for result in sorted(
        results,
        key=lambda item: (-item.score, -(item.end - item.start), item.start),
    ):
        if not any(
            result.start < existing.end and existing.start < result.end
            for existing in selected
        ):
            selected.append(result)

    masked_text = text
    for result in sorted(selected, key=lambda item: item.start, reverse=True):
        if not 0 <= result.start < result.end <= len(text):
            raise ValueError("Analyzer returned a span outside the input text.")
        label = LABELS.get(result.entity_type, result.entity_type)
        masked_text = (
            masked_text[:result.start]
            + f"<{label}>"
            + masked_text[result.end:]
        )
    return masked_text, sorted(selected, key=lambda item: item.start)


def mask_messages(
    messages: Sequence[dict[str, Any]],
    analyzer: AnalyzerEngine,
    score_threshold: float = 0.5,
    max_input_chars: int = 200_000,
) -> tuple[list[dict[str, Any]], list[RecognizerResult]]:
    text_length = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text_length += len(content)
        elif isinstance(content, list):
            text_length += sum(
                len(part["text"])
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )

    if text_length > max_input_chars:
        raise InputTooLongError(
            f"Input is {text_length} characters; maximum is {max_input_chars}."
        )

    masked_messages: list[dict[str, Any]] = []
    all_results: list[RecognizerResult] = []
    for message in messages:
        masked_message = dict(message)
        content = message.get("content")
        if isinstance(content, str):
            masked_content, results = _mask_text(
                content,
                analyzer,
                score_threshold,
            )
            masked_message["content"] = masked_content
            all_results.extend(results)
        elif isinstance(content, list):
            masked_parts: list[Any] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    masked_text, results = _mask_text(
                        part["text"],
                        analyzer,
                        score_threshold,
                    )
                    masked_part = dict(part)
                    masked_part["text"] = masked_text
                    masked_parts.append(masked_part)
                    all_results.extend(results)
                else:
                    masked_parts.append(part)
            masked_message["content"] = masked_parts
        masked_messages.append(masked_message)

    return masked_messages, all_results


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
