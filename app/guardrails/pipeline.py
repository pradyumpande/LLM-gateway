from dataclasses import dataclass

from app.config import Settings
from app.guardrails.injection import score_injection
from app.guardrails.pii import InputTooLongError, mask_pii
from app.guardrails.recognizers import Recognition, recognize_pii


@dataclass(frozen=True)
class GuardrailResult:
    text: str
    pii_matches: tuple[Recognition, ...]
    pii_score: float
    injection_score: float
    injection_flagged: bool
    blocked: bool


def inspect_input(text: str, settings: Settings | None = None) -> GuardrailResult:
    settings = settings or Settings()
    if len(text) > settings.max_input_chars:
        raise InputTooLongError(
            f"Input is {len(text)} characters; maximum is "
            f"{settings.max_input_chars}."
        )

    pii_matches = recognize_pii(text, settings.pii_score_threshold)
    masked_text, pii_matches = mask_pii(text, pii_matches)
    injection_score = score_injection(text)

    return GuardrailResult(
        text=masked_text,
        pii_matches=tuple(pii_matches),
        pii_score=max(
            (match.score for match in pii_matches),
            default=0.0,
        ),
        injection_score=injection_score,
        injection_flagged=injection_score >= settings.injection_flag_threshold,
        blocked=injection_score >= settings.injection_block_threshold,
    )
