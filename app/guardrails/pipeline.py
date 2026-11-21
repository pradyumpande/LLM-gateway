import asyncio
from dataclasses import dataclass
import time
from typing import Any, Mapping

from app.auth import AuthContext
from app.config import Settings
from app.guardrails.injection import score_injection, score_messages
from app.guardrails.pii import (
    InputTooLongError,
    mask_messages,
    mask_pii,
    total_text_chars,
)
from app.guardrails.recognizers import Recognition, recognize_pii


@dataclass(frozen=True)
class GuardrailResult:
    payload: dict[str, Any]
    pii_counts: dict[str, int]
    pii_total: int
    injection_score: float
    injection_rules: tuple[str, ...]
    injection_flagged: bool
    block: bool
    block_code: str | None
    block_message: str | None
    block_status: int | None
    elapsed_ms: float


async def run_input_guardrails(
    payload: Mapping[str, Any],
    ctx: AuthContext,
    state: Any,
) -> GuardrailResult:
    started_at = time.perf_counter()
    copied_payload = dict(payload)
    injection_score = 0.0
    injection_rules: tuple[str, ...] = ()
    injection_flagged = False

    def result(
        *,
        pii_counts: dict[str, int] | None = None,
        pii_total: int = 0,
        block_code: str | None = None,
        block_message: str | None = None,
        block_status: int | None = None,
    ) -> GuardrailResult:
        return GuardrailResult(
            payload=copied_payload,
            pii_counts=pii_counts or {},
            pii_total=pii_total,
            injection_score=injection_score,
            injection_rules=injection_rules,
            injection_flagged=injection_flagged,
            block=block_code is not None,
            block_code=block_code,
            block_message=block_message,
            block_status=block_status,
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
        )

    messages = copied_payload.get("messages", [])
    settings: Settings = state.settings
    if total_text_chars(messages) > settings.max_input_chars:
        return result(
            block_code="request_too_large",
            block_message=(
                f"Request exceeds the maximum input size of "
                f"{settings.max_input_chars} characters."
            ),
            block_status=413,
        )

    injection_result = await asyncio.to_thread(
        score_messages,
        messages,
        ("user", "tool"),
        settings.injection_flag_threshold,
        settings.injection_block_threshold,
    )
    injection_score = injection_result.score
    injection_rules = injection_result.rules
    injection_flagged = injection_result.flagged
    if injection_result.blocked:
        return result(
            block_code="prompt_injection_detected",
            block_message="Request blocked by the prompt-injection guardrail.",
            block_status=400,
        )

    masked_messages, report = await asyncio.to_thread(
        mask_messages,
        state.analyzer,
        messages,
        settings.pii_entities,
        settings.pii_score_threshold,
        settings.max_input_chars,
    )
    pii_counts = report["counts"]
    pii_total = report["total"]

    if ctx.pii_action == "mask":
        if pii_total:
            copied_payload["messages"] = masked_messages
    elif ctx.pii_action == "block":
        if pii_total:
            types = ", ".join(pii_counts)
            return result(
                pii_counts=pii_counts,
                pii_total=pii_total,
                block_code="pii_detected",
                block_message=(
                    f"Request contains sensitive data (types: {types}). "
                    "Remove it and retry."
                ),
                block_status=400,
            )
    elif ctx.pii_action != "log_only":
        return result(
            pii_counts=pii_counts,
            pii_total=pii_total,
            block_code="invalid_pii_action",
            block_message=f"Unsupported PII action {ctx.pii_action!r}.",
            block_status=500,
        )

    return result(pii_counts=pii_counts, pii_total=pii_total)


@dataclass(frozen=True)
class LegacyGuardrailResult:
    text: str
    pii_matches: tuple[Recognition, ...]
    pii_score: float
    injection_score: float
    injection_flagged: bool
    blocked: bool


def inspect_input(text: str, settings: Settings | None = None) -> LegacyGuardrailResult:
    settings = settings or Settings()
    if len(text) > settings.max_input_chars:
        raise InputTooLongError(
            f"Input is {len(text)} characters; maximum is "
            f"{settings.max_input_chars}."
        )

    pii_matches = recognize_pii(text, settings.pii_score_threshold)
    masked_text, pii_matches = mask_pii(text, pii_matches)
    injection_result = score_injection(
        text,
        settings.injection_flag_threshold,
        settings.injection_block_threshold,
    )

    return LegacyGuardrailResult(
        text=masked_text,
        pii_matches=tuple(pii_matches),
        pii_score=max((match.score for match in pii_matches), default=0.0),
        injection_score=injection_result.score,
        injection_flagged=injection_result.flagged,
        blocked=injection_result.blocked,
    )
