from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import re
import unicodedata
from typing import Any


_ZERO_WIDTH = dict.fromkeys(
    map(ord, "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff"),
    None,
)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).lower()
    return re.sub(r"[ \t\r\f\v]+", " ", text)


_RULE_FLAGS = re.IGNORECASE | re.MULTILINE
_RULES = (
    (
        "override_instructions",
        0.6,
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b[^\n]{0,40}"
            r"\b(?:previous|prior|above|earlier|preceding|system|your|all)\b"
            r"[^\n]{0,30}\b(?:instructions?|prompts?|rules|guidelines|directions)\b",
            _RULE_FLAGS,
        ),
    ),
    (
        "reveal_prompt",
        0.5,
        re.compile(
            r"\b(?:reveal|show|print|repeat|display|output|leak|disclose)\b"
            r"[^\n]{0,30}\b(?:system|hidden|initial|original|secret|developer)\b"
            r"[^\n]{0,15}\b(?:prompt|instructions?|message)\b",
            _RULE_FLAGS,
        ),
    ),
    (
        "ask_system_prompt",
        0.4,
        re.compile(
            r"\bwhat (?:is|are) your (?:system|initial|original|hidden) "
            r"(?:prompt|instructions?)\b",
            _RULE_FLAGS,
        ),
    ),
    (
        "persona_switch",
        0.3,
        re.compile(
            r"\b(?:you are now|from now on,? you|pretend (?:to be|that you)|"
            r"act as if you)\b",
            _RULE_FLAGS,
        ),
    ),
    (
        "no_restrictions",
        0.3,
        re.compile(
            r"\b(?:no|without|free (?:of|from)) (?:any )?"
            r"(?:restrictions?|limitations?|filters?|guidelines|censorship|safety)\b",
            _RULE_FLAGS,
        ),
    ),
    (
        "jailbreak_terms",
        0.3,
        re.compile(
            r"\b(?:dan mode|do anything now|jailbreak(?:ed|ing)?)\b",
            _RULE_FLAGS,
        ),
    ),
    ("developer_mode", 0.3, re.compile(r"\bdeveloper mode\b", _RULE_FLAGS)),
    (
        "chat_delimiters",
        0.3,
        re.compile(
            r"<\|(?:im_start|im_end|system|user|assistant|endoftext)\|>",
            _RULE_FLAGS,
        ),
    ),
    (
        "llama_delimiters",
        0.3,
        re.compile(r"\[/?inst\]|<</?sys>>|</?system>", _RULE_FLAGS),
    ),
    (
        "fake_role_line",
        0.4,
        re.compile(r"^[ \t]{0,20}(?:system|assistant)[ \t]{0,10}:", _RULE_FLAGS),
    ),
    (
        "base64_blob",
        0.3,
        re.compile(r"[a-z0-9+/]{100,}={0,2}", _RULE_FLAGS),
    ),
)


@dataclass(frozen=True)
class InjectionResult:
    score: float
    rules: tuple[str, ...]
    flagged: bool
    blocked: bool


def score_injection(
    text: str,
    flag_threshold: float = 0.4,
    block_threshold: float = 0.8,
) -> InjectionResult:
    normalized_text = normalise(text)
    matched_rules = tuple(
        name
        for name, _, pattern in _RULES
        if pattern.search(normalized_text) is not None
    )
    weight_by_name = {name: weight for name, weight, _ in _RULES}
    score = min(1.0, sum(weight_by_name[name] for name in matched_rules))
    return InjectionResult(
        score=score,
        rules=matched_rules,
        flagged=score >= flag_threshold,
        blocked=score >= block_threshold,
    )


def iter_texts(
    messages: Iterable[Mapping[str, Any]],
    roles: Iterable[str],
) -> Iterable[str]:
    allowed_roles = frozenset(roles)
    for message in messages:
        if message.get("role") not in allowed_roles:
            continue
        content = message.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    yield part["text"]


def score_messages(
    messages: Sequence[Mapping[str, Any]],
    roles: Iterable[str] = ("user", "tool"),
    flag_threshold: float = 0.4,
    block_threshold: float = 0.8,
) -> InjectionResult:
    scores = (
        score_injection(text, flag_threshold, block_threshold)
        for text in iter_texts(messages, roles)
    )
    return max(scores, key=lambda result: result.score, default=InjectionResult(
        score=0.0,
        rules=(),
        flagged=0.0 >= flag_threshold,
        blocked=0.0 >= block_threshold,
    ))
