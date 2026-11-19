from dataclasses import dataclass
import re


@dataclass(frozen=True)
class InjectionFinding:
    start: int
    end: int
    score: float


_INJECTION_PATTERNS = (
    (re.compile(r"\bignore\s+(?:all\s+)?previous\s+instructions?\b", re.I), 0.85),
    (re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous\s+)?instructions?\b", re.I), 0.85),
    (re.compile(r"\bignore\s+all\s+rules\b", re.I), 0.85),
    (re.compile(r"\breveal\s+(?:the\s+)?system\s+prompt\b", re.I), 0.9),
    (re.compile(r"\b(?:system|developer)\s+prompt\b", re.I), 0.55),
    (re.compile(r"\bdo\s+anything\s+now\b", re.I), 0.8),
    (re.compile(r"\bjailbreak\b", re.I), 0.7),
)


def find_injection_signals(text: str) -> list[InjectionFinding]:
    findings = [
        InjectionFinding(match.start(), match.end(), score)
        for pattern, score in _INJECTION_PATTERNS
        for match in pattern.finditer(text)
    ]
    return sorted(findings, key=lambda finding: finding.start)


def score_injection(text: str) -> float:
    return max(
        (finding.score for finding in find_injection_signals(text)),
        default=0.0,
    )
