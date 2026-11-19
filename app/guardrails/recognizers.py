from dataclasses import dataclass
import re

from presidio_analyzer import Pattern, PatternRecognizer

from app.guardrails import verhoeff


class AadhaarRecognizer(PatternRecognizer):
    PATTERNS = [
        Pattern(
            name="aadhaar_12_digits",
            regex=r"(?<![0-9])[2-9][0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}(?![0-9])",
            score=0.3,
        )
    ]
    CONTEXT = ["aadhaar", "aadhar", "uidai", "uid"]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="AADHAAR",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )

    def validate_result(self, pattern_text: str) -> bool:
        digits = re.sub(r"[\s\-]", "", pattern_text)
        return verhoeff.validate(digits)


class IndianPhoneRecognizer(PatternRecognizer):
    PATTERNS = [
        Pattern(
            name="in_mobile",
            regex=(
                r"(?<![0-9])(?:(?:\+|00)91[\s\-]?|91[\s\-]?|0)?"
                r"[6-9][0-9]{4}[\s\-]?[0-9]{5}(?![0-9])"
            ),
            score=0.6,
        )
    ]
    CONTEXT = [
        "phone",
        "mobile",
        "mob",
        "cell",
        "call",
        "contact",
        "whatsapp",
        "tel",
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="IN_PHONE",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )


class IndianPanRecognizer(PatternRecognizer):
    PATTERNS = [
        Pattern(
            name="in_pan",
            regex=r"\b[A-Z]{3}[ABCFGHLJPT][A-Z][0-9]{4}[A-Z]\b",
            score=0.85,
        )
    ]
    CONTEXT = ["pan", "permanent account number", "income tax"]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="IN_PAN",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )


@dataclass(frozen=True)
class Recognition:
    entity: str
    start: int
    end: int
    score: float


_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w-])")
_SSN = re.compile(r"(?<!\d)(\d{3})[- ]?(\d{2})[- ]?(\d{4})(?!\d)")
_NUMBER_SEQUENCE = re.compile(r"(?<!\w)(?:\+?\d[\d(). -]{5,}\d)(?!\w)")
_CUSTOM_RECOGNIZER_TYPES = (
    AadhaarRecognizer,
    IndianPhoneRecognizer,
    IndianPanRecognizer,
)


def _luhn_valid(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False

    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _ssn_valid(match: re.Match[str]) -> bool:
    area, group, serial = match.groups()
    return (
        area != "000"
        and area != "666"
        and int(area) < 900
        and group != "00"
        and serial != "0000"
    )


def recognize_pii(
    text: str,
    score_threshold: float = 0.5,
) -> list[Recognition]:
    if not 0 <= score_threshold <= 1:
        raise ValueError("PII score threshold must be between 0 and 1.")

    recognitions: list[Recognition] = []
    for match in _EMAIL.finditer(text):
        recognitions.append(Recognition("email", match.start(), match.end(), 0.99))

    for match in _SSN.finditer(text):
        if _ssn_valid(match):
            recognitions.append(Recognition("ssn", match.start(), match.end(), 0.98))

    for match in _NUMBER_SEQUENCE.finditer(text):
        digits = "".join(character for character in match.group() if character.isdigit())
        if _luhn_valid(digits):
            recognitions.append(
                Recognition("credit_card", match.start(), match.end(), 0.95)
            )

    for recognizer_type in _CUSTOM_RECOGNIZER_TYPES:
        recognizer = recognizer_type()
        for result in recognizer.analyze(text, entities=[recognizer.supported_entities[0]]):
            recognitions.append(
                Recognition(result.entity_type, result.start, result.end, result.score)
            )

    candidates = sorted(
        (
            recognition
            for recognition in recognitions
            if recognition.score >= score_threshold
        ),
        key=lambda item: (-item.score, -(item.end - item.start), item.start),
    )
    selected: list[Recognition] = []
    for candidate in candidates:
        if not any(
            candidate.start < existing.end and existing.start < candidate.end
            for existing in selected
        ):
            selected.append(candidate)

    return sorted(selected, key=lambda item: item.start)
