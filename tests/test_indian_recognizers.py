import pytest

from app.guardrails.pipeline import inspect_input
from app.guardrails.recognizers import recognize_pii


@pytest.mark.parametrize(
    "value",
    ["2345 6789 0124", "234567890124"],
)
def test_valid_aadhaar_formats_are_recognized(value):
    matches = recognize_pii(value)

    assert len(matches) == 1
    assert matches[0].entity == "AADHAAR"
    assert matches[0].score == 1.0


def test_aadhaar_with_invalid_checksum_is_ignored():
    assert recognize_pii("2345 6789 0123") == []


@pytest.mark.parametrize(
    "value",
    [
        "98765 43210",
        "+91 98765 43210",
        "+919876543210",
        "09876543210",
        "98765-43210",
    ],
)
def test_indian_mobile_formats_are_recognized(value):
    matches = recognize_pii(value)

    assert len(matches) == 1
    assert matches[0].entity == "IN_PHONE"
    assert matches[0].score == 0.6


def test_mobile_number_starting_with_one_is_not_recognized():
    assert recognize_pii("12345 67890") == []


def test_pan_format_is_recognized():
    matches = recognize_pii("ABCPE1234F")

    assert len(matches) == 1
    assert matches[0].entity == "IN_PAN"
    assert matches[0].score == 0.85


def test_pipeline_masks_custom_indian_pii_entities():
    result = inspect_input("IDs: 234567890124, 98765 43210, ABCPE1234F")

    assert result.text == "IDs: <AADHAAR>, <PHONE>, <PAN>"


def test_phone_inside_longer_ascii_digit_sequence_is_not_recognized():
    assert recognize_pii("123456789876543210") == []
