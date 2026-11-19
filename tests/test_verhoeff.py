import pytest

from app.guardrails.verhoeff import check_digit, validate


def test_known_verhoeff_vectors():
    assert check_digit("236") == "3"
    assert validate("2363")
    assert not validate("2364")
    assert validate("234567890124")
    assert not validate("234567890123")


def test_valid_number_rejects_each_single_digit_substitution():
    base = "34567890123"
    number = base + check_digit(base)

    assert number.startswith(("2", "3", "4", "5", "6", "7", "8", "9"))
    assert validate(number)

    for index, original in enumerate(number):
        for replacement in "0123456789":
            if replacement == original:
                continue
            changed = number[:index] + replacement + number[index + 1 :]
            assert not validate(changed), (index, replacement, changed)


def test_valid_number_rejects_each_different_adjacent_swap():
    base = "34567890123"
    number = base + check_digit(base)

    for index in range(len(number) - 1):
        if number[index] == number[index + 1]:
            continue
        swapped = (
            number[:index]
            + number[index + 1]
            + number[index]
            + number[index + 2 :]
        )
        assert not validate(swapped), (index, swapped)


@pytest.mark.parametrize("value", ["", "abc", "²", "１２３", "123a"])
def test_validate_returns_false_for_non_ascii_digits(value):
    assert not validate(value)


@pytest.mark.parametrize("value", ["", "abc", "²", "１２３", "123a"])
def test_check_digit_rejects_non_ascii_digits(value):
    with pytest.raises(ValueError, match="digits only"):
        check_digit(value)
