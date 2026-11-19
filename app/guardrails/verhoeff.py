_MULTIPLICATION_TABLE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

_PERMUTATION_TABLE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)

_INVERSE_TABLE = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def _is_ascii_digits(value: str) -> bool:
    return value.isascii() and value.isdigit()


def validate(num: str) -> bool:
    """Return whether a full number, including its check digit, is valid."""
    if not _is_ascii_digits(num):
        return False

    checksum = 0
    for index, character in enumerate(reversed(num)):
        checksum = _MULTIPLICATION_TABLE[checksum][
            _PERMUTATION_TABLE[index % 8][int(character)]
        ]
    return checksum == 0


def check_digit(num: str) -> str:
    """Return the Verhoeff digit to append to ``num``."""
    if not _is_ascii_digits(num):
        raise ValueError("digits only")

    checksum = 0
    for index, character in enumerate(reversed(num)):
        checksum = _MULTIPLICATION_TABLE[checksum][
            _PERMUTATION_TABLE[(index + 1) % 8][int(character)]
        ]
    return str(_INVERSE_TABLE[checksum])


def verhoeff_check_digit(number: str) -> str:
    return check_digit(number)


def is_valid_verhoeff(value: str) -> bool:
    return validate(value)
