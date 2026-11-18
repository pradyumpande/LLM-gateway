from app.auth import generate_key, hash_key, key_prefix


def test_generated_keys_are_unique_and_prefix_is_short():
    first_key = generate_key()
    second_key = generate_key()

    assert first_key != second_key
    assert first_key.startswith("gw_")
    assert key_prefix(first_key) == first_key[:8]


def test_hash_key_is_deterministic():
    key = "gw_example-key"

    assert hash_key(key) == hash_key(key)
