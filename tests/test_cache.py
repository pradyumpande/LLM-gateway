import json
import sys
import time
import types

import pytest
from qdrant_client import AsyncQdrantClient

from app.cache.embedder import FastEmbedEmbedder
from app.cache.keys import (
    context_hash,
    exact_key,
    normalise_messages,
    num_sig,
    params_hash,
    semantic_scope,
    sha,
)
from app.cache.policy import CacheInfo, build_cache_info
from app.cache.semantic import SemanticCache, collection_name
from app.cache.service import (
    CacheService,
    LookupResult,
    cacheable_response,
    materialise,
)
from app.config import Settings


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.get_error = None
        self.set_error = None

    async def get(self, key):
        if self.get_error is not None:
            raise self.get_error
        return self.values.get(key)

    async def set(self, key, value, ex):
        if self.set_error is not None:
            raise self.set_error
        _ = ex
        self.values[key] = value.encode("utf-8")
        return True


class FakeEmbedder:
    name = "test-model"
    dim = 2

    def __init__(self):
        self.calls = []

    async def embed(self, text):
        self.calls.append(text)
        return [1.0, 0.0]


def _settings(tmp_path, **values):
    return Settings(_env_file=None, fastembed_cache_dir=tmp_path, **values)


def _build_info(settings, payload, team_id=1, **overrides):
    options = {
        "enabled": settings.cache_enabled,
        "max_temperature": settings.cache_max_temperature,
        "pii_found": False,
        "injection_flagged": False,
        "cache_control": None,
        "semantic_enabled": settings.semantic_enabled,
        "sem_min": settings.semantic_min_chars,
        "sem_max": settings.semantic_max_chars,
    }
    options.update(overrides)
    return build_cache_info(payload, team_id, **options)


def _cache_service(redis, settings, semantic=None, embedder=None, **options):
    return CacheService(
        redis,
        semantic,
        embedder or FakeEmbedder(),
        ttl_s=settings.cache_ttl_seconds,
        max_bytes=options.get("max_bytes", settings.cache_max_bytes),
        lookup_timeout_s=settings.cache_lookup_timeout_s,
    )


def _complete_response(content="A complete answer."):
    return {
        "id": "chatcmpl-original",
        "created": 123,
        "model": "provider-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }


def test_normalized_messages_strip_only_outer_whitespace_without_mutating():
    messages = [
        {"role": "user", "content": "  first  line\n  second  line  "},
        {"role": "assistant", "content": None},
    ]
    original = [dict(message) for message in messages]

    normalized = normalise_messages(messages)

    assert normalized[0]["content"] == "first  line\n  second  line"
    assert normalized[1] == messages[1]
    assert messages == original


def test_exact_keys_and_hashes_are_stable_and_properly_scoped():
    payload = {
        "model": "alias",
        "messages": [{"role": "user", "content": "  hello  "}],
        "temperature": 0.4,
        "user": "ignored-for-params",
    }
    p_hash = params_hash(payload)

    assert p_hash == params_hash(
        {
            "user": "different",
            "temperature": 0.4,
            "messages": payload["messages"],
            "model": "other-alias",
        }
    )
    assert p_hash != params_hash({**payload, "temperature": 0.5})
    assert exact_key(4, "alias", payload["messages"], p_hash) == exact_key(
        4, "alias", [{"role": "user", "content": "hello"}], p_hash
    )
    assert exact_key(4, "alias", payload["messages"], p_hash) != exact_key(
        5, "alias", payload["messages"], p_hash
    )
    assert exact_key(4, "other-alias", payload["messages"], p_hash) != exact_key(
        4, "alias", payload["messages"], p_hash
    )
    assert exact_key(
        4,
        "alias",
        payload["messages"],
        params_hash({**payload, "temperature": 0.5}),
    ) != exact_key(4, "alias", payload["messages"], p_hash)
    assert exact_key(
        4,
        "alias",
        [{"role": "user", "content": "different"}],
        p_hash,
    ) != exact_key(4, "alias", payload["messages"], p_hash)
    reordered_payload = {
        "user": "ignored-for-params",
        "temperature": 0.4,
        "messages": payload["messages"],
        "model": "alias",
    }
    assert params_hash(reordered_payload) == params_hash(payload)
    assert sha("cache") == sha("cache")


def test_context_hash_excludes_last_message_but_scopes_prior_turns():
    first = [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Earlier question"},
        {"role": "user", "content": "  Current question  "},
    ]
    changed_last = [*first[:-1], {"role": "user", "content": "Different question"}]
    changed_context = [
        {"role": "system", "content": "Be detailed"},
        *first[1:],
    ]
    changed_earlier_turn = [
        first[0],
        {"role": "user", "content": "A different earlier question"},
        *first[2:],
    ]

    assert context_hash(first) == context_hash(changed_last)
    assert context_hash(first) != context_hash(changed_context)
    assert context_hash(first) != context_hash(changed_earlier_turn)


def test_number_signature_is_ordered_and_ignores_prompts_without_numbers():
    assert num_sig("2+2") != num_sig("2+3")
    assert num_sig("no digits") == "none"
    assert num_sig("5 to 3") != num_sig("3 to 5")
    assert num_sig("change 5 to 3") != num_sig("change 3 to 5")
    assert num_sig("version 1.2, then 3") == num_sig("version 1.2, then 3")
    assert num_sig("no numeric values") == "none"


def test_cache_policy_allows_absent_temperature_and_builds_semantic_metadata(
    tmp_path,
):
    settings = _settings(tmp_path)
    payload = {
        "model": "alias",
        "messages": [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "  Explain cache keys using 3 examples.  "},
        ],
    }

    info = _build_info(settings, payload)

    assert info.read and info.write
    assert info.bypass_reason is None
    assert info.model_requested == "alias"
    assert info.exact_key.startswith("cache:exact:1:")
    assert info.semantic_text == "Explain cache keys using 3 examples."
    assert info.num_sig == num_sig(info.semantic_text)
    assert info.text_hash == sha(info.semantic_text)[:32]
    assert info.context_hash == context_hash(payload["messages"])


@pytest.mark.parametrize(
    ("payload_update", "reason"),
    [
        ({"stream": True}, "stream"),
        ({"tools": []}, "tools"),
        ({"functions": []}, "tools"),
        ({"tool_choice": "auto"}, "tools"),
        ({"function_call": "auto"}, "tools"),
        ({"n": 2}, "n_gt_1"),
        ({"temperature": 0.8}, "temperature"),
    ],
)
def test_cache_policy_bypasses_unsupported_request_options(
    tmp_path, payload_update, reason
):
    settings = _settings(tmp_path)
    payload = {
        "model": "alias",
        "messages": [{"role": "user", "content": "Tell me about caching"}],
        **payload_update,
    }

    info = _build_info(settings, payload)

    assert info.bypass_reason == reason
    assert not info.read and not info.write
    assert info.exact_key is None


def test_cache_control_sensitive_content_and_disabled_policy(tmp_path):
    settings = _settings(tmp_path)
    payload = {
        "model": "alias",
        "messages": [{"role": "user", "content": "Tell me about caching"}],
    }

    no_cache = _build_info(settings, payload, cache_control="private, no-cache")
    assert no_cache.read is False
    assert no_cache.write is True
    assert no_cache.bypass_reason == "no-cache"

    no_store = _build_info(settings, payload, cache_control="no-store")
    assert no_store.bypass_reason == "no-store"
    assert no_store.read is False and no_store.write is False

    pii = _build_info(settings, payload, pii_found=True)
    injection = _build_info(settings, payload, injection_flagged=True)
    disabled = _build_info(settings, payload, enabled=False)
    assert pii.bypass_reason == "pii"
    assert injection.bypass_reason == "injection_flag"
    assert disabled.bypass_reason == "disabled"
    assert all(not info.read and not info.write for info in (pii, injection, disabled))


@pytest.mark.parametrize(
    ("payload_update", "policy_update", "bypass_reason", "read", "write"),
    [
        ({"stream": True}, {}, "stream", False, False),
        ({"tools": []}, {}, "tools", False, False),
        ({"n": 2}, {}, "n_gt_1", False, False),
        ({"temperature": 0.9}, {}, "temperature", False, False),
        ({}, {}, None, True, True),
        ({"temperature": 0.2}, {}, None, True, True),
        ({}, {"pii_found": True}, "pii", False, False),
        ({}, {"injection_flagged": True}, "injection_flag", False, False),
        (
            {},
            {"cache_control": "no-cache"},
            "no-cache",
            False,
            True,
        ),
        (
            {},
            {"cache_control": "no-store"},
            "no-store",
            False,
            False,
        ),
    ],
)
def test_build_cache_info_policy_table(
    tmp_path,
    payload_update,
    policy_update,
    bypass_reason,
    read,
    write,
):
    settings = _settings(tmp_path)
    payload = {
        "model": "alias",
        "messages": [{"role": "user", "content": "What is semantic caching?"}],
        **payload_update,
    }

    info = _build_info(settings, payload, **policy_update)

    assert info.bypass_reason == bypass_reason
    assert info.read is read
    assert info.write is write


@pytest.mark.parametrize(
    "content",
    [
        "abc",
        "x" * 5000,
        [{"type": "text", "text": "What is semantic caching?"}],
    ],
)
def test_build_cache_info_disables_semantic_text_for_ineligible_content(
    tmp_path,
    content,
):
    settings = _settings(tmp_path)
    payload = {
        "model": "alias",
        "messages": [{"role": "user", "content": content}],
    }

    assert _build_info(settings, payload).semantic_text is None


def test_semantic_eligibility_uses_only_short_plain_text_last_user_message(tmp_path):
    settings = _settings(tmp_path, semantic_min_chars=8, semantic_max_chars=30)
    base = {"model": "alias", "messages": []}

    assert _build_info(
        settings,
        {**base, "messages": [{"role": "user", "content": "short"}]},
    ).semantic_text is None
    assert _build_info(
        settings,
        {
            **base,
            "messages": [
                {"role": "user", "content": "A prior user request"},
                {"role": "assistant", "content": "A reply"},
                {"role": "user", "content": "Explain cache keys"},
            ],
        },
    ).semantic_text == "Explain cache keys"
    assert _build_info(
        settings,
        {**base, "messages": [{"role": "assistant", "content": "Explain cache keys"}]},
    ).semantic_text is None
    assert _build_info(
        settings,
        {
            **base,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "long enough"}]}
            ],
        },
    ).semantic_text is None
    assert _build_info(
        settings,
        {**base, "messages": [{"role": "user", "content": "x" * 31}]},
    ).semantic_text is None


@pytest.mark.asyncio
async def test_fastembed_measures_dimension_and_embeds_off_the_event_loop(
    monkeypatch, tmp_path
):
    captured = {}
    embedded_texts = []

    class FakeTextEmbedding:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def embed(self, texts):
            embedded_texts.extend(texts)
            yield [0.25, 0.75]

    fake_module = types.ModuleType("fastembed")
    fake_module.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)

    embedder = FastEmbedEmbedder("test-model", tmp_path / "embeddings")

    assert embedder.name == "test-model"
    assert embedder.dim == 2
    assert embedded_texts == ["warm up"]
    assert await embedder.embed("some query") == [0.25, 0.75]
    assert embedded_texts == ["warm up", "some query"]
    assert captured == {
        "model_name": "test-model",
        "cache_dir": str(tmp_path / "embeddings"),
    }


@pytest.mark.asyncio
async def test_semantic_cache_scopes_and_supports_team_and_global_purges():
    client = AsyncQdrantClient(location=":memory:")
    cache = SemanticCache(
        client=client,
        collection=collection_name("test-model"),
        dim=2,
        threshold=0.9,
        ttl_s=60,
        embed_model="test-model",
    )
    scope = semantic_scope(1, "model-a", "params", "context", "3")
    response = {"choices": [{"message": {"content": "A cache stores results."}}]}
    entry = {
        "response": response,
        "cost_usd": 0.002,
        "model_alias": "model-a",
        "provider": "fake",
        "created_at": int(time.time()),
    }

    try:
        await cache.store(
            [1.0, 0.0],
            text_hash=sha("tell me about cache")[:32],
            entry=entry,
            **scope,
        )
        assert cache.collection == collection_name("test-model")
        await cache.ensure_collection()

        hit = await cache.lookup(
            [1.0, 0.0],
            **scope,
        )
        assert hit is not None
        assert hit.score == pytest.approx(1.0)
        assert hit.payload["response"] == response
        assert hit.payload["cost_usd"] == 0.002
        assert await cache.count_team(1) == 1
        assert await cache.count_team(2) == 0

        for other_scope in (
            semantic_scope(2, "model-a", "params", "context", "3"),
            semantic_scope(1, "model-a", "params", "other-context", "3"),
            semantic_scope(1, "model-a", "other-params", "context", "3"),
            semantic_scope(1, "model-a", "params", "context", "4"),
        ):
            assert await cache.lookup([1.0, 0.0], **other_scope) is None

        await cache.purge_team(1)
        assert await cache.count_team(1) == 0

        await cache.store(
            [1.0, 0.0],
            text_hash=sha("tell me about cache")[:32],
            entry=entry,
            **scope,
        )
        await cache.purge_all()
        assert await cache.count_team(1) == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_semantic_cache_ttl_filters_and_purges_expired_points():
    client = AsyncQdrantClient(location=":memory:")
    cache = SemanticCache(
        client=client,
        collection=collection_name("ttl-test"),
        dim=2,
        threshold=0.9,
        ttl_s=1,
        embed_model="ttl-test",
    )
    scope = semantic_scope(1, "model-a", "params", "context", "none")
    expired_entry = {
        "response": {"choices": []},
        "cost_usd": 0.0,
        "model_alias": "model-a",
        "provider": "fake",
        "created_at": int(time.time()) - 10,
    }

    try:
        await cache.store(
            [1.0, 0.0],
            text_hash="expired",
            entry=expired_entry,
            **scope,
        )
        assert await cache.lookup([1.0, 0.0], **scope) is None
        await cache.purge_expired()
        assert await cache.count_team(1) == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_semantic_cache_round_trips_a_four_dimensional_point():
    client = AsyncQdrantClient(location=":memory:")
    cache = SemanticCache(
        client=client,
        collection=collection_name("four-dimensional-test"),
        dim=4,
        threshold=0.9,
        ttl_s=60,
        embed_model="four-dimensional-test",
    )
    scope = semantic_scope(3, "model-a", "params", "context", "none")
    vector = [0.1, 0.2, 0.3, 0.4]

    try:
        await cache.store(
            vector,
            text_hash="four-dimensional-point",
            entry={
                "response": {"choices": []},
                "cost_usd": 0.0,
                "model_alias": "model-a",
                "provider": "fake",
                "created_at": int(time.time()),
            },
            **scope,
        )

        hit = await cache.lookup(vector, **scope)

        assert hit is not None
        assert hit.payload["response"] == {"choices": []}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cache_service_honors_no_cache_read_but_refreshes_exact_entry(tmp_path):
    settings = _settings(tmp_path)
    redis = FakeRedis()
    service = _cache_service(redis, settings)
    payload = {
        "model": "model-a",
        "temperature": 0.4,
        "messages": [{"role": "user", "content": "Explain cache keys"}],
    }
    info = _build_info(settings, payload, cache_control="no-cache")
    response = _complete_response()

    assert await service.lookup(info) == LookupResult()
    await service.store(
        info,
        None,
        response,
        cost_usd=0.0,
        model_alias="model-a",
        provider="fake",
    )
    cached = json.loads(redis.values[info.exact_key])
    assert cached["response"] == response
    assert cached["cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_cache_service_does_not_store_sensitive_requests(tmp_path):
    settings = _settings(tmp_path)
    redis = FakeRedis()
    service = _cache_service(redis, settings)
    payload = {
        "model": "model-a",
        "temperature": 0.1,
        "messages": [{"role": "user", "content": "Explain cache keys"}],
    }
    info = _build_info(settings, payload, pii_found=True)

    await service.store(
        info,
        None,
        _complete_response(),
        cost_usd=0.0,
        model_alias="model-a",
        provider="fake",
    )
    assert redis.values == {}


@pytest.mark.asyncio
async def test_cache_service_materializes_exact_response_without_mutation(tmp_path):
    settings = _settings(tmp_path)
    redis = FakeRedis()
    service = _cache_service(redis, settings)
    payload = {
        "model": "model-a",
        "temperature": 0.4,
        "messages": [{"role": "user", "content": "Explain cache keys"}],
    }
    info = _build_info(settings, payload)
    response = _complete_response()
    original_response = json.loads(json.dumps(response))

    assert (await service.lookup(info)).kind is None
    await service.store(
        info,
        None,
        response,
        cost_usd=0.002,
        model_alias="model-a",
        provider="fake",
    )
    result = await service.lookup(info)

    assert result.kind == "exact"
    assert result.entry["response"] == response
    assert result.age_s is not None and result.age_s >= 0
    materialized = materialise(result.entry)
    assert materialized["id"].startswith("chatcmpl-cache-")
    assert materialized["id"] != response["id"]
    assert materialized["created"] >= 123
    assert materialized["usage"] == response["usage"]
    assert response == original_response


@pytest.mark.asyncio
async def test_semantic_lookup_vector_is_reused_when_storing(tmp_path):
    settings = _settings(tmp_path)
    redis = FakeRedis()
    client = AsyncQdrantClient(location=":memory:")
    semantic_cache = SemanticCache(
        client=client,
        collection=collection_name("test-model"),
        dim=2,
        threshold=0.9,
        ttl_s=settings.cache_ttl_seconds,
        embed_model="test-model",
    )
    embedder = FakeEmbedder()
    service = CacheService(
        redis,
        semantic_cache,
        embedder,
        ttl_s=settings.cache_ttl_seconds,
        max_bytes=settings.cache_max_bytes,
        lookup_timeout_s=settings.cache_lookup_timeout_s,
    )
    payload = {
        "model": "model-a",
        "temperature": 0.4,
        "messages": [{"role": "user", "content": "Explain semantic caching"}],
    }
    info = _build_info(settings, payload)
    response = _complete_response("A semantic result.")

    try:
        miss = await service.lookup(info)
        assert miss.kind is None
        assert miss.vector == [1.0, 0.0]
        assert embedder.calls == ["Explain semantic caching"]

        await service.store(
            info,
            miss.vector,
            response,
            cost_usd=0.002,
            model_alias="model-a",
            provider="fake",
        )
        assert embedder.calls == ["Explain semantic caching"]
        redis.values.clear()

        result = await service.lookup(info)
        assert result.kind == "semantic"
        assert result.entry["response"] == response
        assert result.entry["cost_usd"] == 0.002
        assert result.entry["model_alias"] == "model-a"
        assert result.entry["provider"] == "fake"
        assert result.score == pytest.approx(1.0)
        assert len(embedder.calls) == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cache_failures_are_logged_and_fail_open(tmp_path, caplog):
    settings = _settings(tmp_path)
    redis = FakeRedis()
    redis.get_error = RuntimeError("redis unavailable")
    redis.set_error = RuntimeError("redis unavailable")
    service = _cache_service(redis, settings)
    payload = {
        "model": "model-a",
        "messages": [{"role": "user", "content": "Explain cache failures"}],
    }
    info = _build_info(settings, payload)

    result = await service.lookup(info)
    await service.store(
        info,
        None,
        _complete_response(),
        cost_usd=0.0,
        model_alias="model-a",
        provider="fake",
    )

    assert result == LookupResult()
    assert "exact cache lookup failed" in caplog.text
    assert "exact cache store failed" in caplog.text


@pytest.mark.asyncio
async def test_semantic_backend_failures_are_misses_not_request_errors(tmp_path):
    class FailingSemantic:
        async def lookup(self, vector, **scope):
            _ = vector, scope
            raise RuntimeError("qdrant unavailable")

        async def store(self, vector, **kwargs):
            _ = vector, kwargs
            raise RuntimeError("qdrant unavailable")

    settings = _settings(tmp_path)
    redis = FakeRedis()
    service = CacheService(
        redis,
        FailingSemantic(),
        FakeEmbedder(),
        ttl_s=settings.cache_ttl_seconds,
        max_bytes=settings.cache_max_bytes,
        lookup_timeout_s=settings.cache_lookup_timeout_s,
    )
    payload = {
        "model": "model-a",
        "messages": [{"role": "user", "content": "Explain semantic caching"}],
    }
    info = _build_info(settings, payload)

    result = await service.lookup(info)
    await service.store(
        info,
        result.vector,
        _complete_response(),
        cost_usd=0.0,
        model_alias="model-a",
        provider="fake",
    )

    assert result == LookupResult()
    assert json.loads(redis.values[info.exact_key])["response"] == _complete_response()


def test_only_complete_text_responses_are_cacheable():
    response = _complete_response()

    assert cacheable_response(response)
    assert not cacheable_response(
        {
            **response,
            "choices": [{**response["choices"][0], "finish_reason": "length"}],
        }
    )
    assert not cacheable_response(_complete_response("  "))
    assert not cacheable_response(
        {
            **response,
            "choices": [
                {
                    **response["choices"][0],
                    "message": {
                        "role": "assistant",
                        "content": "A tool is being called.",
                        "tool_calls": [{"id": "call"}],
                    },
                }
            ],
        }
    )
    assert not cacheable_response({"choices": []})
    assert not cacheable_response(
        {"choices": [{"finish_reason": "stop", "message": None}]}
    )


@pytest.mark.asyncio
async def test_cache_size_cap_skips_exact_and_semantic_store(tmp_path):
    settings = _settings(tmp_path)
    redis = FakeRedis()
    service = _cache_service(redis, settings, max_bytes=32)
    payload = {
        "model": "model-a",
        "messages": [{"role": "user", "content": "Explain cache size limits"}],
    }
    info = _build_info(settings, payload)

    await service.store(
        info,
        None,
        _complete_response("A" * 100),
        cost_usd=0.0,
        model_alias="model-a",
        provider="fake",
    )

    assert redis.values == {}


@pytest.mark.asyncio
async def test_malformed_exact_entry_is_a_miss(tmp_path):
    settings = _settings(tmp_path)
    redis = FakeRedis()
    service = _cache_service(redis, settings)
    payload = {
        "model": "model-a",
        "messages": [{"role": "user", "content": "Explain malformed cache entries"}],
    }
    info = _build_info(settings, payload)
    redis.values[info.exact_key] = (
        b'{"created_at":"not-an-integer","response":{"choices":[]}}'
    )

    result = await service.lookup(info)

    assert result == LookupResult()


def test_cache_info_uses_requested_model_not_provider_model(tmp_path):
    settings = _settings(tmp_path)
    payload = {
        "model": "public-alias",
        "messages": [{"role": "user", "content": "What is semantic caching?"}],
    }

    info = _build_info(settings, payload, team_id=42)

    assert info == CacheInfo(
        team_id=42,
        model_requested="public-alias",
        read=True,
        write=True,
        bypass_reason=None,
        exact_key=info.exact_key,
        params_hash=info.params_hash,
        context_hash=info.context_hash,
        num_sig="none",
        semantic_text="What is semantic caching?",
        text_hash=sha("What is semantic caching?")[:32],
    )
