# LLM-gateway

Authentication context is cached in Redis for 60 seconds. Changes to team
limits may therefore take up to 60 seconds to apply to requests.

Rate limiting uses Redis-backed token buckets. If Redis is unavailable during
the rate-limit check, requests fail open and are allowed through; the failure
is logged with the request ID.

Monthly budget checks are soft limits: concurrent requests can all pass the
pre-check and slightly overshoot the budget. A stricter follow-up could reserve
estimated cost before each provider call and settle it afterwards. Spend
counters are held in Redis and can be lost on a Redis restart; reseeding them
from `usage_logs` is planned once request logging is implemented. Cached
responses are intended to cost $0.

Guardrail defaults limit inputs to 200,000 characters, flag prompt-injection
scores at 0.4, block at 0.8, and recognize PII at scores of 0.5 or higher.
These settings are available through environment variables. PII detection is
initialized and warmed once during gateway startup, then run off the event loop
for chat messages. Install the small English spaCy model before starting the
gateway with `python -m spacy download en_core_web_sm`.

PII masks are stable per request and entity: repeated occurrences of the same
value share a numbered placeholder. Detection reports and logs include entity
counts only, never matched values. `PERSON` recognition is opt-in by adding
`"PERSON"` to the JSON `PII_ENTITIES` environment setting; it is excluded by
default to avoid masking ordinary names unnecessarily.

Prompt-injection checks normalize Unicode compatibility forms and zero-width
characters, then score distinct heuristic rules. Only user and tool message
text is scanned; the highest score across those texts is used. A flagged score
is logged with rule names only, and a blocked score is rejected before model
resolution. These starter rules can produce false positives: for example,
"Ignore the typo in my previous message and fix the instructions" can trigger
the override rule. Tune weights and thresholds against the Step 10 dataset.
Input guardrails run after rate-limit and budget checks; rejected requests keep
their rate-limit token and are recorded in `audit_log`. Masked requests expose
the number of replacements through `X-PII-Masked`, and flagged injection
signals set `X-Injection-Flag: true`.

### Guardrail pipeline order

The request path is `authenticate -> rate limit -> budget -> guardrails -> cache
lookup -> provider -> spend -> cache store`. Input-size checks run before content
analysis to reject oversized input cheaply. Bounded injection rules run before
spaCy so a blocked request avoids the more expensive PII analysis. PII analysis
runs in a worker thread to keep synchronous NLP work off the event loop. Cache
hits return before model resolution or provider work; successful provider
responses are stored in a background task after spend is recorded.

### Provider and routing configuration

`config/models.yaml` maps public model aliases to providers and groups aliases
into routing tiers. Each provider declares the environment variable containing
its API key; each configured alias must also have an entry in
`config/pricing.yaml`. `MODELS_CONFIG_PATH` selects the models file and defaults
to `config/models.yaml`. Startup rejects unknown providers or tier aliases,
reserved/colliding tier names, missing pricing entries, and tiers left empty
after aliases without provider keys are dropped with warnings.

For local offline development, set `MODELS_CONFIG_PATH=config/models.mock.yaml`
and provide `MOCK_API_KEY`. That config maps the cheap and strong tiers across
three mock-provider endpoints; `cheap-b` and `strong-b` intentionally share
mock provider B.

Start the three mock servers from PowerShell with
`.\.venv313\Scripts\python.exe scripts\run_mocks.py`. Each server exposes
`/_control` (GET/POST), `/_stats`, and `/_reset_stats`. For example, POST
`{"fail_rate":1.0,"fail_status":503}` to `http://localhost:9000/_control` to
make provider A return injected 503 errors, then restore it with
`{"fail_rate":0}`. Streaming supports `stream_options.include_usage`; set
`stream_fail_after` through `/_control` to inject a mid-stream abort.

### Limitations

- Indian mobile recognition supports contiguous and 5-5 formats, with optional
  country-code or leading-zero prefixes. It does not cover 3-3-4 phone formats
  or landlines.
- The recognizers use ASCII digits, so native-script numerals are not matched.
- Aadhaar values with a failing Verhoeff checksum are intentionally ignored.
  A valid checksum is not proof that a number is an Aadhaar; generic 12-digit
  identifiers can still be false positives.
- Prompt-injection detection is heuristic and cannot cover every attack or
  obfuscation. Unicode compatibility normalization handles some forms such as
  full-width text, but cross-script homoglyph attacks are not covered.
- `PERSON` recognition remains opt-in to avoid masking ordinary names
  unnecessarily.

### Curated guardrail evaluation

Run `python tests/eval_guardrails.py` from the repository root. The hand-written
JSONL datasets contain 40 PII-positive and 40 clean cases, plus 20 injection
attacks and 20 benign prompts. Results below use the default PII threshold
(0.50) and injection thresholds (flag 0.40, block 0.80):

| PII entity | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aadhaar | 13 | 6 | 0 | 0.684 | 1.000 |
| Credit card | 6 | 0 | 0 | 1.000 | 1.000 |
| Email | 6 | 0 | 0 | 1.000 | 1.000 |
| PAN | 4 | 0 | 0 | 1.000 | 1.000 |
| Indian phone | 19 | 0 | 0 | 1.000 | 1.000 |
| Overall | 48 | 6 | 0 | 0.889 | 1.000 |

All six PII false positives are synthetic, random-looking, checksum-valid
12-digit numbers in generic `UID` tracking/order records. Verhoeff validates
the check digit, not the identity or intended use of a number, and the nearby
`UID` context raises the score. This illustrates the known Aadhaar ambiguity;
these small curated-set metrics are not a production error-rate estimate.

| Injection decision | TP | FP | FN | TN | Precision | Recall | False-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flag at 0.40 | 20 | 1 | 0 | 19 | 0.952 | 1.000 | 0.050 |
| Block at 0.80 | 14 | 0 | 6 | 20 | 1.000 | 0.700 | 0.000 |

The remaining flag false positive is the benign request, “Ignore the typo in
my previous message and fix the instructions.” Single generic jailbreak and
chat-template delimiter rules have lower weights (0.3) so educational mentions
alone do not trigger a flag. The higher block threshold intentionally favors
precision: several single-signal attacks are flagged but not blocked.

### Masking latency

In one local run, after five warm-up calls, 200 `mask_messages` calls on a
199-character prompt containing three synthetic PII values had a **19.6 ms
median and 29.9 ms p95**. Calls used `asyncio.to_thread`, so the measurement
includes thread dispatch and PII analysis/masking, but excludes input-size and
injection checks, database work, and provider/network latency. This is a local
baseline, not an end-to-end request-latency guarantee; model, hardware, and
workload affect it.

### Embedding model cache

FastEmbed's first model initialization downloads its model files (about 130 MB).
`FASTEMBED_CACHE_PATH` defaults to `.cache/fastembed` under the project root so
the model cache persists with the project instead of relying on a temporary OS
cache. The `.cache/` directory is ignored by Git.
FastEmbed depends on ONNX Runtime; use a stable Python release with an available
ONNX Runtime wheel. The Python 3.15.0 release candidate in this development
environment had no matching wheel, while installation succeeded under Python
3.13.17.

### Response cache policy

Exact cache keys use SHA-256 over canonical JSON, the requested model alias,
team ID, normalized messages, and generation parameters. Message normalization
strips only leading and trailing whitespace from string content; inner
whitespace is preserved because it can affect code and prompts. Parameter
hashes omit model/messages and transport or client metadata fields. Python's
process-randomized built-in `hash()` is not used.

An omitted temperature remains cache-eligible. OpenAI-style providers commonly
default it to 1.0, so treating the omitted value as an explicit temperature
above the configured limit would disable caching for many clients. This means
a cache hit can return one earlier sample for a request whose provider default
is stochastic; this is intended for FAQ-style traffic.

Requests marked with PII or an injection flag bypass both cache reads and
writes. Tool/function requests, streaming requests, and requests asking for
multiple completions also bypass caching. `Cache-Control: no-cache` skips
reading but still permits refreshing the cache with the new response;
`Cache-Control: no-store` skips both reading and writing.

Semantic lookup is limited to a plain-text last user message between 8 and
1000 characters. Short messages are too ambiguous, and embedding models can
truncate long text so prompts that differ only near the end may appear
identical. Semantic matches are additionally scoped by team, requested model,
generation parameters, prior conversation context, and the ordered numeric
signature so different numbers cannot silently share an answer. Requests that
are ineligible for semantic lookup can still use the exact cache.

Semantic vectors are stored in a Qdrant collection derived from the embedding
model name, and startup checks the collection's vector dimension. Payload
indexes support the team, model, context, parameter, and numeric-scope filters.
Qdrant does not expire individual points, so lookups filter out entries older
than the configured TTL and periodic cleanup purges those expired points.
Semantic Qdrant operations use the async client; embedding runs off the event
loop with at most two concurrent FastEmbed calls.
The model is initialized once during gateway startup. If Qdrant is unavailable,
startup logs the failure and continues with exact caching only; a restart is
needed to re-enable semantic caching. The cleanup task and Qdrant client are
stopped during shutdown.

Cache reads and writes are best effort: Redis, embedding, or Qdrant errors are
logged and treated as cache misses rather than request failures. Only complete
text answers with `finish_reason == "stop"` and no tool calls are cached, and
the serialized entry is size-capped. A hit is materialized with a fresh
completion ID and timestamp while retaining the original response usage data.
The entry also preserves the provider cost and model metadata for cache-hit
accounting.

Rate limiting and the monthly budget check still run before cache lookup. A hit
returns `X-Cache: HIT`, costs `$0`, and reports the original cost in
`X-Saved-Cost`; misses and bypasses report `X-Cache: MISS` or `BYPASS`.
Successful provider responses are stored after sending the response. Admins can
invalidate exact and semantic entries with `DELETE /admin/cache?team_id=<id>`,
or clear all entries with `DELETE /admin/cache?everything=true`.

### Cache layers and scope

Lookup checks the Redis exact cache first. An exact hit avoids embedding and
Qdrant; after an exact miss, eligible requests embed the last user message and
query Qdrant for a sufficiently similar response. The vector produced during
lookup is reused if the provider response is stored.

| Field | Layer | Purpose |
| --- | --- | --- |
| `team_id` | Both | Prevents one tenant from reading another tenant's response. |
| `model_requested` | Both | Keeps responses isolated by the model alias requested by the client. |
| `context_hash` | Semantic | Hashes system and earlier-turn context, so a matching last question under different instructions does not collide. |
| `params_hash` | Both | Separates generation settings such as temperature and token limits that can change the answer. |
| `num_sig` | Semantic | Preserves the ordered numeric literals as a guard against matching different values. It cannot catch every direction/unit swap. |
| `text_hash` | Semantic point ID | Makes an identical scoped prompt upsert the same Qdrant point instead of duplicating it. |

The exact Redis key also hashes the full normalized message list, requested
model, team, and generation parameters. Semantic lookups use the last user
message for similarity, but require every semantic scope field above to match.

| Condition | Cache behavior |
| --- | --- |
| Cache disabled, streaming, tools/functions, `n > 1`, temperature over the configured maximum, PII, or injection flag | Bypass reads and writes. |
| `Cache-Control: no-cache` | Skip reads, call the provider, then refresh the cache. |
| `Cache-Control: no-store` | Skip reads and writes. |
| Temperature omitted | Remains eligible; the gateway does not assume the provider's default is a particular value. |

### Cache tests

The unit tests in `tests/test_cache.py` need no external services. The tests in
`tests/test_cache_integration.py` use Redis and Qdrant from the configured
service URLs; start the Docker services with `docker compose up -d redis qdrant`
before running `python -m pytest tests/test_cache_integration.py`. These tests
create unique Qdrant collections and remove them during teardown. The API test
also initializes the real FastEmbed model on first use.

A local `BAAI/bge-small-en-v1.5` run produced 384-dimensional vectors and these
cosine similarities:

| Pair | Similarity |
| --- | ---: |
| “What is the capital of France?” / “Which city is the capital of France?” | 0.9593 |
| “What is the capital of France?” / “What is the capital of Germany?” | 0.7949 |
| “What is the capital of France?” / “How do I bake bread?” | 0.3442 |

The changed-country question is much closer to the paraphrase than the
unrelated question, illustrating why a semantic threshold needs evaluation on
the curated positive and negative pairs rather than selection from a single
example.

### Semantic threshold calibration

The hand-built [cache-pair dataset](./tests/data/cache_pairs.jsonl) contains 40
same-intent pairs, 44 different-intent pairs, 34 hard negatives, and 10
unrelated negatives. `tests/tune_threshold.py` uses a seeded development split
and reserves every unit-direction and time-word negative for a held-out
challenge set. The dev split had zero false hits at `0.94`; adding a `0.01`
margin selected `SEMANTIC_THRESHOLD=0.95`.

At `0.95`, the 46-pair held-out test had a 0.273 hit rate, 0.292 false-hit
rate, and 0.462 precision. The France paraphrase (similarity `0.9593`) hits
and the Germany hard negative (`0.7949`) misses, but the held-out challenge
exposed false hits for unit-direction conversions and some time-word changes.
The ordered number guard does not distinguish unit direction when both
requests contain the same numbers. This small, curated evaluation does not
certify general-purpose semantic caching as safe: use it only for FAQ-style
traffic, choose a stricter threshold if false hits matter more than recall, or
add a verifier step before returning semantic hits.

The sweep shows the recall trade-off as the threshold rises; the complete
0.70–1.00 sweep is in [`docs/threshold_sweep.csv`](./docs/threshold_sweep.csv).
Selected points (hit rate / false-hit rate) are:

| Threshold | Dev split | Held-out test |
| ---: | ---: | ---: |
| 0.90 | 0.722 / 0.050 | 0.545 / 0.417 |
| 0.94 | 0.500 / 0.000 | 0.364 / 0.333 |
| 0.95 (chosen) | 0.500 / 0.000 | 0.273 / 0.292 |
| 0.97 | 0.333 / 0.000 | 0.227 / 0.167 |
| 1.00 | 0.167 / 0.000 | 0.045 / 0.000 |

### Cache latency benchmark

One warmed local run used `mock-fast`, Redis and Qdrant on localhost, and
`BAAI/bge-small-en-v1.5`. The HTTP phases were 30 sequential requests each;
the exact pass repeated the cold questions, and the semantic pass used 30
manually written paraphrases that all exceeded the configured threshold (minimum
cosine `0.9687`). A 500 ms pause after the cold pass allowed background stores
to finish. Component timings are medians and p95s over 100 calls.

| Path | Samples | Median | p95 |
| --- | ---: | ---: | ---: |
| Unique misses, cache enabled | 30 | 296.48 ms | 382.94 ms |
| Unique misses, `CACHE_ENABLED=false` | 30 | 251.46 ms | 302.82 ms |
| Estimated miss overhead (difference between run summaries) | 30 | +45.02 ms | +80.12 ms |
| Exact hits | 30 | 39.81 ms | 63.10 ms |
| Semantic paraphrase hits | 30 | 80.99 ms | 97.34 ms |
| `embed_sync` (warmed) | 100 | 18.17 ms | 22.77 ms |
| Filtered Qdrant query | 100 | 15.36 ms | 21.04 ms |

All 30 exact and semantic requests hit. The Qdrant component measures a
filtered lookup with no matching result and excludes embedding time; it is not
an end-to-end semantic-hit measurement. Miss overhead is an estimate from
subtracting the separately measured medians and p95s, not paired per-request
differences. The first request after startup can be slower; the benchmark
excludes its unmeasured warm-up request.

### Cache limitations and deployment

Redis gives exact entries a hard TTL. Qdrant does not expire points itself:
semantic lookups apply a soft TTL filter, and the periodic cleanup task later
deletes expired points. There is no stampede protection, so concurrent cold
requests for the same prompt can all call the provider; a Redis `SET NX`
single-flight lock is a possible follow-up.

The cache currently shares Redis with budgets and rate limits. Do not apply an
`allkeys-lru` eviction policy to that shared instance, because it could evict
gateway state unrelated to caching. Production should use a separate cache
Redis; the planned consistent-hashing cache distribution builds on that
separation.
