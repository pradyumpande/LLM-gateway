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

The Indian PII recognizers match Aadhaar only when its Verhoeff checksum is
valid, and match Indian mobile numbers in contiguous or 5-5 formats (with
optional country/leading-zero prefixes), plus PAN-format identifiers. They do
not match 3-3-4 phone formats, landlines, or native-script digits. Aadhaar
values with typos are intentionally ignored to reduce false positives; a
future possible-Aadhaar tier could use a nearby Aadhaar keyword with a lower
score.
