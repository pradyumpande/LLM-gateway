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
