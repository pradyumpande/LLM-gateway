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

The request path is `authenticate -> rate limit -> budget -> input size ->
injection -> PII -> resolve model -> provider`. The size check runs before
content analysis to reject oversized input cheaply. Bounded injection rules run
before spaCy so a blocked request avoids the more expensive PII analysis. PII
analysis runs in a worker thread to keep synchronous NLP work off the event
loop. Model resolution and provider work happen only after the input passes
guardrails.

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
