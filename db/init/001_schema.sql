CREATE TABLE teams (
  id SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  monthly_budget_usd NUMERIC(10,4) NOT NULL DEFAULT 10,
  rate_limit_rpm INT NOT NULL DEFAULT 60,
  burst INT NOT NULL DEFAULT 10,
  pii_action TEXT NOT NULL DEFAULT 'mask',      -- mask | block | log_only
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE api_keys (
  id SERIAL PRIMARY KEY,
  team_id INT NOT NULL REFERENCES teams(id),
  key_prefix TEXT NOT NULL,
  key_hash TEXT UNIQUE NOT NULL,                -- sha256 of the key, never the key
  active BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE usage_logs (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  request_id TEXT NOT NULL,
  team_id INT, key_id INT,
  model_requested TEXT, model_used TEXT, provider TEXT,
  cache_status TEXT,                            -- miss | exact_hit | semantic_hit
  status INT,
  prompt_tokens INT DEFAULT 0, completion_tokens INT DEFAULT 0,
  cost_usd NUMERIC(12,6) DEFAULT 0, saved_usd NUMERIC(12,6) DEFAULT 0,
  latency_ms INT, overhead_ms INT,
  fallback_chain TEXT,
  pii_found BOOLEAN DEFAULT false, injection_flag BOOLEAN DEFAULT false
);
CREATE INDEX ON usage_logs (ts);
CREATE INDEX ON usage_logs (team_id, ts);

CREATE TABLE audit_log (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  actor TEXT, event TEXT NOT NULL, details JSONB,
  prev_hash TEXT, hash TEXT
);