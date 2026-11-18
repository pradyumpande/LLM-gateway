-- KEYS[1] = bucket key
-- ARGV[1] = capacity, ARGV[2] = refill per second, ARGV[3] = cost

local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])

local time = redis.call('TIME')
local now_ms = math.floor(tonumber(time[1]) * 1000 + tonumber(time[2]) / 1000)

local data = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens
local last_ts
if not data[1] or not data[2] then
    tokens = capacity
    last_ts = now_ms
else
    tokens = tonumber(data[1])
    last_ts = tonumber(data[2])
end

local elapsed_ms = math.max(0, now_ms - last_ts)
tokens = math.min(capacity, tokens + elapsed_ms / 1000 * refill_rate)

local allowed = 0
local retry_after_ms = 0
if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
elseif refill_rate > 0 then
    retry_after_ms = math.ceil((cost - tokens) / refill_rate * 1000)
else
    retry_after_ms = 2147483647
end

redis.call('HSET', KEYS[1], 'tokens', string.format('%.8f', tokens), 'ts', now_ms)
redis.call('PEXPIRE', KEYS[1], 120000)

return { allowed, retry_after_ms }