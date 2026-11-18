import redis.asyncio as redis


def create_client(url: str) -> redis.Redis:
    return redis.from_url(url, decode_responses=True)