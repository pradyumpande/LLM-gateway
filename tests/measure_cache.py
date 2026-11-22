import argparse
import asyncio
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cache.embedder import FastEmbedEmbedder
from app.cache.keys import context_hash, num_sig, params_hash
from app.cache.semantic import SemanticCache, collection_name
from app.config import Settings


QUESTIONS = [
    "What is a semantic cache?",
    "How do I reset my password?",
    "What does Redis TTL mean?",
    "How can I write a unit test?",
    "What is the capital of France?",
    "How does HTTPS protect a website?",
    "How do I reverse a string in Python?",
    "Why does Earth have seasons?",
    "How can I make a SQL query faster?",
    "What does idempotent mean in an API?",
    "What is OAuth used for?",
    "How do I create a Git branch?",
    "What is photosynthesis?",
    "How do I stop a Docker container?",
    "What is encryption at rest?",
    "How does DNS resolve a domain?",
    "What is a Python virtual environment?",
    "How do I sort a list in Python?",
    "What is a REST API?",
    "How can I improve an email subject?",
    "What is a database index?",
    "How do I rename a file?",
    "What does HTTP 404 mean?",
    "How do I configure a Redis cache?",
    "What is a Docker image?",
    "How do I add pagination to an API?",
    "What is a regular expression?",
    "How does a TLS certificate work?",
    "How can I reduce memory usage in Python?",
    "What is a background task?",
]

SEMANTIC_VARIANTS = [
    "What exactly is a semantic cache?",
    "How can I reset my password?",
    "What does Redis TTL stand for?",
    "How do I write a unit test?",
    "Which city is the capital of France?",
    "How does HTTPS secure a website?",
    "How can I reverse a string in Python?",
    "Why do we have seasons on Earth?",
    "How can I speed up a SQL query?",
    "What does idempotent mean for an API?",
    "What is OAuth for?",
    "How can I create a branch in Git?",
    "What is photosynthesis, exactly?",
    "How can I stop a Docker container?",
    "What does encryption at rest mean?",
    "How does DNS resolve a domain name?",
    "What is a virtual environment in Python?",
    "How can I sort a Python list?",
    "What is a REST API, exactly?",
    "How can I make an email subject better?",
    "What does a database index do?",
    "How can I change a file's name?",
    "What is the meaning of HTTP 404?",
    "How can I set up a Redis cache?",
    "What is a Docker image, exactly?",
    "How can I add pagination to an API?",
    "What is a regular expression, exactly?",
    "How does a certificate for TLS work?",
    "How can I lower Python memory usage?",
    "What does a background task do?",
]


def summary(values: list[float]) -> tuple[float, float]:
    return (
        statistics.median(values),
        float(np.percentile(values, 95)),
    )


def print_summary(name: str, values: list[float]) -> None:
    median, p95 = summary(values)
    print(f"{name:28} n={len(values):2} median={median:7.2f} ms p95={p95:7.2f} ms")


def post(client: httpx.Client, key: str, text: str) -> tuple[float, httpx.Response]:
    started = time.perf_counter()
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "mock-fast",
            "messages": [{"role": "user", "content": text}],
        },
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return elapsed_ms, response


def benchmark_http(
    mode: str,
    key: str,
    endpoint: str,
    settings: Settings,
    embedder: FastEmbedEmbedder,
) -> None:
    run_id = uuid.uuid4().hex[:10]
    base_questions = [
        f"Benchmark {run_id} case {index:02d}: {question}"
        for index, question in enumerate(QUESTIONS, 1)
    ]
    semantic_questions = [
        f"Benchmark {run_id} case {index:02d}: {question}"
        for index, question in enumerate(SEMANTIC_VARIANTS, 1)
    ]

    if mode == "all":
        scores = []
        for base, variant in zip(base_questions, semantic_questions):
            base_vector = np.asarray(embedder.embed_sync(base), dtype=np.float32)
            variant_vector = np.asarray(embedder.embed_sync(variant), dtype=np.float32)
            similarity = float(
                base_vector
                @ variant_vector
                / (np.linalg.norm(base_vector) * np.linalg.norm(variant_vector))
            )
            if num_sig(base) != num_sig(variant):
                raise RuntimeError("A semantic pair has different numeric signatures.")
            if similarity < settings.semantic_threshold:
                raise RuntimeError(
                    f"Semantic benchmark pair scores {similarity:.4f}, below "
                    f"threshold {settings.semantic_threshold:.2f}."
                )
            scores.append(similarity)
        print(
            "Verified 30 paraphrases above threshold; "
            f"minimum cosine={min(scores):.4f}"
        )

    with httpx.Client(base_url=endpoint, timeout=30) as client:
        _, warmup = post(client, key, f"Benchmark warmup {run_id}: explain caching.")
        if warmup.headers.get("x-cache") == "HIT":
            raise RuntimeError("The per-run warm-up question unexpectedly hit cache.")

        cold_times = []
        cold_responses = []
        for question in base_questions:
            elapsed, response = post(client, key, question)
            cold_times.append(elapsed)
            cold_responses.append(response)
        time.sleep(0.5)

        print_summary("cold unique misses", cold_times)
        print(
            "  cache outcomes: "
            f"{sum(r.headers.get('x-cache') == 'MISS' for r in cold_responses)} MISS, "
            f"{sum(r.headers.get('x-cache') == 'BYPASS' for r in cold_responses)} BYPASS"
        )

        if mode == "misses":
            return
        if sum(r.headers.get("x-cache") == "MISS" for r in cold_responses) != 30:
            raise RuntimeError("The enabled-cache cold pass did not produce 30 misses.")

        exact_times = []
        exact_responses = []
        for question in base_questions:
            elapsed, response = post(client, key, question)
            exact_times.append(elapsed)
            exact_responses.append(response)
        print_summary("exact hits", exact_times)
        exact_hits = sum(
            response.headers.get("x-cache") == "HIT"
            and response.headers.get("x-cache-type") == "exact"
            for response in exact_responses
        )
        print(f"  exact hits: {exact_hits}/30")
        if sum(
            response.headers.get("x-cache") == "HIT"
            and response.headers.get("x-cache-type") == "exact"
            for response in exact_responses
        ) != 30:
            raise RuntimeError("The exact-hit pass did not produce 30 exact hits.")

        semantic_times = []
        semantic_responses = []
        for question in semantic_questions:
            elapsed, response = post(client, key, question)
            semantic_times.append(elapsed)
            semantic_responses.append(response)
        print_summary("semantic paraphrases", semantic_times)
        semantic_hits = [
            response
            for response in semantic_responses
            if response.headers.get("x-cache") == "HIT"
            and response.headers.get("x-cache-type") == "semantic"
        ]
        print(f"  semantic hits: {len(semantic_hits)}/30")
        if semantic_hits:
            scores = [float(response.headers["x-cache-score"]) for response in semantic_hits]
            print(f"  semantic score range: {min(scores):.4f}..{max(scores):.4f}")
        if len(semantic_hits) != 30:
            raise RuntimeError("The semantic-hit pass did not produce 30 semantic hits.")


def benchmark_embedder(embedder: FastEmbedEmbedder) -> list[float]:
    text = "Benchmark the warmed synchronous embedding path."
    embedder.embed_sync(text)
    timings = []
    for _ in range(100):
        started = time.perf_counter()
        embedder.embed_sync(text)
        timings.append((time.perf_counter() - started) * 1000)
    print_summary("embed_sync", timings)
    return embedder.embed_sync(text)


async def qdrant_timings(settings: Settings, vector: list[float]) -> list[float]:
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(
        url=settings.qdrant_url or "http://localhost:6333",
        api_key=settings.qdrant_api_key or None,
        timeout=10,
    )
    cache = SemanticCache(
        client=client,
        collection=collection_name(settings.embed_model),
        dim=len(vector),
        threshold=settings.semantic_threshold,
        ttl_s=settings.cache_ttl_seconds,
        embed_model=settings.embed_model,
    )
    text = "Measure filtered Qdrant semantic lookup latency."
    payload = {
        "model": "mock-fast",
        "messages": [{"role": "user", "content": text}],
    }
    scope = {
        "team_id": 0,
        "model_requested": "mock-fast",
        "context_hash": context_hash(payload["messages"]),
        "params_hash": params_hash(payload),
        "num_sig": num_sig(text),
    }
    try:
        await cache.ensure_collection()
        timings = []
        for _ in range(100):
            started = time.perf_counter()
            await cache.lookup(vector, **scope)
            timings.append((time.perf_counter() - started) * 1000)
        return timings
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure local exact/semantic cache latency and miss overhead."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "misses"),
        default="all",
        help="Use misses for the second run with gateway CACHE_ENABLED=false.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="Gateway base URL.",
    )
    args = parser.parse_args()

    key = os.environ.get("KEY")
    if not key:
        raise SystemExit('Set KEY to the generated gateway key (for example: $env:KEY="gw_...").')

    settings = Settings()
    embedder = FastEmbedEmbedder(
        "BAAI/bge-small-en-v1.5",
        str(settings.fastembed_cache_dir),
    )
    benchmark_http(args.mode, key, args.endpoint, settings, embedder)
    if args.mode == "all":
        benchmark_embedder(embedder)
        vector = embedder.embed_sync(QUESTIONS[0])
        timings = asyncio.run(qdrant_timings(settings, vector))
        print_summary("Qdrant filtered query", timings)


if __name__ == "__main__":
    main()
