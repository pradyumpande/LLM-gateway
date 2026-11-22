import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cache.embedder import FastEmbedEmbedder
from app.cache.keys import num_sig


DATASET = ROOT / "tests" / "data" / "cache_pairs.jsonl"
SWEEP_OUTPUT = ROOT / "docs" / "threshold_sweep.csv"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(a @ b / denominator)


def classify(row: dict, threshold: float) -> bool:
    return row["sim"] >= threshold and row["num_ok"]


def metrics(rows: list[dict], threshold: float) -> dict[str, float]:
    positives = [row for row in rows if row["same"]]
    negatives = [row for row in rows if not row["same"]]
    true_positives = sum(classify(row, threshold) for row in positives)
    false_positives = sum(classify(row, threshold) for row in negatives)
    predicted_positive = true_positives + false_positives
    return {
        "hit_rate": true_positives / len(positives) if positives else 0.0,
        "false_hit_rate": false_positives / len(negatives) if negatives else 0.0,
        "precision": (
            true_positives / predicted_positive if predicted_positive else 1.0
        ),
    }


def load_rows() -> list[dict]:
    with DATASET.open(encoding="utf-8") as dataset:
        rows = [json.loads(line) for line in dataset if line.strip()]

    positive_count = sum(row["same"] is True for row in rows)
    negative_count = sum(row["same"] is False for row in rows)
    hard_negative_count = sum(
        row["same"] is False and row["type"] != "unrelated" for row in rows
    )
    if positive_count < 35 or negative_count < 35 or hard_negative_count < 20:
        raise ValueError(
            "Dataset must contain at least 35 same pairs, 35 different pairs, "
            "and 20 hard negatives."
        )
    return rows


def print_distributions(rows: list[dict]) -> None:
    groups: dict[tuple[bool, str], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["same"], row["type"])].append(row["sim"])

    print("Similarity distributions by label and type (min / median / max):")
    for (same, pair_type), scores in sorted(
        groups.items(),
        key=lambda item: (not item[0][0], item[0][1]),
    ):
        minimum, median, maximum = np.min(scores), np.median(scores), np.max(scores)
        print(
            f"  {'same' if same else 'different':9} {pair_type:16} "
            f"n={len(scores):2} {minimum:.4f} / {median:.4f} / {maximum:.4f}"
        )


def main() -> None:
    rows = load_rows()
    embedder = FastEmbedEmbedder(
        EMBED_MODEL,
        str(ROOT / ".cache" / "fastembed"),
    )
    texts = sorted({text for row in rows for text in (row["a"], row["b"])})
    vectors = {
        text: np.asarray(embedder.embed_sync(text), dtype=np.float32)
        for text in texts
    }
    for row in rows:
        row["sim"] = cosine(vectors[row["a"]], vectors[row["b"]])
        row["num_ok"] = num_sig(row["a"]) == num_sig(row["b"])

    print(f"Dataset: {len(rows)} pairs ({len(texts)} unique texts)")
    print_distributions(rows)

    challenge_types = {"swap", "time"}
    dev_pool = [row for row in rows if row["type"] not in challenge_types]
    challenge_holdout = [
        row for row in rows if row["type"] in challenge_types
    ]
    random.Random(42).shuffle(dev_pool)
    random.Random(42).shuffle(challenge_holdout)
    split = len(dev_pool) // 2
    dev = dev_pool[:split]
    test = dev_pool[split:] + challenge_holdout
    print(
        "Split: seeded random half; unit-direction swaps and time-word "
        "negatives reserved for the held-out challenge set."
    )
    thresholds = [round(index / 100, 2) for index in range(70, 101)]
    sweep = []
    for threshold in thresholds:
        dev_metrics = metrics(dev, threshold)
        test_metrics = metrics(test, threshold)
        sweep.append(
            {
                "threshold": threshold,
                **{f"dev_{key}": value for key, value in dev_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )

    zero_false_hit_thresholds = [
        row["threshold"]
        for row in sweep
        if row["dev_false_hit_rate"] == 0
    ]
    if zero_false_hit_thresholds:
        dev_cutoff = min(zero_false_hit_thresholds)
        threshold = min(1.0, round(dev_cutoff + 0.01, 2))
        print(
            f"Lowest zero-false-hit dev threshold: {dev_cutoff:.2f}; "
            f"using a +0.01 margin: {threshold:.2f}"
        )
    else:
        threshold = 1.0
        print(
            "No threshold from 0.70 to 1.00 had zero false hits on dev; "
            "using 1.00 as the most conservative measurable setting."
        )

    test_metrics = metrics(test, threshold)
    print(
        f"Held-out test at {threshold:.2f}: "
        f"hit_rate={test_metrics['hit_rate']:.3f}, "
        f"false_hit_rate={test_metrics['false_hit_rate']:.3f}, "
        f"precision={test_metrics['precision']:.3f}"
    )

    false_hits = [
        row for row in test if not row["same"] and classify(row, threshold)
    ]
    missed_same = [
        row for row in test if row["same"] and not classify(row, threshold)
    ]
    print("Held-out false hits:")
    for row in false_hits:
        print(
            f"  [{row['type']}] sim={row['sim']:.4f} num_ok={row['num_ok']} "
            f"a={row['a']!r} | b={row['b']!r}"
        )
    print("Held-out missed same-intent pairs:")
    for row in missed_same:
        print(
            f"  [{row['type']}] sim={row['sim']:.4f} num_ok={row['num_ok']} "
            f"a={row['a']!r} | b={row['b']!r}"
        )

    SWEEP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with SWEEP_OUTPUT.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=sweep[0].keys())
        writer.writeheader()
        writer.writerows(sweep)
    print(f"Saved threshold sweep to {SWEEP_OUTPUT.relative_to(ROOT)}")
    print(f"Set SEMANTIC_THRESHOLD={threshold:.2f} in .env to use this result.")


if __name__ == "__main__":
    main()
