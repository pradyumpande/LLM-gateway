import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.guardrails.injection import score_injection
from app.guardrails.pii import DEFAULT_ENTITIES, build_analyzer, resolve_overlaps


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as data_file:
        return [json.loads(line) for line in data_file if line.strip()]


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def print_metrics(
    name: str,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    true_negatives: int | None = None,
) -> None:
    precision = ratio(true_positives, true_positives + false_positives)
    recall = ratio(true_positives, true_positives + false_negatives)
    line = (
        f"{name}: TP={true_positives} FP={false_positives} "
        f"FN={false_negatives} precision={precision:.3f} recall={recall:.3f}"
    )
    if true_negatives is not None:
        line += (
            f" TN={true_negatives} "
            f"FPR={ratio(false_positives, false_positives + true_negatives):.3f}"
        )
    print(line)


def evaluate_pii(analyzer, threshold: float) -> None:
    cases = load_jsonl(ROOT / "tests" / "data" / "pii_cases.jsonl")
    totals: dict[str, list[int]] = {}
    overall = [0, 0, 0]
    all_types = set()

    for case_number, case in enumerate(cases, start=1):
        text = case["text"]
        expected = {tuple(entity) for entity in case["entities"]}
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=DEFAULT_ENTITIES,
            score_threshold=threshold,
        )
        detected = {
            (result.entity_type, text[result.start : result.end])
            for result in resolve_overlaps(results)
        }
        all_types.update(entity_type for entity_type, _ in expected | detected)

        false_positives = detected - expected
        false_negatives = expected - detected
        true_positives = detected & expected
        overall[0] += len(true_positives)
        overall[1] += len(false_positives)
        overall[2] += len(false_negatives)

        for entity_type, _ in expected | detected:
            counts = totals.setdefault(entity_type, [0, 0, 0])
            counts[0] += sum(pair[0] == entity_type for pair in true_positives)
            counts[1] += sum(pair[0] == entity_type for pair in false_positives)
            counts[2] += sum(pair[0] == entity_type for pair in false_negatives)

        for entity_type, value in sorted(false_positives):
            print(
                f"PII FP case {case_number}: {entity_type} {value!r} "
                f"in {text!r}"
            )
        for entity_type, value in sorted(false_negatives):
            print(
                f"PII FN case {case_number}: expected {entity_type} {value!r} "
                f"in {text!r}"
            )

    print(f"\nPII evaluation: {len(cases)} cases, score threshold={threshold:.2f}")
    for entity_type in sorted(all_types):
        tp, fp, fn = totals.get(entity_type, [0, 0, 0])
        print_metrics(entity_type, tp, fp, fn)
    print_metrics("OVERALL", *overall)


def evaluate_injection(settings: Settings) -> None:
    cases = load_jsonl(ROOT / "tests" / "data" / "injection_cases.jsonl")
    scored = []
    for case in cases:
        result = score_injection(
            case["text"],
            settings.injection_flag_threshold,
            settings.injection_block_threshold,
        )
        scored.append((case, result.score))

    print(f"\nInjection evaluation: {len(cases)} cases")
    for name, threshold in (
        ("flag", settings.injection_flag_threshold),
        ("block", settings.injection_block_threshold),
    ):
        tp = fp = fn = tn = 0
        errors = []
        for case_number, (case, score) in enumerate(scored, start=1):
            expected = bool(case["label"])
            predicted = score >= threshold
            if expected and predicted:
                tp += 1
            elif expected:
                fn += 1
                errors.append(("FN", case_number, case["text"], score))
            elif predicted:
                fp += 1
                errors.append(("FP", case_number, case["text"], score))
            else:
                tn += 1
        print_metrics(
            f"{name} threshold={threshold:.2f}",
            tp,
            fp,
            fn,
            true_negatives=tn,
        )
        for kind, case_number, text, score in errors:
            print(
                f"Injection {kind} {name} case {case_number} "
                f"(score={score:.2f}): {text!r}"
            )


def main() -> None:
    settings = Settings()
    analyzer = build_analyzer()
    evaluate_pii(analyzer, settings.pii_score_threshold)
    evaluate_injection(settings)


if __name__ == "__main__":
    main()
