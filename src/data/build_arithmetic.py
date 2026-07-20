import json
import random
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "constructed" / "arithmetic.jsonl"

MIN_10_DIGIT = 10**9
MAX_10_DIGIT = 10**10 - 1


def build_arithmetic(n=1000, seed=123):
    rng = random.Random(seed)
    half = n // 2
    samples = []

    for i in range(n):
        a = rng.randint(MIN_10_DIGIT, MAX_10_DIGIT)
        b = rng.randint(MIN_10_DIGIT, MAX_10_DIGIT)
        op = "+" if i < half else "-"
        answer = a + b if op == "+" else a - b
        samples.append({
            "a": a,
            "b": b,
            "op": op,
            "prompt": f"{a} {op} {b} =",
            "answer": answer,
        })

    rng.shuffle(samples)
    return samples


if __name__ == "__main__":
    samples = build_arithmetic()

    assert len(samples) == 1000
    assert sum(1 for s in samples if s["op"] == "+") == 500
    assert sum(1 for s in samples if s["op"] == "-") == 500

    for s in samples:
        assert MIN_10_DIGIT <= s["a"] <= MAX_10_DIGIT
        assert MIN_10_DIGIT <= s["b"] <= MAX_10_DIGIT
        assert len(str(s["a"])) == 10
        assert len(str(s["b"])) == 10
        expected = s["a"] + s["b"] if s["op"] == "+" else s["a"] - s["b"]
        assert s["answer"] == expected
        assert s["prompt"] == f"{s['a']} {s['op']} {s['b']} ="

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Arithmetic checks passed. Wrote {len(samples)} samples to {OUTPUT_PATH}")
