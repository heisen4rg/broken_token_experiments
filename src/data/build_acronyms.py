import json
import random
import string
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "constructed" / "acronyms.jsonl"

ACRONYM_LENGTH = 5


def build_acronyms(n=3594, seed=123):
    rng = random.Random(seed)
    samples = []

    for _ in range(n):
        acronym = "".join(rng.choice(string.ascii_lowercase) for _ in range(ACRONYM_LENGTH))
        samples.append({
            "acronym": acronym,
            "prompt": f"Come up with a sequence of words where the first letters "
                      f"would form this acronym: {acronym}",
        })

    return samples


if __name__ == "__main__":
    samples = build_acronyms()

    assert len(samples) == 3594

    for s in samples:
        assert len(s["acronym"]) == ACRONYM_LENGTH
        assert all(c in string.ascii_lowercase for c in s["acronym"])
        assert s["acronym"] in s["prompt"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Acronyms checks passed. Wrote {len(samples)} samples to {OUTPUT_PATH}")
