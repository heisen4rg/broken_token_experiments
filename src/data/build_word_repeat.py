import json
import random
from pathlib import Path

WORD_LIST_PATH = Path(__file__).resolve().parents[2] / "data" / "external" / "google-10000-english.txt"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "constructed" / "word_repeat.jsonl"


def load_word_list():
    return [w.strip() for w in WORD_LIST_PATH.read_text().splitlines() if w.strip()]


def build_word_repeat(n=500, seed=123):
    words = load_word_list()
    rng = random.Random(seed)
    sampled = rng.sample(words, n)
    return [{"word": w} for w in sampled]


if __name__ == "__main__":
    samples = build_word_repeat()

    assert len(samples) == 500
    assert len({s["word"] for s in samples}) == 500

    word_list = set(load_word_list())
    for s in samples:
        assert s["word"] in word_list

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Word Repeat checks passed. Wrote {len(samples)} samples to {OUTPUT_PATH}")
