import json
import random
import string
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hf_cache  # noqa: E402, F401  (sets HF_HOME before transformers is imported)

from transformers import AutoTokenizer  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "constructed" / "counting_characters.jsonl"

MIN_LEN = 5
MAX_LEN = 10


def candidate_words(tokenizer):
    vocab_ids = tokenizer.get_vocab().values()
    seen = set()
    words = []
    for token_id in vocab_ids:
        word = tokenizer.decode([token_id])
        if word in seen:
            continue
        seen.add(word)
        if MIN_LEN <= len(word) <= MAX_LEN and all(c in string.ascii_lowercase for c in word):
            words.append(word)
    return words


def most_frequent_char(word):
    counts = Counter(word)
    best_char, best_count = None, -1
    for char in word:  # deterministic tie-break: first max found, left to right
        if counts[char] > best_count:
            best_char, best_count = char, counts[char]
    return best_char, best_count


def build_counting_characters(n=1001, seed=123):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    words = candidate_words(tokenizer)

    rng = random.Random(seed)
    sampled = rng.sample(words, n)

    samples = []
    for word in sampled:
        char, count = most_frequent_char(word)
        samples.append({
            "word": word,
            "char": char,
            "answer": count,
            "prompt": f"Count the number of the letter '{char}' in the word {word}.",
        })
    return samples


if __name__ == "__main__":
    samples = build_counting_characters()

    assert len(samples) == 1001
    assert len({s["word"] for s in samples}) == 1001  # no duplicate words

    for s in samples:
        assert MIN_LEN <= len(s["word"]) <= MAX_LEN
        assert all(c in string.ascii_lowercase for c in s["word"])
        assert s["word"].count(s["char"]) == s["answer"]
        assert s["answer"] == max(Counter(s["word"]).values())
        assert s["prompt"] == f"Count the number of the letter '{s['char']}' in the word {s['word']}."

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Counting Characters checks passed. Wrote {len(samples)} samples to {OUTPUT_PATH}")
