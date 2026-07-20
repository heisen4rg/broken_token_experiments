import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tokenizers"))
from random_token_segmenter import misspell_word  # noqa: E402

from build_word_repeat import build_word_repeat, load_word_list  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "constructed" / "identifying_misspellings.jsonl"
FEWSHOT_PATH = Path(__file__).resolve().parents[2] / "data" / "constructed" / "identifying_misspellings_fewshot.jsonl"

N_TEST = 500
N_FEWSHOT = 10


def build_pair(word, rng):
    misspelling = misspell_word(word)
    correct_option = rng.choice(["A", "B"])
    if correct_option == "A":
        options = {"A": word, "B": misspelling}
    else:
        options = {"A": misspelling, "B": word}
    return {
        "word": word,
        "misspelling": misspelling,
        "options": options,
        "answer": "B" if correct_option == "A" else "A",  # the letter holding the MISSPELLING
        "correctly_spelled_option": correct_option,  # gets non-canonical tokenization at eval time
    }


def build_identifying_misspellings(n_test=N_TEST, n_fewshot=N_FEWSHOT, seed=123):
    random.seed(seed)  # misspell_word() draws from the global random module
    rng = random.Random(seed)

    # word_repeat.py already claims 500 words at this same seed -- stay disjoint from it.
    word_repeat_words = {s["word"] for s in build_word_repeat(n=N_TEST, seed=seed)}
    remaining = [w for w in load_word_list() if w not in word_repeat_words]

    sampled = rng.sample(remaining, n_test + n_fewshot)
    test_words, fewshot_words = sampled[:n_test], sampled[n_test:]

    test_samples = [build_pair(w, rng) for w in test_words]
    fewshot_samples = [build_pair(w, rng) for w in fewshot_words]
    return test_samples, fewshot_samples


if __name__ == "__main__":
    test_samples, fewshot_samples = build_identifying_misspellings()

    assert len(test_samples) == N_TEST
    assert len(fewshot_samples) == N_FEWSHOT
    assert len({s["word"] for s in test_samples} | {s["word"] for s in fewshot_samples}) == N_TEST + N_FEWSHOT

    word_repeat_words = {s["word"] for s in build_word_repeat(n=N_TEST, seed=123)}
    for s in test_samples + fewshot_samples:
        assert s["word"] not in word_repeat_words  # disjoint from word_repeat.jsonl
        assert s["misspelling"] != s["word"]
        assert set(s["options"].keys()) == {"A", "B"}
        assert s["options"][s["answer"]] == s["misspelling"]
        assert s["options"][s["correctly_spelled_option"]] == s["word"]
        assert s["answer"] != s["correctly_spelled_option"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for s in test_samples:
            f.write(json.dumps(s) + "\n")
    with open(FEWSHOT_PATH, "w") as f:
        for s in fewshot_samples:
            f.write(json.dumps(s) + "\n")

    print(f"Identifying Misspellings checks passed. Wrote {len(test_samples)} test samples to "
          f"{OUTPUT_PATH} and {len(fewshot_samples)} few-shot examples to {FEWSHOT_PATH}")
