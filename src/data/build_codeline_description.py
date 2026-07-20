import json
import random
from pathlib import Path

from datasets import load_from_disk

XLCOST_PATH = Path(__file__).resolve().parents[2] / "data" / "sources" / "xlcost"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "constructed" / "codeline_description.jsonl"

LANGUAGES = ["cpp", "csharp", "java", "javascript", "php", "python"]
SAMPLES_PER_LANGUAGE = 800
OPTION_LETTERS = "ABCD"


def extract_description(text):
    # XLCoST's program-level "text" field concatenates a problem title with
    # per-line comments, joined by "|" then ";". The first segment reads
    # like a clean one-sentence description (matches Table 2's style).
    first_part = text.split("|")[0] if "|" in text else text.split(";")[0]
    return first_part.strip()


def detokenize_code(code):
    # XLCoST represents newlines/indentation as literal NEW_LINE/INDENT/
    # DEDENT tokens rather than actual whitespace -- reconstruct real code.
    indent = 0
    lines = []
    current = []
    for tok in code.split():
        if tok == "NEW_LINE":
            lines.append("    " * indent + " ".join(current))
            current = []
        elif tok == "INDENT":
            indent += 1
        elif tok == "DEDENT":
            indent = max(0, indent - 1)
        else:
            current.append(tok)
    if current:
        lines.append("    " * indent + " ".join(current))
    code = "\n".join(lines)
    # XLCoST marks spaces inside string literals with "▁" (SentencePiece
    # convention) rather than real spaces, so they survive its own
    # whitespace-based tokenization -- restore them for readability.
    return code.replace(" ▁ ", " ").replace("▁", " ")


def build_language_pool(lang):
    ds = load_from_disk(str(XLCOST_PATH))[lang]
    seen_descriptions = set()
    pool = []
    for ex in ds:
        description = extract_description(ex["text"])
        if not description or description in seen_descriptions:
            continue
        seen_descriptions.add(description)
        pool.append({"code": detokenize_code(ex["code"]), "description": description})
    return pool


def build_codeline_description(seed=123):
    rng = random.Random(seed)
    samples = []

    for lang in LANGUAGES:
        pool = build_language_pool(lang)
        chosen = rng.sample(pool, SAMPLES_PER_LANGUAGE)

        for item in chosen:
            distractor_candidates = [p for p in pool if p["description"] != item["description"]]
            distractors = rng.sample(distractor_candidates, len(OPTION_LETTERS) - 1)

            option_texts = [item["description"]] + [d["description"] for d in distractors]
            rng.shuffle(option_texts)
            answer = OPTION_LETTERS[option_texts.index(item["description"])]

            samples.append({
                "language": lang,
                "code": item["code"],
                "correct_description": item["description"],
                "options": dict(zip(OPTION_LETTERS, option_texts)),
                "answer": answer,
            })

    return samples


if __name__ == "__main__":
    samples = build_codeline_description()

    assert len(samples) == SAMPLES_PER_LANGUAGE * len(LANGUAGES)
    for lang in LANGUAGES:
        assert sum(1 for s in samples if s["language"] == lang) == SAMPLES_PER_LANGUAGE

    for s in samples:
        assert s["code"].strip() != ""
        # NEW_LINE/INDENT/DEDENT as their own space-delimited tokens would
        # mean detokenize_code() missed something; they can legitimately
        # appear inside string literals in the actual code (e.g. some Java
        # samples print("...NEW_LINE...") as part of their own output).
        assert not any(tok in ("NEW_LINE", "INDENT", "DEDENT") for tok in s["code"].split())
        assert set(s["options"].keys()) == set(OPTION_LETTERS)
        assert len(set(s["options"].values())) == 4  # all 4 options distinct
        assert s["options"][s["answer"]] == s["correct_description"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Codeline Description checks passed. Wrote {len(samples)} samples to {OUTPUT_PATH}")
