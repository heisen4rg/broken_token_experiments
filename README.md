# Broken Tokens — Reproduction

Trying to reproduce ["Broken Tokens? Your Language Model can Secretly Handle Non-Canonical Tokenizations"](https://arxiv.org/abs/2506.19004) (Zheng, Liu, Ahia, Hayase, Choi, Smith — NeurIPS 2025). The paper's core finding: instruction-tuned LMs are surprisingly robust to text tokenized in ways their tokenizer would never actually produce — random splits, character-level splits, whatever — and in some cases these "wrong" tokenizations even *improve* performance (character-level tokenization on string-manipulation tasks, right-aligned digit grouping on arithmetic).

Original authors' code, for reference when something doesn't line up: [github.com/Brianzhengca/Tokenizer-Robustness](https://github.com/Brianzhengca/Tokenizer-Robustness).

## What's here so far

### The tokenization algorithms (`src/tokenizers/random_token_segmenter.py`)

Everything the paper needs to actually *produce* non-canonical tokenizations, in one file:

- **Random segmentation** (Appendix A's algorithm) — recursively re-splits a canonical token into a uniformly-random finer segmentation, using the tokenizer's own vocab. Comes with a correctness proof in the paper; the implementation mirrors the pseudocode directly.
- **Character-level segmentation** — decomposes a token into single characters. Works correctly for English/ASCII; doesn't yet handle multi-byte scripts like Chinese correctly (it'd over-segment to byte level instead of stopping at character level — a known gap, deliberately parked for now since Appendix C.1's Chinese benchmarks aren't a priority yet).
- **BPE-dropout** — for the granularity/"length ratio" sweep in Figure 2. Turns out the `tokenizers` library already implements this exact algorithm natively (it's literally what the paper cites), so this is a thin, verified wrapper rather than a reimplementation.
- **Right-to-left digit grouping** — for the arithmetic task in §3, regroups digits in threes from the right instead of the tokenizer's default left-to-right grouping.
- **Text → non-canonical token IDs pipeline** — stitches the above together: takes raw text, applies a segmentation scheme to each canonical token, and converts the result to model-ready token IDs. Chat-template special tokens (`<|begin_of_text|>` and friends) are left alone rather than getting shredded into nonsense — found that bug the hard way.
- **Misspelling generator** — random single-character add/remove/substitute, for the §4.3 "Identifying Misspellings" probe.

All of it has inline self-tests (`python src/tokenizers/random_token_segmenter.py`).

### Exp 1's data (`data/manifest.yaml` + `src/data/fetch_benchmarks.py`)

Exp 1 (§2, Table 1 / Figure 1) evaluates 3 instruct models across 20 benchmarks under three tokenization schemes and checks how much performance survives. `manifest.yaml` catalogs where each of those 20 benchmarks actually comes from, and `fetch_benchmarks.py` pulls them all down.

This part had more landmines than expected — a good chunk of the "obvious" HuggingFace dataset IDs turned out to be wrong or broken:

- **COPA** and **TOFU** were silently fetching the *wrong data* (an augmented COPA variant, and TOFU's full forget+retain set instead of just the retain set) — not a crash, just wrong, which is worse.
- **Winograd, PIQA, JeopardyQA, AlpacaEval, and MATH** all failed outright, because their default HuggingFace repos still ship as old-style Python loading scripts, which the `datasets` library no longer executes (security reasons). Each needed either a verified parquet-native mirror or a targeted pull of the exact right file.

Every fix is noted directly in `manifest.yaml` so the reasoning doesn't get lost.

Also added `src/hf_cache.py`, which redirects all HuggingFace downloads (models and datasets) into `.cache/` inside this repo instead of the usual `~/.cache/huggingface` — so this project's disk footprint is self-contained and deletable, without breaking access to gated repos (it carries your HF token over).

**Note:** `data/benchmarks/` and `data/sources/` (raw third-party pulls) plus `.cache/` are gitignored — regenerate anytime with `python src/data/fetch_benchmarks.py`. Still an open question (for you and your mentor) whether the real benchmark data should be committed instead — plain git chokes on a few of the larger files (>100MB), so it'd mean either trimming to just the splits actually evaluated (the train splits currently included aren't used by any experiment in the paper — everything here is zero-shot evaluation, nothing gets finetuned on these 20 benchmarks) or switching to Git LFS.

### §3's constructed tasks + §4.3's probes (`src/data/build_*.py` → `data/constructed/`)

Six task datasets, one builder script each, all committed directly (small — a few KB to ~5MB each, no size concerns like the benchmark pulls above):

- **Arithmetic** (1000), **Acronyms** (3594), **Counting Characters** (1001, sampled straight from Llama-3.1's own vocab) — all self-generated, no external data.
- **Codeline Description** (4800, across 6 languages) — needed real digging: XLCoST's "snippet-level" data turned out to be single code lines, not the substantial blocks Table 2 shows, so switched to program-level and extracted a clean description out of its concatenated-comment field, plus wrote a small detokenizer to turn XLCoST's `NEW_LINE`/`INDENT`/`DEDENT` markers back into readable code. XLCoST itself now lives in `data/manifest.yaml`'s new `sources:` section (raw material for building a task, as opposed to `benchmarks:`, which Exp 1 evaluates directly).
- **Word Repeat** and **Identifying Misspellings** (500 each, kept disjoint) — both sampled from the google-10000-english word list (now saved locally at `data/external/`), the latter also building the 10-example few-shot set Appendix B.5 describes.

Chinese benchmarks (Appendix C.1) are intentionally skipped for now.

## What's left

Everything below still needs either model weights or GPU compute for the actual "run it and get numbers" step — that's deliberately being saved for last. But it's worth being precise about which *parts* of each are heavy vs. just code/small data that could be prepped ahead of time if useful:

1. **Exp 1 (§2).** Needs the eval harness (system prompts from Appendix B.1, per-benchmark answer parsing/scoring, the retention-percentage math for Table 1 and length-ratio math for Figure 2 — all just code, no downloads, but hard to fully trust without a real model to test against) and the actual model weights (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, OLMo-2-7B-Instruct — only tokenizers cached so far, full weights ~45GB combined).
2. **§3's tasks.** Data's done (above); still needs the same kind of harness/scoring code plus the same 3 models' weights to actually run against.
3. **§4.1 (training-stage analysis).** Needs OLMo2/Tulu3 checkpoints across base/SFT/DPO/instruct (~100GB+, genuinely heavy) — plus the grammaticality metric needs LanguageTool (a ~200MB grammar-checker tool, not a model, could be set up independently) and the win-rate metric needs an `OPENAI_API_KEY` for the `alpaca_eval_gpt4` judge (an external credential, not a download).
4. **§4.2 (SFT ablation).** The finetuning run itself needs Llama-3.2-1B + real GPU compute (paper used 8×L40S) — genuinely heavy. The Tulu 3 SFT Personas Instruction Following dataset it trains on, though, is a normal-sized text dataset (~13.7k examples, same ballpark as the 20 benchmarks already fetched) — not heavy on its own.
5. **§4.3's probes.** Data's done; running Word Repeat / Identifying Misspellings against a model needs weights.
6. **The data-storage decision** — commit the trimmed eval-only benchmark data directly, use Git LFS, or keep it as a gitignored "run this script" recipe. Whatever's easiest for you and your mentor to work with together.

## Getting set up

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python src/tokenizers/random_token_segmenter.py   # sanity-checks the tokenization algorithms
./venv/bin/python src/data/fetch_benchmarks.py                # pulls all 20 Exp 1 benchmarks + XLCoST
./venv/bin/python src/data/build_arithmetic.py                 # and the other build_*.py scripts in src/data/
```
