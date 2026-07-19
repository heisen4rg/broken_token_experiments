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

**Note:** `data/benchmarks/` and `.cache/` are gitignored — the actual downloaded data lives locally only, not in the repo. Regenerate it anytime with `python src/data/fetch_benchmarks.py`. Still an open question (for you and your mentor) whether the real data should be committed instead — plain git chokes on a few of the larger files (>100MB), so it'd mean either trimming to just the splits actually evaluated (the train splits currently included aren't used by any experiment in the paper — everything here is zero-shot evaluation, nothing gets finetuned on these 20 benchmarks) or switching to Git LFS.

## What's left

Roughly in the order it'll probably get tackled:

1. **Actually running Exp 1.** Right now we have the tokenization logic and the raw benchmark data, but not the harness that ties them together — loading each model, applying the paper's system prompts (Appendix B.1 spells these out per benchmark type), generating under each tokenization scheme, parsing answers, and computing the retention numbers in Table 1. Also need the actual model weights (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, OLMo-2-7B-Instruct — only their tokenizers are downloaded so far, full weights are ~45GB combined).
2. **§3's "tokenization can help" tasks** — Counting Characters and Acronyms are fully self-generated (no external data needed), Codeline Description needs the XLCoST dataset, Arithmetic just needs random number generation. The digit-grouping and character-segmentation code already exists; what's missing is the task datasets and the eval harness for them.
3. **§4's source-of-robustness experiments** — this is the training-stage analysis (OLMo2/Tulu3 at base/SFT/DPO/instruct) plus the actual SFT ablation finetuning run. Needs its own set of model checkpoints (~100GB+) and the `allenai/open-instruct`-based finetuning setup the paper used.
4. **Appendix C.1's Chinese benchmarks** — lowest priority; also needs the character-segmentation multi-byte fix mentioned above first.
5. **The data-storage decision** — commit the trimmed eval-only data directly, use Git LFS, or keep it as a gitignored "run this script" recipe. Whatever's easiest for you and your mentor to work with together.

## Getting set up

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python src/tokenizers/random_token_segmenter.py   # sanity-checks the tokenization algorithms
./venv/bin/python src/data/fetch_benchmarks.py                # pulls all 20 Exp 1 benchmarks into data/benchmarks/
```
