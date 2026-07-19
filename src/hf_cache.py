"""
Redirects the HuggingFace hub/datasets cache into this repo instead of the
user's home directory (~/.cache/huggingface), so every model/dataset this
project downloads lives under the repo and can be wiped by deleting the
repo -- nothing leaks into shared machine-wide cache state.

Must be imported before `datasets`/`transformers` for the env var to take
effect (those libraries read it once, at import time).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / ".cache" / "huggingface"
GLOBAL_TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"

os.environ.setdefault("HF_HOME", str(CACHE_DIR))

# Redirecting HF_HOME also redirects where huggingface_hub looks for the
# login token (normally $HF_HOME/token), which would silently break access
# to gated repos (e.g. meta-llama models). Carry the existing token over via
# HF_TOKEN so auth still works against the relocated cache.
if "HF_TOKEN" not in os.environ and GLOBAL_TOKEN_FILE.exists():
    os.environ["HF_TOKEN"] = GLOBAL_TOKEN_FILE.read_text().strip()
