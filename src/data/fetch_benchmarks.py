import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hf_cache  # noqa: E402, F401  (sets HF_HOME before datasets is imported)

import yaml  # noqa: E402
from datasets import concatenate_datasets, load_dataset, DatasetDict  # noqa: E402

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "manifest.yaml"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks"


def _fetch_multi_config(hf_dataset, configs, hf_revision):
    # Some datasets (e.g. MATH) don't expose a combined "all" config --
    # only per-subject configs. Fetch each and concatenate matching splits.
    per_config = [load_dataset(hf_dataset, c, revision=hf_revision) for c in configs]
    splits = set()
    for ds in per_config:
        splits.update(ds.keys())
    return DatasetDict({
        split: concatenate_datasets([ds[split] for ds in per_config if split in ds])
        for split in splits
    })


def fetch_all():
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())

    results = []
    for entry in manifest["benchmarks"]:
        name = entry["name"]
        hf_dataset = entry["hf_dataset"]
        hf_config = entry.get("hf_config")
        hf_revision = entry.get("hf_revision")
        hf_data_files = entry.get("hf_data_files")

        print(f"Fetching {name} ({hf_dataset}"
              f"{', ' + str(hf_config) if hf_config else ''}"
              f"{', revision=' + hf_revision if hf_revision else ''}"
              f"{', explicit data_files' if hf_data_files else ''})...")

        try:
            if hf_data_files:
                # Bypasses the dataset's own builder config entirely -- used
                # when the repo's default/auto-converted config mixes
                # incompatible schemas together (see e.g. alpaca_eval).
                dataset = load_dataset("parquet", data_files=hf_data_files)
            elif isinstance(hf_config, list):
                dataset = _fetch_multi_config(hf_dataset, hf_config, hf_revision)
            else:
                dataset = load_dataset(hf_dataset, hf_config, revision=hf_revision)
            dataset.save_to_disk(str(OUTPUT_DIR / name))
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))

    print("\n--- Summary ---")
    for name, ok, error in results:
        status = "OK" if ok else "FAILED"
        print(f"[{status}] {name}" + (f": {error}" if error else ""))

    num_failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - num_failed}/{len(results)} succeeded.")

    return results


if __name__ == "__main__":
    fetch_all()
