from __future__ import annotations

import os
import time
from typing import Optional

from datasets import load_dataset, load_from_disk

try:
    from huggingface_hub.utils import HfHubHTTPError
except Exception:  # pragma: no cover
    from huggingface_hub.utils._errors import HfHubHTTPError

try:
    from huggingface_hub import HfFolder
except Exception:  # pragma: no cover
    HfFolder = None


def default_hf_home() -> str:
    return os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def get_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        t = explicit.strip()
        return t or None
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    if HfFolder is not None:
        try:
            tok = HfFolder.get_token()
            if tok:
                return tok
        except Exception:
            pass
    return None


def is_hf_429(err: Exception) -> bool:
    s = str(err)
    if "429" in s and ("Too Many Requests" in s or "Client Error" in s):
        return True
    resp = getattr(err, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if code == 429:
            return True
    return False


def load_glue_dataset_with_disk_cache(
    task: str,
    *,
    hf_datasets_cache_dir: Optional[str],
    glue_disk_cache_dir: str,
    hf_token: Optional[str],
    offline: bool,
    max_retries_with_token: int = 6,
):
    """Load GLUE/<task> and store it in a save_to_disk cache.

    - If glue_disk_cache_dir/glue_<task> exists: load_from_disk.
    - Else if offline: error.
    - Else download via datasets.load_dataset and save_to_disk.

    Handles 429 rate limits with exponential backoff (works best with HF token).
    """

    os.makedirs(glue_disk_cache_dir, exist_ok=True)
    disk_path = os.path.join(glue_disk_cache_dir, f"glue_{task}")

    if os.path.isdir(disk_path):
        print(f"[INFO] GLUE/{task}: loading from disk cache: {disk_path}", flush=True)
        return load_from_disk(disk_path)

    if offline:
        raise RuntimeError(
            f"[OFFLINE] GLUE/{task} not in disk cache ({disk_path}). "
            "Run once online to populate the cache, then re-run with --offline."
        )

    token = hf_token
    attempt = 0

    while True:
        try:
            # datasets>=2.14 prefers token=...
            try:
                ds = load_dataset("glue", task, cache_dir=hf_datasets_cache_dir, token=token)
            except TypeError:
                ds = load_dataset("glue", task, cache_dir=hf_datasets_cache_dir, use_auth_token=token)

            print(f"[INFO] GLUE/{task}: saving to disk cache: {disk_path}", flush=True)
            ds.save_to_disk(disk_path)
            return ds

        except HfHubHTTPError as e:
            if is_hf_429(e):
                if not token:
                    raise RuntimeError(
                        "HuggingFace Hub returned HTTP 429 (Too Many Requests).\n\n"
                        "Fix:\n"
                        "  1) Create a HF account + access token (read is enough)\n"
                        "  2) `huggingface-cli login` once OR export HF_TOKEN in sbatch\n"
                        "  3) Re-run the job\n"
                    ) from e

                attempt += 1
                if attempt > max_retries_with_token:
                    raise RuntimeError(
                        f"Still getting HTTP 429 after {max_retries_with_token} retries with token. "
                        "Try again later or reduce concurrent downloads."
                    ) from e

                wait_s = min(15 * (2 ** (attempt - 1)), 300)
                print(
                    f"[WARN] HF 429 for GLUE/{task}. Retry {attempt}/{max_retries_with_token} in {wait_s}s...",
                    flush=True,
                )
                time.sleep(wait_s)
                continue

            raise
