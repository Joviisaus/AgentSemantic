"""Fetch rows from the HF datasets-server API and decode 2D part masks."""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=CoMPaT%2F3DCoMPaT200&config=default&split={split}&offset={offset}&length={length}"
)

DEFAULT_VOCAB = Path(r"E:\Proj\MeshReductionAgent\configs\vocab\3dcompat_coarse.json")


def _urlopen(url: str, *, timeout: int = 120, retries: int = 5) -> bytes:
    last_exc: Optional[Exception] = None
    ctx = ssl.create_default_context()
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=ctx) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ssl.SSLError) as exc:
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last_exc}") from last_exc


def load_vocab(vocab_path: Optional[Path] = None) -> dict:
    path = Path(vocab_path) if vocab_path else DEFAULT_VOCAB
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_rows(split: str = "train", offset: int = 0, length: int = 100) -> List[dict]:
    url = HF_ROWS_URL.format(split=split, offset=offset, length=length)
    payload = json.loads(_urlopen(url).decode())
    return [item["row"] for item in payload.get("rows", [])]


def _download_url(url: str) -> bytes:
    return _urlopen(url)


def load_image_field(field: dict) -> Image.Image:
    src = field["src"]
    raw = _download_url(src)
    return Image.open(BytesIO(raw)).convert("RGB")


def decode_part_stored(mask_rgb: np.ndarray) -> np.ndarray:
    """Return per-pixel stored part ids (0=background, else global_part_idx+1).

    Coarse part-only PNGs use small RGB triplets; material-composed pixels keep
    the part index in the blue channel (see 3DCoMPaT supplementary).
    """
    arr = np.asarray(mask_rgb, dtype=np.uint8)
    stored = arr[..., 2].astype(np.int32)
    stored[arr.sum(axis=-1) == 0] = 0
    return stored


def category_for_model(model_id: str, vocab: dict) -> Tuple[int, str]:
    cat_id = int(model_id[:2], 16)
    name = vocab["categories"].get(str(cat_id), "object")
    return cat_id, name


def category_part_names(model_id: str, vocab: dict) -> List[str]:
    cat_id, _ = category_for_model(model_id, vocab)
    part_ids = vocab["category_to_parts"].get(str(cat_id), [])
    names = vocab["part_names"]
    return [names[i] for i in part_ids]


def valid_stored_ids(model_id: str, vocab: dict) -> set[int]:
    cat_id, _ = category_for_model(model_id, vocab)
    return {int(i) + 1 for i in vocab["category_to_parts"].get(str(cat_id), [])}


def part_name_for_stored(stored: int, vocab: dict) -> str:
    if stored <= 0:
        return ""
    idx = int(stored) - 1
    names = vocab["part_names"]
    return names[idx] if 0 <= idx < len(names) else f"part_{idx}"


def dominant_category_part(
    mask_rgb: np.ndarray, model_id: str, vocab: dict,
) -> Tuple[int, str, int]:
    """Pick the largest in-category part region in a view."""
    stored = decode_part_stored(mask_rgb)
    valid = valid_stored_ids(model_id, vocab)
    best_stored, best_count = 0, 0
    for sid in valid:
        count = int((stored == sid).sum())
        if count > best_count:
            best_stored, best_count = int(sid), count
    return best_stored, part_name_for_stored(best_stored, vocab), best_count


def group_rows_by_shape(rows: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        key = f"{row['model_id.txt'].strip()}_{int(row['style_id.cls'])}"
        groups.setdefault(key, []).append(row)
    for key in groups:
        groups[key].sort(key=lambda r: int(r["view_id.cls"]))
    return groups
