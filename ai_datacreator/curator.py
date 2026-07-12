"""Dedupe, quality-filter, and split raw generated pairs before export."""

from __future__ import annotations

import random
import re
from difflib import SequenceMatcher

from rich.console import Console

from .config import CurateConfig
from .generator import RawPair

console = Console()

# Above this many pairs, the O(n^2) near-dupe check is skipped (exact-match
# dedupe still runs) to keep curation fast on large datasets.
NEAR_DUPE_MAX_PAIRS = 4000


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _length_filter(pairs: list[RawPair], cfg: CurateConfig) -> list[RawPair]:
    kept = [
        p
        for p in pairs
        if cfg.min_output_length <= len(p.output) <= cfg.max_output_length
        and len(p.instruction) >= 5
    ]
    console.print(f"Length filter: {len(pairs)} -> {len(kept)}")
    return kept


def _exact_dedupe(pairs: list[RawPair]) -> list[RawPair]:
    seen: set[str] = set()
    kept: list[RawPair] = []
    for p in pairs:
        key = _normalize(p.instruction) + "||" + _normalize(p.output)
        if key in seen:
            continue
        seen.add(key)
        kept.append(p)
    console.print(f"Exact dedupe: {len(pairs)} -> {len(kept)}")
    return kept


def _near_dupe_filter(pairs: list[RawPair], threshold: float) -> list[RawPair]:
    if len(pairs) > NEAR_DUPE_MAX_PAIRS:
        console.print(
            f"[yellow]Skipping near-dupe filter: {len(pairs)} pairs exceeds "
            f"{NEAR_DUPE_MAX_PAIRS} limit (exact dedupe still applied)[/yellow]"
        )
        return pairs

    kept: list[RawPair] = []
    kept_norm: list[str] = []
    for p in pairs:
        norm = _normalize(p.instruction)
        is_dupe = any(
            SequenceMatcher(None, norm, other).ratio() >= threshold for other in kept_norm
        )
        if not is_dupe:
            kept.append(p)
            kept_norm.append(norm)
    console.print(f"Near-dupe filter: {len(pairs)} -> {len(kept)}")
    return kept


def curate(pairs: list[RawPair], cfg: CurateConfig) -> list[RawPair]:
    result = _length_filter(pairs, cfg)
    if cfg.dedupe:
        result = _exact_dedupe(result)
        result = _near_dupe_filter(result, cfg.similarity_threshold)
    return result


def split(pairs: list[RawPair], val_split: float, seed: int = 42) -> tuple[list[RawPair], list[RawPair]]:
    if val_split <= 0:
        return pairs, []
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split)) if shuffled else 0
    return shuffled[n_val:], shuffled[:n_val]
