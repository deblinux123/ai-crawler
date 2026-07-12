"""Write curated pairs to JSONL files on disk."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from .formats import convert
from .generator import RawPair

console = Console()


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    console.print(f"[green]Wrote {len(records)} records -> {path}[/green]")


def export_dataset(
    train: list[RawPair],
    val: list[RawPair],
    fmt: str,
    output_dir: Path,
    *,
    include_source: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths: dict[str, Path] = {}

    train_path = output_dir / f"train.{fmt}.jsonl"
    write_jsonl(convert(train, fmt, include_source=include_source), train_path)
    paths["train"] = train_path

    if val:
        val_path = output_dir / f"val.{fmt}.jsonl"
        write_jsonl(convert(val, fmt, include_source=include_source), val_path)
        paths["val"] = val_path

    return paths
