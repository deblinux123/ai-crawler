"""Convert curated pairs into standard fine-tuning dataset formats."""

from __future__ import annotations

from typing import Any

from .generator import RawPair


def to_alpaca(pair: RawPair, *, include_source: bool = False) -> dict[str, Any]:
    record = {
        "instruction": pair.instruction,
        "input": "",
        "output": pair.output,
    }
    if include_source:
        record["source_url"] = pair.source_url
        record["task_type"] = pair.task
    return record


def to_sharegpt(pair: RawPair, *, include_source: bool = False) -> dict[str, Any]:
    record = {
        "conversations": [
            {"from": "human", "value": pair.instruction},
            {"from": "gpt", "value": pair.output},
        ]
    }
    if include_source:
        record["source_url"] = pair.source_url
        record["task_type"] = pair.task
    return record


CONVERTERS = {
    "alpaca": to_alpaca,
    "sharegpt": to_sharegpt,
}


def convert(pairs: list[RawPair], fmt: str, *, include_source: bool = False) -> list[dict]:
    converter = CONVERTERS[fmt]
    return [converter(p, include_source=include_source) for p in pairs]
