"""Generate training pairs from crawled chunks using a local Ollama model."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import GenerateConfig
from .crawler import Chunk
from .prompts import build_messages

console = Console()


class GeneratedPair(BaseModel):
    instruction: str = Field(description="Self-contained question or task")
    output: str = Field(description="Accurate, grounded answer or response")


class GeneratedPairs(BaseModel):
    pairs: list[GeneratedPair]


@dataclass
class RawPair:
    instruction: str
    output: str
    source_url: str
    source_title: str
    task: str  # "qa" or "instruction"


async def _generate_for_chunk(
    client, chunk: Chunk, task: str, cfg: GenerateConfig
) -> list[RawPair]:
    messages = build_messages(
        task=task,
        title=chunk.title,
        text=chunk.text,
        n=cfg.pairs_per_chunk,
        extra_system=cfg.system_prompt,
    )

    @retry(stop=stop_after_attempt(cfg.max_retries), wait=wait_exponential(min=1, max=10))
    async def _call():
        return await client.chat(
            model=cfg.model,
            messages=messages,
            format=GeneratedPairs.model_json_schema(),
            options={"temperature": cfg.temperature},
        )

    try:
        response = await _call()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Generation failed for {chunk.url} ({task}): {exc}[/red]")
        return []

    try:
        parsed = GeneratedPairs.model_validate_json(response.message.content)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Could not parse model output for {chunk.url}: {exc}[/yellow]")
        return []

    return [
        RawPair(
            instruction=p.instruction.strip(),
            output=p.output.strip(),
            source_url=chunk.url,
            source_title=chunk.title,
            task=task,
        )
        for p in parsed.pairs
        if p.instruction.strip() and p.output.strip()
    ]


async def generate_pairs(chunks: list[Chunk], cfg: GenerateConfig) -> list[RawPair]:
    from ollama import AsyncClient

    client = AsyncClient(host=cfg.ollama_host)
    tasks_to_run = ["qa", "instruction"] if cfg.task == "both" else [cfg.task]

    jobs = [(chunk, task) for chunk in chunks for task in tasks_to_run]
    results: list[RawPair] = []
    semaphore = asyncio.Semaphore(cfg.concurrency)

    progress = Progress(
        TextColumn("[cyan]Generating pairs[/cyan]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    )

    async def _worker(chunk: Chunk, task: str, progress_task_id):
        async with semaphore:
            pairs = await _generate_for_chunk(client, chunk, task, cfg)
            results.extend(pairs)
            progress.advance(progress_task_id)

    with progress:
        progress_task_id = progress.add_task("generate", total=len(jobs))
        await asyncio.gather(*(_worker(c, t, progress_task_id) for c, t in jobs))

    console.print(f"[green]Generated {len(results)} raw pairs[/green] from {len(chunks)} chunks")
    return results
