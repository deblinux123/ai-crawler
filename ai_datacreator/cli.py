"""ai-datacreator CLI.

Typical usage:

    # one-shot: crawl -> generate -> curate -> export
    ai-datacreator run https://docs.example.com --model llama3.1 --format alpaca

    # step by step
    ai-datacreator crawl https://docs.example.com --out chunks.jsonl
    ai-datacreator generate chunks.jsonl --model llama3.1 --out-dir ./output
    ai-datacreator push ./output/train.alpaca.jsonl --repo you/my-dataset
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import CrawlConfig, CurateConfig, GenerateConfig, HFConfig, PipelineConfig
from .crawler import Chunk, crawl_to_chunks
from .curator import curate, split
from .exporter import export_dataset
from .generator import RawPair, generate_pairs

app = typer.Typer(
    name="ai-datacreator",
    help="Crawl the web, generate fine-tuning data with a local Ollama model, curate it, and optionally push it to HuggingFace.",
    add_completion=False,
)
console = Console()


def _chunks_to_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")


def _chunks_from_jsonl(path: Path) -> list[Chunk]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk(**json.loads(line)))
    return chunks


def _print_summary(train: list[RawPair], val: list[RawPair]) -> None:
    table = Table(title="Dataset summary")
    table.add_column("Split")
    table.add_column("Pairs", justify="right")
    table.add_row("train", str(len(train)))
    table.add_row("val", str(len(val)))
    console.print(table)


@app.command()
def crawl(
    urls: list[str] = typer.Argument(..., help="One or more seed URLs to crawl"),
    depth: int = typer.Option(1, help="Link-following depth (0 = only the given URLs)"),
    max_pages: int = typer.Option(50, help="Max total pages to crawl"),
    chunk_size: int = typer.Option(1500, help="Characters per chunk fed to the LLM"),
    external: bool = typer.Option(False, help="Allow following links to other domains"),
    out: Path = typer.Option(Path("chunks.jsonl"), help="Where to save crawled chunks"),
):
    """Crawl one or more URLs and save clean text chunks to a JSONL file."""
    cfg = CrawlConfig(
        urls=urls,
        max_depth=depth,
        max_pages=max_pages,
        chunk_size=chunk_size,
        same_domain_only=not external,
    )
    chunks = asyncio.run(crawl_to_chunks(cfg))
    _chunks_to_jsonl(chunks, out)
    console.print(f"[green]Saved {len(chunks)} chunks -> {out}[/green]")


@app.command()
def generate(
    chunks_file: Path = typer.Argument(..., help="JSONL file produced by `crawl`"),
    model: str = typer.Option("llama3.1", help="Ollama model tag, e.g. llama3.1, qwen2.5, mistral"),
    ollama_host: str = typer.Option("http://localhost:11434"),
    task: str = typer.Option("both", help="qa | instruction | both"),
    dataset_format: str = typer.Option("alpaca", "--format", help="alpaca | sharegpt"),
    pairs_per_chunk: int = typer.Option(3, help="How many pairs to generate per chunk"),
    temperature: float = typer.Option(0.7),
    concurrency: int = typer.Option(4, help="Parallel requests to Ollama"),
    val_split: float = typer.Option(0.1, help="Fraction reserved for validation"),
    include_source: bool = typer.Option(False, help="Keep source_url/task_type fields in output"),
    out_dir: Path = typer.Option(Path("./output")),
):
    """Generate, curate, and export a fine-tuning dataset from crawled chunks."""
    if task not in ("qa", "instruction", "both"):
        raise typer.BadParameter("task must be one of: qa, instruction, both")
    if dataset_format not in ("alpaca", "sharegpt"):
        raise typer.BadParameter("format must be one of: alpaca, sharegpt")

    chunks = _chunks_from_jsonl(chunks_file)
    if not chunks:
        console.print("[red]No chunks found in input file.[/red]")
        raise typer.Exit(1)

    gen_cfg = GenerateConfig(
        model=model,
        ollama_host=ollama_host,
        task=task,  # type: ignore[arg-type]
        dataset_format=dataset_format,  # type: ignore[arg-type]
        pairs_per_chunk=pairs_per_chunk,
        temperature=temperature,
        concurrency=concurrency,
    )
    raw_pairs = asyncio.run(generate_pairs(chunks, gen_cfg))

    curate_cfg = CurateConfig(val_split=val_split)
    curated = curate(raw_pairs, curate_cfg)
    train, val = split(curated, val_split)

    export_dataset(train, val, dataset_format, out_dir, include_source=include_source)
    _print_summary(train, val)


@app.command()
def run(
    urls: list[str] = typer.Argument(..., help="One or more seed URLs to crawl"),
    depth: int = typer.Option(1),
    max_pages: int = typer.Option(50),
    model: str = typer.Option("llama3.1"),
    ollama_host: str = typer.Option("http://localhost:11434"),
    task: str = typer.Option("both", help="qa | instruction | both"),
    dataset_format: str = typer.Option("alpaca", "--format", help="alpaca | sharegpt"),
    pairs_per_chunk: int = typer.Option(3),
    concurrency: int = typer.Option(4),
    val_split: float = typer.Option(0.1),
    include_source: bool = typer.Option(False),
    out_dir: Path = typer.Option(Path("./output")),
    push_to: Optional[str] = typer.Option(None, help="HuggingFace repo id to push to, e.g. you/my-dataset"),
    private: bool = typer.Option(False, help="Make the pushed HF dataset private"),
):
    """End-to-end: crawl -> generate -> curate -> export -> (optionally) push."""
    if task not in ("qa", "instruction", "both"):
        raise typer.BadParameter("task must be one of: qa, instruction, both")
    if dataset_format not in ("alpaca", "sharegpt"):
        raise typer.BadParameter("format must be one of: alpaca, sharegpt")

    crawl_cfg = CrawlConfig(urls=urls, max_depth=depth, max_pages=max_pages)
    chunks = asyncio.run(crawl_to_chunks(crawl_cfg))
    if not chunks:
        console.print("[red]Crawl produced no usable content. Try increasing depth/max_pages.[/red]")
        raise typer.Exit(1)

    gen_cfg = GenerateConfig(
        model=model,
        ollama_host=ollama_host,
        task=task,  # type: ignore[arg-type]
        dataset_format=dataset_format,  # type: ignore[arg-type]
        pairs_per_chunk=pairs_per_chunk,
        concurrency=concurrency,
    )
    raw_pairs = asyncio.run(generate_pairs(chunks, gen_cfg))

    curate_cfg = CurateConfig(val_split=val_split)
    curated = curate(raw_pairs, curate_cfg)
    train, val = split(curated, val_split)

    paths = export_dataset(train, val, dataset_format, out_dir, include_source=include_source)
    _print_summary(train, val)

    if push_to:
        from .hf_push import push_dataset

        hf_cfg = HFConfig(repo_id=push_to, private=private)
        push_dataset(paths, hf_cfg)


@app.command()
def push(
    train_file: Path = typer.Argument(..., help="Path to train JSONL file"),
    repo: str = typer.Option(..., help="HuggingFace repo id, e.g. you/my-dataset"),
    val_file: Optional[Path] = typer.Option(None, help="Path to val JSONL file"),
    private: bool = typer.Option(False),
):
    """Push already-exported JSONL file(s) to HuggingFace Hub."""
    from .hf_push import push_dataset

    paths = {"train": train_file}
    if val_file:
        paths["val"] = val_file
    push_dataset(paths, HFConfig(repo_id=repo, private=private))


@app.command("init-config")
def init_config(out: Path = typer.Option(Path("config.yaml"))):
    """Write an example YAML config file you can edit and reuse."""
    PipelineConfig().to_yaml(out)
    console.print(f"[green]Wrote example config -> {out}[/green]")


if __name__ == "__main__":
    app()
