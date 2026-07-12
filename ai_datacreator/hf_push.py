"""Push an exported JSONL dataset to the HuggingFace Hub."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from .config import HFConfig

console = Console()


def push_dataset(paths: dict[str, Path], cfg: HFConfig) -> str:
    """Push train/val JSONL files as a HuggingFace `datasets` dataset.

    Requires the user to be logged in already (`huggingface-cli login`) or
    to have HF_TOKEN set in the environment.
    """
    if not cfg.repo_id:
        raise ValueError("hf.repo_id must be set to push to HuggingFace (e.g. 'username/my-dataset')")

    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()
    console.print(f"[cyan]Logged in to HuggingFace as[/cyan] {user.get('name', 'unknown')}")

    data_files = {split: str(p) for split, p in paths.items()}
    dataset_dict = DatasetDict(
        {split: Dataset.from_json(str(p)) for split, p in data_files.items()}
    )

    console.print(f"[cyan]Pushing dataset to[/cyan] {cfg.repo_id} ...")
    dataset_dict.push_to_hub(
        cfg.repo_id,
        private=cfg.private,
        commit_message=cfg.commit_message,
    )

    url = f"https://huggingface.co/datasets/{cfg.repo_id}"
    console.print(f"[green]Pushed![/green] {url}")
    return url
