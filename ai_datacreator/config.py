"""Central configuration for the ai-datacreator pipeline.

All CLI commands ultimately build a `PipelineConfig` from flags and/or a
YAML file, so behaviour stays consistent whether you use `crawl` + `generate`
separately or the all-in-one `run` command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

TaskType = Literal["qa", "instruction", "both"]
DatasetFormat = Literal["alpaca", "sharegpt"]


class CrawlConfig(BaseModel):
    urls: list[str] = Field(default_factory=list)
    max_depth: int = 1
    max_pages: int = 50
    same_domain_only: bool = True
    min_text_length: int = 200
    chunk_size: int = 1500          # characters per chunk fed to the LLM
    chunk_overlap: int = 150
    exclude_patterns: list[str] = Field(default_factory=lambda: [
        "/login", "/signin", "/cart", "/privacy", "/terms", "/cookie"
    ])


class GenerateConfig(BaseModel):
    model: str = "llama3.1"
    ollama_host: str = "http://localhost:11434"
    task: TaskType = "both"
    dataset_format: DatasetFormat = "alpaca"
    pairs_per_chunk: int = 3
    temperature: float = 0.7
    max_retries: int = 3
    concurrency: int = 4            # parallel requests to Ollama
    system_prompt: str | None = None  # optional domain-specific steering


class CurateConfig(BaseModel):
    dedupe: bool = True
    min_output_length: int = 20
    max_output_length: int = 4000
    similarity_threshold: float = 0.9   # for near-dupe filtering
    val_split: float = 0.1


class HFConfig(BaseModel):
    repo_id: str | None = None
    private: bool = False
    commit_message: str = "Add dataset via ai-datacreator"


class PipelineConfig(BaseModel):
    crawl: CrawlConfig = Field(default_factory=CrawlConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    curate: CurateConfig = Field(default_factory=CurateConfig)
    hf: HFConfig = Field(default_factory=HFConfig)
    output_dir: Path = Path("./output")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data or {})

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
        )
