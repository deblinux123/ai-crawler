"""Crawl websites with crawl4ai and turn them into clean text chunks.

This module is the only place that talks to crawl4ai. Everything downstream
(generator.py) just consumes `Chunk` objects, so swapping the crawling
backend later would not touch the rest of the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from rich.console import Console

from .config import CrawlConfig

console = Console()


@dataclass
class Chunk:
    """A piece of crawled text small enough to feed to the LLM in one shot."""

    url: str
    title: str
    text: str
    chunk_index: int
    source_length: int = field(repr=False, default=0)


def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Paragraph-aware splitter: fills chunks with whole paragraphs, only
    hard-splitting a paragraph if it alone exceeds chunk_size."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
            continue

        if current:
            chunks.append(current)
            # keep a small tail as overlap for context continuity
            current = current[-overlap:] if overlap else ""

        if len(para) > chunk_size:
            for i in range(0, len(para), chunk_size - overlap):
                chunks.append(para[i : i + chunk_size])
            current = ""
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        chunks.append(current)

    return chunks


async def crawl_to_chunks(cfg: CrawlConfig) -> list[Chunk]:
    """Crawl every seed URL (with optional BFS depth) and return text chunks."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
    from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.4, threshold_type="fixed")
    )

    filters = []
    if cfg.exclude_patterns:
        filters.append(
            URLPatternFilter(patterns=[f"*{p}*" for p in cfg.exclude_patterns], reverse=True)
        )

    deep_strategy = None
    if cfg.max_depth > 0:
        deep_strategy = BFSDeepCrawlStrategy(
            max_depth=cfg.max_depth,
            include_external=not cfg.same_domain_only,
            max_pages=cfg.max_pages,
            filter_chain=FilterChain(filters) if filters else FilterChain([]),
        )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_generator,
        deep_crawl_strategy=deep_strategy,
        word_count_threshold=10,
        exclude_external_links=True,
        remove_overlay_elements=True,
        verbose=False,
    )

    browser_config = BrowserConfig(headless=True, verbose=False)

    chunks: list[Chunk] = []
    seen_urls: set[str] = set()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for seed_url in cfg.urls:
            console.print(f"[cyan]Crawling[/cyan] {seed_url} ...")
            try:
                results = await crawler.arun(url=seed_url, config=run_config)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Failed to crawl {seed_url}: {exc}[/red]")
                continue

            # arun returns a single CrawlResult for non-deep crawls,
            # or a list of CrawlResult when a deep_crawl_strategy is set.
            page_results = results if isinstance(results, list) else [results]

            for page in page_results:
                if not getattr(page, "success", True):
                    continue
                if page.url in seen_urls:
                    continue
                seen_urls.add(page.url)

                md_obj = page.markdown
                text = getattr(md_obj, "fit_markdown", None) or str(md_obj)
                text = text.strip()
                if len(text) < cfg.min_text_length:
                    continue

                title = ""
                try:
                    title = (page.metadata or {}).get("title", "")
                except AttributeError:
                    pass
                if not title:
                    title = urlparse(page.url).path.strip("/") or urlparse(page.url).netloc

                for i, piece in enumerate(
                    _split_into_chunks(text, cfg.chunk_size, cfg.chunk_overlap)
                ):
                    chunks.append(
                        Chunk(
                            url=page.url,
                            title=title,
                            text=piece,
                            chunk_index=i,
                            source_length=len(text),
                        )
                    )

    console.print(f"[green]Done crawling.[/green] {len(seen_urls)} pages -> {len(chunks)} chunks")
    return chunks
