# ai-datacreator

Crawl the web with **crawl4ai**, turn it into fine-tuning data with a **local Ollama model**, curate it, and optionally push it straight to **HuggingFace Hub** — all from one CLI.

```
crawl (crawl4ai) → chunk → generate pairs (Ollama, local) → curate (dedupe/filter) → export JSONL → push to HF (optional)
```

## Install

```bash
git clone <this repo>
cd ai-datacreator
pip install -e .
```

Requires:
- Python 3.10+
- [crawl4ai](https://github.com/unclecode/crawl4ai) set up (`crawl4ai-setup` after install, to fetch the Playwright browser)
- [Ollama](https://ollama.com) running locally with at least one model pulled, e.g. `ollama pull llama3.1`

## Quick start

One-shot, end to end:

```bash
ai-datacreator run https://docs.example.com \
  --depth 2 \
  --model llama3.1 \
  --task both \
  --format alpaca \
  --out-dir ./output
```

This crawls the site (depth 2), generates both QA and instruction-style pairs with your local Ollama model, dedupes/filters low-quality pairs, and writes `output/train.alpaca.jsonl` + `output/val.alpaca.jsonl`.

Push to HuggingFace right after generating:

```bash
ai-datacreator run https://docs.example.com --push-to yourname/my-dataset
```
(Requires `huggingface-cli login` first, or `HF_TOKEN` set in your environment.)

## Step-by-step usage

If you want to inspect or reuse the crawled content before spending LLM calls on it:

```bash
# 1. Crawl and cache clean text chunks
ai-datacreator crawl https://docs.example.com --depth 2 --max-pages 100 --out chunks.jsonl

# 2. Generate + curate + export a dataset from those chunks
ai-datacreator generate chunks.jsonl \
  --model llama3.1 \
  --task qa \
  --format sharegpt \
  --pairs-per-chunk 4 \
  --out-dir ./output

# 3. Push an already-exported dataset
ai-datacreator push ./output/train.sharegpt.jsonl \
  --val-file ./output/val.sharegpt.jsonl \
  --repo yourname/my-dataset
```

## Commands

| Command | Purpose |
|---|---|
| `crawl` | Crawl URL(s) → clean text chunks (`chunks.jsonl`) |
| `generate` | Chunks → generated pairs → curated → exported JSONL |
| `run` | Crawl + generate + curate + export (+ optional push) in one go |
| `push` | Push an existing JSONL dataset to HuggingFace Hub |
| `init-config` | Write an example `config.yaml` you can edit and reuse |

Run `ai-datacreator <command> --help` for the full flag list on any command.

## Dataset formats

- **Alpaca**: `{"instruction": ..., "input": "", "output": ...}`
- **ShareGPT**: `{"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}`

Add `--include-source` to keep `source_url` / `task_type` fields in the output for traceability (strip them before training if you don't want them in the dataset).

## Task types

- `qa` — factual question/answer pairs grounded in the crawled text
- `instruction` — instruction-following tasks (summarize, explain, rewrite, compare, etc.) with a grounded response
- `both` — generates both kinds per chunk (default)

## Notes on quality

- Curation does exact-match dedupe always, plus a near-duplicate similarity filter (skipped automatically above 4000 pairs to stay fast — exact dedupe still applies).
- Output length is filtered (`--val-split` doesn't affect this) — tune `min_output_length` / `max_output_length` via a YAML config (see `init-config`) if the defaults don't fit your content.
- Bigger/better local models (e.g. `llama3.1:70b`, `qwen2.5:32b`) produce noticeably higher-quality pairs than small ones — worth the extra wait if quality matters more than speed.

## Using a config file

```bash
ai-datacreator init-config --out config.yaml
```

Edit the YAML, then wire it into your own script using `PipelineConfig.from_yaml("config.yaml")` if you want fully reproducible, version-controlled runs (the CLI flags above cover the common case; the config object exposes every knob for programmatic use).
# ai-crawler
