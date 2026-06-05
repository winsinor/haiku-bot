# haiku-bot

Fetches an "on this day in history" event from Wikipedia and generates a valid 5/7/5 haiku using a local LLM via [Ollama](https://ollama.com).

## Install

Requires Ollama. If it's not installed:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then clone and set up:

```bash
curl -fsSL https://raw.githubusercontent.com/winsinor/haiku-bot/main/setup.sh | bash
```

## Run

```bash
python3 haiku_bot.py
```

Example output:

```
  June 5, 1783 — The Montgolfier brothers made their first public balloon flight

    Silk bag catches wind
    Two brothers reach for the clouds
    Earth falls far below

  5 / 7 / 5  ·  287 tokens  ·  3.8s  ·  via pool
```

## Model

Defaults to `gemma2:2b`. If the model isn't installed, the script will prompt you to pull it automatically.

To use a different model:

```bash
python3 haiku_bot.py --model llama3.2:1b
```

To pull a model manually:

```bash
ollama pull gemma2:2b
```

To see what models you have installed:

```bash
ollama list
```

## Flags

| Flag | Description |
|---|---|
| `--model NAME` | Ollama model to use (default: `gemma2:2b`) |
| `--date MM-DD` | Draw events from a specific date (default: today) |
| `--no-cache` | Disable the Wikipedia response cache |

## Benchmark

`tests/haiku_benchmark_v2.py` runs multi-cycle benchmarks across models and generation strategies.

| Flag | Description |
|---|---|
| `--model NAME` | Ollama model name — skips the interactive picker |
| `--cycles N` | Number of haiku to generate — skips the prompt |
| `--strategy` | Generation strategy: `hybrid` (default), `pool`, or `repair` |
| `--once` | Exit after one batch instead of looping |
| `--seed N` | Seed the RNG for reproducible event draws — use for A/B comparisons |
| `--json PATH` | Write full machine-readable results to a JSON file |
| `--audit` | Flag haikus that passed only due to syllable-counter tolerance |
| `--rerank` | Ask the model to pick the most coherent assembled haiku (extra calls) |
| `--judge` | Score successful haikus aesthetically 1–10 using the model itself |
| `--smart-select` | Use a haiku-ability heuristic to prefer more imageable events |
| `--no-cache` | Disable the Wikipedia response cache |
| `--record` | Save all raw model responses to `~/.haiku_record.jsonl` |
| `--replay` | Replay from `~/.haiku_record.jsonl` instead of calling Ollama |

### Strategies

- **repair** — generate a whole haiku, then fix individual lines that miss the syllable target
- **pool** — generate candidate lines per position, filter to valid ones, assemble a non-repeating combo
- **hybrid** — try pool first, fall back to repair if assembly fails

### Record / Replay

Record a run once, then iterate on verifier or assembler logic at zero model cost:

```bash
python3 tests/haiku_benchmark_v2.py --model qwen2.5:1.5b --cycles 50 --seed 1 --record --once
python3 tests/haiku_benchmark_v2.py --model qwen2.5:1.5b --cycles 50 --seed 1 --replay --json results/after.json --once
```

### Comparing runs

```bash
python3 tests/compare_runs.py results/run_a.json results/run_b.json
```
