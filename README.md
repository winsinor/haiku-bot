# haiku-bot

Benchmark for local LLMs: pull a random "on this day in history" event from Wikipedia, generate a valid 5/7/5 haiku about it, and report success rate + token cost. Runs against any model served by [Ollama](https://ollama.com).

## Requirements

```
pip install requests syllables pronouncing
```

Ollama must be running locally on port 11434.

## Usage

```bash
python3 haiku_benchmark_v2.py
```

Without flags the script prompts you to pick a model and number of cycles interactively.

## Flags

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

## Strategies

- **repair** — generate a whole haiku, then fix individual lines that miss the syllable target
- **pool** — generate candidate lines per position, filter to valid ones, assemble a non-repeating combo
- **hybrid** — try pool first, fall back to repair if assembly fails

## Record / Replay

Record a run once, then iterate on the verifier or assembler logic at zero model cost:

```bash
python3 haiku_benchmark_v2.py --model qwen2.5:1.5b --cycles 50 --seed 1 --record --once
python3 haiku_benchmark_v2.py --model qwen2.5:1.5b --cycles 50 --seed 1 --replay --json results/after.json --once
```

## Comparing runs

```bash
python3 tests/compare_runs.py results/run_a.json results/run_b.json
```
