# TESTING.md — what to actually run on the Pi / desktop

Goal: produce numbers that let us tell whether v2's changes are real wins, not noise. The
two enablers are already built: `--seed` (same events across runs) and `--json` (machine
-readable output that `compare_runs.py` diffs and that you can paste back to Claude).

## The golden rule

**Change one thing, hold a fixed `--seed`, dump `--json`, then `compare_runs.py`.** Without
a fixed seed you're comparing different events and any delta is partly luck.

-----

## Test 1 — Did the verifier change inflate or fix the score? (the big one)

This isolates the dual-counter tolerance, the change most likely to move your headline number.

```bash
python3 haiku_benchmark_v2.py --model qwen2.5:1.5b --cycles 30 --seed 1 \
    --strategy repair --audit --json v2_repair.json --once
```

Look at the summary line **`Tolerance saves : N`**. That's how many haikus passed only
because of the dual counter — i.e. how much of v1's failure rate was the *verifier*, not the
model. If N is large, v1 was under-reporting the model. If N is ~0, tolerance isn't the story
and the gains are coming from elsewhere.

> Honest check: read each tolerance-saved haiku in the DETAILED LOG (`[tolerance-saved]`
> tag). If any look like genuine 6-syllable lines that slipped through, tighten
> `POETIC_SYLLABLES`. The tolerance should only rescue true ambiguity (fire/hour/flower).

-----

## Test 2 — Strategy bake-off (repair vs pool vs hybrid)

Same model, same seed, three strategies. This answers "is pooling actually better?"

```bash
for S in repair pool hybrid; do
  python3 haiku_benchmark_v2.py --model qwen2.5:1.5b --cycles 30 --seed 2 \
      --strategy $S --audit --json run_$S.json --once
done
python3 compare_runs.py run_repair.json run_pool.json
python3 compare_runs.py run_repair.json run_hybrid.json
```

Watch **three** numbers, not one:

- `success rate` — does pooling win on validity?
- `tokens/SUCCESS` — the honest efficiency metric. Pooling makes more calls; is the higher
  success rate worth the tokens? (In my mock, hybrid was +1 success at 2.3× tokens — your
  real model may differ.)
- `PER-EVENT HEAD-TO-HEAD` — which specific events flip, and via which `source` path.

-----

## Test 3 — Pi vs desktop speed (same logic, different hardware)

You run on both. Hold everything constant; the only variable is the machine.

```bash
# on the Pi 4 and on the Ubuntu desktop, identical command:
python3 haiku_benchmark_v2.py --model qwen2.5:1.5b --cycles 20 --seed 3 \
    --strategy hybrid --json pi.json --once     # (name desktop.json on the desktop)
python3 compare_runs.py pi.json desktop.json
```

`success rate` and `tokens/SUCCESS` should be ~identical (same logic, same seed → same
events). **`total time`** is the real comparison. If success rates differ, something is
nondeterministic (model quantization, num_thread) — worth knowing.

-----

## Test 4 — Model sweep

Same seed, swap models. This is what the benchmark is *for*.

```bash
for M in qwen2.5:1.5b llama3.2:1b gemma2:2b phi3:mini; do
  python3 haiku_benchmark_v2.py --model "$M" --cycles 30 --seed 4 \
      --strategy hybrid --audit --judge --json "model_${M//[:.]/_}.json" --once
done
```

Compare pairwise. The `--judge` aesthetic score matters here: a model that hits 5/7/5 with
lifeless stock lines is worse than one at 90% with vivid imagery. (Idea #2's diversity metric
will sharpen this once you add it.)

-----

## What to paste back to Claude

Either:

- the two `*.json` files (best — I can run the exact comparison and read per-event detail), or
- the `compare_runs.py` output block, or
- the `DETAILED RUN LOG` section.

The JSON has everything I need: per-line strict counts AND tolerant ranges, the `source`
path per haiku, per-phase token spend, and the audit flags. That's enough for me to tell you
*why* a delta happened, not just that it did.

## Sample sizes

- 10 cycles: smoke test, don't trust the % .
- 30 cycles: minimum for a believable success-rate comparison.
- 100+ cycles (use Idea #1 replay to make logic iteration free): for tolerance-dict tuning
  and difficulty stratification (Idea #5).
