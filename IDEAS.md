# IDEAS.md — five upgrades, pre-designed for cheap implementation

Each idea below is specified so a smaller/cheaper model (or you in a hurry) can implement
it with minimal thinking: the problem, where it plugs into `haiku_benchmark_v2.py`, working
code, the reporting it adds, and how to verify. Pick them off one at a time.

Difficulty / payoff at a glance:

|#|Idea                                |Effort|Payoff                        |Model cost to test   |
|-|------------------------------------|------|------------------------------|---------------------|
|1|Record/replay harness               |Med   |**Huge** (test logic for free)|one-time only        |
|2|Diversity / mode-collapse detector  |Low   |High (catches gamed scores)   |none (offline metric)|
|3|Data-driven tolerance expansion     |Low   |Med (improves verifier)       |none (offline metric)|
|4|Self-consistency syllable confidence|Med   |Med-High (cuts wasted repairs)|moderate             |
|5|Difficulty-stratified reporting     |Low   |Med (diagnostic)              |none (offline metric)|

-----

## Idea 1 — Record/replay harness  ⭐ do this first

**Problem.** Every time you tweak the verifier, the assembler, the tolerance dict, or the
parser, you re-run the model to see the effect — burning Pi time and conflating "did my
logic change help?" with "did the model happen to sample differently?".

**Insight.** The model layer and the post-processing layer are separable. If you record
every raw model response keyed by its exact prompt, you can **replay** those recordings
through new post-processing code at zero model cost and with zero sampling noise. This is
the single highest-leverage testing tool you can add.

**Where it plugs in.** Wrap `call_ollama()`. Nothing else changes.

```python
# --- add near config ---
RECORD_PATH = os.path.expanduser("~/.haiku_record.jsonl")
import hashlib

def _prompt_key(prompt, temperature, num_predict):
    h = hashlib.sha256(f"{MODEL}|{temperature}|{num_predict}|{prompt}".encode()).hexdigest()[:16]
    return h

# --- replace the body of call_ollama with a record/replay wrapper ---
_REPLAY = None  # dict loaded from RECORD_PATH when --replay is set

def _load_replay():
    global _REPLAY
    _REPLAY = {}
    if os.path.exists(RECORD_PATH):
        with open(RECORD_PATH) as f:
            for line in f:
                d = json.loads(line)
                _REPLAY[d["key"]] = d

def call_ollama(prompt, temperature=0.7, num_predict=80):
    key = _prompt_key(prompt, temperature, num_predict)
    if ARGS is not None and getattr(ARGS, "replay", False):
        if _REPLAY is None:
            _load_replay()
        hit = _REPLAY.get(key)
        if hit:
            return hit["response"], hit["tokens"], 0.0
        # replay miss: deterministic empty-ish response so the run still completes
        return "", 0, 0.0
    # --- real call (existing code) ---
    payload = {"model": MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": temperature, "num_predict": num_predict, "num_thread": 4}}
    t0 = time.time()
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    resp = data["response"].strip()
    tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
    if ARGS is not None and getattr(ARGS, "record", False):
        with open(RECORD_PATH, "a") as f:
            f.write(json.dumps({"key": key, "prompt": prompt, "response": resp,
                                "tokens": tokens}) + "\n")
    return resp, tokens, elapsed
```

Add flags: `--record` (capture during a real run) and `--replay` (re-run from recordings).
Use a fixed `--seed` so the same events are drawn during both record and replay.

**Workflow it unlocks:**

```bash
# once: capture a fixed corpus of model behavior
python3 haiku_benchmark_v2.py --model qwen2.5:1.5b --cycles 50 --seed 1 --record --once
# now iterate on verifier/tolerance/assembler logic for FREE:
python3 haiku_benchmark_v2.py --model qwen2.5:1.5b --cycles 50 --seed 1 --replay --json after.json --once
```

**Verify.** Record a 5-cycle run, then replay it — success count and haikus should be
identical. Change `POETIC_SYLLABLES` (e.g. add a word), replay again — only metrics that
depend on tolerance should move. No network calls during replay.

-----

## Idea 2 — Diversity / mode-collapse detector

**Problem.** A model can score 100% on 5/7/5 by always reaching for the same safe imagery
("silent moon", "cold light"). Your benchmark currently can't see this — it rewards a model
that games the constraint with stock phrases. (You can already smell it in the mock output.)

**Where it plugs in.** New function + one line in `print_summary` / `write_json`. Pure
offline metric over `results`, no model calls.

```python
def diversity_report(results):
    """Inter-haiku content-word overlap. Low overlap = diverse; high = mode collapse."""
    succ = [r for r in results if r["success"]]
    if len(succ) < 2:
        return None
    wordsets = [content_words(" ".join(r["lines"])) for r in succ]
    # mean pairwise Jaccard similarity
    sims = []
    for i in range(len(wordsets)):
        for j in range(i + 1, len(wordsets)):
            a, b = wordsets[i], wordsets[j]
            if a or b:
                sims.append(len(a & b) / len(a | b))
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    # most-overused content words across all haikus
    from collections import Counter
    allw = Counter(w for ws in wordsets for w in ws)
    repeated = [(w, c) for w, c in allw.most_common(8) if c > 1]
    vocab_ratio = len(allw) / max(1, sum(allw.values()))  # unique / total
    return {"mean_jaccard": round(mean_sim, 3),
            "vocab_ratio": round(vocab_ratio, 3),
            "overused": repeated}
```

In `print_summary`:

```python
dv = diversity_report(results)
if dv:
    print(f"  Diversity        : jaccard={dv['mean_jaccard']} (lower=better)  "
          f"vocab_ratio={dv['vocab_ratio']} (higher=better)")
    if dv["overused"]:
        print(f"  Overused words   : " + ", ".join(f'{w}×{c}' for w, c in dv["overused"]))
```

**Interpretation.** mean_jaccard > ~0.25 across distinct events is a red flag for mode
collapse. Report it per model so you can compare "valid AND varied," not just valid.

**Verify.** Feed it 3 identical haikus → jaccard ≈ 1.0. Feed 3 disjoint ones → ≈ 0.0.

-----

## Idea 3 — Data-driven tolerance expansion

**Problem.** `POETIC_SYLLABLES` is hand-curated. Real runs will surface other ambiguous
words (CMU vs estimator disagreements) you haven't hard-coded, causing silent false
failures. Let the benchmark tell you which words to add.

**Where it plugs in.** Instrument `syl_range()` to log disagreements to a module-level
counter, then dump a ranked candidate list at end of run.

```python
from collections import Counter
_DISAGREE = Counter()

# inside syl_range(), in the branch where both cmu and est exist:
    if cmu is not None and cmu != est and wc not in POETIC_SYLLABLES:
        _DISAGREE[(wc, cmu, est)] += 1
    # ...return as before

def disagreement_report(top=15):
    if not _DISAGREE:
        return
    print("\n  -- Counter disagreements seen (candidates for POETIC_SYLLABLES) --")
    for (w, cmu, est), n in _DISAGREE.most_common(top):
        print(f'    "{w}": {{{min(cmu,est)}, {max(cmu,est)}}},   # ×{n}  cmu={cmu} est={est}')
```

Call `disagreement_report()` at the end of `write_log`. After a few hundred cycles you'll
have a copy-pasteable block of real ambiguous words to fold into the dict — turning the
tolerance system from hand-guessed to data-driven.

**Verify.** Run on text containing "fire", "engine", "flame"; confirm they appear in the
report with the right counts.

-----

## Idea 4 — Self-consistency syllable confidence

**Problem.** Some repair cycles are wasted "fixing" lines that are only ambiguous, not
wrong. And you have no measure of whether the *model itself* can count syllables — which is
the actual skill the benchmark is probing.

**Idea.** After generating a line, ask the model to count its own syllables. Compare to your
verifier. Use the agreement three ways: (a) skip repair when the model and a tolerant
verifier both think it's fine; (b) report the model's self-count accuracy as a headline
model-quality metric; (c) flag lines where model-count, CMU, and estimator all disagree as
genuinely ambiguous.

**Where it plugs in.** New helper; optional gate inside `repair_into_valid`; new metric.

```python
def model_syllable_count(line):
    """Ask the model to count syllables in a line. Returns int or None."""
    prompt = (f'Count the total syllables in this line. Reply ONLY a number.\n"{line}"')
    raw, tok, el = call_ollama(prompt, temperature=0.0, num_predict=5)
    m = re.search(r'\d+', raw)
    return (int(m.group()), tok) if m else (None, tok)

# Metric: how often does the model's self-count match CMU? (run with --self-check)
# Accumulate in stats: stats["selfcheck_hits"], stats["selfcheck_total"]
```

Gate (optional, behind `--self-check`): before spending repair budget on line `i`, if
`line_hits(line, target)` is True *and* the model's self-count == target, treat as done.
This avoids "repairing" tolerance-ambiguous lines.

Report in summary: `self-count accuracy = hits/total` — a clean, model-discriminating
number that doesn't depend on your verifier at all.

**Cost.** One extra cheap call (~5 tokens out) per line you check. Keep it behind a flag.

**Verify.** Mock the count call to return target → gate fires, repair budget untouched.
Mock it to return a wrong number → falls through to normal repair.

-----

## Idea 5 — Difficulty-stratified success reporting

**Problem.** A single aggregate success rate hides *where* the model fails. You proved
earlier that surface complexity doesn't cleanly predict failure — so measure it properly
across many runs instead of guessing, and bucket the results.

**Where it plugs in.** Reuse `haikuability_penalty()` (already in the script) as a difficulty
score; bucket results; report success + tokens per bucket. Pure offline.

```python
def difficulty_report(results):
    buckets = {"easy (0-4)": [], "med (5-9)": [], "hard (10+)": []}
    for r in results:
        p = haikuability_penalty(r["inspiration"])
        key = "easy (0-4)" if p <= 4 else "med (5-9)" if p <= 9 else "hard (10+)"
        buckets[key].append(r)
    print("\n  -- Success by event difficulty --")
    for k, rs in buckets.items():
        if rs:
            ok = sum(1 for r in rs if r["success"])
            tok = sum(r["tokens"] for r in rs) / len(rs)
            print(f"    {k:14} {ok}/{len(rs)} ok  ({100*ok//len(rs)}%)  avg {tok:.0f} tok")
```

Call it in `write_log`. Now you can answer "does hard-event success improve with model size,
or just easy-event success?" — far more actionable than one number. Over many seeded runs
this also empirically settles whether `haikuability_penalty` predicts anything (and thus
whether `--smart-select` is worth keeping).

**Verify.** Hand it results with known penalties; confirm bucketing and percentages.

-----

## Suggested order

1 first (it makes 2–5 free to develop and tune via replay), then 2 and 3 (cheap, high-signal
offline metrics), then 5 (diagnostic), then 4 (the only one with real per-line model cost).
