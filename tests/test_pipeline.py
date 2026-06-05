#!/usr/bin/env python3
"""Offline end-to-end test: monkeypatch the network choke points and run the full
pipeline for every strategy. Proves control flow, parsing, assembly, repair,
audit, rerank and judge all execute without touching the network."""
import importlib.util, random, types, re, sys

spec = importlib.util.spec_from_file_location("hb", "haiku_benchmark_v2.py")
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)

random.seed(42)

BANK5 = ["Cold light in the dark","Dust drifts on the moon","Red plains lie still now",
         "Soft winds drift alone","Stone walls hold the night","Pale dawn breaks slowly",
         "Frost clings to the vine","Lone owl calls softly","Deep snow blankets all",
         "Grey clouds hide the sun"]
BANK7 = ["A frozen world drifts unseen","Distant echoes softly fade","Golden embers glow then die",
         "Restless oceans churn below","Ancient shadows stretch and yawn","Crimson sunsets bleed to grey",
         "Gentle thunder rolls afar","Silent mountains hold their breath","Weary travelers seek the shore",
         "Hollow valleys catch the rain"]
BAD = ["Bad","This line is clearly far too long to fit","Nope nope"]

# A tunable knob: fraction of pool lines that are deliberately wrong-length.
JUNK_RATE = 0.35
MODEL_TOKENS = (40, 120)

def mock_call_ollama(prompt, temperature=0.7, num_predict=80):
    tok = random.randint(*MODEL_TOKENS)
    low = prompt.lower()

    # judge
    if "rate this haiku" in low:
        return str(random.randint(5, 9)), tok, 0.1
    # rerank
    if "reads best as one coherent poem" in low:
        return str(random.randint(1, 3)), tok, 0.1
    # pool: "Each line MUST be exactly N syllables"
    m = re.search(r"exactly (\d+) syllables", prompt)
    if m and "different short poetic lines" in low:
        target = int(m.group(1))
        bank = BANK7 if target == 7 else BANK5
        lines = []
        for _ in range(hb.POOL_SIZE):
            if random.random() < JUNK_RATE:
                lines.append(random.choice(BAD))
            else:
                lines.append(random.choice(bank))
        return "\n".join(lines), tok, 0.2
    # repair: "need N" → return a correct-length line
    m2 = re.search(r"need (\d+)", prompt)
    if m2 and "editor fixing one line" in low:
        target = int(m2.group(1))
        bank = BANK7 if target == 7 else BANK5
        return random.choice(bank), tok, 0.15
    # whole-haiku generation
    if "<haiku>" in low:
        # 50/50 a clean haiku vs a slightly-off one to exercise repair
        if random.random() < 0.5:
            return (f"<haiku>\n{random.choice(BANK5)}\n{random.choice(BANK7)}\n"
                    f"{random.choice(BANK5)}\n</haiku>"), tok, 0.3
        return (f"<haiku>\n{random.choice(BAD)}\n{random.choice(BANK7)}\n"
                f"{random.choice(BANK5)}\n</haiku>"), tok, 0.3
    return "unknown", tok, 0.1

CANNED_EVENTS = [
    {"text": "Mariner 4 flyby of Mars takes the first close-up photos of another planet",
     "pages": [{"title": "Mariner_4"}]},
    {"text": "Triton, the largest moon of Neptune, is discovered",
     "pages": [{"title": "Triton_(moon)"}]},
    {"text": "The first National League baseball game is played",
     "pages": [{"title": "Baseball"}]},
]

def mock_events(date=None):
    return CANNED_EVENTS

def mock_summary(event):
    return "A brief mock summary for offline testing."

def make_args(strategy, **kw):
    a = types.SimpleNamespace(
        model="mock:test", cycles=4, strategy=strategy, audit=True, rerank=False,
        judge=False, smart_select=False, no_cache=True, once=True, seed=123, json=None)
    for k, v in kw.items():
        setattr(a, k, v)
    return a

def run(strategy, **kw):
    hb.call_ollama = mock_call_ollama
    hb.get_history_events = mock_events
    hb.fetch_article_summary = mock_summary
    hb.MODEL = "mock:test"
    hb.ARGS = make_args(strategy, **kw)
    print("\n" + "#" * 60)
    print(f"#  TESTING STRATEGY: {strategy}  extra={kw}")
    print("#" * 60)
    results = hb.run_batch(4)
    ok = sum(1 for r in results if r["success"])
    print(f"\n>>> {strategy}: {ok}/{len(results)} succeeded, no crash.")
    return results

if __name__ == "__main__":
    run("pool")
    run("repair")
    run("hybrid")
    run("hybrid", rerank=True)
    run("hybrid", judge=True)
    print("\n\nALL STRATEGIES RAN TO COMPLETION WITHOUT ERROR.")
