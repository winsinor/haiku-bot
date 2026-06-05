#!/usr/bin/env python3
"""Quick Pi timing test: generate 5 haikus and report speed/energy stats."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import types
import haiku_benchmark_v2 as hb

hb.ARGS = types.SimpleNamespace(
    model="gemma2:2b",
    cycles=5,
    strategy="repair",
    audit=True,
    rerank=False,
    judge=False,
    smart_select=False,
    no_cache=False,
    once=True,
    seed=42,
    json=None,
)
hb.MODEL = "gemma2:2b"

import random, time
random.seed(42)

results = []
start = time.time()

for i in range(1, 6):
    t0 = time.time()
    r = hb.run_one_cycle(i, 5)
    elapsed = time.time() - t0
    if r is None:
        print(f"[{i}/5] Skipped - no valid event")
        continue
    results.append(r)
    cs = "/".join(str(c) for c in r["counts"]) if r["counts"] else "?"
    status = "OK" if r["success"] else "XX"
    print(f"\n[{i}/5] {status}  {r['date']}")
    print(f"  Event   : {r['inspiration'][:72]}{'...' if len(r['inspiration']) > 72 else ''}")
    if r["lines"]:
        for line in r["lines"]:
            print(f"  Haiku   : {line}")
    print(f"  Counts  : {cs}  |  Gens: {r['gens']}  |  Tokens: {r['tokens']}  |  Time: {elapsed:.1f}s")

total_time = time.time() - start

# ---- Stats ----
n = len(results)
if not n:
    print("No results.")
    sys.exit(1)

ok = sum(1 for r in results if r["success"])
total_tokens = sum(r["tokens"] for r in results)
avg_tokens = total_tokens / n
avg_time = total_time / n
hours = total_time / 3600
mwh = hb.PI4_TDP_WATTS * hours * 1000

print("\n" + "=" * 50)
print(f"  PI TIMING TEST — {hb.MODEL}")
print("=" * 50)
print(f"  Haikus generated : {n}")
print(f"  Successes        : {ok}/{n} ({100*ok//n}%)")
print(f"  Total time       : {total_time:.1f}s")
print(f"  Avg time/haiku   : {avg_time:.1f}s")
print(f"  Avg tokens/haiku : {avg_tokens:.0f}")
print(f"  Total tokens     : {total_tokens}")
print(f"  Est. energy      : {mwh:.3f} mWh  ({hb.PI4_TDP_WATTS}W TDP assumed)")
print("=" * 50)
