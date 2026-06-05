#!/usr/bin/env python3
"""Compare two haiku-benchmark JSON result files (from --json).

Usage:
    python3 compare_runs.py run_a.json run_b.json

Designed for A/B testing: run both configs with the SAME --seed so the event
draws match, then this prints a per-metric delta and (if seeds match) a
per-event head-to-head. Paste the output back to Claude for analysis.
"""
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def pct(x):
    return f"{100 * x:.1f}%"


def bar(label, a, b, fmt=str, better="higher"):
    """Print 'label : A  vs  B   (delta)'."""
    da = fmt(a) if a is not None else "n/a"
    db = fmt(b) if b is not None else "n/a"
    arrow = ""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = b - a
        if abs(d) > 1e-9:
            up = d > 0
            good = (up and better == "higher") or (not up and better == "lower")
            arrow = f"   {'+' if up else ''}{d:.3g}  {'✓' if good else '✗'}"
    print(f"  {label:22} {da:>12}  vs {db:>12}{arrow}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    A, B = load(sys.argv[1]), load(sys.argv[2])
    ma, mb = A["meta"], B["meta"]
    aa, ab = A["aggregate"], B["aggregate"]

    print("=" * 64)
    print(f"  A: {ma['model']} [{ma['strategy']}] seed={ma['seed']} n={ma['cycles']}")
    print(f"  B: {mb['model']} [{mb['strategy']}] seed={mb['seed']} n={mb['cycles']}")
    print("=" * 64)

    same_seed = ma["seed"] == mb["seed"] and ma["seed"] is not None
    print(f"  Seeds match (true A/B): {same_seed}")
    print("-" * 64)
    print(f"  {'metric':22} {'A':>12}     {'B':>12}")
    print("-" * 64)
    bar("success rate", aa["success_rate"], ab["success_rate"], pct, "higher")
    bar("successes", aa["success"], ab["success"], str, "higher")
    bar("avg tokens/haiku", aa["avg_tokens"], ab["avg_tokens"], lambda x: f"{x:.0f}", "lower")
    bar("tokens/SUCCESS", aa["tokens_per_success"], ab["tokens_per_success"], lambda x: f"{x:.0f}", "lower")
    bar("tolerance saves", aa["tolerance_saves"], ab["tolerance_saves"], str, "higher")
    bar("bailouts", aa["bailouts"], ab["bailouts"], str, "lower")
    bar("total time (s)", ma["total_time_s"], mb["total_time_s"], lambda x: f"{x:.1f}", "lower")
    print("-" * 64)
    print(f"  line failures  A={aa['line_failures']}  B={ab['line_failures']}")
    print(f"  phase tokens   A={ {k: v for k, v in aa['tokens_by_phase'].items() if v} }")
    print(f"                 B={ {k: v for k, v in ab['tokens_by_phase'].items() if v} }")

    # Per-event head-to-head when seeds match (same events, same order)
    if same_seed and len(A["cycles"]) == len(B["cycles"]):
        print("\n" + "-" * 64)
        print("  PER-EVENT HEAD-TO-HEAD (same seed => same events)")
        print("-" * 64)
        flips_win, flips_lose = 0, 0
        for ca, cb in zip(A["cycles"], B["cycles"]):
            sa, sb = ca["success"], cb["success"]
            mark = "  ="
            if sa and not sb:
                mark, flips_lose = "B✗", flips_lose + 1
            elif sb and not sa:
                mark, flips_win = "B✓", flips_win + 1
            tag = "OK" if sb else "XX"
            print(f"   [{mark}] {tag} {cb['date']:13} A:{'OK' if sa else 'XX'} B:{'OK' if sb else 'XX'}"
                  f"  (A {ca['tokens']}tok via {ca['source']} | B {cb['tokens']}tok via {cb['source']})")
        print("-" * 64)
        print(f"  B fixed {flips_win} that A failed; B broke {flips_lose} that A passed.")
        print(f"  NET: {flips_win - flips_lose:+d} haikus")

    print("=" * 64)


if __name__ == "__main__":
    main()
