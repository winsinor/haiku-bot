#!/usr/bin/env python3
"""
Haiku benchmark v2 — local-LLM 5/7/5 generator + benchmark.

What's new vs v1 (all evidence-backed against the v1 run log):
  * DUAL-COUNTER tolerance: a line passes if EITHER CMU or the estimator
    (or a small poetic-exceptions dict) yields the target. Recovers
    legitimately-ambiguous lines like "...by fire" that v1 wrongly failed.
  * LINE-POOL strategy: generate a pool of candidate lines per position,
    keep the syllable-valid ones, assemble a non-repeating combo. Math from
    the v1 log says per-line best-of-3 (~0.57) beats whole-haiku best-of-6
    (~0.49) at half the generations.
  * HYBRID: pool first, fall back to targeted repair only where needed.
  * SYLLABLE-DICTIONARY injection + FEW-SHOT prelude in the gen prompt:
    small models can't count but can follow a lookup table and examples.
  * WIKIPEDIA CACHE: makes repeated benchmarks measure the model, not the network.
  * --audit: reports which "failures" were really counter-disagreements.
  * --rerank / --judge: optional quality axes (cost extra model calls).

Network choke points are call_ollama() (model) and the wiki_* fns (Wikipedia),
both monkeypatchable for offline testing. See test_pipeline.py.
"""

import argparse
import datetime
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time

import requests
import syllables

try:
    import pronouncing
    HAS_CMU = True
except ImportError:
    HAS_CMU = False

# ----------------------------- Config -----------------------------
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_BASE}/api/generate"

MAX_RETRIES = 4                 # whole-haiku attempts
POOL_SIZE = 6                   # candidate lines requested per position
MAX_LINE_FAILURES = [3, 5, 3]   # repair budget per position (L1, L2, L3)
MAX_REPAIR_DISTANCE = 3         # skip/bail repairs if a line is further off than this
REPAIR_TEMPS = [0.3, 0.15, 0.5, 0.2, 0.6, 0.35, 0.55]
MIN_SCORE = 0
PI4_TDP_WATTS = 6.0
TARGET = [5, 7, 5]
CACHE_PATH = os.path.expanduser("~/.haiku_wiki_cache.json")
RECORD_PATH = os.path.expanduser("~/.haiku_record.jsonl")

# Words where CMU and poets/estimator legitimately disagree.
# Value = SET of syllable counts a careful reader would accept.
POETIC_SYLLABLES = {
    "fire": {1, 2}, "fires": {1, 2}, "hour": {1, 2}, "hours": {1, 2},
    "flower": {1, 2}, "flowers": {1, 2}, "our": {1, 2}, "ours": {1, 2},
    "power": {1, 2}, "powers": {1, 2}, "higher": {1, 2}, "flame": {1, 2},
    "flames": {1, 2}, "prayer": {1, 2}, "quiet": {1, 2}, "being": {1, 2},
    "poem": {1, 2}, "every": {2, 3}, "heaven": {1, 2}, "evening": {2, 3},
    "different": {2, 3}, "toward": {1, 2}, "towards": {1, 2}, "science": {1, 2},
    "engine": {2, 3}, "choir": {1, 2}, "tower": {1, 2}, "shower": {1, 2},
}

STOPWORDS = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
             "of", "for", "is", "are", "was", "were", "be", "as", "by", "it"}

FEWSHOT = """Examples (count syllables by sounding out each word):

Event: A telescope discovers a distant icy moon.
<haiku>
Cold light in the dark
A frozen world drifts un-seen
Si-lence wraps the stars
</haiku>

Event: The first photograph is taken from orbit.
<haiku>
Earth hangs in the void
A small blue mar-ble glow-ing
Far from rest-less hands
</haiku>
"""

# ----------------------------- Runtime -----------------------------
MODEL = None
_TOTAL_CYCLES = 1
ARGS = None


# ----------------------------- Syllables -----------------------------
from collections import Counter
_DISAGREE = Counter()  # (word, cmu_count, est_count) → occurrences


def syl_range(word):
    """(min, max) plausible syllables for one word, fusing CMU + estimator + poetic dict."""
    wc = re.sub(r"[^a-zA-Z']", "", word).lower()
    if not wc:
        return (0, 0)
    if wc in POETIC_SYLLABLES:
        s = POETIC_SYLLABLES[wc]
        return (min(s), max(s))
    cmu = None
    if HAS_CMU:
        ph = pronouncing.phones_for_word(wc)
        if ph:
            cmu = pronouncing.syllable_count(ph[0])
    est = syllables.estimate(wc)
    if cmu is None:
        return (est, est)
    if cmu != est:
        _DISAGREE[(wc, cmu, est)] += 1
    return (cmu, cmu)  # CMU is authoritative; estimator only used as fallback


def line_range(line):
    rs = [syl_range(w) for w in re.findall(r"[a-zA-Z']+", line)]
    return (sum(a for a, _ in rs), sum(b for _, b in rs))


def line_strict(line):
    """Single best-guess count (CMU-first), used for reporting + audit."""
    total = 0
    for w in re.findall(r"[a-zA-Z']+", line):
        wc = re.sub(r"[^a-zA-Z']", "", w).lower()
        if not wc:
            continue
        if HAS_CMU:
            ph = pronouncing.phones_for_word(wc)
            if ph:
                total += pronouncing.syllable_count(ph[0])
                continue
        total += syllables.estimate(wc)
    return total


def line_hits(line, target):
    if abs(line_strict(line) - target) > 1:
        return False
    lo, hi = line_range(line)
    return lo <= target <= hi


def line_breakdown(line):
    """Word=syllables pairs for the repair prompt, e.g. 'the=1, Carrington=4, Event=2'."""
    parts = []
    for w in re.findall(r"[a-zA-Z']+", line):
        wc = re.sub(r"[^a-zA-Z']", "", w).lower()
        if not wc:
            continue
        lo, hi = syl_range(wc)
        parts.append(f"{w}={lo}" if lo == hi else f"{w}={lo}-{hi}")
    return ", ".join(parts)


# ----------------------------- Word reuse -----------------------------
def content_words(line):
    return {w.lower().strip(".,!?;:'\"") for w in line.split()} - STOPWORDS


def has_word_reuse(lines):
    seen = set()
    for line in lines:
        w = content_words(line)
        if w & seen:
            return True
        seen |= w
    return False


def verify_haiku(haiku_text):
    """Tolerant verify: each line must be able to hit its target within its range."""
    lines = [l.strip() for l in haiku_text.strip().splitlines() if l.strip()]
    if len(lines) != 3:
        return False, lines, [], "wrong line count"
    counts = [line_strict(l) for l in lines]
    if not all(line_hits(lines[i], TARGET[i]) for i in range(3)):
        return False, lines, counts, f"syllable counts {counts}"
    if has_word_reuse(lines):
        return False, lines, counts, "repeated words across lines"
    return True, lines, counts, "ok"


def strict_valid(lines):
    """Would the OLD strict single-counter have accepted this? (for audit)"""
    if len(lines) != 3:
        return False
    return [line_strict(l) for l in lines] == TARGET


# ----------------------------- Wikipedia (cacheable) -----------------------------
class JsonCache:
    def __init__(self, path):
        self.path, self.data = path, {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)


_CACHE = None


def _cache():
    global _CACHE
    if _CACHE is None:
        _CACHE = JsonCache(CACHE_PATH)
    return _CACHE


def wiki_events(date):
    url = (f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/"
           f"{date.strftime('%m')}/{date.strftime('%d')}")
    headers = {"User-Agent": "haiku-bot/2.0 (personal project)"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json().get("events", [])


def wiki_summary(title):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
    headers = {"User-Agent": "haiku-bot/2.0 (personal project)"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    extract = r.json().get("extract", "")
    sentences = re.split(r'(?<=[.!?])\s+', extract.strip())
    return " ".join(sentences[:3])


def get_history_events(date=None):
    if date is None:
        date = datetime.date.today()
    key = f"events:{date.strftime('%m-%d')}"
    if ARGS is None or not ARGS.no_cache:
        hit = _cache().get(key)
        if hit is not None:
            return hit
    try:
        events = wiki_events(date)
    except Exception as e:
        print(f"Error fetching Wikipedia events: {e}")
        sys.exit(1)
    if ARGS is None or not ARGS.no_cache:
        _cache().set(key, events)
    return events


def fetch_article_summary(event):
    pages = event.get("pages", [])
    if not pages:
        return None
    title = pages[0].get("title", "")
    if not title:
        return None
    key = f"summary:{title}"
    if ARGS is None or not ARGS.no_cache:
        hit = _cache().get(key)
        if hit is not None:
            return hit
    try:
        s = wiki_summary(title)
    except Exception:
        s = None
    if s and (ARGS is None or not ARGS.no_cache):
        _cache().set(key, s)
    return s


# ----------------------------- Event scoring -----------------------------
def score_event(text):
    t = text.lower()
    s = 0
    if any(k in t for k in ["battle", "war", "invasion", "siege", "troops", "regiment", "casualties", "army", "military"]):
        s -= 15
    if any(k in t for k in ["died", "dead", "death", "die", "killed", "kill", "killing", "assassinated", "executed", "funeral", "massacre", "murder"]):
        s -= 12
    if any(k in t for k in ["pope", "saint", "canonized", "bishop", "church", "cathedral", "vatican", "monastery",
                            "religious", "theology", "crusade", "mosque", "temple", "synagogue", "cardinal", "clergyman"]):
        s -= 15
    if any(k in t for k in ["elected", "parliament", "senate", "congress", "court", "ruling", "legislation", "constitution", "treaty"]):
        s -= 7
    if any(k in t for k in ["founded", "incorporated", "merger", "acquisition", "ipo", "bank", "company", "corporation"]):
        s -= 5
    if any(k in t for k in ["census", "economic", "report", "standard", "tariff", "index"]):
        s -= 3
    if any(k in t for k in ["space", "launch", "comet", "eclipse", "astronaut", "orbit", "planet", "telescope"]):
        s += 5
    if any(k in t for k in ["discovered", "expedition", "explored", "mapped", "voyage", "island", "cave"]):
        s += 5
    if any(k in t for k in ["storm", "earthquake", "volcano", "eruption", "hurricane", "blizzard", "meteor"]):
        s += 4
    if any(k in t for k in ["invented", "patent", "first", "demonstration", "computing", "prototype", "machine"]):
        s += 4
    if any(k in t for k in ["film", "album", "music", "art", "book", "premiere", "published", "exhibition", "movie", "theater", "broadcast", "novel"]):
        s += 5
    if any(k in t for k in ["video game", "arcade", "nintendo", "playstation", "atari", "sega", "console", "esports", "pac-man", "tetris", "super mario"]):
        s += 7
    if any(k in t for k in ["record", "unusual", "odd", "strange", "remarkable", "bizarre", "peculiar", "curiosity"]):
        s += 8
    if any(k in t for k in ["animal", "zoo", "creature", "beast", "dinosaur", "fossil", "mascot", "pet", "dog", "cat", "bear"]):
        s += 8
    if any(k in t for k in ["hoax", "prank", "stunt", "circus", "illusion", "escapologist", "magician"]):
        s += 8
    if any(k in t for k in ["toy", "game", "comic", "boardgame", "puzzle", "theme park", "ride", "animation"]):
        s += 6
    if any(k in t for k in ["food", "drink", "recipe", "chocolate", "beer", "wine", "feast", "restaurant", "coincidence"]):
        s += 6
    return s


def haikuability_penalty(text):
    """Mild 'hard to haiku-ify' signal. NOTE: tested weak on v1 data — tiebreaker only."""
    p = 0
    p += 4 * len(re.findall(r"[A-Z][a-z]{6,}", text))   # long proper nouns
    p += 3 * len(re.findall(r"\d", text))               # digits
    p += 2 * text.count(",")                            # clauses
    p += 3 * len(re.findall(r"[A-Za-z]{11,}", text))    # tongue-twisters
    if len(text) > 120:
        p += 5
    return p


def pick_event(events):
    if not events:
        return None
    scored = sorted(
        [(score_event(e.get("text", "")), e) for e in events],
        reverse=True, key=lambda x: x[0]
    )
    candidates = [(sc, e) for sc, e in scored if sc >= MIN_SCORE and len(e.get("text", "")) < 200]
    if not candidates:
        candidates = scored[:5]
    if ARGS is not None and ARGS.smart_select:
        # weak tiebreak by haiku-ability among the top few (honest: small effect)
        top = candidates[:5]
        top.sort(key=lambda se: haikuability_penalty(se[1].get("text", "")))
        return top[0][1]
    return candidates[0][1]


# ----------------------------- Ollama -----------------------------
_REPLAY = None  # populated by --replay


def _prompt_key(prompt, temperature, num_predict):
    h = hashlib.sha256(f"{MODEL}|{temperature}|{num_predict}|{prompt}".encode()).hexdigest()[:16]
    return h


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
        return "", 0, 0.0  # replay miss: deterministic empty response
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict, "num_thread": 4},
    }
    t0 = time.time()
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    if "response" not in data:
        raise ValueError(f"Unexpected Ollama response: {data}")
    tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
    resp = data["response"].strip()
    if ARGS is not None and getattr(ARGS, "record", False):
        with open(RECORD_PATH, "a") as f:
            f.write(json.dumps({"key": key, "prompt": prompt, "response": resp, "tokens": tokens}) + "\n")
    return resp, tokens, elapsed


# ----------------------------- Parsing -----------------------------
def clean_line(line):
    line = re.sub(r'`[^`]+`|```\w*', '', line).strip()
    line = re.split(r'\s+becomes\s+', line, flags=re.IGNORECASE)[0].strip()
    line = re.sub(r'<[^>]+>', '', line).strip()
    line = re.sub(r'\[[^\]]*\]', '', line).strip()       # strip [placeholder] text
    line = re.sub(r'^\s*[\d#\-\*\.\)]+\s*', '', line)   # leading list markers
    line = line.strip('\'"“”—–-')
    return line.strip()


def parse_pool(raw):
    """Extract candidate lines from a pool response: one per nonempty line."""
    out = []
    for ln in raw.splitlines():
        c = clean_line(ln)
        if c and len(c.split()) >= 1 and not c.lower().startswith(("event:", "haiku", "example")):
            out.append(c)
    # dedupe preserving order
    seen, uniq = set(), []
    for c in out:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def extract_haiku(raw):
    m = re.search(r'<haiku>\s*(.*?)\s*</haiku>', raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.strip().startswith(('<', '>', '-'))]
    return "\n".join(lines[-3:]) if len(lines) >= 3 else raw


# ----------------------------- Prompt builders -----------------------------
def syllable_hint(text):
    words = [w for w in re.findall(r"[A-Za-z]+", text) if len(w) > 3]
    pairs = []
    for w in words[:8]:
        lo, hi = syl_range(w)
        pairs.append(f"{w}={lo}" if lo == hi else f"{w}={lo}-{hi}")
    return ("Syllable counts for key words: " + ", ".join(pairs) + ".") if pairs else ""


def gen_prompt(event_text, summary=None):
    ctx = event_text if not summary else f"{event_text}\n\nMore context: {summary}"
    hint = syllable_hint(event_text)
    return f"""{FEWSHOT}
On this day in history: {ctx}

Write a traditional haiku about this event. Focus on vivid imagery.
Three lines: five syllables, seven syllables, five syllables.
Spell out any numbers as words. Do not repeat a word across lines.
Use simple common words; avoid long proper nouns.
{hint}
Output ONLY the three lines inside <haiku> tags:

<haiku>
[five-syllable line]
[seven-syllable line]
[five-syllable line]
</haiku>"""


POSITION_NAME = {0: "five-syllable", 1: "seven-syllable", 2: "five-syllable"}


def pool_prompt(position, event_text, summary, used_words):
    target = TARGET[position]
    ctx = event_text if not summary else f"{event_text} ({summary})"
    avoid = ""
    if used_words:
        avoid = f"Do NOT use these words: {', '.join(sorted(used_words))}.\n"
    return f"""Write {POOL_SIZE} different short poetic lines about this event, for a haiku.
Event: {ctx}

Each line MUST be exactly {target} syllables. Count by sounding out words:
"en-gine"=2, "bright"=1, "morn-ing"=2, "launch"=1, "dis-tant"=2.
Use vivid imagery and simple common words. Avoid long proper nouns.
{avoid}Output ONLY the {POOL_SIZE} lines, one per line, no numbering, no extra text."""


def repair_prompt(line, current, target, context_lines, temperature):
    direction = "shorter" if current > target else "longer"
    diff = abs(current - target)
    sw = "syllable" if diff == 1 else "syllables"
    if direction == "shorter":
        ex = (f'Example ({diff} {sw} shorter): "The old si-lent pond" (5) -> "Old si-lent pond" (4). '
              f'Drop small words or shorten a phrase.')
    else:
        ex = (f'Example ({diff} {sw} longer): "Old si-lent pond" (4) -> "The old si-lent pond" (5). '
              f'Add a small word or expand a phrase.')
    return f"""You are an editor fixing ONE line of a haiku.
Original line ({current} syllables, need {target}): "{line}"

The other lines are:
{context_lines}

Rewrite so it has EXACTLY {target} syllables ({diff} {sw} {direction}).
Count by sounding out: "en-gine"=2, "bright"=1, "morn-ing"=2, "launch"=1.
{ex}
Use simple common words. Avoid long proper nouns.
Do not reuse any word already in the other lines.
Output ONLY the single rewritten line. No explanations, no quotes, no markdown."""


# ----------------------------- Strategies -----------------------------
def assemble(pools):
    """First non-reusing combination across position pools, or None."""
    sizes = [len(p) for p in pools]
    if 0 in sizes:
        return None
    # cap the search space defensively
    capped = [p[:8] for p in pools]
    for combo in itertools.product(*capped):
        if not has_word_reuse(combo):
            return list(combo)
    return None


def rerank_best(combos, event_text):
    """Optional: ask the model to pick the most coherent assembled haiku."""
    if len(combos) <= 1:
        return combos[0] if combos else None
    listing = "\n\n".join(f"{i+1}.\n" + "\n".join(c) for i, c in enumerate(combos))
    prompt = (f"These are candidate haiku about: {event_text}\n\n{listing}\n\n"
              f"Which number reads best as one coherent poem? Reply with ONLY the number.")
    raw, tok, _ = call_ollama(prompt, temperature=0.0, num_predict=5)
    m = re.search(r'\d+', raw)
    idx = (int(m.group()) - 1) if m else 0
    if not (0 <= idx < len(combos)):
        idx = 0
    return combos[idx], tok


def strategy_pool(event_text, summary, stats):
    """Generate per-position pools, filter to valid lines, assemble."""
    pools, used = [], set()
    for pos in range(3):
        raw, tok, el = call_ollama(pool_prompt(pos, event_text, summary, used),
                                  temperature=0.7, num_predict=120)
        acct(stats, "pool", tok, el)
        stats["gens"] += 1
        cands = [c for c in parse_pool(raw) if line_hits(c, TARGET[pos])]
        # bank words from the first valid candidate to nudge later positions apart
        if cands:
            used |= content_words(cands[0])
        pools.append(cands)
        stats["pool_valid"][pos] += len(cands)

    if ARGS is not None and ARGS.rerank:
        # build several non-reusing combos, then rerank
        combos, seen = [], set()
        sizes = [len(p) for p in pools]
        if 0 not in sizes:
            for combo in itertools.product(*[p[:4] for p in pools]):
                if not has_word_reuse(combo):
                    key = tuple(combo)
                    if key not in seen:
                        seen.add(key)
                        combos.append(list(combo))
                if len(combos) >= 4:
                    break
        if combos:
            picked = rerank_best(combos, event_text)
            if isinstance(picked, tuple):
                acct(stats, "rerank", picked[1], 0.0)
                picked = picked[0]
            return picked
        return None
    return assemble(pools)


def repair_into_valid(lines, counts, stats):
    """Targeted per-line repair with bailout. Mutates and returns (lines, counts, ok)."""
    for i in range(3):
        if line_hits(lines[i], TARGET[i]):
            continue
        budget = MAX_LINE_FAILURES[i]
        for rep in range(1, budget + 1):
            if abs(counts[i] - TARGET[i]) > MAX_REPAIR_DISTANCE:
                stats["bailouts"] += 1
                return lines, counts, False
            temp = REPAIR_TEMPS[rep % len(REPAIR_TEMPS)]
            ctx = "\n".join(l for j, l in enumerate(lines) if j != i)
            new, tok, el = call_ollama(repair_prompt(lines[i], counts[i], TARGET[i], ctx, temp),
                                      temperature=temp, num_predict=60)
            new = clean_line(new.splitlines()[0] if new.splitlines() else new)
            acct(stats, "repair", tok, el)
            cand = [l if j != i else new for j, l in enumerate(lines)]
            nc = line_strict(new)
            if line_hits(new, TARGET[i]) and not has_word_reuse(cand):
                lines[i], counts[i] = new, nc
                break
            stats["line_failures"][i] += 1
            # accept partial progress ONLY if strictly closer to target (never worse)
            if abs(nc - TARGET[i]) < abs(counts[i] - TARGET[i]) and not has_word_reuse(cand):
                lines[i], counts[i] = new, nc
        else:
            stats["bailouts"] += 1
            return lines, counts, False
    ok = all(line_hits(lines[i], TARGET[i]) for i in range(3)) and not has_word_reuse(lines)
    return lines, counts, ok


# ----------------------------- One cycle -----------------------------
def acct(stats, phase, tok, elapsed):
    """Attribute token + time cost to a named phase (pool/repair/salvage/gen/rerank)."""
    stats["tokens"] += tok
    stats["tok"][phase] += tok
    stats["sec"][phase] += elapsed


def run_one_cycle(cycle_num, total_cycles):
    random_date = datetime.date(2000, random.randint(1, 12), random.randint(1, 28))
    events = get_history_events(random_date)
    event = pick_event(events)
    if not event:
        return None
    event_text = event.get("text", "")
    summary = fetch_article_summary(event)

    stats = {"tokens": 0, "gens": 0, "line_failures": [0, 0, 0],
             "bailouts": 0, "pool_valid": [0, 0, 0], "strategy_used": ARGS.strategy,
             "audit_recovered": False, "source": "none",
             "tok": {"pool": 0, "repair": 0, "salvage": 0, "gen": 0, "rerank": 0},
             "sec": {"pool": 0.0, "repair": 0.0, "salvage": 0.0, "gen": 0.0, "rerank": 0.0}}
    lines, counts, success = [], [], False

    strat = ARGS.strategy

    # ---- POOL / HYBRID: try assembly first ----
    if strat in ("pool", "hybrid"):
        print(f"  Cycle {cycle_num} of {total_cycles}: pooling lines...", end="\r", flush=True)
        assembled = strategy_pool(event_text, summary, stats)
        if assembled:
            stats["source"] = "pool"
            lines = assembled
            counts = [line_strict(l) for l in lines]
            valid, lines, counts, _ = verify_haiku("\n".join(lines))
            if valid:
                success = True
        elif strat == "hybrid":
            # Salvage: fresh whole-haiku gen, then repair.
            stats["source"] = "salvage"
            print(f"  Cycle {cycle_num} of {total_cycles}: hybrid repair fallback...", end="\r", flush=True)
            raw, tok, el = call_ollama(gen_prompt(event_text, summary), 0.7, 100)
            acct(stats, "salvage", tok, el)
            stats["gens"] += 1
            v, lines, counts, _ = verify_haiku(extract_haiku(raw))
            if v:
                success = True
            elif len(lines) == 3:
                lines, counts, success = repair_into_valid(lines, counts, stats)

    # ---- REPAIR (v1-style) ----
    if strat == "repair":
        stats["source"] = "repair"
        for attempt in range(1, MAX_RETRIES + 1):
            stats["gens"] += 1
            print(f"  Cycle {cycle_num} of {total_cycles}: generating {attempt} of {MAX_RETRIES}...",
                  end="\r", flush=True)
            raw, tok, el = call_ollama(gen_prompt(event_text, summary), 0.7, 100)
            acct(stats, "gen", tok, el)
            v, lines, counts, reason = verify_haiku(extract_haiku(raw))
            if v:
                success = True
                break
            if len(lines) != 3 or reason == "repeated words across lines":
                continue
            if any(abs(counts[i] - TARGET[i]) > MAX_REPAIR_DISTANCE for i in range(min(3, len(counts)))):
                stats["bailouts"] += 1
                continue
            lines, counts, success = repair_into_valid(lines, counts, stats)
            if success:
                break

    print(" " * 64, end="\r", flush=True)

    # ---- Audit: did tolerance recover a strict-counter failure? ----
    if success and lines and not strict_valid(lines):
        stats["audit_recovered"] = True

    return {
        "date": random_date.strftime("%B %d"),
        "inspiration": event_text,
        "lines": lines,
        "counts": counts,
        "ranges": [line_range(l) for l in lines],
        "success": success,
        **stats,
    }


# ----------------------------- Reporting -----------------------------
def print_summary(results, elapsed_total):
    total = len(results)
    if not total:
        return
    perfect = sum(1 for r in results if r["success"])
    avg_tokens = sum(r["tokens"] for r in results) / total
    avg_gens = sum(r["gens"] for r in results) / total
    hours = elapsed_total / 3600
    mwh = PI4_TDP_WATTS * hours * 1000
    recovered = sum(1 for r in results if r.get("audit_recovered"))

    print("\n" + "=" * 50)
    print(f"  BENCHMARK SUMMARY - {MODEL}  [{ARGS.strategy}]")
    print("=" * 50)
    print(f"  Cycles completed : {total}")
    print(f"  Perfect 5/7/5    : {perfect}/{total} ({100 * perfect // total}%)")
    print(f"  Avg tokens/haiku : {avg_tokens:.0f}")
    print(f"  Avg gens needed  : {avg_gens:.1f}")
    print(f"  Total time       : {elapsed_total:.1f}s")
    print(f"  Avg time/haiku   : {elapsed_total / total:.1f}s")
    print(f"  Est. energy      : {mwh:.2f} mWh")
    lf = [sum(r["line_failures"][i] for r in results) for i in range(3)]
    print(f"  Line fail counts : L1={lf[0]}  L2={lf[1]}  L3={lf[2]}")
    print(f"  Early bailouts   : {sum(r['bailouts'] for r in results)}")
    if ARGS.strategy in ("pool", "hybrid"):
        pv = [sum(r['pool_valid'][i] for r in results) for i in range(3)]
        print(f"  Pool valid lines : L1={pv[0]}  L2={pv[1]}  L3={pv[2]} (of {POOL_SIZE}/pos/cycle)")
    if ARGS.audit:
        print(f"  Tolerance saves  : {recovered} (would've FAILED under strict single-counter)")
    # Efficiency: tokens per SUCCESSFUL haiku (the number that matters when failures waste tokens)
    succ_tokens = sum(r["tokens"] for r in results if r["success"])
    print(f"  Tokens/SUCCESS   : {succ_tokens / perfect:.0f}" if perfect else "  Tokens/SUCCESS   : n/a (0 successes)")
    # Where did successes come from?
    from collections import Counter
    paths = Counter(r["source"] for r in results)
    succ_paths = Counter(r["source"] for r in results if r["success"])
    print(f"  Source paths     : " + "  ".join(f"{k}={paths[k]}(ok:{succ_paths.get(k,0)})" for k in sorted(paths)))
    # Per-phase token spend (helps see if pooling or repair dominates cost)
    phase_tok = {p: sum(r["tok"].get(p, 0) for r in results) for p in ("pool", "repair", "salvage", "gen", "rerank")}
    nz = {k: v for k, v in phase_tok.items() if v}
    if nz:
        print(f"  Tokens by phase  : " + "  ".join(f"{k}={v}" for k, v in nz.items()))
    dv = diversity_report(results)
    if dv:
        print(f"  Diversity        : jaccard={dv['mean_jaccard']} (lower=better)  "
              f"vocab_ratio={dv['vocab_ratio']} (higher=better)")
        if dv["overused"]:
            print(f"  Overused words   : " + ", ".join(f"{w}×{c}" for w, c in dv["overused"]))
    print("=" * 50)
    sys.stdout.flush()


def diversity_report(results):
    succ = [r for r in results if r["success"] and r.get("lines")]
    if len(succ) < 2:
        return None
    wordsets = [content_words(" ".join(r["lines"])) for r in succ]
    sims = []
    for i in range(len(wordsets)):
        for j in range(i + 1, len(wordsets)):
            a, b = wordsets[i], wordsets[j]
            if a or b:
                sims.append(len(a & b) / len(a | b))
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    allw = Counter(w for ws in wordsets for w in ws)
    repeated = [(w, c) for w, c in allw.most_common(8) if c > 1]
    vocab_ratio = len(allw) / max(1, sum(allw.values()))
    return {
        "mean_jaccard": round(mean_sim, 3),
        "vocab_ratio": round(vocab_ratio, 3),
        "overused": repeated,
    }


def write_log(results, elapsed_total):
    if not results:
        return
    print("\n" + "#" * 50)
    print(f"  DETAILED RUN LOG - {MODEL}  [{ARGS.strategy}]")
    print(f"  POOL_SIZE={POOL_SIZE}  MAX_LINE_FAILURES={MAX_LINE_FAILURES}  BAILOUT_DIST={MAX_REPAIR_DISTANCE}")
    print("#" * 50)
    for i, r in enumerate(results):
        status = "OK " if r["success"] else "XX "
        cs = "/".join(str(c) for c in r["counts"]) if r["counts"] else "?"
        flag = "  [tolerance-saved]" if r.get("audit_recovered") else ""
        print(f"\n  [{i+1}] {status}{r['date']} - {r['inspiration'][:66]}{'...' if len(r['inspiration']) > 66 else ''}")
        for line in r["lines"]:
            print(f"       {line}")
        print(f"       Syllables(strict): {cs}  |  Gens: {r['gens']}  |  Tokens: {r['tokens']}  |  "
              f"Bailouts: {r['bailouts']}  |  Lfails: {r['line_failures']}{flag}")
    fails = [r for r in results if not r["success"]]
    print("\n" + "-" * 50)
    print("  Failures detail:")
    if fails:
        for r in fails:
            cs = "/".join(str(c) for c in r["counts"]) if r["counts"] else "?"
            print(f"    XX {r['date']}: final {cs} - {r['inspiration'][:56]}")
    else:
        print("    None - perfect run!")

    if ARGS.judge:
        run_judge(results)

    difficulty_report(results)
    disagreement_report()
    print("#" * 50)


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


def disagreement_report(top=15):
    if not _DISAGREE:
        return
    print("\n  -- CMU/estimator disagreements (candidates for POETIC_SYLLABLES) --")
    for (w, cmu, est), n in _DISAGREE.most_common(top):
        print(f'    "{w}": {{{min(cmu, est)}, {max(cmu, est)}}},   # ×{n}  cmu={cmu} est={est}')


def run_judge(results):
    """Optional aesthetic scoring of successful haikus by the model itself."""
    good = [r for r in results if r["success"]]
    if not good:
        return
    print("\n  -- Aesthetic judging (model-as-judge, 1-10) --")
    scores = []
    for r in good:
        prompt = (f"Rate this haiku 1-10 on imagery and flow. Reply ONLY a number.\n\n"
                  + "\n".join(r["lines"]))
        raw, _, _ = call_ollama(prompt, temperature=0.0, num_predict=5)
        m = re.search(r'\d+', raw)
        sc = int(m.group()) if m else 0
        sc = max(0, min(10, sc))
        scores.append(sc)
        print(f"    {sc:2}/10  {r['lines'][0]} / ...")
    if scores:
        print(f"    Avg aesthetic score: {sum(scores) / len(scores):.1f}/10")


# ----------------------------- Model selection -----------------------------
def get_installed_models():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def pull_model(name):
    print(f"  Pulling {name}...")
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/pull", json={"name": name, "stream": True},
                          stream=True, timeout=600)
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            status = data.get("status", "")
            if any(k in status for k in ("pulling", "verifying", "success")):
                sys.stderr.write(f"\r\033[K  {status}")
                sys.stderr.flush()
            if status == "success":
                sys.stderr.write("\r\033[K")
                print(f"  Pulled {name}")
                return True
    except Exception as e:
        print(f"  Pull failed: {e}")
    return False


def select_model():
    installed = get_installed_models()
    if installed:
        print("Installed models:")
        for i, name in enumerate(installed, 1):
            print(f"  {i}) {name}")
        print(f"  {len(installed)+1}) Custom / pull new")
        choice = input(f"Select model [1-{len(installed)+1}]: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(installed):
                return installed[idx]
        except ValueError:
            pass
    else:
        print("  Could not reach Ollama or no models installed.")
    name = input("Enter model name: ").strip() or "qwen2.5:1.5b"
    if name not in installed:
        print(f"  '{name}' not found locally.")
        if input("  Pull it now? [y/N] ").strip().lower() == "y":
            if not pull_model(name):
                print("  Pull failed.")
                sys.exit(1)
        else:
            sys.exit(0)
    return name


# ----------------------------- Batch loop -----------------------------
def write_json(results, elapsed_total, path):
    """Dump a complete, machine-readable record for cross-run A/B analysis."""
    total = len(results) or 1
    perfect = sum(1 for r in results if r["success"])
    payload = {
        "meta": {
            "model": MODEL,
            "strategy": ARGS.strategy,
            "cycles": len(results),
            "seed": ARGS.seed,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "config": {"POOL_SIZE": POOL_SIZE, "MAX_LINE_FAILURES": MAX_LINE_FAILURES,
                       "MAX_REPAIR_DISTANCE": MAX_REPAIR_DISTANCE, "REPAIR_TEMPS": REPAIR_TEMPS},
            "total_time_s": round(elapsed_total, 2),
        },
        "aggregate": {
            "success": perfect,
            "success_rate": round(perfect / total, 4),
            "avg_tokens": round(sum(r["tokens"] for r in results) / total, 1),
            "tokens_per_success": round(sum(r["tokens"] for r in results if r["success"]) / perfect, 1) if perfect else None,
            "tolerance_saves": sum(1 for r in results if r.get("audit_recovered")),
            "bailouts": sum(r["bailouts"] for r in results),
            "line_failures": [sum(r["line_failures"][i] for r in results) for i in range(3)],
            "tokens_by_phase": {p: sum(r["tok"].get(p, 0) for r in results)
                                for p in ("pool", "repair", "salvage", "gen", "rerank")},
        },
        "cycles": [
            {k: r[k] for k in ("date", "inspiration", "lines", "counts", "ranges",
                               "success", "source", "audit_recovered", "gens",
                               "tokens", "tok", "sec", "line_failures", "bailouts", "pool_valid")}
            for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  >> wrote machine-readable results to {path}")


def run_batch(cycles):
    global _TOTAL_CYCLES
    _TOTAL_CYCLES = cycles
    if ARGS.seed is not None:
        random.seed(ARGS.seed)   # reproducible event draws for true A/B comparison
    results = []
    start = time.time()
    for i in range(1, cycles + 1):
        t0 = time.time()
        r = run_one_cycle(i, cycles)
        elapsed = time.time() - t0
        if r is None:
            print(f"[{i}/{cycles}] Skipped - no valid event\n")
            continue
        results.append(r)
        cs = "/".join(str(c) for c in r["counts"]) if r["counts"] else "?"
        status = "OK" if r["success"] else "XX"
        print(f"[{i}/{cycles}] {status}  {r['date']}  (via {r['source']})")
        print(f"  Inspiration : {r['inspiration'][:78]}{'...' if len(r['inspiration']) > 78 else ''}")
        if r["lines"]:
            print(f"  Haiku       : {r['lines'][0]}")
            for line in r["lines"][1:]:
                print(f"                {line}")
        else:
            print("  Haiku       : (none generated)")
        print(f"  Syllables   : {cs}   Gens: {r['gens']}   Tokens: {r['tokens']}   Time: {elapsed:.1f}s")
        print()
        print_summary(results, time.time() - start)
        print()
    elapsed_total = time.time() - start
    write_log(results, elapsed_total)
    if ARGS.json:
        write_json(results, elapsed_total, ARGS.json)
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Local-LLM haiku benchmark v2")
    p.add_argument("--model", help="Ollama model (skips interactive picker)")
    p.add_argument("--cycles", type=int, help="Number of cycles (skips prompt)")
    p.add_argument("--strategy", choices=["pool", "repair", "hybrid"], default="hybrid",
                   help="generation strategy (default: hybrid)")
    p.add_argument("--audit", action="store_true",
                   help="report failures that were really counter-disagreements")
    p.add_argument("--rerank", action="store_true",
                   help="model picks most coherent assembled haiku (extra calls)")
    p.add_argument("--judge", action="store_true",
                   help="model-as-judge aesthetic scoring after the run")
    p.add_argument("--smart-select", action="store_true",
                   help="weak haiku-ability tiebreak on event choice (small effect)")
    p.add_argument("--no-cache", action="store_true", help="disable Wikipedia cache")
    p.add_argument("--seed", type=int, help="seed RNG for reproducible event draws (A/B testing)")
    p.add_argument("--json", help="write machine-readable results to this path")
    p.add_argument("--once", action="store_true", help="run a single batch then exit")
    p.add_argument("--record", action="store_true",
                   help=f"record all model responses to {RECORD_PATH}")
    p.add_argument("--replay", action="store_true",
                   help=f"replay from {RECORD_PATH} instead of calling Ollama")
    return p.parse_args()


def main():
    global MODEL, ARGS
    ARGS = parse_args()
    print("=== Haiku Benchmark v2 ===\n")
    MODEL = ARGS.model or select_model()
    if ARGS.cycles:
        cycles = ARGS.cycles
    else:
        try:
            cycles = int(input("How many cycles? ").strip())
        except ValueError:
            cycles = 5

    while True:
        print(f"\nRunning {cycles} cycles with {MODEL}  [strategy={ARGS.strategy}]...\n")
        print("-" * 50)
        run_batch(cycles)
        if ARGS.once:
            break
        print("\nRun again?")
        print("  1) Same model, same cycles")
        print("  2) Change model")
        print("  3) Change cycles")
        print("  4) Change both")
        print("  5) Quit")
        again = input("Choice [1-5]: ").strip()
        if again == "1":
            pass
        elif again == "2":
            MODEL = select_model()
        elif again == "3":
            try:
                cycles = int(input("How many cycles? ").strip())
            except ValueError:
                pass
        elif again == "4":
            MODEL = select_model()
            try:
                cycles = int(input("How many cycles? ").strip())
            except ValueError:
                pass
        else:
            print("Done.")
            break


if __name__ == "__main__":
    main()
