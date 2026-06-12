#!/usr/bin/env python3
"""
haiku_bot.py — daily haiku generator.

Fetches an "on this day in history" event from Wikipedia,
generates a valid 5/7/5 haiku using gemma2:2b via Ollama,
and prints it with brief stats.
"""

import argparse
import datetime
import itertools
import json
import os
import re
import subprocess
import sys
import time

import requests
import syllables

from printer import ReceiptPrinter, SLOW_PRINT_SETTINGS, fit_haiku, wrap_text

try:
    import pronouncing
    HAS_CMU = True
except ImportError:
    HAS_CMU = False

# ----------------------------- Config -----------------------------
OLLAMA_BASE   = "http://localhost:11434"
OLLAMA_URL    = f"{OLLAMA_BASE}/api/generate"
DEFAULT_MODEL = "gemma2:2b"
VERSION       = "1.0"

VERBOSE = True  # set to False by --quiet flag

def status(msg):
    if VERBOSE:
        print(f"  {msg}", flush=True)
TARGET        = [5, 7, 5]
CACHE_PATH    = os.path.expanduser("~/.haiku_wiki_cache.json")

POOL_SIZE          = 6
MAX_LINE_FAILURES  = [3, 5, 3]
MAX_REPAIR_DISTANCE = 3
REPAIR_TEMPS       = [0.3, 0.15, 0.5, 0.2, 0.6, 0.35, 0.55]

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

# ----------------------------- Syllables -----------------------------
def syl_range(word):
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
    return (cmu, cmu)


def line_range(line):
    rs = [syl_range(w) for w in re.findall(r"[a-zA-Z']+", line)]
    return (sum(a for a, _ in rs), sum(b for _, b in rs))


def line_strict(line):
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


def verify_haiku(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) != 3:
        return False, lines, []
    counts = [line_strict(l) for l in lines]
    if not all(line_hits(lines[i], TARGET[i]) for i in range(3)):
        return False, lines, counts
    if has_word_reuse(lines):
        return False, lines, counts
    return True, lines, counts


# ----------------------------- Wikipedia -----------------------------
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


def get_history_events(date, no_cache=False):
    key = f"events:{date.strftime('%m-%d')}"
    if not no_cache:
        hit = _cache().get(key)
        if hit is not None:
            return hit
    url = (f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/"
           f"{date.strftime('%m')}/{date.strftime('%d')}")
    r = requests.get(url, headers={"User-Agent": "haiku-bot/2.0"}, timeout=15)
    r.raise_for_status()
    events = r.json().get("events", [])
    if not no_cache:
        _cache().set(key, events)
    return events


def fetch_article_summary(event, no_cache=False):
    pages = event.get("pages", [])
    if not pages:
        return None
    title = pages[0].get("title", "")
    if not title:
        return None
    key = f"summary:{title}"
    if not no_cache:
        hit = _cache().get(key)
        if hit is not None:
            return hit
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
        r = requests.get(url, headers={"User-Agent": "haiku-bot/2.0"}, timeout=10)
        r.raise_for_status()
        extract = r.json().get("extract", "")
        sentences = re.split(r'(?<=[.!?])\s+', extract.strip())
        s = " ".join(sentences[:3])
    except Exception:
        s = None
    if s and not no_cache:
        _cache().set(key, s)
    return s


def pick_event(events, current_year):
    def score(text):
        t = text.lower()
        s = 0
        if any(k in t for k in ["battle", "war", "invasion", "siege", "army", "military"]):
            s -= 15
        if any(k in t for k in ["die", "death", "died", "killed", "kills", "assassinated", "executed", "massacre"]):
            s -= 15
        if any(k in t for k in ["pope", "saint", "church", "cathedral", "vatican", "crusade"]):
            s -= 15
        if any(k in t for k in ["elected", "parliament", "senate", "congress", "legislation"]):
            s -= 7
        if any(k in t for k in ["space", "launch", "comet", "eclipse", "astronaut", "orbit", "planet"]):
            s += 5
        if any(k in t for k in ["discovered", "expedition", "explored", "voyage", "island", "cave"]):
            s += 5
        if any(k in t for k in ["storm", "earthquake", "volcano", "eruption", "hurricane", "meteor"]):
            s += 4
        if any(k in t for k in ["invented", "patent", "first", "demonstration", "prototype"]):
            s += 4
        if any(k in t for k in ["film", "album", "music", "art", "book", "premiere", "published"]):
            s += 5
        if any(k in t for k in ["video game", "arcade", "nintendo", "atari", "pac-man", "tetris"]):
            s += 7
        if any(k in t for k in ["animal", "dinosaur", "fossil", "creature"]):
            s += 8
        if any(k in t for k in ["record", "unusual", "strange", "remarkable", "bizarre"]):
            s += 8
        return s

    def age_score(year_str):
        try:
            year = int(year_str)
        except (TypeError, ValueError):
            return 0
        age = current_year - year
        if age < 0:
            return 0
        penalty = min(15, age / 10.0)
        if year < 1994:
            penalty += 5
        penalty = min(penalty, 15)
        bonus = 5 if age <= 25 else 0
        return -penalty + bonus

    if not events:
        return None

    scored = []
    for e in events:
        text, year = e.get("text", ""), e.get("year")
        kw, age = score(text), age_score(year)
        total = kw + age
        status(f"  [{year}] score={kw:+.1f} age={age:+.1f} total={total:+.1f}  {text[:70]}")
        scored.append((total, e))
    scored.sort(key=lambda x: x[0], reverse=True)

    candidates = [e for _, e in scored if len(e.get("text", "")) < 200]
    candidates = candidates or [e for _, e in scored]

    top, top_score = candidates[0], scored[0][0]
    status(f"Top candidate: [{top.get('year')}] total={top_score:+.1f}  {top.get('text', '')[:70]}")
    status(f"Selected event: {top.get('year')} — {top.get('text', '')}")
    return top


# ----------------------------- Ollama -----------------------------
def call_ollama(model, prompt, temperature=0.7, num_predict=80, label=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": num_predict, "num_thread": 4},
    }
    if VERBOSE and label:
        status(f"{label} (streaming model output):")
    chunks = []
    tokens = 0
    with requests.post(OLLAMA_URL, json=payload, timeout=300, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            piece = data.get("response", "")
            chunks.append(piece)
            if VERBOSE:
                print(piece, end="", flush=True)
            if data.get("done"):
                tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
    if VERBOSE:
        print(flush=True)
    return "".join(chunks).strip(), tokens


# ----------------------------- Parsing -----------------------------
def clean_line(line):
    line = re.sub(r'`[^`]+`|```\w*', '', line).strip()
    line = re.split(r'\s+becomes\s+', line, flags=re.IGNORECASE)[0].strip()
    line = re.sub(r'<[^>]+>', '', line).strip()
    line = re.sub(r'\[[^\]]*\]', '', line).strip()
    line = re.sub(r'^\s*[\d#\-\*\.\)]+\s*', '', line)
    line = line.strip('\'"""—–-')
    return line.strip()


def parse_pool(raw):
    out = []
    for ln in raw.splitlines():
        c = clean_line(ln)
        if c and len(c.split()) >= 1 and not c.lower().startswith(("event:", "haiku", "example")):
            out.append(c)
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


# ----------------------------- Prompts -----------------------------
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


def pool_prompt(position, event_text, summary, used_words):
    target = TARGET[position]
    ctx = event_text if not summary else f"{event_text} ({summary})"
    avoid = f"Do NOT use these words: {', '.join(sorted(used_words))}.\n" if used_words else ""
    return f"""Write {POOL_SIZE} different short poetic lines about this event, for a haiku.
Event: {ctx}

Each line MUST be exactly {target} syllables. Count by sounding out words:
"en-gine"=2, "bright"=1, "morn-ing"=2, "launch"=1, "dis-tant"=2.
Use vivid imagery and simple common words. Avoid long proper nouns.
{avoid}Output ONLY the {POOL_SIZE} lines, one per line, no numbering, no extra text."""


def repair_prompt(line, current, target, context_lines):
    direction = "shorter" if current > target else "longer"
    diff = abs(current - target)
    sw = "syllable" if diff == 1 else "syllables"
    if direction == "shorter":
        ex = f'Example: "The old si-lent pond" (5) -> "Old si-lent pond" (4). Drop small words.'
    else:
        ex = f'Example: "Old si-lent pond" (4) -> "The old si-lent pond" (5). Add a small word.'
    return f"""You are an editor fixing ONE line of a haiku.
Original line ({current} syllables, need {target}): "{line}"

The other lines are:
{context_lines}

Rewrite so it has EXACTLY {target} syllables ({diff} {sw} {direction}).
Count by sounding out: "en-gine"=2, "bright"=1, "morn-ing"=2, "launch"=1.
{ex}
Use simple common words. Do not reuse any word already in the other lines.
Output ONLY the single rewritten line. No explanations, no quotes, no markdown."""


# ----------------------------- Generation -----------------------------
def assemble(pools):
    if 0 in [len(p) for p in pools]:
        return None
    for combo in itertools.product(*[p[:8] for p in pools]):
        if not has_word_reuse(combo):
            return list(combo)
    return None


POSITION_LABEL = ["line 1 (5)", "line 2 (7)", "line 3 (5)"]

def try_pool(model, event_text, summary):
    pools, used, tokens = [], set(), 0
    for pos in range(3):
        status(f"  Generating candidates for {POSITION_LABEL[pos]}...")
        raw, tok = call_ollama(model, pool_prompt(pos, event_text, summary, used), temperature=0.7, num_predict=120,
                                label=f"  Candidates for {POSITION_LABEL[pos]}")
        tokens += tok
        target = TARGET[pos]
        all_cands = parse_pool(raw)
        cands = []
        for c in all_cands:
            if line_hits(c, target):
                cands.append(c)
                status(f"    OK   ({line_strict(c)}/{target} syl): {c}")
            else:
                status(f"    REJECT ({line_strict(c)}/{target} syl): {c}")
        if not cands:
            status(f"    No valid candidates for {POSITION_LABEL[pos]}")
        if cands:
            used |= content_words(cands[0])
        pools.append(cands)
    assembled = assemble(pools)
    if assembled is None:
        status("  Could not assemble a non-repeating combo from candidate pools.")
    return assembled, tokens


def try_repair(model, lines, counts):
    tokens = 0
    for i in range(3):
        if line_hits(lines[i], TARGET[i]):
            continue
        status(f"  Repairing {POSITION_LABEL[i]}: \"{lines[i]}\" ({counts[i]}/{TARGET[i]} syl)")
        for rep in range(1, MAX_LINE_FAILURES[i] + 1):
            if abs(counts[i] - TARGET[i]) > MAX_REPAIR_DISTANCE:
                status(f"    Giving up: off by {abs(counts[i] - TARGET[i])} syllables (max {MAX_REPAIR_DISTANCE})")
                return lines, counts, False, tokens
            temp = REPAIR_TEMPS[rep % len(REPAIR_TEMPS)]
            ctx = "\n".join(l for j, l in enumerate(lines) if j != i)
            new, tok = call_ollama(model, repair_prompt(lines[i], counts[i], TARGET[i], ctx), temperature=temp, num_predict=60,
                                    label=f"  Repair attempt {rep} for {POSITION_LABEL[i]}")
            tokens += tok
            new = clean_line(new.splitlines()[0] if new.splitlines() else new)
            nc = line_strict(new)
            cand = [l if j != i else new for j, l in enumerate(lines)]
            reuse = has_word_reuse(cand)
            if line_hits(new, TARGET[i]) and not reuse:
                status(f"    Attempt {rep}: \"{new}\" ({nc}/{TARGET[i]} syl) -> ACCEPTED")
                lines[i], counts[i] = new, nc
                break
            if abs(nc - TARGET[i]) < abs(counts[i] - TARGET[i]) and not reuse:
                status(f"    Attempt {rep}: \"{new}\" ({nc}/{TARGET[i]} syl) -> closer, kept as new baseline")
                lines[i], counts[i] = new, nc
            else:
                reason = "word reuse" if reuse else "no improvement"
                status(f"    Attempt {rep}: \"{new}\" ({nc}/{TARGET[i]} syl) -> rejected ({reason})")
        else:
            status(f"    Exhausted {MAX_LINE_FAILURES[i]} attempts for {POSITION_LABEL[i]}")
            return lines, counts, False, tokens
    ok = all(line_hits(lines[i], TARGET[i]) for i in range(3)) and not has_word_reuse(lines)
    return lines, counts, ok, tokens


def explain_failure(lines, counts):
    """Describe why a candidate haiku failed verify_haiku, for verbose logging."""
    if len(lines) != 3:
        return f"expected 3 lines, got {len(lines)}"
    reasons = []
    for i in range(3):
        if not line_hits(lines[i], TARGET[i]):
            reasons.append(f"{POSITION_LABEL[i]} has {counts[i]} syllables (want {TARGET[i]}): \"{lines[i]}\"")
    if has_word_reuse(lines):
        reasons.append("a content word is repeated across lines")
    return "; ".join(reasons) if reasons else "unknown"


def generate(model, event_text, summary, strategy="repair"):
    total_tokens = 0
    status(f"Generating haiku with {model} (strategy={strategy})...")
    status(f"  Event: {event_text}")
    if summary:
        status(f"  Summary: {summary}")

    if strategy in ("pool", "hybrid"):
        assembled, tok = try_pool(model, event_text, summary)
        total_tokens += tok
        if assembled:
            ok, lines, counts = verify_haiku("\n".join(assembled))
            if ok:
                return lines, counts, total_tokens, "pool"
            status(f"  Assembled candidate failed verification: {explain_failure(lines, counts)}")
        if strategy == "pool":
            status("Could not produce a valid haiku.")
            return [], [], total_tokens, "failed"

    # repair / hybrid fallback: generate then fix
    raw, tok = call_ollama(model, gen_prompt(event_text, summary), temperature=0.7, num_predict=100,
                            label="  Direct generation")
    total_tokens += tok
    ok, lines, counts = verify_haiku(extract_haiku(raw))
    if ok:
        return lines, counts, total_tokens, "direct"
    status(f"  Direct generation failed verification: {explain_failure(lines, counts)}")
    if len(lines) == 3:
        status("  Repairing...")
        lines, counts, ok, tok = try_repair(model, lines, counts)
        total_tokens += tok
        if ok:
            return lines, counts, total_tokens, "repair"
        status(f"  Repair failed: {explain_failure(lines, counts)}")

    status("Could not produce a valid haiku.")
    return lines, counts, total_tokens, "failed"


# ----------------------------- Model management -----------------------------
def installed_models():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def pull_model(name):
    print(f"  Pulling {name}...")
    r = requests.post(f"{OLLAMA_BASE}/api/pull", json={"name": name, "stream": True},
                      stream=True, timeout=600)
    r.raise_for_status()
    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        status = data.get("status", "")
        total = data.get("total", 0)
        completed = data.get("completed", 0)
        if total:
            pct = int(100 * completed / total)
            sys.stderr.write(f"\r  {status} {pct}%  ")
        else:
            sys.stderr.write(f"\r  {status}  ")
        sys.stderr.flush()
        if status == "success":
            sys.stderr.write("\r" + " " * 40 + "\r")
            print(f"  Pulled {name}.")
            return
    raise RuntimeError(f"Pull ended without success status")


def ensure_model(name):
    models = installed_models()
    if models is None:
        print("Could not reach Ollama. Is it running?")
        sys.exit(1)
    # match exact name or name without tag against installed list
    def matches(installed):
        return installed == name or installed.split(":")[0] == name.split(":")[0]
    if any(matches(m) for m in models):
        return
    print(f"  Model '{name}' is not installed.")
    try:
        answer = input(f"  Pull it now? [Y/n] ").strip().lower()
    except EOFError:
        answer = "y"
    if answer in ("", "y", "yes"):
        pull_model(name)
    else:
        print(f"  To pull manually: ollama pull {name}")
        sys.exit(1)


# ----------------------------- Self-update -----------------------------
def self_update():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, "pull", "--ff-only"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        status(f"Self-update skipped: {e}")
        return
    if result.returncode != 0:
        status(f"Self-update failed: {result.stderr.strip()}")
        return
    if "Already up to date" not in result.stdout:
        status("Updated to the latest version, restarting...")
        os.execv(sys.executable, [sys.executable] + sys.argv)


# ----------------------------- Printing -----------------------------
def print_receipt(date_str, year, event_text, lines):
    header_lines = ["HAIKU BOT", f"{date_str}, {year}"]
    header_lines += wrap_text(event_text)
    header = "\n".join(header_lines) + "\n\n"

    mult, haiku_lines = fit_haiku(lines)
    haiku_text = "\n".join(haiku_lines) + "\n"

    try:
        with ReceiptPrinter() as printer:
            printer.init()
            printer.set_print_speed(*SLOW_PRINT_SETTINGS)
            # Warm up the print head with a blank line first; the first
            # line printed right after connecting tends to come out faint.
            printer.print_text("\n")
            printer.justify("left")
            printer.set_size(1)
            printer.print_text(header)
            printer.justify("center")
            printer.set_size(mult)
            printer.print_text(haiku_text)
            printer.set_size(1)
            printer.justify("left")
            printer.feed()
            printer.cut()
    except OSError as e:
        status(f"Failed to print receipt: {e}")


# ----------------------------- Main -----------------------------
def main():
    global VERBOSE

    p = argparse.ArgumentParser(description="Daily haiku bot")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    p.add_argument("--date", help="Date to draw events from (MM-DD), default: today")
    p.add_argument("--no-cache", action="store_true", help="Disable Wikipedia cache")
    p.add_argument("--quiet", action="store_true", help="Skip status messages, output haiku only")
    p.add_argument("--strategy", choices=["repair", "pool", "hybrid"], default="repair",
                   help="Generation strategy (default: repair)")
    p.add_argument("--no-print", action="store_true", help="Skip printing to the receipt printer")
    p.add_argument("--no-update", action="store_true", help="Skip self-update via git pull")
    args = p.parse_args()

    VERBOSE = not args.quiet

    if not args.no_update:
        self_update()

    if VERBOSE:
        print(f"\n  {'=' * 30}")
        print(f"  Haiku Bot v{VERSION}")
        print(f"  {'=' * 30}\n")
        print(f"  Model : {args.model}")

    ensure_model(args.model)

    if args.date:
        try:
            today = datetime.datetime.strptime(f"2000-{args.date}", "%Y-%m-%d").date()
        except ValueError:
            print(f"Invalid date format '{args.date}', expected MM-DD")
            sys.exit(1)
    else:
        today = datetime.date.today()

    status(f"Fetching Wikipedia events for {today.strftime('%B %-d')}...")
    try:
        events = get_history_events(today, no_cache=args.no_cache)
    except Exception as e:
        print(f"Failed to fetch Wikipedia events: {e}")
        sys.exit(1)

    event = pick_event(events, datetime.date.today().year)
    if not event:
        print("No suitable event found for today.")
        sys.exit(1)

    event_text = event.get("text", "")
    year = event.get("year", "")
    status(f"Event: {year} — {event_text}")

    summary = fetch_article_summary(event, no_cache=args.no_cache)

    if VERBOSE:
        print()

    t0 = time.time()
    lines, counts, tokens, path = generate(args.model, event_text, summary, args.strategy)
    elapsed = time.time() - t0

    date_str = today.strftime("%B %-d")
    cs = " / ".join(str(c) for c in counts) if counts else "? / ? / ?"

    print()
    print(f"  {date_str}, {year} — {event_text}")
    print()
    if lines:
        for line in lines:
            print(f"    {line}")
    else:
        print("    (no haiku generated)")
    print()
    print(f"  {cs}  ·  {tokens} tokens  ·  {elapsed:.1f}s")
    print()

    if path == "failed":
        sys.exit(1)

    if not args.no_print:
        status("Printing receipt...")
        print_receipt(date_str, year, event_text, lines)


if __name__ == "__main__":
    main()
