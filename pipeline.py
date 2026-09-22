"""prommer.net pitch pipeline.

crawl (code) -> extract claims (LLM) -> verify quotes (code) -> draft pitches (LLM) -> check citations (code) -> write for approval

The LLM never gets to state an opinion for Thomas that isn't a verbatim quote from his own site.
Nothing is sent anywhere: output is a markdown file with status PENDING_APPROVAL.
"""
import json, os, re, subprocess, sys, time
from html.parser import HTMLParser
from pathlib import Path

BASE = "https://prommer.net"
PAGES = [
    "/en/tech/profile/",
    "/en/tech/guides/context-engineering/",
    "/en/tech/guides/ai-strategy/",
    "/en/tech/guides/enterprise-ai-strategy/",
    "/en/tech/press/",
]
OUT = Path(os.environ.get("OUT_DIR", "out"))  # /tmp on Vercel (read-only fs)
MAX_CHARS_PER_PAGE = 12000


def log(step, **kw):
    line = json.dumps({"t": round(time.time(), 1), "step": step, **kw})
    print(line, file=sys.stderr)
    with open(OUT / "run.log", "a") as f:
        f.write(line + "\n")


# ---------- step 1: crawl (deterministic) ----------
class TextExtractor(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "svg", "noscript"}

    def __init__(self):
        super().__init__()
        self.depth, self.chunks = 0, []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if not self.depth and data.strip():
            self.chunks.append(data.strip())


def normalize(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


MIN_CHARS = 1000
REDIRECT_RE = re.compile(r"<title>Redirecting to: (/[^<]+)</title>")


def fetch(path, hops=3):
    """curl, not urllib: python.org builds on macOS ship without CA certs (CERTIFICATE_VERIFY_FAILED).
    The site answers moved pages with HTTP 200 + an HTML "Redirecting to:" stub, so -L alone doesn't follow them."""
    for _ in range(hops):
        r = subprocess.run(["curl", "-sfL", "--max-time", "15", "-A", "prommer-pitch-pipeline/0.1", BASE + path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"curl exit {r.returncode}")
        m = REDIRECT_RE.search(r.stdout[:2000])
        if not m:
            return path, r.stdout
        log("crawl", url=path, redirect_to=m.group(1))
        path = m.group(1)
    raise RuntimeError("too many html redirects")


def crawl():
    pages = {}
    for path in PAGES:
        try:
            final, html = fetch(path)
        except RuntimeError as e:
            log("crawl", url=path, error=str(e))
            continue
        p = TextExtractor()
        p.feed(html)
        text = " ".join(p.chunks)
        if len(text) < MIN_CHARS:  # a stub page would silently shrink the evidence base
            log("crawl", url=final, error=f"thin page ({len(text)} chars), skipped")
            continue
        pages[BASE + final] = text
        log("crawl", url=final, chars=len(text))
    return pages


# ---------- LLM call (Gemini, JSON mode) ----------
def _load_env():
    env = Path(__file__).with_name(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() and v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_client = None
MODEL = None


def _gemini():
    global _client, MODEL
    if _client is None:
        from google import genai
        _load_env()
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    return _client


def llm_json(prompt, stdin_text, retries=1):
    """System prompt + data as contents, JSON mime type. On parse failure, retry once with the error fed back."""
    from google.genai import types
    client = _gemini()
    for attempt in range(retries + 1):
        t0 = time.time()
        resp = client.models.generate_content(
            model=MODEL,
            contents=stdin_text,
            config=types.GenerateContentConfig(
                system_instruction=prompt, response_mime_type="application/json", temperature=0.2),
        )
        raw = resp.text or ""
        u = resp.usage_metadata
        try:
            data = json.loads(raw)
            log("llm", model=MODEL, attempt=attempt, secs=round(time.time() - t0, 1), ok=True,
                tokens_in=u.prompt_token_count, tokens_out=u.candidates_token_count)
            return data
        except json.JSONDecodeError as e:
            log("llm", model=MODEL, attempt=attempt, ok=False, error=str(e), head=raw[:200])
            prompt += f"\n\nYour previous reply was not valid JSON ({e}). Reply with ONLY the JSON."
    raise RuntimeError("LLM did not return valid JSON")


# ---------- step 2: extract claims (LLM) ----------
EXTRACT_PROMPT = """You are extracting Thomas Prommer's stated positions from his own website pages (given on stdin, each prefixed with URL:).
Return ONLY a JSON array. Each item: {"url": <page url>, "quote": <an EXACT verbatim substring of that page, 10-40 words>, "position": <one-line paraphrase of the opinion the quote expresses>}.
Only include quotes that express an opinion or claim Thomas makes (not navigation, not bios written in third person unless they state a fact about his work).
Max 6 per page. Do not fix typos or punctuation in quotes."""


def extract_claims(pages):
    corpus = "\n\n".join(f"URL: {u}\n{t[:MAX_CHARS_PER_PAGE]}" for u, t in pages.items())
    claims = llm_json(EXTRACT_PROMPT, corpus)
    for i, c in enumerate(claims):
        c["id"] = f"C{i + 1}"
    log("extract", claims=len(claims))
    return claims


# ---------- step 3: verify quotes (deterministic) ----------
def verify_claims(claims, pages):
    norm_pages = {u: normalize(t) for u, t in pages.items()}
    kept, dropped = [], []
    for c in claims:
        page = norm_pages.get(c.get("url"))
        if page and normalize(c.get("quote", "")) in page:
            kept.append(c)
        else:
            dropped.append(c)
    log("verify", kept=len(kept), dropped=[{"id": d["id"], "quote": d.get("quote", "")[:80]} for d in dropped])
    return kept


# ---------- step 4: draft pitches (LLM, verified claims only) ----------
DRAFT_PROMPT = """You draft podcast / press pitch angles for Thomas Prommer. The ONLY material you may use is the verified claims list on stdin (JSON).
Return ONLY a JSON array of 3 pitches: {"angle": <episode/article title>, "audience": <which kind of show or outlet and why>, "talking_points": [{"text": <one sentence>, "claim_ids": [<ids from the list that support this sentence>]}]}.
Every talking point must cite at least one claim id. Do not attribute any opinion to Thomas that is not in a cited claim. No invented stats, clients, or anecdotes."""


def draft_pitches(claims):
    slim = [{"id": c["id"], "position": c["position"], "quote": c["quote"]} for c in claims]
    pitches = llm_json(DRAFT_PROMPT, json.dumps(slim, indent=1))
    log("draft", pitches=len(pitches))
    return pitches


# ---------- step 5: citation check (deterministic) ----------
def check_citations(pitches, claims):
    valid = {c["id"] for c in claims}
    out = []
    for p in pitches:
        good = [tp for tp in p.get("talking_points", [])
                if tp.get("claim_ids") and set(tp["claim_ids"]) <= valid]
        bad = len(p.get("talking_points", [])) - len(good)
        log("check", angle=p.get("angle"), kept_points=len(good), dropped_points=bad)
        if len(good) >= 2:  # a pitch with <2 grounded points is not worth Thomas's review time
            out.append({**p, "talking_points": good})
    return out


# ---------- step 6: write for human approval ----------
def render(pitches, claims):
    by_id = {c["id"]: c for c in claims}
    lines = ["# Pitch pack for Thomas Prommer", "", "**status: PENDING_APPROVAL** - nothing below has been sent.", ""]
    for p in pitches:
        lines += [f"## {p['angle']}", f"_Audience:_ {p['audience']}", ""]
        for tp in p["talking_points"]:
            lines.append(f"- {tp['text']} [{', '.join(tp['claim_ids'])}]")
        lines.append("")
    lines += ["## Sources (verbatim, verified against the live page)", ""]
    for cid in sorted({i for p in pitches for tp in p["talking_points"] for i in tp["claim_ids"]},
                      key=lambda x: int(x[1:])):
        c = by_id[cid]
        lines.append(f"- **{cid}** \"{c['quote']}\" - {c['url']}")
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "run.log").write_text("")
    pages = crawl()
    if not pages:
        sys.exit("crawl returned nothing")
    claims = extract_claims(pages)
    (OUT / "claims_raw.json").write_text(json.dumps(claims, indent=1))
    claims = verify_claims(claims, pages)
    (OUT / "claims_verified.json").write_text(json.dumps(claims, indent=1))
    if len(claims) < 3:
        sys.exit("too few verified claims to draft from")
    pitches = check_citations(draft_pitches(claims), claims)
    (OUT / "pitches.json").write_text(json.dumps(pitches, indent=1))
    (OUT / "pitch_pack.md").write_text(render(pitches, claims))
    log("done", pitches=len(pitches))


if __name__ == "__main__":
    main()
