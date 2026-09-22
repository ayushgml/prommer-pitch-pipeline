"""Vercel entrypoint. GET / shows the graph and the last committed run; POST /api/run runs the LangGraph pipeline live.

/api/run needs the RUN_TOKEN header so a public URL can't burn the Gemini key.
Approval resume uses the in-process InMemorySaver, so it only works if the same instance serves /api/approve.
A durable checkpointer (Postgres) is the fix; not built.
"""
import html, json, os, uuid
from pathlib import Path

os.environ.setdefault("OUT_DIR", "/tmp/out")

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from langgraph.types import Command

import graph as g
import pipeline as p

app = FastAPI()
GRAPH = g.build()
RESULTS = Path(__file__).with_name("results")

DIAGRAM = """START -> crawl (code) -> extract (Gemini) -> verify (code) --[< 8 verified, retries left]--> extract
                                                         \\-> draft (Gemini) -> cite_check (code) -> judge (Gemini)
                                                                 ^                                    |
                                                                 +------[unsupported points]-----------+
                                                                                                      \\-> approve [interrupt: Thomas] -> publish -> END"""


def _read(name):
    f = RESULTS / name
    return f.read_text() if f.exists() else "(missing)"


@app.get("/", response_class=HTMLResponse)
def index():
    log_lines = "\n".join(_read("run.log").splitlines())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Prommer Pitch Pipeline</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1b1b1b;--muted:#666;--card:#f1efe9;--accent:#1f5f8b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141414;--fg:#e8e6e1;--muted:#9a9a9a;--card:#1f1f1f;--accent:#7fb3d9}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif;max-width:960px;margin:0 auto;padding:24px 16px}}
pre{{background:var(--card);padding:12px;border-radius:8px;overflow-x:auto;font-size:12.5px;white-space:pre}}
.md{{white-space:pre-wrap}} a{{color:var(--accent)}} .muted{{color:var(--muted)}}
</style></head><body>
<h1>Prommer pitch pipeline</h1>
<p>A LangGraph workflow that turns Thomas Prommer's own published positions on prommer.net into podcast and press pitch angles.
Every talking point cites a verbatim quote that code has checked against the live page, and a judge model checks that the point follows from its quote.
Nothing is sent: the graph stops at a human-approval interrupt.</p>
<p class="muted">Model: {p.MODEL or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")} ·
<a href="https://github.com/ayushgml/prommer-pitch-pipeline">source</a> ·
live run: <code>POST /api/run</code> (token-gated)</p>
<h2>Graph</h2><pre>{html.escape(DIAGRAM)}</pre>
<h2>Last run: trace</h2><pre>{html.escape(log_lines)}</pre>
<h2>Last run: pitch pack sent for approval</h2><pre class="md">{html.escape(_read("pitch_pack.md"))}</pre>
<h2>Last run: approved and published</h2><pre class="md">{html.escape(_read("approved_pitches.md"))}</pre>
</body></html>"""


def _check(token):
    if not os.environ.get("RUN_TOKEN") or token != os.environ["RUN_TOKEN"]:
        raise HTTPException(401, "missing or bad x-run-token")


@app.post("/api/run")
def run(x_run_token: str = Header(default="")):
    _check(x_run_token)
    Path(os.environ["OUT_DIR"]).mkdir(parents=True, exist_ok=True)
    (p.OUT / "run.log").write_text("")
    thread_id = str(uuid.uuid4())
    result = GRAPH.invoke({}, {"configurable": {"thread_id": thread_id}})
    trace = [json.loads(l) for l in (p.OUT / "run.log").read_text().splitlines() if l]
    if "__interrupt__" not in result:
        return {"thread_id": thread_id, "status": "ended_without_pitches", "trace": trace}
    payload = result["__interrupt__"][0].value
    return {"thread_id": thread_id, "status": "PENDING_APPROVAL", "angles": payload["angles"],
            "pitch_pack": payload["pitch_pack"], "trace": trace}


@app.post("/api/approve/{thread_id}")
def approve(thread_id: str, body: dict, x_run_token: str = Header(default="")):
    _check(x_run_token)
    cfg = {"configurable": {"thread_id": thread_id}}
    if not GRAPH.get_state(cfg).next:
        raise HTTPException(409, "no pending interrupt for this thread on this instance (in-memory checkpointer)")
    result = GRAPH.invoke(Command(resume={"approve": body.get("approve", [])}), cfg)
    path = result.get("published")
    return {"published": bool(path), "content": Path(path).read_text() if path else ""}
