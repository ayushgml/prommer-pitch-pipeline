"""LangGraph version of the prommer.net pitch pipeline.

START -> crawl -> extract -> verify --(too few verified, retries left)--> extract (with rejected quotes as feedback)
                                   \\-> draft -> cite_check -> judge --(unsupported points, retries left)--> draft (with judge feedback)
                                                                     \\-> approve [interrupt: Thomas] -> publish -> END

LLM nodes: extract, draft, judge. Everything else is code.
The graph exists for the two loops and the human interrupt; the step order itself is fixed.
"""
import json, sys
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

import pipeline as p

MIN_VERIFIED = 8
MAX_EXTRACT_TRIES = 2
MAX_DRAFT_TRIES = 2


class State(TypedDict, total=False):
    pages: dict
    claims: list          # verified only
    rejected: list        # quotes that failed verification, fed back to extract
    extract_tries: int
    pitches: list
    judge_feedback: list  # unsupported talking points, fed back to draft
    draft_tries: int
    decision: dict
    published: str


# ---------- nodes ----------
def crawl(s: State):
    pages = p.crawl()
    if not pages:
        raise RuntimeError("crawl returned nothing")
    return {"pages": pages, "extract_tries": 0, "draft_tries": 0, "rejected": [], "judge_feedback": []}


def extract(s: State):
    prompt = p.EXTRACT_PROMPT
    if s["rejected"]:
        prompt += ("\n\nThese quotes from your last attempt were NOT found verbatim on the page. "
                   "Do not paraphrase; copy exact text:\n" + "\n".join(f"- {q}" for q in s["rejected"]))
    corpus = "\n\n".join(f"URL: {u}\n{t[:p.MAX_CHARS_PER_PAGE]}" for u, t in s["pages"].items())
    claims = p.llm_json(prompt, corpus)
    for i, c in enumerate(claims):
        c["id"] = f"C{i + 1}"
    p.log("extract", attempt=s["extract_tries"], claims=len(claims))
    return {"claims": claims, "extract_tries": s["extract_tries"] + 1}


def verify(s: State):
    kept = p.verify_claims(s["claims"], s["pages"])
    kept_ids = {c["id"] for c in kept}
    rejected = [c.get("quote", "") for c in s["claims"] if c["id"] not in kept_ids]
    return {"claims": kept, "rejected": rejected}


REPAIR_PROMPT = """A fact-checker rejected these talking points (JSON on stdin) because they say more than their quotes.
Rewrite each one so it is fully supported: reuse the quote's own wording, keep its hedges ("rarely", "most", "in my experience"), add nothing.
If it can't be fixed, set text to null. Return ONLY a JSON array: [{"key": <key>, "text": <string or null>}]."""


def draft(s: State):
    if s["judge_feedback"]:
        # Repair mode: accepted points stay frozen; only rejected ones are rewritten.
        # (First version regenerated every pitch, which threw away accepted points and made the judge reject rate go UP.)
        by_id = {c["id"]: c["quote"] for c in s["claims"]}
        items = [{"key": b["key"], "text": b["text"], "reason": b["reason"],
                  "quotes": [by_id[i] for i in b["claim_ids"]]} for b in s["judge_feedback"]]
        fixes = {f["key"]: f.get("text") for f in p.llm_json(REPAIR_PROMPT, json.dumps(items, indent=1))}
        pitches = [dict(pt, talking_points=list(pt["talking_points"])) for pt in s["pitches"]]
        repaired = 0
        for b in s["judge_feedback"]:
            if fixes.get(b["key"]):
                pitches[b["pitch"]]["talking_points"].append(
                    {"text": fixes[b["key"]], "claim_ids": b["claim_ids"]})
                repaired += 1
        p.log("draft", attempt=s["draft_tries"], mode="repair", rejected=len(items), repaired=repaired)
        return {"pitches": pitches, "draft_tries": s["draft_tries"] + 1}
    slim = [{"id": c["id"], "position": c["position"], "quote": c["quote"]} for c in s["claims"]]
    pitches = p.llm_json(p.DRAFT_PROMPT, json.dumps(slim, indent=1))
    p.log("draft", attempt=s["draft_tries"], pitches=len(pitches))
    return {"pitches": pitches, "draft_tries": s["draft_tries"] + 1}


def cite_check(s: State):
    valid = {c["id"] for c in s["claims"]}
    pitches = [dict(pt, talking_points=[tp for tp in pt.get("talking_points", [])
                                         if tp.get("claim_ids") and set(tp["claim_ids"]) <= valid])
               for pt in s["pitches"]]
    return {"pitches": pitches}


JUDGE_PROMPT = """You are a strict fact-checker. stdin has JSON items: {"key", "text", "quotes"}.
For each item decide if "text" is fully supported by "quotes" (no added numbers, no stronger wording, no new claims, no opinion the quotes don't state).
Return ONLY a JSON array: [{"key": <key>, "supported": true|false, "reason": <short, only if false>}]."""


def judge(s: State):
    by_id = {c["id"]: c["quote"] for c in s["claims"]}
    todo = [(pi, ti, tp) for pi, pt in enumerate(s["pitches"]) for ti, tp in enumerate(pt["talking_points"])
            if not tp.get("ok")]
    items = [{"key": f"{pi}.{ti}", "text": tp["text"], "quotes": [by_id[i] for i in tp["claim_ids"]]}
             for pi, ti, tp in todo]
    verdicts = {v["key"]: v for v in p.llm_json(JUDGE_PROMPT, json.dumps(items, indent=1))} if items else {}
    bad, pitches = [], []
    for pi, pt in enumerate(s["pitches"]):
        good = []
        for ti, tp in enumerate(pt["talking_points"]):
            if tp.get("ok"):
                good.append(tp)
                continue
            v = verdicts.get(f"{pi}.{ti}", {"supported": False, "reason": "judge returned no verdict"})
            if v["supported"]:
                good.append({**tp, "ok": True})
            else:
                bad.append({**tp, "pitch": pi, "key": f"{pi}.{ti}", "reason": v.get("reason")})
        pitches.append({**pt, "talking_points": good})
    p.log("judge", checked=len(items), unsupported=len(bad), reasons=[b["reason"] for b in bad])
    return {"pitches": pitches, "judge_feedback": bad}


def approve(s: State):
    s = {**s, "pitches": [pt for pt in s["pitches"] if len(pt["talking_points"]) >= 2]}
    pack = p.render(s["pitches"], s["claims"])
    (p.OUT / "pitch_pack.md").write_text(pack)
    # Pauses the graph; state is checkpointed. Resume with Command(resume={"approve": [pitch indices]}).
    decision = interrupt({"pitch_pack": pack, "angles": [x["angle"] for x in s["pitches"]]})
    p.log("approve", decision=decision)
    return {"decision": decision, "pitches": s["pitches"]}  # publish indexes the same filtered list Thomas saw


def publish(s: State):
    # "Publish" = write the approved subset. Idempotent: same approval -> same file.
    chosen = [s["pitches"][i] for i in s["decision"].get("approve", []) if i < len(s["pitches"])]
    if not chosen:
        p.log("publish", published=0)
        return {"published": ""}
    path = p.OUT / "approved_pitches.md"
    path.write_text(p.render(chosen, s["claims"]).replace("PENDING_APPROVAL", "APPROVED"))
    p.log("publish", published=len(chosen), path=str(path))
    return {"published": str(path)}


# ---------- routing (code, not the model) ----------
def after_verify(s: State):
    if len(s["claims"]) < MIN_VERIFIED and s["extract_tries"] < MAX_EXTRACT_TRIES:
        return "extract"
    if len(s["claims"]) < 3:
        raise RuntimeError(f"only {len(s['claims'])} verified claims after {s['extract_tries']} tries")
    return "draft"


def after_judge(s: State):
    if s["judge_feedback"] and s["draft_tries"] < MAX_DRAFT_TRIES:
        return "draft"
    return "approve" if any(len(pt["talking_points"]) >= 2 for pt in s["pitches"]) else END


def build():
    g = StateGraph(State)
    for name, fn in [("crawl", crawl), ("extract", extract), ("verify", verify), ("draft", draft),
                     ("cite_check", cite_check), ("judge", judge), ("approve", approve), ("publish", publish)]:
        g.add_node(name, fn)
    g.add_edge(START, "crawl")
    g.add_edge("crawl", "extract")
    g.add_edge("extract", "verify")
    g.add_conditional_edges("verify", after_verify, ["extract", "draft"])
    g.add_edge("draft", "cite_check")
    g.add_edge("cite_check", "judge")
    g.add_conditional_edges("judge", after_judge, ["draft", "approve", END])
    g.add_edge("approve", "publish")
    g.add_edge("publish", END)
    return g.compile(checkpointer=InMemorySaver())


def main():
    p.OUT.mkdir(exist_ok=True)
    (p.OUT / "run.log").write_text("")
    graph = build()
    cfg = {"configurable": {"thread_id": "prommer-pitch-1"}}
    result = graph.invoke({}, cfg)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n" + payload["pitch_pack"])
        for i, a in enumerate(payload["angles"]):
            print(f"[{i}] {a}")
        # --approve 0,2 for non-interactive runs; otherwise ask
        arg = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--approve=")), None)
        raw = arg if arg is not None else input("Approve which pitches (e.g. 0,2; blank = none)? ")
        idx = [int(x) for x in raw.split(",") if x.strip().isdigit()]
        result = graph.invoke(Command(resume={"approve": idx}), cfg)

    print("published:", result.get("published") or "nothing")


if __name__ == "__main__":
    main()
