# prommer-pitch-pipeline

**Live:** https://prommer-pitch-pipeline.vercel.app (last run trace + outputs) · `POST /api/run` runs the graph live (token-gated so the URL can't burn the Gemini key)

LangGraph + Gemini 3.8 Flash. `graph.py` is the graph, `pipeline.py` has the node logic, `app.py` is the FastAPI entrypoint for Vercel.

```
START -> crawl (code) -> extract (LLM) -> verify (code) --[< 8 verified]--> extract (+ rejected quotes as feedback)
                                               \-> draft (LLM) -> cite_check (code) -> judge (LLM) --[unsupported]--> draft (+ judge reasons)
                                                                                                  \-> approve [interrupt()] -> publish
```
LangGraph earns its place through the two feedback loops, checkpointed state, and the `interrupt()` approval gate. The routing itself is plain code, not model-decided.

Run locally: `pip install -r requirements.txt`, put `GEMINI_API_KEY` in `.env`, then `python graph.py` (use `--approve=0,1` to skip the prompt).

## What broke
1. `urllib` failed with `CERTIFICATE_VERIFY_FAILED` (python.org macOS build has no CA bundle), so I switched to curl.
2. Silent one: `/en/tech/guides/enterprise-ai-strategy/` returns HTTP 200 with a 169-char "Redirecting to:" stub, so `curl -L` has nothing to follow. The pipeline ran green on 4 real pages + a stub. The fix: parse the stub's redirect target, follow up to 3 hops, and fail loudly under 1000 chars. Verified claims went from 24 to 30.

3. The judge -> draft loop didn't converge. On the live Vercel run, the judge rejected 2 of 10 points on draft 1 and **5 of 8** on the redraft. The writer regenerates every pitch, so it throws away good points and paraphrases the new ones more loosely ("rarely justify" became "don't justify", "worth more" became "substantially more"). Next fix: only rewrite the rejected points, keep the accepted ones fixed, and require each point to contain an exact span of its quote.
4. Approval resume on Vercel returns 409. `InMemorySaver` lives in one function instance, and `/api/approve` landed on a different one. Local CLI resume works. The fix is a Postgres checkpointer keyed by thread_id.

## Not built (on purpose)
- No sending/posting. Anything in Thomas's voice needs his sign-off.
- No agent framework. The step order is fixed, so a plain function chain is enough.
