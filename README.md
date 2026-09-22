# prommer-pitch-pipeline

Turns Thomas Prommer's own published positions on prommer.net into podcast/press pitch angles, where every talking point traces back to a verbatim, re-verified quote. Output waits for his approval; nothing is sent.

```
crawl (curl, code) -> extract claims (LLM) -> verify quotes (code) -> draft pitches (LLM) -> citation check (code) -> pitch_pack.md [PENDING_APPROVAL]
```

| step | runs on | input | output |
|---|---|---|---|
| 1 crawl | curl + html.parser | 5 tech/authority pages | page text, follows the site's HTML "Redirecting to:" stubs, rejects pages < 1000 chars |
| 2 extract | `claude -p` | page text | `claims_raw.json`: `{url, quote, position}` |
| 3 verify | code | claims + page text | `claims_verified.json`: only quotes that are exact substrings of the page (after quote/dash/whitespace normalization) |
| 4 draft | `claude -p` | verified claims only (never raw pages) | 3 pitches; every talking point carries `claim_ids` |
| 5 check | code | pitches + claim ids | drops uncited points and pitches with < 2 grounded points |
| 6 render | code | | `out/pitch_pack.md` with source quotes + URLs |

Every step appends a JSON line to `out/run.log` (counts, drops, LLM latency, cost).

Run: `python3 pipeline.py` (needs the `claude` CLI logged in; stdlib only).

## What broke
1. `urllib` failed with `CERTIFICATE_VERIFY_FAILED` (python.org macOS build has no CA bundle), so I switched to curl.
2. Silent one: `/en/tech/guides/enterprise-ai-strategy/` returns HTTP 200 with a 169-char "Redirecting to:" stub, so `curl -L` has nothing to follow. The pipeline ran green on 4 real pages + a stub. The fix: parse the stub's redirect target, follow up to 3 hops, and fail loudly under 1000 chars. Verified claims went from 24 to 30.

## Not built (on purpose)
- No sending/posting. Anything in Thomas's voice needs his sign-off.
- No agent framework. The step order is fixed, so a plain function chain is enough.
- Citation check only proves the cited IDs exist. It doesn't prove the sentence is *entailed* by the quote. Next step: a per-point entailment check against the quote.
