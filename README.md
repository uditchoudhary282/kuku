# Promo Finder — Kuku AI Generalist, Problem 1

Reads a microdrama's screenplays and returns ranked 20–60 s promo clips. Each promo comes with its episode, segment timestamps, hook, cliffhanger and a one-line reason.

## Three ways to use it
- **Web app (dark dashboard):** `streamlit run app.py` → upload the screenplay zip, run, review promos.
  Every promo has **clip anatomy** (the cut on the episode timeline, hook + cliffhanger marked) and a **why-this-rank**
  waterfall; the posting list downloads as CSV, JSON or an **ffmpeg cut sheet** that cuts the real videos.
  Tabs: Review · Promos · **Short-term memory** (this thread's checkpoints, thread notes, compare threads, time travel) ·
  **Long-term memory** (editor decisions shared by every thread) · Quality audit · Pipeline.
- **Colab:** open `promo_finder_colab.ipynb`, upload both zips, Runtime → Run all.
- **CLI:** below.

### Deploy the app free (Streamlit Community Cloud)
1. Push this folder to a **public GitHub repo** (the `.gitignore` keeps the confidential input pack out).
2. share.streamlit.io → **New app** → pick the repo → main file `app.py` → **Deploy**.
3. Paste the app URL in the form. Reviewers upload the screenplay zip themselves; API keys are typed into the
   sidebar per session and never stored.

## Run (CLI)

```bash
pip install -r requirements.txt

# free, offline, no API key
python -m promo_finder run --data "path/to/I Married a Broken Soldier. He was a Duke"

# hybrid (recommended with a key): free rules search everything, the LLM judges only the top 30 (~30 calls)
python -m promo_finder run --data "<show folder>" --model hybrid:openai:gpt-4o-mini

# any LLM, by changing one string
export GOOGLE_API_KEY=...      # free tier
python -m promo_finder run --data "<show folder>" --model google_genai:gemini-2.0-flash
#   --model groq:llama-3.3-70b-versatile | openai:gpt-4o-mini | anthropic:claude-sonnet-4-5 | ...

# editor reviews before export (human-in-the-loop); decisions are remembered across runs
python -m promo_finder run --data "<show folder>" --review          # interactive a/r per promo
python -m promo_finder run --data "<show folder>" --review --decisions examples/editor_decisions.json

python -m promo_finder resume  --thread <id>   # continue a crashed run; finished episodes are not redone
python -m promo_finder history --thread <id>   # every checkpoint of a run
python -m promo_finder memory                  # what the system has learned from editors
pytest -q                                      # SHOW_DIR=<show folder> if not at ../zipdata/...
```

Outputs go to `outputs/<show>/<model>/`:

| File | Contents |
|---|---|
| `promos_top.csv` / `.json` | The top 15 promos, in posting order |
| `all_candidates_ranked.json` | Every valid candidate, ranked |
| `report.md` | Readable report with scores and the transcript of each clip |
| `run_log.txt` | Per-episode attempts and rule violations caught |

## How it works

```
load ──Send×50──▶ episode subgraph ──▶ stitch ──▶ rank ──▶ review ──▶ export
                  propose ⇄ validate → critique   cross-ep   score +    interrupt()
                  (retry with rule errors)        promos     memory     editor a/r
```

**Code vs. AI split.**
- **Parsing and timestamps** are plain Python regex. The model never writes a time. It only names beat IDs like `E03-B012`, and the code looks up their real times.
- **Hard rules** are checked in code (`rules.py`):
  - total length 20–60 s
  - hook within the first 3 s
  - promo ends within 3 s of the cliffhanger
  - never cut mid-line
  - segments in order, no overlap

  Any violation goes back to the model as text and it tries again, up to 3 attempts.
- **Judgment** comes from the model, or from the free rule-based backend (`heuristics.py`). That covers which moments, how strong the hook and cliffhanger are, and whether a cold viewer can follow the clip.
- **Ranking:** `0.40·hook + 0.35·cliffhanger + 0.25·standalone`, scaled to 0–100. It subtracts a penalty for distance from 35 s, adds learned editor preferences by hook type, and keeps at most 2 promos per episode in the top list.

**LangGraph concepts and why each is there**

| Concept | Use |
|---|---|
| `Send` map-reduce + `operator.add` reducers | 50 episodes processed in parallel, results merged |
| Subgraph with `output_schema` | Episode logic isolated; only `candidates`/`log` flow up |
| Conditional edge (cycle) | validate → propose again with the errors, until valid or 3 tries |
| `RetryPolicy` | Rate limits and transient API errors retried with backoff |
| `CachePolicy` + `SqliteCache` | Same episode + same model → no new LLM call (cost) |
| `SqliteSaver` checkpointer + `thread_id` | Crash at episode 17 → `resume` reruns only episode 17 (tested: 49/50 kept) |
| `get_state_history` | Time travel and audit of a run |
| `SqliteStore` (long-term memory) | Editor decisions, learned hook-type weights, already-posted and rejected clips, all kept across runs and shows |
| `interrupt()` + `Command(resume=…)` | Editor approves or rejects before export |

## Results on the input pack (heuristic backend, no API cost)
- 96 unique, valid promos from 49 of 50 episodes (72 single, 21 multi-segment, 3 cross-episode). Lengths run 21.4–56.3 s.
- Spot check of the top 15: 7 clearly postable before the scorer fixes, 10 after.
- Top pick: Ep 1, "Because he is marrying me." (sister steals the groom at the altar), then a cut to the episode's closing cliffhanger.
- The memory demo: the editor approved 13 of 15 and rejected 2. On the next run, all 15 were pushed below new clips and violence/threat hooks got a boost.

## Where it picks badly (known failures)
- **Subtext hooks are missed.** Ep 50's hook is a priest caught in bed, with no emotion tags and no violent words, so it gets zero promos. An LLM backend is needed here.
- **Keyword bias.** The scorer over-rewards "violence" words. For example, "slap" in an SFX about a book hitting a puddle counted as a slap.
- **The standalone score is a proxy.** It counts speakers and penalizes pronoun openers. It doesn't know whether "Alexander" needs an introduction.
- **Cross-episode stitching is naive.** It joins episode N's hook with episode N+1's ending without checking that they belong to the same storyline.
- **No ground truth yet.** The next step is having an editor hand-label ~5 episodes, then measuring overlap with the tool's picks.
- **The LLM path is tested only with a scripted fake model** (`tests/`), which covers retries, structured output, JSON fallback and the critic. I haven't run a real API key against it here.
