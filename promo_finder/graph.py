"""LangGraph pipeline.

  load ──Send×N──▶ [episode subgraph: propose ⇄ validate → critique] ──▶ stitch ──▶ rank ──▶ review ──▶ export
                     (map: one per episode, retry loop       (reduce: cross-   (score +   (interrupt:
                      feeds rule errors back to the model)    episode promos)  memory)    editor approves)

LangGraph concepts used
  StateGraph + TypedDict state, reducers (operator.add)   parallel writes from 50 episode workers merge safely
  Send API (map-reduce)                                   one subgraph run per episode, fan-out/fan-in
  Subgraph with its own state + output_schema             episode logic isolated from show-level state
  Conditional edges / cycle                               propose → validate → (retry with errors | critique)
  RetryPolicy                                             transient API errors / rate limits retried with backoff
  CachePolicy + SqliteCache                               unchanged episode + same model → no new LLM call
  Checkpointer (SqliteSaver) + thread_id                  crash mid-show → `resume` reruns only unfinished episodes;
                                                          `history` lists every checkpoint (time travel)
  Store (SqliteStore)                                     long-term memory across runs/shows: editor decisions,
                                                          learned hook-type preferences, already-posted promos
  interrupt() + Command(resume=...)                       human-in-the-loop approval before export
"""
from __future__ import annotations

import csv
import hashlib
import json
import operator
import os
import re
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.config import get_store
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy, RetryPolicy, Send, interrupt

from . import heuristics, llm
from .parser import fmt_ts, load_show
from .rules import index_beats, overlap, resolve
from .schemas import Episode, Promo, ProposedPromo, ProposedSegment

MAX_ATTEMPTS = 3
PREFS_NS = ("editor_prefs",)
WEIGHTS = {"hook": 0.40, "cliff": 0.35, "standalone": 0.25}


# ───────────────────────────── state ─────────────────────────────
class ShowState(TypedDict, total=False):
    data_dir: str
    show: str
    model: str
    review: bool
    top_n: int
    out_dir: str
    episodes: list[dict]
    candidates: Annotated[list[dict], operator.add]   # reducer: 50 parallel workers append here
    log: Annotated[list[str], operator.add]
    ranked: list[dict]
    approved: list[dict]
    outputs: dict
    notes: Annotated[list[dict], operator.add]   # short-term memory: editor notes scoped to this thread_id
    judged: Annotated[list[dict], operator.add]  # hybrid mode: LLM verdicts on the top-K candidates
    audit: dict


def quality_audit(episodes: list[dict], ranked: list[dict]) -> dict:
    """Checks on the input itself, not the promos: did every episode parse cleanly?"""
    nums = [e["number"] for e in episodes]
    with_promo = {s["episode"] for r in ranked for s in r["segments"]}
    per = []
    for e in episodes:
        b = e["beats"]
        per.append({"episode": e["number"], "duration_s": round(e["duration"], 1), "beats": len(b),
                    "dialogue": sum(x["type"] == "dialogue" for x in b), "action": sum(x["type"] == "action" for x in b),
                    "promos": sum(1 for r in ranked if any(s["episode"] == e["number"] for s in r["segments"]))})
    return {
        "episodes": len(episodes), "beats": sum(len(e["beats"]) for e in episodes),
        "missing_episode_numbers": sorted(set(range(1, max(nums, default=0) + 1)) - set(nums)),
        "episodes_with_zero_beats": [e["number"] for e in episodes if not e["beats"]],
        "invalid_beat_bounds": sum(1 for e in episodes for x in e["beats"]
                                   if not (0 <= x["start"] < x["end"] <= e["duration"] + 0.01)),
        "episodes_without_promos": [n for n in nums if n not in with_promo],
        "all_promos_20_to_60s": all(20 <= r["duration"] <= 60 for r in ranked),
        "per_episode": per,
        "limitations": ["Scores are editorial heuristics, not predictions of views or retention.",
                        "Script timestamps are assumed to match the edited video.",
                        "No blinded human labels yet, so precision/recall cannot be reported."],
    }


class EpisodeState(TypedDict, total=False):
    episode: dict
    model: str
    attempt: int
    proposals: list[dict]
    errors: list[str]
    error_history: Annotated[list[str], operator.add]
    valid: list[dict]
    candidates: Annotated[list[dict], operator.add]
    log: Annotated[list[str], operator.add]


class EpisodeOutput(TypedDict):
    candidates: Annotated[list[dict], operator.add]
    log: Annotated[list[str], operator.add]


# ───────────────────────── episode subgraph ─────────────────────────
def propose(state: EpisodeState) -> dict:
    ep = Episode(**state["episode"])
    if os.environ.get("FAIL_EPISODE") == str(ep.number):
        raise RuntimeError(f"Simulated crash on episode {ep.number} (unset FAIL_EPISODE and resume)")
    attempt = state.get("attempt", 0) + 1
    if state["model"] == "heuristic":
        props = heuristics.propose(ep)
    else:
        props = llm.llm_propose(state["model"], ep, state.get("errors", []))
    return {"attempt": attempt, "valid": state.get("valid", []),
            "errors": [], "proposals": [p.model_dump() for p in props.promos]}


def validate(state: EpisodeState) -> dict:
    ep = Episode(**state["episode"])
    beats, by_no = index_beats([ep]), {ep.number: ep}
    valid = list(state.get("valid", []))
    errors: list[str] = []
    for k, raw in enumerate(state.get("proposals", [])):
        promo, errs = resolve(ProposedPromo(**raw), beats, by_no, pid=f"E{ep.number:02d}-A{state['attempt']}-P{k + 1}")
        if promo:
            if all(overlap(promo, Promo(**v)) < 0.5 for v in valid):
                valid.append(promo.model_dump())
        else:
            errors += [f"Promo {k + 1}: {e}" for e in errs]
    return {"valid": valid, "errors": errors, "error_history": errors}


def after_validate(state: EpisodeState) -> str:
    if state["errors"] and len(state["valid"]) < 2 and state["attempt"] < MAX_ATTEMPTS:
        return "propose"   # cycle: send rule violations back to the model
    return "critique"


def critique(state: EpisodeState) -> dict:
    ep = Episode(**state["episode"])
    by_id = {b.id: b for b in ep.beats}
    out = []
    for v in state["valid"]:
        p = Promo(**v)
        c = heuristics.critique(ep, by_id[p.hook_beat], by_id[p.cliffhanger_beat],
                                p.segments[0].start, p.segments[-1].end)
        scores = {"hook": c.hook_strength, "cliff": c.cliffhanger_strength, "standalone": c.standalone}
        if state["model"] != "heuristic":
            lc = llm.llm_critique(state["model"], ep.logline, p.transcript, p.hook, p.cliffhanger)
            scores = {"hook": lc.hook_strength, "cliff": lc.cliffhanger_strength, "standalone": lc.standalone}
            p.critic_note = lc.note
        p.scores = scores
        out.append(p.model_dump())
    fixed = len(state.get("error_history", []))
    return {"candidates": out,
            "log": [f"E{ep.number:02d}: {len(out)} promos, {state['attempt']} attempt(s), {fixed} rule violations caught"]}


def _episode_cache_key(inp: dict) -> str:
    ep = inp["episode"]
    blob = json.dumps({"beats": ep["beats"], "model": inp["model"], "v": 3}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def build_episode_graph():
    g = StateGraph(EpisodeState, output_schema=EpisodeOutput)
    g.add_node("propose", propose, retry_policy=RetryPolicy(max_attempts=4, initial_interval=2.0,
                                                            retry_on=lambda e: not isinstance(e, RuntimeError) or "Simulated" not in str(e)))
    g.add_node("validate", validate)
    g.add_node("critique", critique, retry_policy=RetryPolicy(max_attempts=3))
    g.add_edge(START, "propose")
    g.add_edge("propose", "validate")
    g.add_conditional_edges("validate", after_validate, ["propose", "critique"])
    g.add_edge("critique", END)
    return g.compile()


# ───────────────────────────── show graph ─────────────────────────────
def load(state: ShowState) -> dict:
    eps = load_show(state["data_dir"])
    return {"episodes": [e.model_dump() for e in eps],
            "log": [f"Loaded {len(eps)} episodes, {sum(len(e.beats) for e in eps)} beats from '{state['show']}'"]}


HYBRID_TOP_K = 30


def _retryable(e: Exception) -> bool:
    """Retry rate limits and network blips; never retry a missing/wrong key or a simulated crash."""
    txt = f"{type(e).__name__} {e}"
    return not any(w in txt for w in ("Authentication", "Missing credentials", "api_key", "PermissionDenied", "Simulated"))


def is_hybrid(model: str) -> bool:
    return model.startswith("hybrid:")


def fan_out(state: ShowState) -> list[Send]:
    # hybrid: the free rules do the per-episode search; the LLM only judges the shortlist later
    m = "heuristic" if is_hybrid(state["model"]) else state["model"]
    return [Send("episode", {"episode": e, "model": m}) for e in state["episodes"]]


class JudgeIn(TypedDict):
    promo: dict
    model: str
    logline: str


def judge_targets(state: ShowState):
    """Hybrid mode: fan out one LLM judgement per shortlisted promo (Send), else go straight to rank."""
    if not is_hybrid(state["model"]):
        return "rank"
    cands = sorted((Promo(**c) for c in state["candidates"]), key=lambda c: -score(c, {}))
    short, hooks = [], set()
    for c in cands:
        if c.hook_beat not in hooks:
            short.append(c)
            hooks.add(c.hook_beat)
        if len(short) >= HYBRID_TOP_K:
            break
    logl = {e["number"]: e["logline"] for e in state["episodes"]}
    llm_spec = state["model"].split(":", 1)[1]
    return [Send("llm_judge", {"promo": c.model_dump(), "model": llm_spec,
                               "logline": logl.get(c.segments[0].episode, "")}) for c in short]


def llm_judge(state: JudgeIn) -> dict:
    p = state["promo"]
    c = llm.llm_critique(state["model"], state["logline"], p["transcript"], p["hook"], p["cliffhanger"])
    cliff = min(c.cliffhanger_strength, 2) if c.ending == "settled" else c.cliffhanger_strength
    return {"judged": [{"id": p["id"], "scores": {"hook": c.hook_strength, "cliff": cliff, "standalone": c.standalone},
                        "note": c.note, "why": c.why, "ending": c.ending}]}


def _judge_cache_key(inp: dict) -> str:
    blob = json.dumps({"t": inp["promo"]["transcript"], "id": inp["promo"]["id"], "m": inp["model"], "v": 1})
    return hashlib.sha256(blob.encode()).hexdigest()


def stitch(state: ShowState) -> dict:
    """Cross-episode promos: open on episode N's strongest hook, cut to episode N+1's closing cliffhanger."""
    eps = [Episode(**e) for e in state["episodes"]]
    by_no = {e.number: e for e in eps}
    beats = index_beats(eps)
    cands = [Promo(**c) for c in state["candidates"]]
    best_hook: dict[int, Promo] = {}
    for c in cands:
        if c.source == "single" and (c.segments[0].episode not in best_hook or
                                     c.scores["hook"] > best_hook[c.segments[0].episode].scores["hook"]):
            best_hook[c.segments[0].episode] = c
    made = []
    for n, hp in best_hook.items():
        nxt = by_no.get(n + 1)
        if not nxt:
            continue
        hb = beats[hp.hook_beat]
        hook_end = max((b for b in by_no[n].beats if hb.start < b.end <= hb.start + 12), key=lambda b: b.end, default=None)
        tail = [b for b in nxt.beats if b.type != "camera"]
        if not hook_end or not tail:
            continue
        last = max(tail, key=lambda b: b.end)
        budget = 52 - (hook_end.end - hb.start)
        starts = [b for b in nxt.beats if b.type in ("dialogue", "action") and 15 <= last.end - b.start <= budget]
        if not starts:
            continue
        st = min(starts, key=lambda b: abs((last.end - b.start) - min(30, budget)))
        prop = ProposedPromo(segments=[ProposedSegment(start_beat=hb.id, end_beat=hook_end.id),
                                       ProposedSegment(start_beat=st.id, end_beat=last.id)],
                             hook_beat=hb.id, cliffhanger_beat=last.id, hook=hp.hook,
                             cliffhanger=heuristics._short(last), hook_type=hp.hook_type,
                             why=f"Episode {n}'s strongest hook, then straight to how episode {n + 1} leaves it hanging.")
        promo, errs = resolve(prop, beats, by_no, pid=f"X{n:02d}-{n + 1:02d}", source="cross_episode")
        if promo:
            promo.scores = {"hook": hp.scores["hook"],
                            "cliff": heuristics.critique(nxt, last, last, st.start, last.end).cliffhanger_strength,
                            "standalone": max(1, min(hp.scores["standalone"], 5) - 1)}  # jumps cost clarity
            made.append(promo.model_dump())
    made.sort(key=lambda p: -(p["scores"]["hook"] + p["scores"]["cliff"]))
    return {"candidates": made[:8], "log": [f"Stitched {len(made[:8])} cross-episode promos (of {len(made)} possible)"]}


def score_parts(p: Promo, prefs: dict[str, float]) -> dict[str, float]:
    """Each term of the score, so the UI can show *why* a promo ranks where it does."""
    sc = {k: float(p.scores.get(k, 1)) for k in WEIGHTS}
    return {
        "hook (40%)": round(WEIGHTS["hook"] * (sc["hook"] - 1) / 4 * 100, 1),
        "cliffhanger (35%)": round(WEIGHTS["cliff"] * (sc["cliff"] - 1) / 4 * 100, 1),
        "standalone (25%)": round(WEIGHTS["standalone"] * (sc["standalone"] - 1) / 4 * 100, 1),
        "length vs 35 s": round(-abs(p.duration - 35) / 5, 1),           # sweet spot for Reels/TikTok
        "editor memory": round(prefs.get(p.hook_type, 0.0), 1),          # learned from editor decisions (Store)
    }


def score(p: Promo, prefs: dict[str, float]) -> float:
    return round(sum(score_parts(p, prefs).values()), 1)


def rank(state: ShowState) -> dict:
    store = get_store()
    prefs_item = store.get(PREFS_NS, "hook_type_bonus")
    prefs = prefs_item.value if prefs_item else {}
    posted_items = store.search(("posted", slug(state["show"])), limit=1000)
    rejected_items = store.search(("rejected", slug(state["show"])), limit=1000)

    def remembered(items):
        return [(i.key, [tuple(x) for x in i.value.get("segments", [])]) for i in items]

    mem_posted, mem_rejected = remembered(posted_items), remembered(rejected_items)

    def hits(p: Promo, mem) -> bool:
        """Same clip, or a re-cut that shares >50% of its footage with a remembered one."""
        mine = [(s.episode, s.start, s.end) for s in p.segments]
        for key, segs in mem:
            if key == sig(p):
                return True
            if segs:
                shared = sum(max(0.0, min(a[2], b[2]) - max(a[1], b[1])) for a in mine for b in segs if a[0] == b[0])
                other = sum(b[2] - b[1] for b in segs)
                if shared / max(1e-6, min(p.duration, other)) > 0.5:
                    return True
        return False

    def sig(p: Promo) -> str:
        return "|".join(f"{s.episode}:{s.start_beat}:{s.end_beat}" for s in p.segments)

    last_ep = max((e["number"] for e in state["episodes"]), default=1)
    cands = [Promo(**c) for c in state["candidates"]]
    judged = {j["id"]: j for j in state.get("judged", []) or []}
    for c in cands:   # hybrid: the LLM's verdict replaces the rule-based scores for the shortlist
        j = judged.get(c.id)
        if j:
            c.scores = j["scores"]
            c.critic_note = j["note"] + (" (settled ending)" if j["ending"] == "settled" else "")
            c.why = j["why"] or c.why
    spoilers = 0
    for c in cands:
        c.breakdown = score_parts(c, prefs)
        # late-show reveals give away the twist a new viewer should discover by watching
        if c.hook_type == "reveal" and min(s.episode for s in c.segments) > 0.6 * last_ep:
            c.breakdown["spoiler risk"] = -6.0
            spoilers += 1
        c.score = round(sum(c.breakdown.values()), 1)
    cands.sort(key=lambda c: (bool(judged) and c.id not in judged, -c.score))   # judged shortlist first
    kept: list[Promo] = []
    for c in cands:                             # dedupe: same footage, or the same opening hook line
        if all(overlap(c, k) < 0.5 and c.hook_beat != k.hook_beat for k in kept):
            kept.append(c)

    flags = {id(c): (hits(c, mem_posted), hits(c, mem_rejected)) for c in kept}
    fresh = [c for c in kept if not any(flags[id(c)])]
    stale = [c for c in kept if any(flags[id(c)])]
    top, rest, per_ep = [], [], {}
    for c in fresh:                             # diversity: max 2 per episode in the top list
        e = c.segments[0].episode
        (top if per_ep.get(e, 0) < 2 and len(top) < state["top_n"] else rest).append(c)
        per_ep[e] = per_ep.get(e, 0) + 1
    ordered = top + rest + stale
    out = []
    for i, c in enumerate(ordered, 1):
        d = c.model_dump()
        d.update(rank=i, already_posted=flags[id(c)][0], editor_rejected=flags[id(c)][1], signature=sig(c))
        out.append(d)
    return {"ranked": out, "log": [f"Ranked {len(out)} unique promos ({len(cands) - len(kept)} near-duplicates dropped, "
                                   f"{len(stale)} already posted/rejected (or re-cuts of them) pushed down, {spoilers} late-show reveals "
                                   f"penalised as spoilers, prefs={prefs or 'none yet'})"
                                   + (f"; LLM judged the top {len(judged)}, "
                                      f"{sum(j['ending'] == 'settled' for j in judged.values())} had settled endings" if judged else "")]}


def review(state: ShowState) -> dict:
    top = [r for r in state["ranked"] if r["rank"] <= state["top_n"]]
    if not state.get("review"):
        return {"approved": top}
    decisions: dict[str, str] = interrupt({
        "question": "Approve or reject each promo (a/r). Rejections teach the ranker.",
        "promos": [{"id": r["id"], "rank": r["rank"], "score": r["score"], "hook_type": r["hook_type"],
                    "segments": [f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in r["segments"]],
                    "hook": r["hook"], "cliffhanger": r["cliffhanger"]} for r in top]})
    store = get_store()
    item = store.get(PREFS_NS, "hook_type_bonus")
    prefs: dict[str, float] = dict(item.value) if item else {}
    approved = []
    for r in top:
        d = decisions.get(r["id"], "a").lower()[:1]
        segs = [[x["episode"], x["start"], x["end"]] for x in r["segments"]]
        store.put(("feedback", slug(state["show"])), r["id"], {"decision": d, "hook_type": r["hook_type"],
                                                        "signature": r["signature"], "score": r["score"]})
        prefs[r["hook_type"]] = max(-10.0, min(10.0, prefs.get(r["hook_type"], 0.0) + (1.5 if d == "a" else -3.0)))
        if d == "a":
            approved.append(r)
            store.put(("posted", slug(state["show"])), r["signature"], {"id": r["id"], "segments": segs})
        else:
            store.put(("rejected", slug(state["show"])), r["signature"], {"id": r["id"], "segments": segs})
    store.put(PREFS_NS, "hook_type_bonus", prefs)
    return {"approved": approved, "log": [f"Editor approved {len(approved)}/{len(top)}; prefs now {prefs}"]}


def export(state: ShowState) -> dict:
    out = Path(state["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    audit = quality_audit(state["episodes"], state["ranked"])
    (out / "quality_audit.json").write_text(json.dumps(audit, indent=2))
    (out / "all_candidates_ranked.json").write_text(json.dumps(state["ranked"], indent=2))
    (out / "promos_top.json").write_text(json.dumps(state["approved"], indent=2))
    with open(out / "promos_top.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "score", "duration_s", "segments", "hook", "cliffhanger", "why", "hook_type", "source"])
        for r in state["approved"]:
            w.writerow([r["rank"], r["id"], r["score"], r["duration"],
                        " + ".join(f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in r["segments"]),
                        r["hook"], r["cliffhanger"], r["why"], r["hook_type"], r["source"]])
    (out / "run_log.txt").write_text("\n".join(state["log"]))
    (out / "cut_sheet.sh").write_text(cut_sheet(state["approved"]))
    (out / "report.md").write_text(_report(state))
    return {"audit": audit, "outputs": {"dir": str(out), "files": sorted(p.name for p in out.iterdir())}}


def cut_sheet(promos: list[dict], pattern: str = "episode-{ep:03d}.mp4") -> str:
    """ffmpeg commands that turn the ranked list into actual clips. Multi-segment promos are cut, then joined."""
    L = ["#!/usr/bin/env bash", "# Promo cut sheet: run in the folder with the episode videos.",
         f"# Expected video names: {pattern.format(ep=1)}, {pattern.format(ep=2)}, ... (rename your files, or edit the names in the commands below)",
         "# Times come from the screenplay; nudge by a few frames in your editor if needed.", "set -e", "mkdir -p promos", ""]
    for r in promos:
        tag = f"promo_{r['rank']:02d}_{r['id']}"
        L.append(f"# #{r['rank']}  {r['duration']:.0f}s  score {r['score']}  |  {r['hook'][:70]}")
        parts = []
        for i, sgm in enumerate(r["segments"]):
            src, out = pattern.format(ep=sgm["episode"]), f"promos/{tag}_part{i + 1}.mp4"
            L.append(f'ffmpeg -y -loglevel error -ss {sgm["start"]:.2f} -to {sgm["end"]:.2f} -i "{src}" '
                     f'-c:v libx264 -preset veryfast -crf 20 -c:a aac "{out}"')
            parts.append(out)
        if len(parts) == 1:
            L.append(f'mv "{parts[0]}" "promos/{tag}.mp4"')
        else:
            L.append(f'printf "%s\\n" {" ".join(chr(34) + "file " + chr(39) + p.split("/")[-1] + chr(39) + chr(34) for p in parts)} '
                     f'> "promos/{tag}.txt"')
            L.append(f'(cd promos && ffmpeg -y -loglevel error -f concat -safe 0 -i "{tag}.txt" -c copy "{tag}.mp4" '
                     f'&& rm {" ".join(chr(34) + p.split("/")[-1] + chr(34) for p in parts)} "{tag}.txt")')
        L.append("")
    L.append('echo "Done: $(ls promos/*.mp4 | wc -l) promos in ./promos"')
    return "\n".join(L)


def _report(state: ShowState) -> str:
    L = [f"# Promos — {state['show']}", f"Model: `{state['model']}` · candidates: {len(state['ranked'])} · "
         f"top list: {len(state['approved'])}", ""]
    for r in state["approved"]:
        segs = " + ".join(f"Ep {s['episode']} {s['start_ts']}–{s['end_ts']}" for s in r["segments"])
        sc = r["scores"]
        L += [f"## #{r['rank']} · {r['id']} · score {r['score']} · {r['duration']:.0f}s · {r['source']}",
              f"- **Segments:** {segs}", f"- **Hook ({r['hook_type']}):** {r['hook']}",
              f"- **Cliffhanger:** {r['cliffhanger']}", f"- **Why:** {r['why']}",
              f"- **Scores:** hook {sc.get('hook')}/5 · cliffhanger {sc.get('cliff')}/5 · standalone {sc.get('standalone')}/5"
              + (f" · critic: {r['critic_note']}" if r.get("critic_note") else ""),
              "", "<details><summary>Transcript</summary>", "", "```", r["transcript"], "```", "</details>", ""]
    return "\n".join(L)


def build_graph(checkpointer=None, store=None, cache=None):
    g = StateGraph(ShowState)
    g.add_node("load", load)
    g.add_node("episode", build_episode_graph(),
               retry_policy=RetryPolicy(max_attempts=2, retry_on=lambda e: "Simulated" not in str(e)),
               cache_policy=CachePolicy(key_func=_episode_cache_key))
    g.add_node("stitch", stitch)
    g.add_node("llm_judge", llm_judge, retry_policy=RetryPolicy(max_attempts=4, initial_interval=2.0, retry_on=_retryable),
               cache_policy=CachePolicy(key_func=_judge_cache_key))
    g.add_node("rank", rank)
    g.add_node("review", review)
    g.add_node("export", export)
    g.add_edge(START, "load")
    g.add_conditional_edges("load", fan_out, ["episode"])
    g.add_edge("episode", "stitch")
    g.add_conditional_edges("stitch", judge_targets, ["llm_judge", "rank"])
    g.add_edge("llm_judge", "rank")
    g.add_edge("rank", "review")
    g.add_edge("review", "export")
    g.add_edge("export", END)
    return g.compile(checkpointer=checkpointer, store=store, cache=cache)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]
