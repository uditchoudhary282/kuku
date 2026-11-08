"""CLI.

  python -m promo_finder run     --data "<show folder>" [--model heuristic] [--thread NAME] [--review] [--top 15]
  python -m promo_finder resume  --thread NAME            # continue a crashed / interrupted run from its checkpoint
  python -m promo_finder history --thread NAME            # list checkpoints (time travel)
  python -m promo_finder memory                           # show long-term memory (editor prefs, feedback, posted)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from langgraph.cache.sqlite import SqliteCache
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.types import Command

from .graph import build_graph, slug

STATE_DIR = Path(".promo_state")


def open_runtime():
    STATE_DIR.mkdir(exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(STATE_DIR / "checkpoints.db", check_same_thread=False))
    store = SqliteStore(sqlite3.connect(STATE_DIR / "memory.db", check_same_thread=False, isolation_level=None))
    store.setup()
    cache = SqliteCache(path=str(STATE_DIR / "llm_cache.db"))
    return build_graph(checkpointer=saver, store=store, cache=cache), store


def drive(graph, inp, cfg, decisions_file: str | None):
    """Stream the run; when the graph pauses at interrupt(), collect editor decisions and resume."""
    cfg = {**cfg, "max_concurrency": int(os.environ.get("PROMO_CONCURRENCY", "8"))}  # stay under API rate limits
    while True:
        for chunk in graph.stream(inp, cfg, stream_mode="updates"):
            for node, upd in chunk.items():
                if node == "__interrupt__":
                    continue
                for line in (upd or {}).get("log", []) if isinstance(upd, dict) else []:
                    print(f"  [{node}] {line}")
        snap = graph.get_state(cfg)
        if not snap.interrupts:
            return snap
        payload = snap.interrupts[0].value
        if decisions_file:
            decisions = json.loads(Path(decisions_file).read_text())
        else:
            print(f"\n⏸  {payload['question']}")
            decisions = {}
            for p in payload["promos"]:
                print(f"\n#{p['rank']} {p['id']} ({p['score']}, {p['hook_type']}) {' + '.join(p['segments'])}\n"
                      f"   HOOK: {p['hook']}\n   CLIFF: {p['cliffhanger']}")
                decisions[p["id"]] = (input("   approve? [a]/r: ").strip() or "a")
        inp = Command(resume=decisions)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="promo_finder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--model", default="heuristic")
    r.add_argument("--thread")
    r.add_argument("--review", action="store_true")
    r.add_argument("--decisions", help="JSON {promo_id: a|r} for non-interactive review")
    r.add_argument("--top", type=int, default=15)
    r.add_argument("--out", default="outputs")
    for name in ("resume", "history"):
        x = sub.add_parser(name)
        x.add_argument("--thread", required=True)
        x.add_argument("--decisions")
    sub.add_parser("memory")
    a = ap.parse_args(argv)

    graph, store = open_runtime()
    if a.cmd == "memory":
        for ns in store.list_namespaces():
            items = store.search(ns, limit=1000)
            print(f"{'/'.join(ns)}: {len(items)} item(s)")
            for it in items[:5]:
                print(f"   {it.key}: {json.dumps(it.value)[:120]}")
        return

    if a.cmd == "run":
        show = Path(a.data).resolve().name
        thread = a.thread or f"{slug(show)}--{slug(a.model)}--{time.strftime('%Y%m%d-%H%M%S')}"
        cfg = {"configurable": {"thread_id": thread}}
        if graph.get_state(cfg).values:
            sys.exit(f"Thread '{thread}' already exists. Use `resume --thread {thread}` or pick a new --thread.")
        inp = {"data_dir": str(Path(a.data).resolve()), "show": show, "model": a.model, "review": a.review,
               "top_n": a.top, "out_dir": str(Path(a.out) / slug(show) / slug(a.model)), "candidates": [], "log": []}
        print(f"▶ thread={thread} model={a.model}  (if it stops: python -m promo_finder resume --thread {thread})")
        try:
            snap = drive(graph, inp, cfg, a.decisions)
        except Exception as e:  # noqa: BLE001
            nxt = sorted(set(graph.get_state(cfg).next))
            print(f"\n✖ Run stopped at {nxt}: {type(e).__name__}: {str(e)[:160]}\n"
                  f"  Finished work is saved in thread '{thread}'. Fix the cause (e.g. set the API key), then:\n"
                  f"  python -m promo_finder resume --thread {thread}")
            return
    elif a.cmd == "resume":
        cfg = {"configurable": {"thread_id": a.thread}}
        snap = graph.get_state(cfg)
        if not snap.next:
            print("Nothing to resume: this thread already finished.")
            return
        print(f"↻ resuming thread={a.thread} at {snap.next}")
        snap = drive(graph, None, cfg, a.decisions)
    else:  # history
        cfg = {"configurable": {"thread_id": a.thread}}
        for s in graph.get_state_history(cfg):
            n = len(s.values.get("candidates", []))
            nxt = list(s.next)
            nxt = [f"{nxt[0]} ×{len(nxt)}"] if len(nxt) > 1 and len(set(nxt)) == 1 else nxt
            print(f"{s.config['configurable']['checkpoint_id'][-12:]}  step={s.metadata.get('step'):>3}  "
                  f"next={nxt}  candidates={n}")
        return

    out = snap.values.get("outputs")
    if out:
        print(f"\n✔ done → {out['dir']}  ({', '.join(out['files'])})")
        for p in snap.values["approved"][:5]:
            segs = " + ".join(f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"])
            print(f"  #{p['rank']:>2} {p['score']:>5}  {segs:<38} {p['hook'][:60]}")


if __name__ == "__main__":
    sys.exit(main())
