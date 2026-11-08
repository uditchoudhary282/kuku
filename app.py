"""Promo Finder — dark dashboard (Streamlit).

Run locally:  streamlit run app.py          (optional: SHOW_DIR=<episode folder> to skip the upload)
Deploy free:  push repo to GitHub → share.streamlit.io → New app → app.py
The input pack is never stored in the repo; users upload it in the app.

Memory model shown in the UI
  • Short-term memory = one LangGraph thread (thread_id). Every node writes a checkpoint; switching threads
    restores that run exactly; notes written to a thread live only in that thread.
  • Long-term memory  = the LangGraph Store. Editor decisions written here change every future run.
"""
from __future__ import annotations

import html
import io
import json
import os
import re
import shutil
import sqlite3
import statistics
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from langgraph.types import Command

APP_VERSION = "1.4.0"
try:   # all imports up front: a half-updated deploy fails here with a clear message, not deep inside a tab
    import promo_finder
    from promo_finder import llm
    from promo_finder.__main__ import STATE_DIR, open_runtime
    from promo_finder.graph import build_episode_graph, cut_sheet, quality_audit, slug
    _ver = getattr(promo_finder, "__version__", "unknown")
    if _ver != APP_VERSION:
        raise ImportError(f"promo_finder package is version {_ver}, app.py expects {APP_VERSION}")
except ImportError as _e:
    st.set_page_config(page_title="Promo Finder", page_icon="🎬")
    st.error("**Files are out of sync.** `app.py` and the `promo_finder/` folder come from different versions.\n\n"
             "Fix: upload the **whole project** again (app.py, requirements.txt and the entire `promo_finder/` folder) "
             f"from the same zip.\n\nDetails: `{_e}`")
    st.stop()

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

OR, OR2, BG, CARD, LINE, MUTED = "#ff7a1a", "#c2410c", "#0f0f11", "#18181b", "#27272a", "#8b8b94"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
html, body, .stApp, button, input, textarea, select, [class*="css"] {{ font-family:'Inter',system-ui,sans-serif !important; }}
.stApp {{ background:radial-gradient(1200px 500px at 85% -10%, rgba(255,122,26,.10), transparent 60%), {BG}; color:#f4f4f5; }}
header[data-testid="stHeader"] {{ background:transparent; height:0; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display:none !important; }}
.block-container {{ padding-top:1.1rem; padding-bottom:3rem; max-width:1320px; }}
h1,h2,h3,h4,p,span,div,label {{ color:inherit; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background:#121214; border-right:1px solid {LINE}; }}
[data-testid="stSidebarHeader"] {{ height:.5rem; padding:0; }}
[data-testid="stSidebarUserContent"] {{ padding-top:.3rem; }}
.brand {{ display:flex; align-items:center; gap:.6rem; font-weight:800; font-size:1.02rem; margin:.2rem 0 .9rem; color:#fff; }}
.brand .logo {{ width:32px; height:32px; border-radius:10px; background:linear-gradient(135deg,{OR},{OR2}); display:grid; place-items:center; font-size:15px;
               box-shadow:0 6px 18px rgba(255,122,26,.35); }}
.side-h {{ font-size:.66rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:#6b6b74; margin:1rem 0 .35rem; }}
section[data-testid="stSidebar"] [data-testid="stButton"] button {{ justify-content:flex-start; border:0; background:transparent; color:#c4c4cc;
  font-weight:500; font-size:.84rem; padding:.45rem .6rem; border-radius:10px; min-height:0; box-shadow:none; }}
section[data-testid="stSidebar"] [data-testid="stButton"] button > div {{ justify-content:flex-start; width:100%; }}
section[data-testid="stSidebar"] [data-testid="stButton"] button p {{ text-align:left; }}
section[data-testid="stSidebar"] [data-testid="stButton"] button:hover {{ background:#1f1f23; color:#fff; }}
section[data-testid="stSidebar"] [class*="st-key-run_active"] button, section[data-testid="stSidebar"] [class*="st-key-show_active"] button {{ background:rgba(255,122,26,.12); color:{OR}; font-weight:600; }}
section[data-testid="stSidebar"] .st-key-rm_show button {{ color:#71717a; font-size:.76rem; }}
section[data-testid="stSidebar"] .st-key-new_run button {{ background:linear-gradient(135deg,{OR},{OR2}); color:#fff; font-weight:700; justify-content:center; }}
section[data-testid="stSidebar"] .st-key-new_run button > div {{ justify-content:center; }}
.pref {{ display:flex; justify-content:space-between; font-size:.8rem; color:#c4c4cc; padding:.16rem .6rem; }}
.pos {{ color:#4ade80; }} .neg {{ color:#f87171; }}

/* widgets on dark: fallbacks so the look holds even if .streamlit/config.toml was not deployed */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {{ color:#d4d4d8 !important; opacity:1 !important; }}
[data-baseweb="select"] div {{ background-color:#18181b !important; color:#f4f4f5 !important; }}
[data-baseweb="select"] svg {{ fill:#a1a1aa !important; }}
[data-testid="stFileUploaderDropzone"] button {{ background:#1f1f23 !important; color:#e4e4e7 !important; border:1px solid #3f3f46 !important; }}
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small {{ color:#8b8b94 !important; }}
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFile"] * {{ color:#e4e4e7 !important; }}
[data-testid="stSliderThumbValue"] p {{ color:#ff7a1a !important; }}
/* widgets on dark */
[data-testid="stFileUploaderDropzone"] {{ background:#18181b; border:1px dashed #3f3f46; }}
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stTextInput input {{ background:#18181b !important; border-color:#2e2e33 !important; color:#f4f4f5 !important; }}
[data-testid="stExpander"] {{ background:{CARD}; border:1px solid {LINE} !important; border-radius:14px !important; }}
[data-testid="stTabs"] [role="tablist"] {{ gap:.3rem; border-bottom:1px solid {LINE}; }}
[data-testid="stTabs"] button[role="tab"] {{ font-weight:600; color:#a1a1aa; }}
[data-testid="stTabs"] button[aria-selected="true"] {{ color:#fff; }}
.stButton button[kind="primary"], .stFormSubmitButton button {{ background:linear-gradient(135deg,{OR},{OR2}) !important; border:0 !important; color:#fff !important;
  font-weight:700 !important; border-radius:12px !important; }}
.stButton button[kind="secondary"], .stDownloadButton button {{ background:#1f1f23; border:1px solid #2e2e33; color:#e4e4e7; border-radius:12px; }}
[data-testid="stMarkdownContainer"] code {{ background:#222226; color:#fdba74; }}

/* cards */
[class*="st-key-card_"] {{ background:{CARD}; border:1px solid {LINE}; border-radius:20px; padding:1.05rem 1.15rem 1rem; }}
.ttl {{ font-size:.92rem; font-weight:700; color:#fafafa; display:flex; align-items:center; justify-content:space-between; }}
.big {{ font-size:1.9rem; font-weight:800; letter-spacing:-.03em; color:#fff; margin:.2rem 0 .1rem; }}
.up {{ font-size:.76rem; font-weight:700; color:#4ade80; background:rgba(74,222,128,.1); padding:.12rem .45rem; border-radius:999px; margin-left:.35rem; vertical-align:middle; }}
.muted {{ color:{MUTED}; font-size:.78rem; }}
.mono {{ font-family:'JetBrains Mono',monospace; font-size:.74rem; color:#a1a1aa; }}
.topbar {{ display:flex; justify-content:space-between; align-items:center; margin:.1rem 0 1rem; }}
.topbar .h {{ font-size:1.25rem; font-weight:800; color:#fff; letter-spacing:-.02em; }}
.topbar .h small {{ display:block; font-size:.78rem; font-weight:500; color:{MUTED}; letter-spacing:0; margin-top:.1rem; }}
.pill {{ display:inline-flex; align-items:center; gap:.4rem; font-size:.74rem; font-weight:600; color:#e4e4e7; background:#1c1c20; border:1px solid {LINE};
         padding:.35rem .7rem; border-radius:999px; margin-left:.35rem; }}
.pill .dot {{ width:7px; height:7px; border-radius:50%; background:#4ade80; box-shadow:0 0 0 3px rgba(74,222,128,.15); }}

.hot {{ background:radial-gradient(400px 200px at 100% 0%, rgba(255,255,255,.18), transparent 60%), linear-gradient(140deg,#ff8a3d 0%, #f26b1d 40%, #9a3412 100%);
        border-radius:20px; padding:1.1rem 1.2rem; color:#fff; position:relative; overflow:hidden; box-shadow:0 20px 40px -18px rgba(255,122,26,.55); }}
.hot .k {{ font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; opacity:.85; }}
.hot .v {{ font-size:2.1rem; font-weight:800; letter-spacing:-.03em; margin:.25rem 0; }}
.hot .q {{ font-size:.92rem; font-weight:600; line-height:1.4; margin:.35rem 0 .6rem; }}
.hot .row {{ display:flex; gap:.4rem; flex-wrap:wrap; }}
.hot .chip {{ background:rgba(0,0,0,.22); border:1px solid rgba(255,255,255,.18); color:#fff; font-size:.72rem; font-weight:600; padding:.22rem .55rem; border-radius:999px; }}
.hot .seg {{ font-family:'JetBrains Mono',monospace; font-size:.74rem; background:rgba(0,0,0,.25); padding:.22rem .5rem; border-radius:8px; }}
.hot .ring {{ position:absolute; right:-40px; bottom:-60px; width:180px; height:180px; border-radius:50%; border:28px solid rgba(255,255,255,.08); }}

.list {{ display:flex; flex-direction:column; gap:.15rem; }}
.item {{ display:grid; grid-template-columns:38px 1fr auto; gap:.7rem; align-items:center; padding:.5rem .2rem; border-bottom:1px solid #222226; }}
.item:last-child {{ border-bottom:0; }}
.av {{ width:36px; height:36px; border-radius:11px; display:grid; place-items:center; font-size:15px; background:#232327; }}
.av.o {{ background:rgba(255,122,26,.14); color:{OR}; }}
.item .n {{ font-size:.84rem; font-weight:600; color:#f4f4f5; }} .item .s {{ font-size:.72rem; color:{MUTED}; }}
.item .r {{ font-size:.82rem; font-weight:700; text-align:right; }} .g {{ color:#4ade80; }} .o {{ color:{OR}; }}

.pcard {{ background:{CARD}; border:1px solid {LINE}; border-radius:18px; padding:.95rem 1.05rem .85rem; margin-bottom:.35rem; }}
.pcard .head {{ display:flex; gap:.4rem; align-items:center; flex-wrap:wrap; margin-bottom:.55rem; }}
.rk {{ width:28px; height:28px; border-radius:9px; background:linear-gradient(135deg,{OR},{OR2}); color:#fff; display:grid; place-items:center; font-weight:800; font-size:.8rem; }}
.c {{ font-size:.7rem; font-weight:600; padding:.18rem .52rem; border-radius:999px; background:#232327; color:#d4d4d8; }}
.c.o {{ background:rgba(255,122,26,.14); color:#fdba74; }} .c.g {{ background:rgba(74,222,128,.12); color:#86efac; }} .c.r {{ background:rgba(248,113,113,.12); color:#fca5a5; }}
.sg {{ font-family:'JetBrains Mono',monospace; font-size:.72rem; color:#d4d4d8; background:#111113; border:1px solid {LINE}; padding:.16rem .45rem; border-radius:7px; }}
.ln {{ display:grid; grid-template-columns:92px 1fr; gap:.4rem; font-size:.86rem; color:#e4e4e7; margin:.28rem 0; line-height:1.45; }}
.ln .l {{ color:#71717a; font-size:.66rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; padding-top:.2rem; }}
.why {{ font-size:.79rem; color:#a1a1aa; margin-top:.4rem; }}
.bars {{ display:flex; gap:.8rem; margin-top:.55rem; }} .bar {{ flex:1; }}
.bar .k {{ font-size:.62rem; color:#71717a; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }}
.bar .t {{ height:5px; background:#27272a; border-radius:9px; margin-top:.25rem; overflow:hidden; }}
.bar .f {{ height:100%; background:linear-gradient(90deg,#fdba74,{OR}); border-radius:9px; }}

.timeline {{ display:flex; gap:.35rem; flex-wrap:wrap; align-items:center; margin:.5rem 0 .2rem; }}
.step {{ font-size:.72rem; font-weight:600; padding:.32rem .6rem; border-radius:10px; background:#202024; border:1px solid {LINE}; color:#d4d4d8; }}
.step.o {{ background:rgba(255,122,26,.12); border-color:rgba(255,122,26,.35); color:#fdba74; }}
.arrow {{ color:#52525b; font-size:.8rem; }}
.mem2 {{ display:grid; grid-template-columns:1fr 1fr; gap:.8rem; }}
.memb {{ background:#141416; border:1px solid {LINE}; border-radius:16px; padding:.85rem 1rem; }}
.memb .t {{ font-weight:700; font-size:.9rem; color:#fff; }} .memb .d {{ font-size:.78rem; color:#a1a1aa; margin-top:.25rem; line-height:1.5; }}
.note {{ background:#141416; border-left:3px solid {OR}; border-radius:8px; padding:.45rem .7rem; margin:.3rem 0; font-size:.84rem; color:#e4e4e7; }}
.note .w {{ font-size:.68rem; color:#71717a; }}
.chk {{ display:flex; gap:.55rem; align-items:center; font-size:.86rem; color:#e4e4e7; padding:.38rem 0; border-bottom:1px solid #222226; }}
.hero {{ text-align:left; }}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════ runtime ═════════════════════════════════
@st.cache_resource
def runtime():
    return open_runtime()   # graph + Store, both SQLite-backed (checkpoints = short-term, store = long-term)


graph, store = runtime()
ss = st.session_state
ss.setdefault("thread", None)
ss.setdefault("show_dir", None)


def esc(x) -> str:
    return html.escape(str(x or ""))


def threads() -> list[str]:
    try:
        con = sqlite3.connect(STATE_DIR / "checkpoints.db")
        rows = con.execute("SELECT thread_id, MAX(checkpoint_id) FROM checkpoints WHERE checkpoint_ns='' "
                           "GROUP BY thread_id ORDER BY 2 DESC LIMIT 10").fetchall()
        return [r[0] for r in rows]
    except sqlite3.Error:
        return []


def tinfo(t: str) -> dict:
    parts = t.split("--")
    model = parts[1] if len(parts) > 1 else "run"
    stamp = parts[2] if len(parts) > 2 else ""
    try:
        when = datetime.strptime(stamp, "%Y%m%d-%H%M%S").strftime("%d %b, %H:%M")
    except ValueError:
        when = stamp
    v = graph.get_state({"configurable": {"thread_id": t}})
    status = "review" if v.interrupts else ("done" if v.values.get("outputs") else ("stopped" if v.next else "—"))
    show_short = " ".join(parts[0].split("-")[:3]).title() if parts else ""
    return {"show": show_short, "model": model.replace("openai-", "").replace("google-genai-", "").replace("hybrid-", "hybrid · "), "when": when,
            "promos": len(v.values.get("ranked", []) or []), "status": status,
            "notes": len(v.values.get("notes", []) or [])}


def chart_cfg(c):
    return (c.configure_view(strokeWidth=0).configure(background="transparent")
            .configure_axis(labelColor="#71717a", titleColor="#71717a", gridColor="#232327", domain=False, tickColor="#232327",
                            labelFont="Inter", titleFont="Inter", labelFontSize=11)
            .configure_legend(labelColor="#a1a1aa", titleColor="#a1a1aa"))


EP_FILE = re.compile(r"episode-\d+[^/\\]*\.txt", re.I)


LIB = STATE_DIR / "library"


def show_name(zip_name: str, members: list[str]) -> str:
    """Prefer the folder name inside the zip; else the zip name without '-scripts' / ' (1)' suffixes."""
    parents = {Path(m).parent.name for m in members if Path(m).parent.name}
    if len(parents) == 1:
        return parents.pop()
    return re.sub(r"(-scripts)?(\s*\(\d+\))*$", "", Path(zip_name).stem).strip() or "show"


def add_to_library(upload) -> str | None:
    """Safe ZIP handling (only episode-*.txt, size limits, no archive paths) → saved under the library."""
    payload = upload.getvalue()
    if len(payload) > 20 * 1024 * 1024:
        st.error("ZIP is larger than 20 MB.")
        return None
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        members = [m for m in z.infolist() if not m.is_dir() and EP_FILE.fullmatch(Path(m.filename).name)]
        if not members or len(members) > 300 or sum(m.file_size for m in members) > 40 * 1024 * 1024:
            st.error("Expected up to 300 files named episode-001-….txt (≤40 MB total).")
            return None
        name = show_name(upload.name, [m.filename for m in members])
        key = slug(name)
        dest = LIB / key / name
        shutil.rmtree(LIB / key, ignore_errors=True)        # re-upload of the same show replaces it
        dest.mkdir(parents=True)
        for m in members:
            (dest / Path(m.filename).name).write_bytes(z.read(m))
    (LIB / key / "meta.json").write_text(json.dumps({"name": name, "episodes": len(members),
                                                     "added": time.strftime("%d %b %H:%M"), "zip": upload.name}))
    return str(dest)


def library() -> list[dict]:
    items = []
    if LIB.exists():
        for d in sorted(LIB.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            meta = d / "meta.json"
            if d.is_dir() and meta.exists():
                m = json.loads(meta.read_text())
                items.append({"key": d.name, "name": m["name"], "episodes": m["episodes"], "added": m["added"],
                              "path": str(d / m["name"]), "removable": True})
    local_dir = os.environ.get("SHOW_DIR", "")
    if local_dir and Path(local_dir).exists():
        items.append({"key": "local", "name": Path(local_dir).name, "path": local_dir, "removable": False,
                      "episodes": len(list(Path(local_dir).glob("episode-*.txt"))), "added": "local folder"})
    return items


# ═════════════════════════════════ sidebar ═════════════════════════════════
with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">🎬</div>Promo Finder</div>', unsafe_allow_html=True)
    if st.button("＋  New run", key="new_run", width="stretch"):
        ss.thread = None
        st.rerun()

    ts = threads()
    if ts:
        st.markdown('<div class="side-h">Threads · short-term memory</div>', unsafe_allow_html=True)
        for t in ts:
            i = tinfo(t)
            icon = {"review": "⏸", "done": "●", "stopped": "⚠"}.get(i["status"], "○")
            label = f"{icon}  {i['show']} · {i['model']} · {i['when'][-5:]}"
            if st.button(label, key=("run_active_" if t == ss.thread else "run_") + t, width="stretch"):
                ss.thread = t
                st.rerun()

    st.markdown('<div class="side-h">Upload a show</div>', unsafe_allow_html=True)
    up = st.file_uploader("Screenplay zip", type="zip", label_visibility="collapsed")
    if up is not None and ss.get("last_upload") != (up.name, up.size):
        saved = add_to_library(up)
        ss.last_upload = (up.name, up.size)
        if saved:
            ss.show_dir = saved
            st.toast(f"Saved to library: {Path(saved).name}", icon="📁")

    lib = library()
    if lib:
        st.markdown('<div class="side-h">Library · uploaded shows</div>', unsafe_allow_html=True)
        for item in lib:
            active = item["path"] == ss.show_dir
            label = f"{'📂' if active else '📁'}  {item['name'][:22]}{'…' if len(item['name']) > 22 else ''} · {item['episodes']} ep"
            if st.button(label, key=("show_active_" if active else "show_") + item["key"], width="stretch",
                         help=f"{item['name']} · {item['episodes']} episodes · added {item['added']}"):
                ss.show_dir = item["path"]
                st.rerun()
        cur = next((i for i in lib if i["path"] == ss.show_dir and i["removable"]), None)
        if cur and st.button("🗑  Remove selected show", key="rm_show", width="stretch"):
            shutil.rmtree(LIB / cur["key"], ignore_errors=True)
            ss.show_dir = None
            st.rerun()

    st.markdown('<div class="side-h">Model</div>', unsafe_allow_html=True)
    choice = st.selectbox("Backend", ["heuristic (free, offline)", "hybrid:openai:gpt-4o-mini", "hybrid:google_genai:gemini-2.0-flash",
                                      "openai:gpt-4o-mini", "openai:gpt-4o",
                                      "google_genai:gemini-2.0-flash", "groq:llama-3.3-70b-versatile", "custom…"],
                          label_visibility="collapsed")
    model = "heuristic" if choice.startswith("heuristic") else choice
    if choice == "custom…":
        model = st.text_input("LangChain model string", "anthropic:claude-sonnet-4-5")
    if model != "heuristic":
        env = {"openai": "OPENAI_API_KEY", "google_genai": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY",
               "anthropic": "ANTHROPIC_API_KEY"}.get(model.removeprefix("hybrid:").split(":")[0], "API_KEY")
        key = st.text_input(env, type="password", help="Used for this session only, never stored.")
        if model.startswith("hybrid:"):
            st.caption("Hybrid: free rules search all episodes, the LLM judges only the top 30 (~30 calls).")
        if key:
            os.environ[env] = key
            llm.get_model.cache_clear()
    top_n = st.slider("Promos in posting list", 5, 30, 15)
    review = st.toggle("Editor review before export", value=True)

    item = store.get(("editor_prefs",), "hook_type_bonus")
    if item and item.value:
        st.markdown('<div class="side-h">Long-term memory · learned</div>', unsafe_allow_html=True)
        st.markdown("".join(f'<div class="pref"><span>{k.replace("_", " ")}</span><b class="{"pos" if v > 0 else "neg"}">{v:+.1f}</b></div>'
                            for k, v in sorted(item.value.items(), key=lambda x: -x[1])), unsafe_allow_html=True)


# ═════════════════════════════════ helpers ═════════════════════════════════


def data_dir() -> str | None:
    if ss.get("show_dir") and Path(ss.show_dir).exists():
        return ss.show_dir
    lib_ = library()
    return lib_[0]["path"] if lib_ else None


def stream(inp, cfg, label="Reading episodes…"):
    with st.status(label, expanded=False) as box:
        n = 0
        for chunk in graph.stream(inp, {**cfg, "max_concurrency": 8}, stream_mode="updates"):
            for node, upd in chunk.items():
                if isinstance(upd, dict):
                    for line in upd.get("log", []):
                        if node == "episode":
                            n += 1
                            box.update(label=f"Finding promos… {n} episodes done")
                        else:
                            box.write(f"**{node}** · {line}")
        box.update(label="Done", state="complete")


def start_run(m: str):
    d = data_dir()
    if not d:
        st.toast("Upload a screenplay zip (or pick a show from the Library) first.", icon="📁")
        return
    show = Path(d).name
    ss.thread = f"{slug(show)}--{slug(m)}--{time.strftime('%Y%m%d-%H%M%S')}"
    cfg = {"configurable": {"thread_id": ss.thread}}
    try:
        stream({"data_dir": d, "show": show, "model": m, "review": review, "top_n": top_n,
                "out_dir": str(Path("outputs") / slug(show) / slug(m)), "candidates": [], "log": [], "notes": []}, cfg)
    except Exception as e:  # noqa: BLE001
        st.error(f"Run stopped: {e}")
    st.rerun()


def promo_card(r: dict, status: bool = False) -> str:
    segs = "".join(f'<span class="sg">Ep{s["episode"]} {s["start_ts"]}–{s["end_ts"]}</span>' for s in r["segments"])
    sc = r.get("scores", {})
    bars = "".join(f'<div class="bar"><div class="k">{k}</div><div class="t"><div class="f" style="width:{float(sc.get(key, 0)) / 5 * 100:.0f}%"></div></div></div>'
                   for k, key in (("hook", "hook"), ("cliffhanger", "cliff"), ("standalone", "standalone")))
    st_chip = ""
    if status and r.get("already_posted"):
        st_chip = '<span class="c g">posted</span>'
    elif status and r.get("editor_rejected"):
        st_chip = '<span class="c r">rejected</span>'
    critic = f'<div class="why">🧠 {esc(r["critic_note"])}</div>' if r.get("critic_note") else ""
    return (f'<div class="pcard"><div class="head"><div class="rk">{r["rank"]}</div><span class="c o">{r["score"]:.0f} pts</span>'
            f'<span class="c">{r["duration"]:.0f}s</span><span class="c">{esc(r["hook_type"]).replace("_", " ")}</span>'
            f'<span class="c">{r["source"].replace("_", " ")}</span>{st_chip}</div><div class="head">{segs}</div>'
            f'<div class="ln"><span class="l">Hook</span>{esc(r["hook"])}</div>'
            f'<div class="ln"><span class="l">Cliffhanger</span>{esc(r["cliffhanger"])}</div>'
            f'<div class="why">💡 {esc(r["why"])}</div>{critic}<div class="bars">{bars}</div></div>')


def anatomy_chart(r: dict, ep_map: dict):
    """The whole episode as a strip: script lines in grey, the cut in orange, hook and cliffhanger marked."""
    rows, marks = [], []
    for sg in r["segments"]:
        ep = ep_map.get(sg["episode"])
        if not ep:
            continue
        lab = f"Ep {sg['episode']}"
        for b in ep["beats"]:
            if b["type"] in ("dialogue", "action"):
                rows.append({"ep": lab, "a": b["start"], "b": b["end"], "kind": b["type"], "text": b["text"][:80]})
        marks.append({"ep": lab, "a": sg["start"], "b": sg["end"], "what": "cut"})
        marks.append({"ep": lab, "a": sg["start"], "b": min(sg["start"] + 3, sg["end"]), "what": "first 3 s"})
        for bid, name in ((r["hook_beat"], "hook"), (r["cliffhanger_beat"], "cliffhanger")):
            hit = next((b for b in ep["beats"] if b["id"] == bid), None)
            if hit:
                marks.append({"ep": lab, "a": hit["start"] if name == "hook" else hit["end"], "b": None, "what": name})
    lines = alt.Chart(pd.DataFrame(rows)).mark_bar(height=10, opacity=.55).encode(
        x=alt.X("a:Q", title="seconds into episode"), x2="b:Q", y=alt.Y("ep:N", title=None),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["dialogue", "action"], range=["#71717a", "#3f3f46"]),
                        legend=alt.Legend(orient="top", title=None)), tooltip=["kind", "text"])
    md = pd.DataFrame(marks)
    cut = alt.Chart(md[md.what == "cut"]).mark_bar(height=26, color=OR, opacity=.28, cornerRadius=4).encode(x="a:Q", x2="b:Q", y="ep:N")
    w3 = alt.Chart(md[md.what == "first 3 s"]).mark_bar(height=26, color=OR, opacity=.65, cornerRadius=4).encode(
        x="a:Q", x2="b:Q", y="ep:N", tooltip=alt.value("hook window: first 3 seconds"))
    pts = md[md.what.isin(["hook", "cliffhanger"])]
    pin = alt.Chart(pts).mark_point(shape="triangle-down", filled=True, size=140, color="#fff", yOffset=-17).encode(
        x="a:Q", y="ep:N", tooltip="what:N")
    lab = alt.Chart(pts).mark_text(dy=-30, color="#fdba74", fontSize=10, fontWeight="bold", font="Inter").encode(
        x="a:Q", y="ep:N", text="what:N")
    return chart_cfg((lines + cut + w3 + pin + lab).properties(height=max(1, len(r["segments"])) * 70 + 40))


def why_chart(r: dict):
    """Waterfall: each term of the score, adding up to the final number."""
    parts = r.get("breakdown") or {}
    if not parts:
        return None
    rows, run = [], 0.0
    for k, val in parts.items():
        rows.append({"term": k, "a": run, "b": run + val, "v": val, "kind": "up" if val >= 0 else "down"})
        run += val
    rows.append({"term": "= score", "a": 0, "b": run, "v": run, "kind": "total"})
    df_ = pd.DataFrame(rows)
    order = list(df_["term"])
    bars = alt.Chart(df_).mark_bar(cornerRadius=5, size=26).encode(
        x=alt.X("term:N", sort=order, title=None, axis=alt.Axis(labelAngle=-22, labelLimit=120, labelFontSize=10, labelOverlap=False)),
        y=alt.Y("a:Q", title=None), y2="b:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["up", "down", "total"], range=[OR, "#ef4444", "#f4f4f5"]), legend=None),
        tooltip=["term", alt.Tooltip("v:Q", format="+.1f")])
    txt = alt.Chart(df_).mark_text(dy=-8, color="#e4e4e7", fontSize=10, fontWeight="bold", font="Inter").encode(
        x=alt.X("term:N", sort=order), y="b:Q", text=alt.Text("v:Q", format="+.1f"))
    return chart_cfg((bars + txt).properties(height=190))


def details(r: dict, ep_map: dict, key: str):
    with st.expander("🔬 Clip anatomy · why this rank · transcript"):
        st.markdown('<div class="muted">The cut (orange) on the episode timeline. Bright orange = the 3-second hook window; '
                    '▼ marks where the hook and the cliffhanger land.</div>', unsafe_allow_html=True)
        st.altair_chart(anatomy_chart(r, ep_map), width="stretch", key=f"an_{key}")
        wc = why_chart(r)
        if wc is not None:
            st.markdown('<div class="muted">How the score adds up (each bar is one term).</div>', unsafe_allow_html=True)
            st.altair_chart(wc, width="stretch", key=f"wy_{key}")
        st.code(r["transcript"], language=None)


# ═════════════════════════════════ welcome ═════════════════════════════════
if not ss.thread:
    st.markdown('<div class="topbar"><div class="h">Dashboard<small>Find 20–60 s promo clips in a microdrama’s screenplays</small></div>'
                '<div><span class="pill"><span class="dot"></span>Ready</span></div></div>', unsafe_allow_html=True)
    a, b = st.columns([1.5, 1], gap="medium")
    with a:
        st.markdown(f"""<div class="hot" style="min-height:250px"><div class="ring"></div>
          <div class="k">✦ Kuku · Promo Finder</div>
          <div class="v" style="font-size:2.3rem;line-height:1.1;margin-top:.6rem">Find the moment.<br>Make them watch.</div>
          <div class="q" style="max-width:520px;font-weight:500;opacity:.92">Upload a show's screenplays. Get ranked promo clips that open on a hook
          and cut on a cliffhanger, with exact timestamps. An editor approves, and the tool remembers.</div>
          <div class="row"><span class="chip">⚡ Hook in 3 s</span><span class="chip">◷ 20–60 s</span><span class="chip">🧵 Threads & memory</span>
          <span class="chip">👤 Editor in the loop</span></div></div>""", unsafe_allow_html=True)
    with b:
        with st.container(key="card_start"):
            st.markdown('<div class="ttl">Start a run</div><div class="muted" style="margin:.2rem 0 .8rem">Each run gets its own '
                        '<b>thread_id</b>: its own short-term memory you can come back to.</div>', unsafe_allow_html=True)
            if st.button("⚡  Run with the free model", type="primary", width="stretch"):
                start_run("heuristic")
            if st.button(f"✦  Run with {model if model != 'heuristic' else 'an LLM (pick in sidebar)'}", width="stretch"):
                if model == "heuristic":
                    st.toast("Choose an LLM backend in the sidebar and paste its key.", icon="✦")
                else:
                    start_run(model)
            st.markdown('<div class="muted" style="margin-top:.7rem">🔒 Upload only scripts you are allowed to process. '
                        'Nothing is stored in the repo.</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3, gap="medium")
    for col, (ic, t, d) in zip((c1, c2, c3), [
        ("📜", "1 · Parse", "3,700+ timestamped lines from 50 scripts. The model never writes a timestamp."),
        ("🎯", "2 · Find & check", "Hooks and cliffhangers per episode, rule-checked in code, retried on failure."),
        ("🧠", "3 · Rank & remember", "Editor approves; long-term memory reshapes the next ranking.")]):
        with col:
            with st.container(key=f"card_step_{t[0]}"):
                st.markdown(f'<div class="av o" style="margin-bottom:.6rem">{ic}</div><div class="ttl">{t}</div>'
                            f'<div class="muted" style="margin-top:.3rem">{d}</div>', unsafe_allow_html=True)
    st.stop()


# ═════════════════════════════════ a run (thread) ═════════════════════════════════
cfg = {"configurable": {"thread_id": ss.thread}}
snap = graph.get_state(cfg)
v = snap.values
ranked = v.get("ranked", []) or []
info = tinfo(ss.thread)

top_l, top_r = st.columns([1.4, 1])
with top_l:
    st.markdown(f'<div class="topbar"><div class="h">{esc(v.get("show", "Run"))}<small>Promo dashboard</small></div></div>',
                unsafe_allow_html=True)
with top_r:
    st.markdown(f'<div style="text-align:right;margin-top:.35rem"><span class="pill">🧵 <span class="mono" style="color:#e4e4e7">'
                f'{esc(ss.thread[-28:])}</span></span><span class="pill"><span class="dot"></span>{esc(info["status"])}</span></div>',
                unsafe_allow_html=True)

if snap.next and not snap.interrupts:
    st.warning("This run stopped before finishing. Finished episodes are saved in the thread's checkpoint.")
    if st.button("↻ Resume from checkpoint", type="primary"):
        stream(None, cfg)
        st.rerun()
    st.stop()
if not ranked:
    st.info("No results in this thread yet.")
    st.stop()

# ── row 1: area chart + top pick ──
eps = [e["number"] for e in v.get("episodes", [])]
EPM = {e["number"]: e for e in v.get("episodes", [])}
best = {n: 0.0 for n in eps}
count = {n: 0 for n in eps}
for r in ranked:
    e = r["segments"][0]["episode"]
    best[e] = max(best.get(e, 0), r["score"])
    count[e] = count.get(e, 0) + 1
df = pd.DataFrame({"episode": list(best), "best": list(best.values()), "count": [count[n] for n in best]})
avg = statistics.mean([x for x in df["best"] if x > 0] or [0])
top = ranked[0]

r1a, r1b = st.columns([2.05, 1], gap="medium")
with r1a:
    with st.container(key="card_chart"):
        h1, h2 = st.columns([1.4, 1])
        h1.markdown(f'<div class="ttl">Promo strength by episode</div><div class="big">{top["score"]:.0f}'
                    f'<span class="up">↑ {top["score"] - avg:.0f} vs avg</span></div>'
                    f'<div class="muted">Best promo score in each of {len(eps)} episodes</div>', unsafe_allow_html=True)
        mode = h2.segmented_control("View", ["Best score", "Promos found"], default="Best score", label_visibility="collapsed",
                                    key="chart_mode")
        y = "best" if mode != "Promos found" else "count"
        base = alt.Chart(df).encode(x=alt.X("episode:Q", title=None, scale=alt.Scale(domain=[1, max(eps or [1])]),
                                            axis=alt.Axis(values=[1] + list(range(5, max(eps or [5]) + 1, 5)), format="d", labelExpr="'Ep ' + datum.label")))
        area = base.mark_area(line={"color": OR, "strokeWidth": 2.2}, interpolate="monotone",
                              color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                                                 stops=[alt.GradientStop(color="rgba(255,122,26,0)", offset=0),
                                                        alt.GradientStop(color="rgba(255,122,26,.55)", offset=1)])
                              ).encode(y=alt.Y(f"{y}:Q", title=None), tooltip=["episode", alt.Tooltip(f"{y}:Q", format=".0f")])
        rule = alt.Chart(pd.DataFrame({"y": [avg if y == "best" else df["count"].mean()]})).mark_rule(
            color="#52525b", strokeDash=[4, 4]).encode(y="y:Q")
        peak = df.loc[df[y].idxmax()]
        dot = alt.Chart(pd.DataFrame([peak])).mark_point(filled=True, size=90, color="#fff", stroke=OR, strokeWidth=3).encode(
            x="episode:Q", y=f"{y}:Q")
        lbl = alt.Chart(pd.DataFrame([{**peak, "t": f"Ep {int(peak['episode'])} · {peak[y]:.0f}"}])).mark_text(
            dy=-14, dx=8, align="left", color="#fff", fontWeight="bold", fontSize=11, font="Inter").encode(x="episode:Q", y=f"{y}:Q", text="t:N")
        st.altair_chart(chart_cfg((area + rule + dot + lbl).properties(height=235)), width="stretch")
with r1b:
    segs = "".join(f'<span class="seg">Ep{s["episode"]} {s["start_ts"]}–{s["end_ts"]}</span> ' for s in top["segments"])
    st.markdown(f"""<div class="hot" style="min-height:338px"><div class="ring"></div>
      <div class="k">#1 · post this first</div><div class="v">{top['score']:.0f}<span style="font-size:1rem;opacity:.8"> / 100</span></div>
      <div class="row">{segs}</div>
      <div class="q">🪝 {esc(top['hook'])}</div>
      <div style="font-size:.84rem;opacity:.92;margin-bottom:.7rem">⏸ {esc(top['cliffhanger'])}</div>
      <div class="row"><span class="chip">{top['duration']:.0f} s</span><span class="chip">{esc(top['hook_type']).replace('_', ' ')}</span>
      <span class="chip">{top['source'].replace('_', ' ')}</span></div></div>""", unsafe_allow_html=True)

st.write("")
# ── row 2: review CTA / types bar / threads list ──
r2a, r2b, r2c = st.columns([1, 1.15, 1.15], gap="medium")
with r2a:
    if snap.interrupts:
        st.markdown(f"""<div class="hot" style="min-height:228px"><div class="ring"></div><div class="k">Editor in the loop</div>
          <div class="v" style="font-size:1.45rem;line-height:1.2">{len(snap.interrupts[0].value['promos'])} promos waiting<br>for your call</div>
          <div class="q" style="font-weight:500;font-size:.84rem;opacity:.92">The run is paused at <b>interrupt()</b>. Post or reject in the
          <b>Review</b> tab; your decisions go to long-term memory.</div></div>""", unsafe_allow_html=True)
    else:
        posted = len(store.search(("posted", slug(v.get("show", ""))), limit=1000))
        st.markdown(f"""<div class="hot" style="min-height:228px"><div class="ring"></div><div class="k">Posting list ready</div>
          <div class="v">{len(v.get('approved', []))}</div><div class="q" style="font-weight:500;font-size:.84rem;opacity:.92">promos approved in
          this thread · {posted} clips remembered as posted for this show</div></div>""", unsafe_allow_html=True)
with r2b:
    with st.container(key="card_types"):
        tdf = pd.DataFrame(ranked).groupby("hook_type").size().reset_index(name="n").sort_values("n", ascending=False)
        tdf["hook_type"] = tdf["hook_type"].str.replace("_", " ")
        st.markdown(f'<div class="ttl">Hook types<span class="muted">{len(ranked)} promos</span></div>', unsafe_allow_html=True)
        mx = tdf["n"].max()
        bars = alt.Chart(tdf).mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7, size=26).encode(
            x=alt.X("hook_type:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-28, labelLimit=90, labelOverlap=False, labelFontSize=10)),
            y=alt.Y("n:Q", title=None, axis=alt.Axis(tickCount=4)),
            color=alt.condition(alt.datum.n == mx, alt.value(OR), alt.value("#3a2a20")), tooltip=["hook_type", "n"])
        txt = bars.mark_text(dy=-8, fontSize=11, font="Inter", fontWeight="bold").encode(text="n:Q", color=alt.value("#e4e4e7"))
        st.altair_chart(chart_cfg((bars + txt).properties(height=178)), width="stretch")
with r2c:
    with st.container(key="card_threads"):
        st.markdown('<div class="ttl">Recent threads<span class="muted">short-term memory</span></div>', unsafe_allow_html=True)
        rows = ""
        for t in ts[:4]:
            i = tinfo(t)
            cur = " o" if t == ss.thread else ""
            rows += (f'<div class="item"><div class="av{cur}">{"⚡" if i["model"] == "heuristic" else "✦"}</div>'
                     f'<div><div class="n">{esc(i["show"])} · {esc(i["model"])}{" · this run" if t == ss.thread else ""}</div>'
                     f'<div class="s">{esc(i["when"])} · {i["status"]}{" · " + str(i["notes"]) + " notes" if i["notes"] else ""}</div></div>'
                     f'<div class="r {"o" if t == ss.thread else "g"}">{i["promos"]}</div></div>')
        st.markdown(f'<div class="list">{rows}</div>', unsafe_allow_html=True)

st.write("")
q = st.text_input("Search", placeholder="🔍  Search promos: “slap”, “poison”, “ep 21”, “betrayal”…", label_visibility="collapsed")


def match(r: dict) -> bool:
    if not q.strip():
        return True
    hay = " ".join([r["hook"], r["cliffhanger"], r["why"], r["hook_type"], r["source"],
                    " ".join(f"ep {s['episode']} ep{s['episode']}" for s in r["segments"])]).lower()
    return all(w in hay for w in q.lower().split())


tabs = (["⏸  Review"] if snap.interrupts else []) + ["✨  Promos", "🧵  Short-term memory", "🧠  Long-term memory",
                                                       "✅  Quality audit", "⚙  Pipeline"]
T = dict(zip(tabs, st.tabs(tabs)))

# ── review (interrupt) ──
if snap.interrupts:
    with T["⏸  Review"]:
        payload = snap.interrupts[0].value
        by_id = {r["id"]: r for r in ranked}
        with st.form("review", border=False):
            decisions = {}
            cols = st.columns(2, gap="medium")
            for i, p in enumerate(payload["promos"]):
                r = by_id[p["id"]]
                with cols[i % 2]:
                    st.markdown(promo_card(r), unsafe_allow_html=True)
                    decisions[p["id"]] = st.segmented_control("Decision", ["post", "reject"], default="post", key=p["id"],
                                                              label_visibility="collapsed")
                    details(r, EPM, f"rv_{p['id']}")
            if st.form_submit_button("Submit review → save to long-term memory", type="primary"):
                stream(Command(resume={k: "r" if d == "reject" else "a" for k, d in decisions.items()}), cfg, "Saving decisions…")
                st.rerun()

# ── promos ──
with T["✨  Promos"]:
    shown = [r for r in (v.get("approved") or ranked[:top_n]) if match(r)]
    st.caption(f"{'Posting list' if v.get('approved') else 'Top candidates (awaiting review)'} · {len(shown)} shown"
               + (f" · filter “{q}”" if q.strip() else ""))
    cols = st.columns(2, gap="medium")
    for i, r in enumerate(shown):
        with cols[i % 2]:
            st.markdown(promo_card(r, status=True), unsafe_allow_html=True)
            details(r, EPM, f"pr_{r['id']}")
    with st.expander(f"All {len(ranked)} candidates"):
        st.dataframe(pd.DataFrame([{"rank": r["rank"], "score": r["score"], "dur": round(r["duration"]),
                                    "clip": " + ".join(f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in r["segments"]),
                                    "hook": r["hook"], "type": r["hook_type"], "posted": r.get("already_posted"),
                                    "rejected": r.get("editor_rejected")} for r in ranked if match(r)]),
                     hide_index=True, width="stretch")
    if v.get("approved"):
        rows = [{"rank": r["rank"], "score": r["score"], "duration_s": round(r["duration"]),
                 "segments": " + ".join(f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in r["segments"]),
                 "hook": r["hook"], "cliffhanger": r["cliffhanger"], "why": r["why"]} for r in v["approved"]]
        d1, d2, d3, _ = st.columns([1, 1, 1.5, 1.5])
        d1.download_button("⬇ CSV", pd.DataFrame(rows).to_csv(index=False), "promos_top.csv", width="stretch")
        d2.download_button("⬇ JSON", json.dumps(v["approved"], indent=2), "promos_top.json", width="stretch")
        d3.download_button("🎬 Cut sheet (ffmpeg)", cut_sheet(v["approved"]), "cut_sheet.sh", width="stretch",
                           help="Run next to episode-001.mp4, episode-002.mp4… to cut every promo, joining multi-part ones.")

# ── short-term memory: this thread ──
with T["🧵  Short-term memory"]:
    st.markdown(f"""<div class="mem2">
      <div class="memb"><div class="t">🧵 Short-term memory = this thread</div><div class="d">Identified by <span class="mono">thread_id</span>.
      Every step of the run is checkpointed. Switch threads in the sidebar and you get that run back exactly as it was: results,
      pause point, notes. Crash at episode 17? Resume continues from the last checkpoint.</div></div>
      <div class="memb"><div class="t">🧠 Long-term memory = the Store</div><div class="d">Shared across every thread and every show.
      Holds what the editor taught the tool: posted clips, rejected clips, preferred hook types. Every new thread reads it when ranking.</div></div>
    </div>""", unsafe_allow_html=True)

    hist = list(graph.get_state_history(cfg))
    st.markdown(f'<div class="ttl" style="margin-top:1.1rem">Checkpoints in this thread<span class="muted">{len(hist)} saved states · '
                f'<span class="mono">{esc(ss.thread)}</span></span></div>', unsafe_allow_html=True)
    steps = []
    for hh in reversed(hist):
        nxt = hh.next
        label = (f"episode ×{len(nxt)}" if len(nxt) > 1 else (nxt[0] if nxt else "end")).replace("__start__", "input")
        steps.append(f'<span class="step{" o" if label in ("review", "end") else ""}">#{hh.metadata.get("step")} {esc(label)}</span>')
    st.markdown('<div class="timeline">' + '<span class="arrow">→</span>'.join(steps[-14:]) + "</div>", unsafe_allow_html=True)

    a, b = st.columns(2, gap="medium")
    with a:
        with st.container(key="card_notes"):
            notes = v.get("notes", []) or []
            st.markdown(f'<div class="ttl">Thread notes<span class="muted">{len(notes)} · only in this thread</span></div>'
                        '<div class="muted" style="margin:.2rem 0 .5rem">Written into the thread\'s state with '
                        '<span class="mono">update_state()</span>. Open another thread: these disappear. Come back: they return.</div>',
                        unsafe_allow_html=True)
            with st.form("note", clear_on_submit=True, border=False):
                txt = st.text_input("Note", placeholder="e.g. #3 needs a tighter cut at 01:26", label_visibility="collapsed")
                if st.form_submit_button("Add note to thread") and txt.strip():
                    note = {"notes": [{"text": txt.strip(), "at": time.strftime("%d %b %H:%M")}]}
                    if snap.interrupts:
                        # writing during a paused review creates a new checkpoint before 'review';
                        # replay review so the run pauses again at the same interrupt (it is deterministic)
                        graph.update_state(cfg, note, as_node="rank")
                        graph.invoke(None, cfg)
                    else:
                        graph.update_state(cfg, note, as_node="export")
                    st.rerun()
            for n in reversed(notes[-6:]):
                st.markdown(f'<div class="note">{esc(n["text"])}<div class="w">{esc(n["at"])}</div></div>', unsafe_allow_html=True)
    with b:
        with st.container(key="card_compare"):
            st.markdown('<div class="ttl">Compare with another thread</div><div class="muted" style="margin:.2rem 0 .5rem">'
                        'Each thread remembers its own ranking. Pick one to see what changed (e.g. before vs after review).</div>',
                        unsafe_allow_html=True)
            others = [t for t in ts if t != ss.thread]
            if others:
                o = st.selectbox("Other thread", others, format_func=lambda t: f"{tinfo(t)['model']} · {tinfo(t)['when']}",
                                 label_visibility="collapsed")
                ov = graph.get_state({"configurable": {"thread_id": o}}).values
                mine = {r["signature"]: r for r in (v.get("approved") or ranked[:top_n])}
                theirs = {r["signature"]: r for r in (ov.get("approved") or (ov.get("ranked") or [])[:top_n])}
                new, gone = [mine[k] for k in mine if k not in theirs], [theirs[k] for k in theirs if k not in mine]
                st.markdown(f'<div class="list">'
                            f'<div class="item"><div class="av o">＋</div><div><div class="n">Only in this thread</div>'
                            f'<div class="s">{esc(", ".join(r["id"] for r in new[:6]) or "none")}</div></div><div class="r o">{len(new)}</div></div>'
                            f'<div class="item"><div class="av">－</div><div><div class="n">Only in the other</div>'
                            f'<div class="s">{esc(", ".join(r["id"] for r in gone[:6]) or "none")}</div></div><div class="r">{len(gone)}</div></div>'
                            f'<div class="item"><div class="av">＝</div><div><div class="n">In both</div><div class="s">same clip, same cut</div></div>'
                            f'<div class="r g">{len(set(mine) & set(theirs))}</div></div></div>', unsafe_allow_html=True)
            else:
                st.caption("Run again to create a second thread.")

    before_rank = next((hh for hh in hist if hh.next == ("rank",)), None)
    if before_rank:
        c1, c2 = st.columns([1.3, 3])
        with c1:
            if st.button("⏪  Time travel: re-rank from checkpoint", type="primary", width="stretch"):
                stream(None, before_rank.config, "Replaying rank → review → export…")
                st.rerun()
        c2.caption("Forks this thread from the checkpoint just before ‘rank’ and replays only the last steps with today's "
                   "long-term memory. The 50 episodes are not re-processed.")

# ── long-term memory ──
with T["🧠  Long-term memory"]:
    spaces = store.list_namespaces()
    lab = {"editor_prefs": ("⭐", "Learned preferences"), "feedback": ("🗳", "Editor decisions"),
           "posted": ("✅", "Posted clips"), "rejected": ("⛔", "Rejected clips")}
    cols = st.columns(max(1, len(spaces)), gap="medium")
    for col, ns in zip(cols, spaces):
        ic, name = lab.get(ns[0], ("•", ns[0]))
        with col:
            with st.container(key=f"card_ns_{ns[0]}"):
                st.markdown(f'<div class="av o" style="margin-bottom:.5rem">{ic}</div><div class="muted">{name}</div>'
                            f'<div class="big">{len(store.search(ns, limit=1000))}</div><div class="mono">{esc("/".join(ns))[:30]}</div>',
                            unsafe_allow_html=True)
    if not spaces:
        st.info("Empty. Post or reject promos in a review: the decisions are written here and shape every future thread.")
    item = store.get(("editor_prefs",), "hook_type_bonus")
    if item and item.value:
        pdf = pd.DataFrame([{"type": k.replace("_", " "), "bonus": v_} for k, v_ in item.value.items()])
        st.markdown('<div class="ttl" style="margin-top:1rem">Ranking bonus learned per hook type</div>', unsafe_allow_html=True)
        ch = alt.Chart(pdf).mark_bar(cornerRadius=6, size=18).encode(
            y=alt.Y("type:N", sort="-x", title=None), x=alt.X("bonus:Q", title="points added to score", axis=alt.Axis(tickCount=6)),
            color=alt.condition(alt.datum.bonus > 0, alt.value(OR), alt.value("#7f1d1d")), tooltip=["type", "bonus"])
        st.altair_chart(chart_cfg(ch.properties(height=34 * len(pdf) + 20)), width="stretch")
    for ns in spaces:
        with st.expander(f"{lab.get(ns[0], ('', ns[0]))[1]} · {'/'.join(ns)}"):
            st.dataframe(pd.DataFrame([{"key": i.key, **(i.value if isinstance(i.value, dict) else {"value": i.value})}
                                       for i in store.search(ns, limit=200)]), hide_index=True, width="stretch")

# ── quality audit ──
with T["✅  Quality audit"]:
    au = v.get("audit") or quality_audit(v.get("episodes", []), ranked)
    checks = [("Every promo is 20–60 s", au["all_promos_20_to_60s"]),
              ("No invalid timestamps in the scripts", au["invalid_beat_bounds"] == 0),
              ("No episode parsed to zero lines", not au["episodes_with_zero_beats"]),
              ("No missing episode numbers", not au["missing_episode_numbers"]),
              (f"Every episode has a promo (missing: {', '.join(map(str, au['episodes_without_promos'])) or 'none'})",
               not au["episodes_without_promos"])]
    a, b = st.columns([1, 1.6], gap="medium")
    with a:
        with st.container(key="card_checks"):
            st.markdown(f'<div class="ttl">Checks<span class="muted">{au["episodes"]} episodes · {au["beats"]:,} lines</span></div>'
                        + "".join(f'<div class="chk">{"✅" if ok else "⚠️"} {esc(t)}</div>' for t, ok in checks)
                        + '<div class="muted" style="margin-top:.7rem">' + "<br>".join("• " + esc(x) for x in au["limitations"]) + "</div>",
                        unsafe_allow_html=True)
    with b:
        with st.container(key="card_audit"):
            pe = pd.DataFrame(au["per_episode"])
            st.markdown('<div class="ttl">Lines per episode<span class="muted">dialogue vs action</span></div>', unsafe_allow_html=True)
            long = pe.melt(id_vars="episode", value_vars=["dialogue", "action"], var_name="kind", value_name="lines")
            ch = alt.Chart(long).mark_bar(size=7).encode(
                x=alt.X("episode:O", title=None, axis=alt.Axis(labelAngle=0, values=list(range(5, 51, 5)))),
                y=alt.Y("lines:Q", title=None, stack=True),
                color=alt.Color("kind:N", scale=alt.Scale(range=[OR, "#52525b"]), legend=alt.Legend(orient="top", title=None)),
                tooltip=["episode", "kind", "lines"])
            st.altair_chart(chart_cfg(ch.properties(height=220)), width="stretch")

# ── pipeline ──
with T["⚙  Pipeline"]:
    sub = build_episode_graph().get_graph()
    N = ('node [shape=box, style="rounded,filled", fillcolor="#1c1c20", color="#3f3f46", fontcolor="#e4e4e7", '
         'fontname="Inter", fontsize=11];')
    dot = ['digraph G { rankdir=LR; bgcolor="transparent"; ' + N + ' edge [color="#71717a", arrowsize=.6, fontcolor="#a1a1aa", fontsize=9];',
           'start [label="", shape=circle, width=.18, fillcolor="#ff7a1a", color="#ff7a1a"];',
           'end [label="", shape=doublecircle, width=.14, fillcolor="#ff7a1a", color="#ff7a1a"];',
           'subgraph cluster_ep { label="episode subgraph ×50 (Send)"; fontname="Inter"; fontsize=10; fontcolor="#fdba74"; '
           'style="rounded,filled"; fillcolor="#1f150e"; color="#7c2d12";']
    for e in sub.edges:
        if "__" in e.source or "__" in e.target:
            continue
        lab_ = ("retry" if e.target == "propose" else "valid") if e.conditional else ""
        dot.append(f'"ep_{e.source}" -> "ep_{e.target}" [{"style=dashed, label=" + chr(34) + lab_ + chr(34) if lab_ else ""}];')
    for n_ in ("propose", "validate", "critique"):
        dot.append(f'"ep_{n_}" [label="{n_}", fillcolor="#2a1a10", color="#c2410c", fontcolor="#fdba74"];')
    dot += ["}", 'start -> load;', 'load -> "ep_propose" [style=dashed, label="Send"];', '"ep_critique" -> stitch;',
            'stitch -> rank;', 'stitch -> llm_judge [style=dashed, label="hybrid: Send ×30"];', 'llm_judge -> rank;',
            'llm_judge [label="llm_judge\n(LLM, top 30)", fillcolor="#2a1a10", color="#c2410c", fontcolor="#fdba74"];', 'rank [label="rank\\n(long-term Store)"];', 'rank -> review;',
            'review [label="review\\n(interrupt)", fillcolor="#ff7a1a", color="#ff7a1a", fontcolor="#ffffff"];',
            'review -> export;', 'export -> end;', "}"]
    st.graphviz_chart("\n".join(dot), width="stretch")
    st.markdown("".join(f'<span class="c o" style="margin:.15rem .2rem;display:inline-block">{c}</span>' for c in
                        ["hybrid LLM judge ×30", "thread_id + checkpointer", "short-term memory", "long-term Store", "time travel", "update_state notes",
                         "interrupt() review", "Send ×50", "retry loop", "RetryPolicy", "CachePolicy", "streaming", "swappable model"]),
                unsafe_allow_html=True)
    with st.expander("Run log"):
        st.code("\n".join(v.get("log", [])), language=None)
