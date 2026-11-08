"""Deterministic screenplay parser. No AI here: timestamps always come from the script."""
from __future__ import annotations

import re
from pathlib import Path

from .schemas import Beat, Episode

TS = r"\((\d+):(\d+(?:\.\d+)?)\)"
TIME_RE = re.compile(rf"Start:\s*{TS}\s*\|\s*End:\s*{TS}")
HEAD_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)")
EP_NUM_RE = re.compile(r"episode-(\d+)", re.I)


def _secs(m: str, s: str) -> float:
    return int(m) * 60 + float(s)


def _classify(head: str) -> tuple[str, str | None, str | None]:
    """Return (type, speaker, emotion) for a block header like 'LADY JULIANA (Gasping, Devastated)'."""
    h = head.strip()
    if h in {"CAMERA", "ACTION", "SOUND FX"}:
        return {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx"}[h], None, None
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", h)
    if m:
        return "dialogue", m.group(1).strip(), m.group(2).strip()
    return "dialogue", h, None


def parse_episode(path: str | Path) -> Episode:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = {"EPISODE NUMBER": "", "EPISODE LOGLINE": "", "CONTINUITY NOTES": ""}
    for ln in lines[:10]:
        for k in meta:
            if ln.startswith(k + ":"):
                meta[k] = ln.split(":", 1)[1].strip()

    m = EP_NUM_RE.search(path.name)
    ep_no = int(m.group(1)) if m else int(re.sub(r"\D", "", meta["EPISODE NUMBER"]) or 0)

    beats: list[Beat] = []
    scene, scene_heading = 0, ""
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        sm = SCENE_RE.match(ln)
        if sm:
            scene = int(sm.group(1))
            if i + 1 < len(lines) and lines[i + 1].startswith("Scene heading:"):
                scene_heading = lines[i + 1].split(":", 1)[1].strip()
        hm = HEAD_RE.match(ln)
        if hm and i + 1 < len(lines):
            tm = TIME_RE.search(lines[i + 1])
            if tm:
                btype, speaker, emotion = _classify(hm.group(1))
                text_lines = []
                j = i + 2
                while j < len(lines) and lines[j].strip() and not HEAD_RE.match(lines[j].strip()):
                    text_lines.append(lines[j].strip())
                    j += 1
                text = " ".join(text_lines).strip().strip('"')
                beats.append(Beat(
                    id=f"E{ep_no:02d}-B{len(beats):03d}",
                    episode=ep_no, scene=scene, scene_heading=scene_heading,
                    type=btype, speaker=speaker, emotion=emotion,
                    start=_secs(tm.group(1), tm.group(2)), end=_secs(tm.group(3), tm.group(4)),
                    text=text,
                ))
                i = j
                continue
        i += 1

    beats.sort(key=lambda b: (b.start, b.end))
    # re-number after sort so IDs follow screen order
    for k, b in enumerate(beats):
        b.id = f"E{ep_no:02d}-B{k:03d}"
    duration = max((b.end for b in beats), default=0.0)
    return Episode(number=ep_no, logline=meta["EPISODE LOGLINE"],
                   continuity=meta["CONTINUITY NOTES"], duration=duration, beats=beats)


def load_show(folder: str | Path) -> list[Episode]:
    files = sorted(Path(folder).glob("*.txt"))
    eps = [parse_episode(f) for f in files]
    return sorted(eps, key=lambda e: e.number)


def fmt_ts(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"
