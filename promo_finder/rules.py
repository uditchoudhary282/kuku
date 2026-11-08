"""Hard promo rules, enforced in code. The model proposes beat IDs; code decides if they are legal."""
from __future__ import annotations

from .parser import fmt_ts
from .schemas import Beat, Episode, Promo, ProposedPromo, Segment

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WINDOW = 3.0       # hook must land in the first 3 s
CLIFF_TAIL = 3.0        # cut must come within 3 s after the cliffhanger beat ends


def index_beats(episodes: list[Episode]) -> dict[str, Beat]:
    return {b.id: b for e in episodes for b in e.beats}


def _snap(ep_beats: list[Beat], start: float, end: float) -> tuple[float, float]:
    """Never cut in the middle of a spoken line: widen the cut to the line boundary."""
    for b in ep_beats:
        if b.type == "dialogue" and b.start < start < b.end:
            start = b.start
        if b.type == "dialogue" and b.start < end < b.end:
            end = b.end
    return round(start, 2), round(end, 2)


def transcript(ep_by_no: dict[int, Episode], segs: list[Segment], max_chars: int = 1200) -> str:
    out = []
    for s in segs:
        for b in ep_by_no[s.episode].beats:
            if b.start >= s.start - 0.01 and b.end <= s.end + 0.01 and b.type in ("dialogue", "action", "sfx"):
                who = f"{b.speaker}: " if b.speaker else f"[{b.type.upper()}] "
                out.append(f"({fmt_ts(b.start)}) {who}{b.text}")
        out.append("--- cut ---")
    return "\n".join(out[:-1])[:max_chars]


def resolve(p: ProposedPromo, beats: dict[str, Beat], ep_by_no: dict[int, Episode],
            pid: str, source: str = "single") -> tuple[Promo | None, list[str]]:
    errs: list[str] = []
    ids = [x for s in p.segments for x in (s.start_beat, s.end_beat)] + [p.hook_beat, p.cliffhanger_beat]
    missing = [i for i in ids if i not in beats]
    if missing:
        return None, [f"Unknown beat IDs {missing}. Use only IDs from the list."]
    if not 1 <= len(p.segments) <= 3:
        return None, ["A promo must have 1 to 3 segments."]

    segs: list[Segment] = []
    for s in p.segments:
        a, b = beats[s.start_beat], beats[s.end_beat]
        if a.episode != b.episode:
            errs.append(f"Segment {s.start_beat}->{s.end_beat} spans two episodes; split it.")
            continue
        start, end = _snap(ep_by_no[a.episode].beats, a.start, max(b.end, a.end))
        if end <= start:
            errs.append(f"Segment {s.start_beat}->{s.end_beat} ends before it starts.")
            continue
        segs.append(Segment(episode=a.episode, start=start, end=end, start_ts=fmt_ts(start),
                            end_ts=fmt_ts(end), start_beat=s.start_beat, end_beat=s.end_beat))
    if errs:
        return None, errs

    for x, y in zip(segs, segs[1:]):
        if x.episode == y.episode and y.start < x.end:
            errs.append("Segments overlap or are out of order; keep them in playback order.")
    dur = round(sum(s.end - s.start for s in segs), 2)
    if dur < MIN_LEN:
        errs.append(f"Promo is {dur:.1f}s; it must be at least {MIN_LEN:.0f}s. Extend the end.")
    if dur > MAX_LEN:
        errs.append(f"Promo is {dur:.1f}s; it must be at most {MAX_LEN:.0f}s. Start later or cut earlier.")

    hook, cliff, first, last = beats[p.hook_beat], beats[p.cliffhanger_beat], segs[0], segs[-1]
    if not (hook.episode == first.episode and first.start - 0.01 <= hook.start <= first.start + HOOK_WINDOW):
        errs.append(f"Hook beat {hook.id} starts {hook.start - first.start:.1f}s into the promo; "
                    f"it must start within the first {HOOK_WINDOW:.0f}s. Start the first segment at or just before the hook.")
    if not (cliff.episode == last.episode and last.start <= cliff.start and last.end - CLIFF_TAIL <= cliff.end <= last.end + 0.01):
        errs.append(f"Cliffhanger beat {cliff.id} is not at the end of the last segment; end the promo right after it.")
    if errs:
        return None, errs

    promo = Promo(id=pid, segments=segs, duration=dur, hook=p.hook, cliffhanger=p.cliffhanger, why=p.why,
                  hook_type=p.hook_type, hook_beat=p.hook_beat, cliffhanger_beat=p.cliffhanger_beat,
                  source=source if len(segs) == 1 or source != "single" else "multi")
    if len({s.episode for s in segs}) > 1:
        promo.source = "cross_episode"
    promo.transcript = transcript(ep_by_no, segs)
    return promo, []


def overlap(a: Promo, b: Promo) -> float:
    """Fraction of the shorter promo that overlaps the other (same episode only)."""
    tot = 0.0
    for x in a.segments:
        for y in b.segments:
            if x.episode == y.episode:
                tot += max(0.0, min(x.end, y.end) - max(x.start, y.start))
    return tot / max(1e-6, min(a.duration, b.duration))
