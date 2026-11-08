"""Free, offline backend: rule-based beat scoring. Used when no LLM key is set, and as a
cheap prefilter / tie-breaker when an LLM is used."""
from __future__ import annotations

import re

from .schemas import Beat, Critique, Episode, ProposalList, ProposedPromo, ProposedSegment
from .rules import MAX_LEN, MIN_LEN, _snap

HOOK_WORDS = {
    "violence": r"\b(slap|slaps|slapped|punch|punches|strike|strikes|struck|hit|hits|kick|shove|shoves|throws?|grab|grabs|chok|stab|blade|dagger|sword|knife|blood|collaps|crash|fist|thug|attack|seiz|drags?)\w*",
    "humiliation": r"\b(kneel|knees|humiliat|laugh(s|ing)? at|mock|spit|tear(s|ing)? (the|her|his)|rip|beg|whore|disgrace|worthless|trash|beggar|abandon|thrown out|cast out|penniless|scrubbing|servant|maid|no respectable|plainness|low value)\w*",
    "reveal": r"\b(reveal|duke|truth|secret|identity|real name|who you really|pendant|seal|signet|crest|letter|realiz|recogni)\w*",
    "betrayal": r"\b(betray|marrying me|married|fianc|engaged|affair|cheat|liar|lied|disown|cast (you|her) out)\w*",
    "threat": r"\b(poison|arsenic|fire|burn|dying|bleed|kill|die|dead|death|destroy|ruin|hang|prison|arrest|guards?|chains|handcuff|you will pay|regret)\w*",
    "romance": r"\b(kiss|kisses|kissed|embrace|my wife|my husband|love you|caress)\w*",
}
HOT_EMOTIONS = {"furious", "screaming", "shouting", "venomous", "cruel", "threatening", "outraged", "gloating",
                "mocking", "smug", "spiteful", "taunting", "scornful", "devastated", "horrified", "shocked",
                "enraged", "menacing", "desperate", "accusing", "gasping", "contemptuous", "triumphant"}
# (2) a violence word only counts as a hook if a person is on the receiving end, not a book or a door
PERSON = (r"\b(her|his|him|she|he|they|them|face|cheek|jaw|body|throat|chest|arm|shoulder|skull|bone|flesh|neck|"
          r"man|woman|girl|boy|guard|thug|servant|maid|lady|lord|duke|duchess|priest|husband|wife|mother|father)\b")
NAME = r"(?<!^)(?<![.!?] )\b[A-Z][a-z]{2,}"   # a capitalised word mid-sentence = a named character
# (1) cues that the scene has resolved: the opposite of a cliffhanger
RESOLVE = (r"\b(together|side by side|smil\w*|relie\w*|triumphant\w*|fad(e|es|ing) (out|away)|peaceful\w*|"
           r"gentl\w*|hug\w*|embrac\w*|exhales?|calm\w*|warm\w*|walk(s|ing)? (away|off|down|out)|"
           r"begin their|sets? off|safe now|thank you|grateful|reconcil\w*|tender\w*|softly|are you (hurt|all right|alright|okay))\b")
CLIFF_SFX = r"(cliffhanger|cut(s)? to black|ominous|abrupt|sudden silence|swell|drone|boom|stinger|hold)"
CLIFF_LINE = r"(\?$|\.\.\.$|\bwho\b|\bwhat\b.*\?|\bnow\b|\bbefore\b|\bunless\b|\bor else\b|\bwill\b|\bnever\b|\bstop\b|\bwait\b)"


def beat_hook_score(b: Beat) -> tuple[float, str]:
    t = b.text.lower()
    s, kind = 0.0, "other"
    has_person = b.type == "dialogue" or bool(re.search(PERSON, t)) or bool(re.search(NAME, b.text))
    found = {}
    for k, pat in HOOK_WORDS.items():
        n = len(re.findall(pat, t))
        if n:
            w = {"violence": 1.6, "betrayal": 1.4, "humiliation": 1.3, "reveal": 1.2, "threat": 1.1, "romance": 0.8}[k]
            if k == "violence" and not has_person:
                w *= 0.35   # "the book crashes into the puddle" is not a slap
            found[k] = w * min(n, 2)
    if found:
        kind = max(found, key=found.get)
        s = found[kind]
        # spoken lines are usually verbal attacks: label the insult, not the verb ("thrown out of her home")
        if b.type == "dialogue" and kind == "violence":
            for k in ("humiliation", "betrayal", "threat"):
                if k in found:
                    kind = k
                    break
    if b.type == "dialogue":
        emos = {e.strip().lower() for e in (b.emotion or "").split(",")}
        s += 1.0 * len(emos & HOT_EMOTIONS)
        s += 0.5 * b.text.count("!")
        if len(b.text.split()) <= 10:
            s += 0.4  # short, punchy lines read in under 3 s
        if kind == "other" and s >= 1.5:
            kind = "shocking_line"
    elif b.type == "sfx" and re.search(r"(slap|punch|crash|gasp|scream|shatter|impact|thud)", t):
        s += 1.2 if has_person else 0.3
        kind = kind if kind != "other" else "violence"
    elif b.type == "camera":
        s *= 0.3  # camera notes describe, they rarely are the hook themselves
    return s, kind


def beat_cliff_score(b: Beat, ep: Episode) -> float:
    t = b.text.lower().strip()
    s = 0.0
    if b.type == "sfx" and re.search(CLIFF_SFX, t):
        s += 2.0 + (1.0 if "cliffhanger" in t else 0)
    if b.type == "camera" and "cut to black" in t:
        s += 1.5
    if b.type == "dialogue":
        if re.search(CLIFF_LINE, t):
            s += 1.2
        emos = {e.strip().lower() for e in (b.emotion or "").split(",")}
        s += 0.6 * len(emos & HOT_EMOTIONS)
        for k, pat in HOOK_WORDS.items():
            if k in ("threat", "reveal", "betrayal") and re.search(pat, t):
                s += 0.8
    if b.type == "action" and re.search(r"(freez|stunned|eyes widen|turns|reaches|raises|lunges|steps forward|grips)", t):
        s += 0.8
    tail = [x for x in ep.beats if b.end - 6 <= x.end <= b.end + 0.01]
    resolved = any(re.search(RESOLVE, x.text.lower()) for x in tail)
    if ep.duration - b.end < 8 and not resolved:
        s += 1.5  # episodes are written to end on a cliffhanger
    if resolved:
        s -= 2.5   # a warm or settled ending is a payoff, not a cliffhanger
    return s


def _speakers(ep: Episode, a: float, b: float) -> set[str]:
    return {x.speaker for x in ep.beats if x.type == "dialogue" and x.speaker and a <= x.start <= b}


def standalone_score(ep: Episode, start: float, end: float) -> float:
    """Rough proxy: clear conflict between 2-3 people reads cold; a crowd of names or a mid-sentence
    pronoun opener does not. Earlier episodes need less backstory."""
    sp = _speakers(ep, start, end)
    s = 3.0
    s += 1.0 if 2 <= len(sp) <= 3 else (-0.5 if len(sp) > 4 else -1.0 if len(sp) == 0 else 0)
    first = next((x for x in ep.beats if x.type == "dialogue" and x.start >= start), None)
    if first and re.match(r"^(he|she|they|it|that|this|him|her)\b", first.text.lower()):
        s -= 0.8
    s -= min(1.0, (ep.number - 1) / 50)  # later episodes lean on more backstory
    return max(1.0, min(5.0, s))


def propose(ep: Episode, k: int = 4) -> ProposalList:
    beats = ep.beats
    hooks = sorted(((beat_hook_score(b), b) for b in beats if b.type != "camera"), key=lambda x: -x[0][0])
    out: list[tuple[float, ProposedPromo]] = []
    # quiet episodes still get candidates: threshold is relative to the episode's own top hooks
    floor = min(1.5, hooks[min(4, len(hooks) - 1)][0][0]) if hooks else 1.5
    for (hs, kind), hb in hooks[:15]:
        if hs < max(0.8, floor):
            break
        # start on the beat that opens the hook moment (<=1.5 s before), but never so early that
        # snapping to a line boundary pushes the hook past the 3 s window
        lead = sorted((b for b in beats if hb.start - 1.5 <= b.start <= hb.start), key=lambda b: b.start)
        sb = hb
        for b in lead:
            if hb.start - _snap(beats, b.start, b.end)[0] <= 2.8:
                sb = b
                break
        best = None
        for cb in beats:
            d = cb.end - sb.start
            if MIN_LEN <= d <= MAX_LEN - 1 and cb.start > hb.end:
                cs = beat_cliff_score(cb, ep) - abs(d - 38) / 40  # gentle preference for ~35-40 s
                if best is None or cs > best[0]:
                    best = (cs, cb)
        if not best:
            continue
        cs, cb = best
        out.append((hs + cs, ProposedPromo(
            segments=[ProposedSegment(start_beat=sb.id, end_beat=cb.id)],
            hook_beat=hb.id, cliffhanger_beat=cb.id,
            hook=_short(hb), cliffhanger=_short(cb),
            why=f"Opens on a {kind.replace('_', ' ')} beat and cuts on an unresolved {cb.type} moment.",
            hook_type=kind if kind in HOOK_WORDS or kind == "shocking_line" else "other")))

    # multi-segment: a strong early hook (6-12 s) + the episode's closing cliffhanger
    if hooks and hooks[0][0][0] >= 2.0:
        (hs, kind), hb = hooks[0]
        hook_end = max((b for b in beats if hb.start <= b.end <= hb.start + 12), key=lambda b: b.end, default=hb)
        tail_end = beats[-1]
        if tail_end.type == "camera" and "black" in tail_end.text.lower() and len(beats) > 1:
            tail_end = max(beats[:-1], key=lambda b: b.end)
        hook_len = hook_end.end - hb.start
        tail_starts = [b for b in beats if b.start > hook_end.end + 5 and b.type in ("dialogue", "action")
                       and 18 <= tail_end.end - b.start <= 50 - hook_len]
        if tail_starts and hook_len >= 4:
            ts = min(tail_starts, key=lambda b: abs((tail_end.end - b.start) - 28))
            out.append((hs + beat_cliff_score(tail_end, ep) + 0.5, ProposedPromo(
                segments=[ProposedSegment(start_beat=hb.id, end_beat=hook_end.id),
                          ProposedSegment(start_beat=ts.id, end_beat=tail_end.id)],
                hook_beat=hb.id, cliffhanger_beat=tail_end.id, hook=_short(hb), cliffhanger=_short(tail_end),
                why="Cold-opens on the episode's biggest beat, then jumps to its closing cliffhanger.",
                hook_type=kind if kind in HOOK_WORDS or kind == "shocking_line" else "other")))

    out.sort(key=lambda x: -x[0])
    chosen: list[ProposedPromo] = []
    used: list[tuple[float, float]] = []
    by_id = {b.id: b for b in beats}
    for _, p in out:
        a, z = by_id[p.segments[0].start_beat].start, by_id[p.segments[-1].end_beat].end
        if any(min(z, u2) - max(a, u1) > 0.5 * (z - a) for u1, u2 in used):
            continue
        chosen.append(p)
        used.append((a, z))
        if len(chosen) >= k:
            break
    return ProposalList(promos=chosen)


def critique(ep: Episode, hook: Beat, cliff: Beat, start: float, end: float) -> Critique:
    hs, _ = beat_hook_score(hook)
    cs = beat_cliff_score(cliff, ep)
    return Critique(hook_strength=int(max(1, min(5, round(1 + hs)))),
                    cliffhanger_strength=int(max(1, min(5, round(1 + cs)))),
                    standalone=int(round(standalone_score(ep, start, end))),
                    note="heuristic critic")


def _short(b: Beat, n: int = 90) -> str:
    who = f'{b.speaker.title()}: "' if b.speaker else ""
    txt = b.text if len(b.text) <= n else b.text[: n - 1] + "…"
    return f"{who}{txt}{'\"' if b.speaker else ''}"
