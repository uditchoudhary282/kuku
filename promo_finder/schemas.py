from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Beat(BaseModel):
    id: str
    episode: int
    scene: int
    scene_heading: str = ""
    type: Literal["camera", "action", "sfx", "dialogue"]
    speaker: Optional[str] = None
    emotion: Optional[str] = None
    start: float
    end: float
    text: str


class Episode(BaseModel):
    number: int
    logline: str = ""
    continuity: str = ""
    duration: float
    beats: list[Beat]


# ---------- What the model is allowed to return (beat IDs only, never timestamps) ----------
class ProposedSegment(BaseModel):
    start_beat: str = Field(description="Beat ID where this segment starts, e.g. E03-B012")
    end_beat: str = Field(description="Beat ID where this segment ends (inclusive)")


class ProposedPromo(BaseModel):
    segments: list[ProposedSegment] = Field(description="1-3 segments in playback order")
    hook_beat: str = Field(description="Beat ID of the hook; must be within first 3s of the promo")
    cliffhanger_beat: str = Field(description="Beat ID of the unresolved moment the promo ends on")
    hook: str = Field(description="What the hook is, one short line")
    cliffhanger: str = Field(description="What is left unresolved, one short line")
    why: str = Field(description="One line on why it works for a cold viewer")
    hook_type: Literal["violence", "humiliation", "reveal", "betrayal", "threat", "romance", "shocking_line", "other"] = "other"


class ProposalList(BaseModel):
    promos: list[ProposedPromo]


class Critique(BaseModel):
    hook_strength: int = Field(ge=1, le=5)
    cliffhanger_strength: int = Field(ge=1, le=5)
    standalone: int = Field(ge=1, le=5, description="Would a never-seen-the-show viewer follow it?")
    note: str = ""
    ending: Literal["open", "settled"] = Field("open", description="open = unresolved question/threat; settled = resolved, warm or calm")
    why: str = Field("", description="One line, for an editor: why a stranger would stop scrolling and want the next episode")


# ---------- Resolved, validated output ----------
class Segment(BaseModel):
    episode: int
    start: float
    end: float
    start_ts: str
    end_ts: str
    start_beat: str
    end_beat: str


class Promo(BaseModel):
    id: str
    segments: list[Segment]
    duration: float
    hook: str
    cliffhanger: str
    why: str
    hook_type: str = "other"
    hook_beat: str
    cliffhanger_beat: str
    source: str = "single"  # single | multi | cross_episode
    scores: dict[str, float] = {}
    critic_note: str = ""
    breakdown: dict[str, float] = {}   # how the score was built: shown as "why this rank"
    score: float = 0.0
    transcript: str = ""
