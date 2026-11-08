"""Model layer. One string picks the backend, so a stronger model is a flag change:
  heuristic                         free, offline, rule-based (default)
  google_genai:gemini-2.0-flash     free tier (GOOGLE_API_KEY)
  groq:llama-3.3-70b-versatile      free tier (GROQ_API_KEY)
  openai:gpt-4o-mini / anthropic:claude-...   any LangChain chat model
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from .parser import fmt_ts
from .schemas import Critique, Episode, ProposalList

PROPOSE_SYSTEM = """You are a promo editor for vertical microdramas (ReelShort / DramaBox style).
Find the best 20-60 second promo clips in ONE episode. A promo:
- opens on a HOOK within its first 3 seconds (slap, reveal, betrayal, shocking line) that stops a scrolling viewer
- ends on a CLIFFHANGER: cut at an unresolved moment, so the viewer must know what happens next
- makes sense to someone who has never seen the show
Rules:
- Refer to moments ONLY by beat IDs from the list (e.g. E03-B012). Never write timestamps.
- 1 segment is normal; use 2-3 segments only to join a hook with a later payoff in the same episode.
- Start the first segment on the hook beat or at most 3 s before it. End the last segment on the cliffhanger beat.
- Total length 20-60 s. Prefer 30-45 s. Promos must not overlap each other.
- Return 2-4 promos, best first."""

CRITIC_SYSTEM = """You are a senior promo editor for vertical microdramas, judging one clip for a stranger scrolling TikTok.
Score 1-5 (be strict, 5 is rare):
- hook_strength: do the first 3 seconds stop a thumb? Physical shock, public humiliation, a shocking line > exposition.
- cliffhanger_strength: does the clip END on an unanswered question or threat? A warm, calm, resolved or
  "are you okay?" ending is NOT a cliffhanger (score 1-2 and set ending="settled").
- standalone: could someone who never saw the show follow who wants what? Unexplained names, objects or backstory lower it.
  A clip that gives away a late-season twist is a spoiler: lower standalone.
Also return: ending ("open" or "settled"), why (one plain line for the editor), note (the biggest weakness)."""


def beats_block(ep: Episode) -> str:
    rows = []
    for b in ep.beats:
        tag = b.type.upper() if b.type != "dialogue" else f"{b.speaker}" + (f" ({b.emotion})" if b.emotion else "")
        rows.append(f"{b.id} [{fmt_ts(b.start)}-{fmt_ts(b.end)}] {tag}: {b.text}")
    return "\n".join(rows)


def propose_prompt(ep: Episode, errors: list[str]) -> str:
    msg = (f"EPISODE {ep.number} (duration {fmt_ts(ep.duration)})\nLogline: {ep.logline}\n"
           f"Context so far: {ep.continuity}\n\nBEATS:\n{beats_block(ep)}")
    if errors:
        msg += ("\n\nYour previous answer broke these rules. Fix them and answer again:\n- "
                + "\n- ".join(errors[:12]))
    return msg


_REGISTRY: dict[str, object] = {}


def register_model(spec: str, model) -> None:
    """Plug in any LangChain-compatible chat model under a name (used by tests and custom APIs)."""
    _REGISTRY[spec] = model
    get_model.cache_clear()


@lru_cache(maxsize=8)
def get_model(spec: str):
    if spec == "heuristic":
        return None
    if spec in _REGISTRY:
        return _REGISTRY[spec]
    from langchain.chat_models import init_chat_model  # lazy: heuristic mode needs no provider SDKs
    return init_chat_model(spec, temperature=0.2)


def _structured(model, schema, system: str, user: str):
    """Use native structured output when the provider has it; else parse JSON from text."""
    msgs = [("system", system), ("human", user)]
    try:
        try:   # tool/function calling: supported by OpenAI, Gemini, Groq, Anthropic; avoids strict-schema limits
            runnable = model.with_structured_output(schema, method="function_calling")
        except (TypeError, ValueError):
            runnable = model.with_structured_output(schema)
        out = runnable.invoke(msgs)
        if out is not None:
            return out
    except NotImplementedError:
        pass
    except Exception as e:  # noqa: BLE001  bad tool output → fall back to plain JSON; auth/network errors re-raise
        if any(w in type(e).__name__ for w in ("Authentication", "Permission", "RateLimit", "Connection", "Timeout")):
            raise
    raw = model.invoke(msgs + [("human", f"Reply with JSON only, matching: {json.dumps(schema.model_json_schema())}")])
    txt = raw.content if isinstance(raw.content, str) else str(raw.content)
    m = re.search(r"\{.*\}", txt, re.S)
    return schema.model_validate_json(m.group(0) if m else txt)


def llm_propose(spec: str, ep: Episode, errors: list[str]) -> ProposalList:
    return _structured(get_model(spec), ProposalList, PROPOSE_SYSTEM, propose_prompt(ep, errors))


def llm_critique(spec: str, logline: str, transcript: str, hook: str, cliff: str) -> Critique:
    user = (f"Show premise (viewer does NOT know this): {logline}\n\nCLIP TRANSCRIPT:\n{transcript}\n\n"
            f"Claimed hook: {hook}\nClaimed cliffhanger: {cliff}")
    return _structured(get_model(spec), Critique, CRITIC_SYSTEM, user)
