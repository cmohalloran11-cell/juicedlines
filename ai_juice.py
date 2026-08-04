"""
ai_juice.py — grounded AI explanations for Juiced projections.

Spec Vol VII (AI Juice). The SMALLEST safe surface: a single grounded LLM call that turns a
projection's real numbers into a plain-language "why this number / how uncertain" explanation.
It is NOT a chatbot and has NO database access — every fact it may state is passed to it in the
context block, so it cannot fabricate a statistic (spec Ch. 10, 193, 204).

Provider-agnostic: AI_PROVIDER selects Gemini (default here — the product's chosen provider)
or Anthropic. Either way the grounding + honesty prompt and the degradation contract are the
same. With no key configured, available()=False and callers get a clean "unavailable" response.

Config (env):
  AI_PROVIDER            gemini | anthropic         (default: gemini)
  GEMINI_API_KEY         Google Gemini key          (gemini)
  ANTHROPIC_API_KEY /
  AI_API_KEY             Anthropic key              (anthropic)
  AI_MODEL               model id                   (default per provider)
"""
from __future__ import annotations

import os
from typing import Any, Optional

import requests

PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()
_DEFAULT_MODEL = {"gemini": "gemini-2.5-flash", "anthropic": "claude-opus-4-8"}
AI_MODEL = os.getenv("AI_MODEL") or _DEFAULT_MODEL.get(PROVIDER, "gemini-2.5-flash")

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _gemini_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY") or None


def _anthropic_key() -> Optional[str]:
    return os.getenv("ANTHROPIC_API_KEY") or os.getenv("AI_API_KEY") or None


def available() -> bool:
    """True only if the selected provider has a key configured — else degrade cleanly."""
    if PROVIDER == "gemini":
        return _gemini_key() is not None
    if PROVIDER == "anthropic":
        try:
            import anthropic  # noqa: F401
        except Exception:
            return False
        return _anthropic_key() is not None
    return False


# Gemini free-tier request-per-minute cap for gemini-2.5-flash — measured live from the
# API's own 429 error body (2026-08-04): "generate_content_free_tier_requests, limit: 20".
# Real, sourced number, not a guess — but Google can change it, and it goes away entirely
# once billing is enabled on the key (paid tier has no equivalent published RPM cap this
# codebase knows to display). No equivalent published number exists for Anthropic here.
GEMINI_FREE_TIER_RPM = 20


def status() -> dict:
    return {
        "available": available(),
        "provider": PROVIDER,
        "model": AI_MODEL if available() else None,
        "reason": None if available() else (
            f"AI Juice ({PROVIDER}) is not configured — set the provider's API key."),
        "rpm_limit": GEMINI_FREE_TIER_RPM if (available() and PROVIDER == "gemini") else None,
    }


# ── grounding prompt (provider-independent) ───────────────────────────────────

_SYSTEM = """You are AI Juice, the analyst voice of the Juiced sports-prop research \
platform. You explain, in plain language, WHY a statistical projection landed where it did \
and HOW UNCERTAIN it is.

Hard rules — follow exactly:
1. Use ONLY the numbers in the DATA block. Never invent a statistic, rate, rank, matchup \
detail, or source that is not present there. If a fact isn't in DATA, say you don't have it.
2. Never state or imply an outcome is guaranteed. Frame everything in terms of probability \
and uncertainty. A projection is a distribution, not a prediction.
3. Explain what drives the number using the fields provided (sample size, model method, \
floor/ceiling band, edge vs the line, any listed drivers), and how wide the uncertainty is.
4. Be concise and concrete: 2-4 short sentences. No hype, no betting advice, no \
"lock"/"guaranteed"/"free money" language. This is research, not a pick.
5. If the data is thin (small sample, no drivers), say the projection is low-confidence."""


def _context_block(line: dict[str, Any]) -> str:
    prov = line.get("provenance") or {}
    fields = {
        "player": line.get("player"), "team": line.get("team"), "sport": line.get("sport"),
        "stat": line.get("stat_type"), "book_line": line.get("line"),
        "model_projection": line.get("model_proj"), "edge_vs_line": line.get("model_edge"),
        "prob_over": line.get("model_prob"), "floor_p10": line.get("model_floor"),
        "ceiling_p90": line.get("model_ceiling"), "sample_games": line.get("model_n"),
        "method": line.get("proj_kind"), "drivers": line.get("drivers"),
        "model_version": prov.get("model_version"), "data_as_of": prov.get("data_snapshot"),
    }
    rows = [f"- {k}: {v}" for k, v in fields.items() if v is not None]
    return "DATA (the only facts you may use):\n" + "\n".join(rows)


def _prompt(line: dict[str, Any]) -> str:
    ctx = _context_block(line)
    return (
        f"{ctx}\n\nExplain this projection for a sharp but non-technical reader: why the model "
        f"landed on {line.get('model_proj')} for {line.get('player')} {line.get('stat_type')} "
        f"(book line {line.get('line')}), and how confident it is.")


# ── provider calls ────────────────────────────────────────────────────────────

def _gemini_error_detail(r: "requests.Response") -> str:
    """The API's own error.status + message (e.g. "RESOURCE_EXHAUSTED: You exceeded your
    current quota...") instead of a bare HTTP code — a 429 read as just "(429)" gives no way
    to tell a spent daily quota from a transient per-minute throttle without re-testing."""
    try:
        err = r.json().get("error", {}) or {}
        status, msg = err.get("status"), err.get("message")
        if status and msg:
            return f"{status}: {msg}"
        return status or str(r.status_code)
    except Exception:
        return str(r.status_code)


def _call_gemini(prompt: str) -> dict:
    url = _GEMINI_URL.format(model=AI_MODEL)
    body = {
        "systemInstruction": {"parts": [{"text": _SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 600, "temperature": 0.4},
    }
    try:
        r = requests.post(url, params={"key": _gemini_key()}, json=body, timeout=25)
    except Exception as exc:
        return {"available": False, "reason": f"AI request failed: {exc}"}
    if r.status_code != 200:
        return {"available": False, "reason": f"AI request rejected ({_gemini_error_detail(r)})."}
    data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        fb = (data.get("promptFeedback") or {}).get("blockReason")
        return {"available": False, "reason": f"AI declined ({fb})." if fb else
                "AI returned no candidates."}
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        return {"available": False, "reason": "AI returned an empty explanation."}
    return {"available": True, "explanation": text}


def _call_anthropic(prompt: str) -> dict:
    try:
        import anthropic
    except Exception as exc:  # pragma: no cover
        return {"available": False, "reason": f"AI SDK unavailable: {exc}"}
    client = anthropic.Anthropic(api_key=_anthropic_key())
    try:
        resp = client.messages.create(model=AI_MODEL, max_tokens=600, system=_SYSTEM,
                                       messages=[{"role": "user", "content": prompt}])
    except Exception as exc:
        return {"available": False, "reason": f"AI request failed: {exc}"}
    if getattr(resp, "stop_reason", None) == "refusal":
        return {"available": False, "reason": "AI declined to answer this request."}
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    return ({"available": True, "explanation": text} if text
            else {"available": False, "reason": "AI returned an empty explanation."})


_COACH_SYSTEM = """You are the AI Coach for the Juiced sports-prop research platform. You give \
a short, actionable STRATEGY recommendation for today's slate as a whole — not a single prop.

Hard rules — follow exactly:
1. Use ONLY the numbers in the DATA block. Never invent a count, percentage, player name, or \
market condition that is not present there.
2. Never guarantee an outcome or use "lock"/"guaranteed"/"free money" language. This is \
strategy guidance (how aggressively to play, what to avoid), not a pick.
3. Be concrete and directive: 2-4 short sentences, plain language, aimed at someone deciding \
how many entries to play and how big. Reference the actual counts from DATA when you make a \
recommendation (e.g. "only N props clear 80+ Juice today" rather than a vague "a few").
4. If the slate has fewer high-confidence edges than usual (few 80+ Juice props relative to \
the total pool, or the top entry's projected win% is low), say so plainly and recommend \
playing smaller/fewer entries. If the slate is strong, say so and explain what makes it strong \
using the real numbers."""


def _coach_context(summary: dict[str, Any]) -> str:
    rows = [f"- {k}: {v}" for k, v in summary.items() if v is not None]
    return "DATA (the only facts you may use):\n" + "\n".join(rows)


def coach(summary: dict[str, Any]) -> dict:
    """Grounded slate-level strategy recommendation — the AI Coach panel. `summary` is a
    small dict of already-computed real aggregate numbers (see dashboard.coach_context());
    this function never touches the DB or the board directly, same no-fabrication contract
    as explain(). Never raises; degrades cleanly when unconfigured."""
    if not available():
        return {"available": False, "reason": status()["reason"]}
    ctx = _coach_context(summary)
    prompt = f"{ctx}\n\nGive today's strategy recommendation."
    if PROVIDER == "gemini":
        url = _GEMINI_URL.format(model=AI_MODEL)
        body = {
            "systemInstruction": {"parts": [{"text": _COACH_SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 400, "temperature": 0.4},
        }
        try:
            r = requests.post(url, params={"key": _gemini_key()}, json=body, timeout=25)
        except Exception as exc:
            return {"available": False, "reason": f"AI request failed: {exc}"}
        if r.status_code != 200:
            return {"available": False, "reason": f"AI request rejected ({_gemini_error_detail(r)})."}
        data = r.json()
        cands = data.get("candidates") or []
        if not cands:
            return {"available": False, "reason": "AI returned no candidates."}
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            return {"available": False, "reason": "AI returned an empty recommendation."}
        return {"available": True, "strategy": text, "provider": PROVIDER, "model": AI_MODEL,
                "grounded_on": summary}
    if PROVIDER == "anthropic":
        try:
            import anthropic
        except Exception as exc:  # pragma: no cover
            return {"available": False, "reason": f"AI SDK unavailable: {exc}"}
        client = anthropic.Anthropic(api_key=_anthropic_key())
        try:
            resp = client.messages.create(model=AI_MODEL, max_tokens=400, system=_COACH_SYSTEM,
                                          messages=[{"role": "user", "content": prompt}])
        except Exception as exc:
            return {"available": False, "reason": f"AI request failed: {exc}"}
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        if not text:
            return {"available": False, "reason": "AI returned an empty recommendation."}
        return {"available": True, "strategy": text, "provider": PROVIDER, "model": AI_MODEL,
                "grounded_on": summary}
    return {"available": False, "reason": "No AI provider configured."}


def explain(line: dict[str, Any]) -> dict:
    """Grounded explanation of one projection. Never raises. On success returns
    {available, explanation, provider, model, grounded_on}."""
    if not available():
        return {"available": False, "reason": status()["reason"]}
    if line.get("model_proj") is None:
        return {"available": False, "reason": "No projection to explain for this line."}

    prompt = _prompt(line)
    result = _call_gemini(prompt) if PROVIDER == "gemini" else _call_anthropic(prompt)
    if not result.get("available"):
        return result

    result.update({
        "provider": PROVIDER,
        "model": AI_MODEL,
        # The exact facts the model was permitted to use — the receipts + fabrication guard.
        "grounded_on": {k: v for k, v in {
            "player": line.get("player"), "stat": line.get("stat_type"),
            "book_line": line.get("line"), "model_projection": line.get("model_proj"),
            "prob_over": line.get("model_prob"), "floor": line.get("model_floor"),
            "ceiling": line.get("model_ceiling"), "sample_games": line.get("model_n"),
            "method": line.get("proj_kind"),
        }.items() if v is not None},
    })
    return result
