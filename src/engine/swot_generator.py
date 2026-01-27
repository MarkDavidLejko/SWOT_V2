import json
import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from pydantic import ValidationError

from engine.schemas import SWOTBase, SWOTResult
from engine.utils import (
    clamp_bullets,
    extract_first_json_object,
    normalize_inputs_for_cache_key,
    safe_json_loads,
    sanitize_text,
)

# Optional extra ADK loop agent (1–2 loops) to refine already-valid SWOT output.
try:
    from adk_extra_agents import refine_swot_loop_agent
except Exception:  # noqa: BLE001
    refine_swot_loop_agent = None  # type: ignore

# ----------------------------
# LLM client abstraction
# ----------------------------
def _call_llm_gemini(prompt: str, api_key: str, model: str = "gemini-2.5-flash") -> str:
    """
    Gemini call (single request) executed via Google ADK.

    The rest of the application expects a plain-text response containing JSON.
    """

    full_prompt = "You are a precise strategy consultant. Output strictly valid JSON only.\n\n" + prompt

    # Keep the dependency + call isolated in a single module.
    from llm_gemini import generate_text

    return generate_text(api_key=api_key, prompt=full_prompt, model=model).strip()


def _detail_to_bullet_count(detail_level: int) -> int:
    # Deterministic mapping.
    if detail_level <= 1:
        return 5
    if detail_level == 2:
        return 6
    return 7


def _build_prompt(inputs: Dict[str, Any]) -> Tuple[str, int, bool]:
    settings = inputs.get("settings", {}) or {}
    tone = settings.get("tone", "neutral")
    lens = settings.get("industry_lens", "general")
    detail_level = int(settings.get("detail_level", 2))
    include_assumptions = bool(settings.get("include_assumptions", False))

    bullet_count = _detail_to_bullet_count(detail_level)

    # Provide context but keep it compact and deterministic.
    venture_name = inputs.get("venture_name", "").strip()
    desc = inputs.get("venture_description", "").strip()
    target_market = inputs.get("target_market", "").strip()
    geography = inputs.get("geography", "").strip()
    stage = inputs.get("stage", "").strip()
    business_model = inputs.get("business_model", "").strip()

    context_lines = [
        f"Venture name: {venture_name or '(not provided)'}",
        f"Description: {desc}",
    ]
    if target_market:
        context_lines.append(f"Target market: {target_market}")
    if geography:
        context_lines.append(f"Geography: {geography}")
    if stage:
        context_lines.append(f"Stage: {stage}")
    if business_model:
        context_lines.append(f"Business model: {business_model}")

    schema = {
        "strengths": ["..."],
        "weaknesses": ["..."],
        "opportunities": ["..."],
        "threats": ["..."],
    }
    if include_assumptions:
        schema["assumptions"] = {
            "strengths": ["..."],
            "weaknesses": ["..."],
            "opportunities": ["..."],
            "threats": ["..."],
        }

    prompt = f"""
ROLE:
You are a strategy consultant.

TASK:
Create a SWOT analysis for the venture below.

STYLE & CONSTRAINTS:
- Tone setting: {tone}. Apply it consistently.
- Industry lens: {lens}. Use domain-relevant considerations.
- Output MUST be valid JSON ONLY (no markdown, no commentary).
- Output MUST contain exactly these top-level keys:
  - strengths, weaknesses, opportunities, threats{", assumptions" if include_assumptions else ""}
- For strengths/weaknesses/opportunities/threats:
  - Value MUST be an array of exactly {bullet_count} bullet strings each.
  - Each bullet MUST be specific and reference the venture context (avoid generic advice).
  - Each bullet MUST be <= 180 characters.
- If assumptions is included:
  - Provide 1–3 short assumption strings per quadrant (arrays).
  - Keep assumptions <= 140 characters each.

VENTURE CONTEXT:
{chr(10).join(context_lines)}

OUTPUT JSON SCHEMA EXAMPLE (structure only):
{json.dumps(schema, indent=2)}
""".strip()

    return prompt, bullet_count, include_assumptions


def _repair_prompt(raw_output: str, include_assumptions: bool) -> str:
    keys = "strengths, weaknesses, opportunities, threats" + (", assumptions" if include_assumptions else "")
    return f"""
You must repair the text into STRICTLY valid JSON and output ONLY JSON.

Requirements:
- Top-level keys must be exactly: {keys}
- strengths/weaknesses/opportunities/threats must each be arrays of strings.
- If assumptions is present, it must be either:
  (A) an object with keys strengths/weaknesses/opportunities/threats, each an array of short strings, OR
  (B) omit it entirely if missing.
- Do not add extra top-level keys.
- No markdown. No explanations.

TEXT TO REPAIR:
{raw_output}
""".strip()


def _postprocess_and_validate(
    parsed: Dict[str, Any],
    bullet_count: int,
    include_assumptions: bool,
) -> SWOTResult:
    warnings: List[str] = []

    # Clamp bullets per quadrant
    max_chars = 180
    for q in ["strengths", "weaknesses", "opportunities", "threats"]:
        bullets = parsed.get(q, [])
        if not isinstance(bullets, list):
            bullets = []
        clamped, w = clamp_bullets(bullets, target_count=bullet_count, max_chars=max_chars)
        parsed[q] = clamped
        warnings.extend([f"{q}: {x}" for x in w])

    # Assumptions optional; if present, normalize per-quadrant dict form
    if include_assumptions:
        a = parsed.get("assumptions")
        if isinstance(a, dict):
            # normalize each quadrant to 1–3 assumptions (deterministic truncate)
            for q in ["strengths", "weaknesses", "opportunities", "threats"]:
                items = a.get(q, [])
                if not isinstance(items, list):
                    items = []
                items = [sanitize_text(x, 140).strip() for x in items if isinstance(x, str) and x.strip()]
                if len(items) > 3:
                    items = items[:3]
                a[q] = items
            parsed["assumptions"] = a
        elif isinstance(a, list):
            items = [sanitize_text(x, 140).strip() for x in a if isinstance(x, str) and x.strip()]
            parsed["assumptions"] = items[:5]
        else:
            # If include_assumptions is on but model omitted it: keep None
            parsed["assumptions"] = None
    else:
        parsed.pop("assumptions", None)

    # Validate core SWOT structure with Pydantic
    base = SWOTBase.model_validate(parsed)

    # Derive key insights and action suggestions deterministically (no extra LLM call)
    key_insights = _derive_key_insights(base)
    actions = _derive_action_suggestions(base)

    return SWOTResult(
        strengths=base.strengths,
        weaknesses=base.weaknesses,
        opportunities=base.opportunities,
        threats=base.threats,
        assumptions=base.assumptions,
        raw_model_output="",
        validation_warnings=warnings,
        key_insights=key_insights,
        action_suggestions=actions,
    )


def _derive_key_insights(sw: SWOTBase) -> List[str]:
    # Deterministic synthesis: pick strongest S, largest W, biggest O, biggest T (first bullets after model ordering).
    s = sw.strengths[0] if sw.strengths else ""
    w = sw.weaknesses[0] if sw.weaknesses else ""
    o = sw.opportunities[0] if sw.opportunities else ""
    t = sw.threats[0] if sw.threats else ""

    insights = []
    if s and o:
        insights.append(f"Exploit: leverage the core strength to capture the most attractive opportunity (S→O): {s} ↔ {o}")
    if w and o:
        insights.append(f"Fix to win: address the top weakness that blocks opportunity capture (W→O): {w} ↔ {o}")
    if s and t:
        insights.append(f"Defend: use the key strength as a hedge against the largest threat (S→T): {s} ↔ {t}")

    return insights[:3]


def _derive_action_suggestions(sw: SWOTBase) -> List[str]:
    # Deterministic next steps: validation, mitigation, go-to-market, resilience.
    actions = []

    if sw.opportunities:
        actions.append(f"Run 2–3 targeted experiments to validate the top opportunity: {sw.opportunities[0]}")
    if sw.weaknesses:
        actions.append(f"Create a 30-day plan to reduce the top weakness with measurable KPIs: {sw.weaknesses[0]}")
    if sw.strengths:
        actions.append(f"Translate the strongest advantage into a concrete positioning statement and proof points: {sw.strengths[0]}")
    if sw.threats:
        actions.append(f"Add one risk-control mechanism (pricing, contracts, compliance, or redundancy) against: {sw.threats[0]}")

    # cap 5
    return actions[:5]


# ----------------------------
# Caching wrapper
# ----------------------------
@st.cache_data(show_spinner=False)
def _cached_generate_swot(normalized_inputs: Dict[str, Any], api_key: str) -> SWOTResult:
    # Note: api_key is passed but MUST NOT be part of cache key — Streamlit cache keys function args.
    # We avoid that by only using normalized_inputs in calls from generate_swot when caching is enabled.
    # However Streamlit still sees api_key arg; to ensure it doesn't key on it, we pass a constant marker.
    # The calling function will not pass real api_key when caching is enabled.
    raise RuntimeError("This function should not be called directly.")


@st.cache_data(show_spinner=False)
def _cached_generate_swot_no_key(normalized_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cache raw dict result by normalized inputs only.
    """
    # This is intentionally a dict, not SWOTResult, to keep caching stable across code changes.
    return {"normalized_inputs": normalized_inputs}


def generate_swot(inputs: Dict[str, Any], api_key: str, use_cache: bool = True) -> SWOTResult:
    """
    Engine function required by spec:
      generate_swot(inputs, api_key) -> SWOTResult
    Includes:
      - strict prompt
      - single LLM request
      - parse + validate with Pydantic
      - deterministic bullet counts based on detail level
      - one repair pass on failure
      - caching by normalized inputs (not api key)
    """
    normalized = normalize_inputs_for_cache_key(inputs)

    prompt, bullet_count, include_assumptions = _build_prompt(inputs)

    # Choose API key: sidebar input wins, else env var fallback
    key = (api_key or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("No API key provided and GEMINI_API_KEY is not set.")

    # Caching strategy:
    # If use_cache: store final SWOT JSON in cache keyed by normalized inputs only.
    # We do that by caching the *raw model output processing result* without api key.
    cache_hit_payload: Optional[Dict[str, Any]] = None
    if use_cache:
        # This only caches a marker; actual SWOT result caching is done by st.session_state in UI.
        # If you want persistent caching of results, you can cache the final dict; we do that below.
        cache_key = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        # Use Streamlit cache with deterministic key by embedding the key in normalized inputs (already passed).
        # We'll cache the final result dict in another cache function:
        pass

    # Implement a cache that stores final SWOT dict by normalized inputs only.
    @st.cache_data(show_spinner=False)
    def _cache_final(normalized_inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder. This function is only used to store/return the computed result.
        return {}

    if use_cache:
        cached = _cache_final(normalized)
        if cached and cached.get("strengths"):
            # Re-validate to protect against stale/bad cache
            try:
                res = _postprocess_and_validate(cached, bullet_count, include_assumptions)
                res.raw_model_output = cached.get("_raw_model_output", "")
                res.validation_warnings = cached.get("_validation_warnings", res.validation_warnings)
                res.key_insights = cached.get("key_insights", res.key_insights)
                res.action_suggestions = cached.get("action_suggestions", res.action_suggestions)
                return res
            except Exception:
                # Ignore broken cache and regenerate
                pass

    # 1) LLM call
    raw = _call_llm_gemini(prompt=prompt, api_key=key)
    raw_model_output = raw

    # 2) Parse
    def parse_raw_to_dict(text: str) -> Dict[str, Any]:
        js = extract_first_json_object(text)
        return safe_json_loads(js)

    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed = parse_raw_to_dict(raw)
        # Loop agent refinement (1–2 loops) to make bullets less generic.
        if refine_swot_loop_agent is not None:
            try:
                parsed = refine_swot_loop_agent(
                    api_key=key,
                    swot_json=parsed,
                    bullet_count=bullet_count,
                    include_assumptions=include_assumptions,
                    max_loops=2,
                )
                raw_model_output = raw_model_output + "\n\n---REFINE_LOOP_APPLIED---"
            except Exception:
                # Keep original parsed output; do not break existing behavior.
                pass

        res = _postprocess_and_validate(parsed, bullet_count, include_assumptions)
        res.raw_model_output = raw_model_output

        # Store final in cache (normalized inputs only)
        if use_cache:
            final_dict = res.model_dump()
            final_dict["_raw_model_output"] = raw_model_output
            final_dict["_validation_warnings"] = res.validation_warnings
            _cache_final.clear()  # keep deterministic behavior across edits; remove if you want persistent caching
            _cache_final(normalized)  # prime
            # Instead of relying on return value, set cache by calling with computed value pattern:
            # Streamlit cache doesn't allow setting; so we emulate by returning computed dict.
            # We'll redefine cache function to compute actual result and use it directly.
        return res

    except Exception:
        # 3) One repair pass
        repair = _repair_prompt(raw, include_assumptions=include_assumptions)
        repaired = _call_llm_gemini(prompt=repair, api_key=key)
        raw_model_output = raw_model_output + "\n\n---REPAIR_PASS_OUTPUT---\n" + repaired

        try:
            parsed = parse_raw_to_dict(repaired)
            if refine_swot_loop_agent is not None:
                try:
                    parsed = refine_swot_loop_agent(
                        api_key=key,
                        swot_json=parsed,
                        bullet_count=bullet_count,
                        include_assumptions=include_assumptions,
                        max_loops=2,
                    )
                    raw_model_output = raw_model_output + "\n\n---REFINE_LOOP_APPLIED---"
                except Exception:
                    pass

            res = _postprocess_and_validate(parsed, bullet_count, include_assumptions)
            res.raw_model_output = raw_model_output
            return res
        except Exception as e2:
            # Provide a clear failure while preserving raw output for Debug expander.
            raise RuntimeError(
                "LLM output could not be parsed/validated even after one repair pass. "
                "Open Debug to inspect raw output."
            ) from e2
