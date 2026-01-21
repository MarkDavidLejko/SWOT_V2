import json
import re
from typing import Any, Dict, List, Tuple

# Conservative sanitization: strip tags and control chars.
TAG_RE = re.compile(r"<[^>]+>")
CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

WHITESPACE_RE = re.compile(r"\s+")


def sanitize_text(text: str, max_len: int) -> str:
    if text is None:
        return ""
    t = str(text)
    t = TAG_RE.sub("", t)
    t = CTRL_RE.sub("", t)
    t = t.replace("\u2028", " ").replace("\u2029", " ")
    t = t.strip()
    if len(t) > max_len:
        t = t[:max_len].rstrip() + "…"
    return t


def normalize_inputs_for_cache_key(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a normalized, deterministic version of inputs for caching.
    Must NOT include api_key.
    """
    def norm(s: Any, max_len: int) -> str:
        return sanitize_text("" if s is None else str(s), max_len=max_len).strip().lower()

    settings = inputs.get("settings", {}) or {}
    normalized = {
        "venture_name": norm(inputs.get("venture_name", ""), 120),
        "venture_description": norm(inputs.get("venture_description", ""), 2500),
        "target_market": norm(inputs.get("target_market", ""), 220),
        "geography": norm(inputs.get("geography", ""), 120),
        "stage": norm(inputs.get("stage", ""), 40),
        "business_model": norm(inputs.get("business_model", ""), 120),
        "settings": {
            "tone": norm(settings.get("tone", "neutral"), 30),
            "industry_lens": norm(settings.get("industry_lens", "general"), 40),
            "detail_level": int(settings.get("detail_level", 2)),
            "include_assumptions": bool(settings.get("include_assumptions", False)),
        },
    }
    return normalized


def extract_first_json_object(text: str) -> str:
    """
    Extract the first top-level JSON object from a text blob.
    Deterministic scan that respects brace depth and strings.
    """
    if not text:
        raise ValueError("Empty model output.")

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start '{' found in model output.")

    in_str = False
    esc = False
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    raise ValueError("Unclosed JSON object in model output.")


def clamp_bullets(
    bullets: List[str],
    target_count: int,
    max_chars: int,
) -> Tuple[List[str], List[str]]:
    """
    Enforces:
    - exactly target_count items (deterministic truncate/pad)
    - each bullet <= max_chars by deterministic split or trim
    Returns (bullets, warnings)
    """
    warnings: List[str] = []

    # normalize whitespace
    cleaned = [WHITESPACE_RE.sub(" ", b).strip(" -•\t\r\n") for b in bullets if isinstance(b, str)]
    cleaned = [b for b in cleaned if b]

    # length control: deterministic split then trim
    processed: List[str] = []
    for b in cleaned:
        if len(b) <= max_chars:
            processed.append(b)
            continue

        # deterministic split: try " — ", then ": ", then "; ", then ", "
        splitters = [" — ", ": ", "; ", ", "]
        parts = None
        for sp in splitters:
            if sp in b:
                tmp = b.split(sp, 1)
                if len(tmp[0]) >= 40 and len(tmp[1]) >= 20:
                    parts = (tmp[0].strip(), tmp[1].strip())
                    break

        if parts and len(parts[0]) <= max_chars and len(parts[1]) <= max_chars:
            processed.extend([parts[0], parts[1]])
            warnings.append("A long bullet was split into two for readability.")
        else:
            processed.append(b[: max_chars - 1].rstrip() + "…")
            warnings.append("A long bullet was trimmed for readability.")

    # enforce exact count deterministically
    if len(processed) > target_count:
        processed = processed[:target_count]
        warnings.append("Extra bullets were truncated to match the selected detail level.")
    elif len(processed) < target_count:
        # pad with deterministic placeholders if model under-delivers
        missing = target_count - len(processed)
        processed.extend([f"(Add: specific point {i+1})" for i in range(missing)])
        warnings.append("Model returned too few bullets; placeholders were added to preserve deterministic layout.")

    return processed, warnings


def swot_to_txt(swot: Dict[str, Any]) -> str:
    def section(name: str, items: List[str]) -> str:
        lines = [name]
        for x in items:
            lines.append(f"- {x}")
        return "\n".join(lines)

    parts = [
        section("Strengths", swot.get("strengths", [])),
        section("Weaknesses", swot.get("weaknesses", [])),
        section("Opportunities", swot.get("opportunities", [])),
        section("Threats", swot.get("threats", [])),
    ]

    if swot.get("key_insights"):
        parts.append(section("Key insight", swot.get("key_insights", [])))

    if swot.get("action_suggestions"):
        parts.append(section("Action suggestions", swot.get("action_suggestions", [])))

    assumptions = swot.get("assumptions")
    if assumptions:
        if isinstance(assumptions, dict):
            parts.append("Assumptions used")
            for q in ["strengths", "weaknesses", "opportunities", "threats"]:
                items = assumptions.get(q, [])
                if items:
                    parts.append(section(q.capitalize(), items))
        elif isinstance(assumptions, list):
            parts.append(section("Assumptions used", assumptions))

    return "\n\n".join(parts).strip() + "\n"


def swot_to_md(swot: Dict[str, Any]) -> str:
    def section(name: str, items: List[str]) -> str:
        lines = [f"## {name}"]
        for x in items:
            lines.append(f"- {x}")
        return "\n".join(lines)

    parts = [
        "# SWOT",
        section("Strengths", swot.get("strengths", [])),
        section("Weaknesses", swot.get("weaknesses", [])),
        section("Opportunities", swot.get("opportunities", [])),
        section("Threats", swot.get("threats", [])),
    ]

    if swot.get("key_insights"):
        parts.append(section("Key insight", swot.get("key_insights", [])))

    if swot.get("action_suggestions"):
        parts.append(section("Action suggestions", swot.get("action_suggestions", [])))

    assumptions = swot.get("assumptions")
    if assumptions:
        if isinstance(assumptions, dict):
            parts.append("## Assumptions used")
            for q in ["strengths", "weaknesses", "opportunities", "threats"]:
                items = assumptions.get(q, [])
                if items:
                    parts.append(section(q.capitalize(), items))
        elif isinstance(assumptions, list):
            parts.append(section("Assumptions used", assumptions))

    return "\n\n".join(parts).strip() + "\n"


def safe_json_loads(s: str) -> Dict[str, Any]:
    return json.loads(s)
