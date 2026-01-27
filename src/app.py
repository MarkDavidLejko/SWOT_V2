import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional
import html
import streamlit as st

from adk_extra_agents import (
    pdf_agent_create_pdf_base64,
    email_agent_draft_email,
    send_email_smtp,
    followup_agent_get_questions,
    action_plan_30d_agent,
)

from engine.swot_generator import generate_swot
from engine.utils import (
    sanitize_text,
    normalize_inputs_for_cache_key,
    swot_to_txt,
    swot_to_md,
)
from engine.schemas import SWOTResult


st.set_page_config(page_title="SWOT Builder", layout="wide")

# Minimal CSS for 2×2 responsive grid cards
st.markdown(
    """
    <style>
      .swot-grid { margin-top: 0.5rem; }
      .swot-card {
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 16px;
        border: 1px solid rgba(0,0,0,0.07);
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
      }
      .swot-title {
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 10px;
        letter-spacing: 0.2px;
      }
      .swot-bullets {
        font-size: 0.98rem;
        line-height: 1.45;
        margin: 0;
        padding-left: 1.15rem;
      }
      .swot-subtle {
        font-size: 0.88rem;
        opacity: 0.9;
        margin-top: 10px;
      }
      .theme-strengths { background: rgba(46, 204, 113, 0.14); }
      .theme-weaknesses { background: rgba(243, 156, 18, 0.16); }
      .theme-opportunities { background: rgba(52, 152, 219, 0.14); }
      .theme-threats { background: rgba(231, 76, 60, 0.14); }

      /* Responsiveness: Streamlit columns stack automatically, but we keep spacing consistent */
      @media (max-width: 900px) {
        .swot-card { padding: 14px 14px; }
        .swot-title { font-size: 1.1rem; }
        .swot-bullets { font-size: 0.96rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Session state initialization
# ----------------------------
DEFAULT_STATE = {
    "initialized": False,
    "last_request": {},
    "swot": None,  # will hold dict version of SWOTResult
    "raw_model_output": "",
    "error": "",
    "api_key": "",
    "validation_warnings": [],
    "time_text": "",
    "pdf_base64": "",
    "email_subject": "",
    "email_body": "",
    "logged_in": False,
    "followup_questions": [],
    "followup_answers": {},
    "action_plan_30d": "",
}

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_state() -> None:
    keep_api_key = st.session_state.get("api_key", "")
    for k, v in DEFAULT_STATE.items():
        st.session_state[k] = v
    st.session_state["api_key"] = keep_api_key


def _append_login_audit(username: str, success: bool) -> None:
    """Append login attempts to a CSV file (lightweight 'database')."""

    import csv
    from datetime import datetime

    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "login_log.csv")

    # ISO timestamp in Europe/Berlin if available; fallback to local time.
    try:
        from zoneinfo import ZoneInfo

        ts = datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")
    except Exception:
        ts = datetime.now().isoformat(timespec="seconds")

    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "username", "success"])
        w.writerow([ts, username, "1" if success else "0"])


def _render_login_gate() -> None:
    """Block access to the app until a successful login."""

    st.title("Login")
    st.write("Enter credentials to access the SWOT Builder interface.")

    u = st.text_input("Username", value="", key="login_username")
    p = st.text_input("Password", value="", type="password", key="login_password")

    if st.button("Login", type="primary", use_container_width=True, key="login_btn"):
        ok = (u or "") == "admin" and (p or "") == "admin"
        _append_login_audit(username=(u or ""), success=ok)
        if ok:
            st.session_state.logged_in = True
            # Do NOT modify widget state here; Streamlit will rerun anyway.
            st.rerun()
        else:
            st.error("Invalid credentials.")



# Login gate (single hard-coded account as requested).
if not st.session_state.get("logged_in", False):
    _render_login_gate()
    st.stop()


# -------------
# Sidebar
# -------------
with st.sidebar:
    st.header("Settings")

    st.session_state.api_key = st.text_input(
        "API key",
        type="password",
        value=st.session_state.api_key or "",
        help="Stored only in session state. Never printed.",
    )

    tone = st.selectbox(
        "Tone",
        options=["conservative", "neutral", "aggressive"],
        index=1,
        help="Controls how optimistic/pessimistic assumptions are.",
    )

    industry_lens = st.selectbox(
        "Industry lens",
        options=["general", "SaaS", "consumer goods", "healthcare", "fintech", "industrial", "education", "travel"],
        index=0,
    )

    detail_level = st.slider(
        "Detail level",
        min_value=1,
        max_value=3,
        value=2,
        help="1 = shorter bullets; 3 = deeper bullets. Bullet count stays deterministic.",
    )

    include_assumptions = st.checkbox(
        "Include risk assumptions",
        value=False,
        help="Adds assumptions (small line) per quadrant and export sections.",
    )

    use_cache = st.checkbox(
        "Use caching",
        value=True,
        help="Caches results by normalized inputs + settings (not API key).",
    )

    if st.button("Clear / Restart", use_container_width=True):
        reset_state()
        st.rerun()

# -------------
# Main UI
# -------------
st.title("SWOT Builder")
st.write(
    "Enter a venture description and optionally provide context. Click **Generate SWOT** to receive a validated SWOT rendered in a colorful 2×2 grid."
)

colA, colB = st.columns([1, 1])

with colA:
    venture_name = st.text_input("Venture name", value="")
with colB:
    stage = st.selectbox("Stage (optional)", options=["", "idea", "MVP", "growth"], index=0)

venture_description = st.text_area(
    "Venture description (required)",
    value="",
    height=160,
    help="Describe what it does, why it matters, and what differentiates it. Length is limited for reliability.",
)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    target_market = st.text_input("Target market (optional)", value="")
with c2:
    geography = st.text_input("Geography (optional)", value="")
with c3:
    business_model = st.text_input("Business model (optional)", value="")

generate = st.button("Generate SWOT", type="primary", use_container_width=True)

# Follow-up questions (asked by an ADK agent) are shown here after the first click.
if st.session_state.followup_questions:
    st.subheader("Clarifying questions")
    st.write("Answer these to improve output quality. Then continue.")

    for i, q in enumerate(st.session_state.followup_questions, start=1):
        key = f"followup_{i}"
        prev = st.session_state.followup_answers.get(key, "")
        st.session_state.followup_answers[key] = st.text_input(f"Q{i}: {q}", value=prev)

    continue_btn = st.button("Continue", type="secondary", use_container_width=True, key="continue_after_followups")
else:
    continue_btn = False

# ----------------------------
# Generation flow
# ----------------------------
if generate or continue_btn:
    st.session_state.error = ""
    st.session_state.validation_warnings = []
    st.session_state.raw_model_output = ""

    # Sanitize and limit lengths
    venture_name_s = sanitize_text(venture_name, max_len=120)
    venture_description_s = sanitize_text(venture_description, max_len=2500)
    target_market_s = sanitize_text(target_market, max_len=220)
    geography_s = sanitize_text(geography, max_len=120)
    stage_s = sanitize_text(stage, max_len=40)
    business_model_s = sanitize_text(business_model, max_len=120)

    if not venture_description_s.strip():
        st.session_state.error = "Venture description is required."
    elif not (st.session_state.api_key or "").strip():
        # Allow env var usage inside engine, but user asked for sidebar input; keep this strict.
        st.session_state.error = "API key is required in the sidebar."
    else:
        # Ask follow-up questions once on the first run; then wait for user answers.
        if generate and not st.session_state.followup_questions:
            try:
                venture_context = "\n".join(
                    [
                        f"Venture name: {venture_name_s.strip() or '(not provided)'}",
                        f"Description: {venture_description_s.strip()}",
                        f"Target market: {target_market_s.strip() or '(not provided)'}",
                        f"Geography: {geography_s.strip() or '(not provided)'}",
                        f"Stage: {stage_s.strip() or '(not provided)'}",
                        f"Business model: {business_model_s.strip() or '(not provided)'}",
                    ]
                )
                qs = followup_agent_get_questions(api_key=st.session_state.api_key, venture_context=venture_context)
                if qs:
                    st.session_state.followup_questions = qs
                    st.session_state.error = "Please answer the clarifying questions and click Continue."
                    st.rerun()
            except Exception:
                # If agent fails, we still continue with existing behavior.
                pass

        # If we have follow-up questions, require answers before continuing.
        if st.session_state.followup_questions:
            # Collect answers already stored by the input widgets.
            answers = []
            for i, q in enumerate(st.session_state.followup_questions, start=1):
                key = f"followup_{i}"
                a = (st.session_state.followup_answers.get(key) or "").strip()
                if not a:
                    st.session_state.error = "Please answer all clarifying questions, then click Continue."
                    st.session_state.initialized = False
                    st.stop()
                answers.append((q, a))

            # Merge answers into the venture description (keeps engine schema unchanged).
            qa_block = "\n".join([f"Q: {q}\nA: {a}" for q, a in answers])
            venture_description_s = (venture_description_s.strip() + "\n\nClarifications:\n" + qa_block).strip()

        request: Dict[str, Any] = {
            "venture_name": venture_name_s.strip(),
            "venture_description": venture_description_s.strip(),
            "target_market": target_market_s.strip(),
            "geography": geography_s.strip(),
            "stage": stage_s.strip(),
            "business_model": business_model_s.strip(),
            "settings": {
                "tone": tone,
                "industry_lens": industry_lens,
                "detail_level": int(detail_level),
                "include_assumptions": bool(include_assumptions),
            },
        }

        st.session_state.last_request = request

        with st.spinner("Generating SWOT..."):
            try:
                result: SWOTResult = generate_swot(
                    inputs=request,
                    api_key=st.session_state.api_key,
                    use_cache=use_cache,
                )

                st.session_state.swot = result.model_dump()
                st.session_state.raw_model_output = result.raw_model_output or ""
                st.session_state.validation_warnings = result.validation_warnings or []
                st.session_state.initialized = True

                # Sequential main agent #2: 30-day action plan (placed after the SWOT results).
                try:
                    md_for_plan = swot_to_md(st.session_state.swot)
                    st.session_state.action_plan_30d = action_plan_30d_agent(
                        api_key=st.session_state.api_key,
                        swot_markdown=md_for_plan,
                    )
                except Exception:
                    st.session_state.action_plan_30d = ""

                # Reset follow-up state once we have a final result.
                st.session_state.followup_questions = []
                st.session_state.followup_answers = {}

            except Exception as e:
                st.session_state.error = str(e)
                st.session_state.initialized = False

# ----------------------------
# Display errors
# ----------------------------
if st.session_state.error:
    st.error(st.session_state.error)

# ----------------------------
# Render SWOT grid if available
# ----------------------------
def render_card(title: str, bullets: list[str], theme_class: str, assumptions: Optional[list[str]] = None) -> None:
    bullets_html = "".join([f"<li>{html.escape(b)}</li>" for b in bullets])
    assumptions_html = ""
    if assumptions:
        safe_assumptions = "; ".join([html.escape(a) for a in assumptions])
        assumptions_html = f'<div class="swot-subtle"><b>Assumptions:</b> {safe_assumptions}</div>'

    st.markdown(
        f"""
        <div class="swot-card {theme_class}">
          <div class="swot-title">{title}</div>
          <ul class="swot-bullets">{bullets_html}</ul>
          {assumptions_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


if st.session_state.initialized and st.session_state.swot:
    sw = st.session_state.swot

    strengths = sw.get("strengths", [])
    weaknesses = sw.get("weaknesses", [])
    opportunities = sw.get("opportunities", [])
    threats = sw.get("threats", [])
    assumptions = sw.get("assumptions", None)  # dict quadrant->list or global list

    st.markdown('<div class="swot-grid"></div>', unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        render_card(
            "Strengths",
            strengths,
            "theme-strengths",
            assumptions.get("strengths") if isinstance(assumptions, dict) else None,
        )
    with r1c2:
        render_card(
            "Weaknesses",
            weaknesses,
            "theme-weaknesses",
            assumptions.get("weaknesses") if isinstance(assumptions, dict) else None,
        )

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        render_card(
            "Opportunities",
            opportunities,
            "theme-opportunities",
            assumptions.get("opportunities") if isinstance(assumptions, dict) else None,
        )
    with r2c2:
        render_card(
            "Threats",
            threats,
            "theme-threats",
            assumptions.get("threats") if isinstance(assumptions, dict) else None,
        )

    # Optional extras
    st.subheader("Key insight")
    key_insights = sw.get("key_insights", [])
    if key_insights:
        st.write("\n".join([f"- {k}" for k in key_insights]))
    else:
        st.write("- No key insights available.")

    st.subheader("Action suggestions")
    actions = sw.get("action_suggestions", [])
    if actions:
        st.write("\n".join([f"- {a}" for a in actions]))
    else:
        st.write("- No action suggestions available.")

    # 30-day action plan (generated by a second sequential ADK main agent).
    if st.session_state.action_plan_30d:
        st.subheader("30-day Action Plan")
        st.text(st.session_state.action_plan_30d)

    if include_assumptions:
        st.subheader("Assumptions used")
        if isinstance(assumptions, dict):
            for q in ["strengths", "weaknesses", "opportunities", "threats"]:
                items = assumptions.get(q, [])
                if items:
                    st.write(f"**{q.capitalize()}**")
                    st.write("\n".join([f"- {x}" for x in items]))
        elif isinstance(assumptions, list) and assumptions:
            st.write("\n".join([f"- {x}" for x in assumptions]))
        else:
            st.write("- None.")

    # ----------------------------
    # Export / download
    # ----------------------------
    st.subheader("Download")

    txt_data = swot_to_txt(sw)
    md_data = swot_to_md(sw)
    json_data = json.dumps(sw, ensure_ascii=False, indent=2)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            label="Download TXT",
            data=txt_data,
            file_name="swot.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="swot.json",
            mime="application/json",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            label="Download Markdown",
            data=md_data,
            file_name="swot.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # ----------------------------
    # Additional ADK Agents
    # ----------------------------
    st.subheader("Agents")
    a1, a2 = st.columns(2)

    # Erstelle PDF Agent
    with a1:
        if st.button("PDF Agent: Create PDF", use_container_width=True):
            try:
                pdf_obj = pdf_agent_create_pdf_base64(
                    api_key=st.session_state.api_key,
                    title=f"SWOT – {sw.get('venture_name','').strip() or 'Venture'}",
                    content=(
                        md_data
                        + ("\n\n\n30-day Action Plan\n" + st.session_state.action_plan_30d if st.session_state.action_plan_30d else "")
                    ),
                )
                st.session_state.pdf_base64 = (pdf_obj or {}).get("base64", "")
            except Exception as e:
                st.session_state.pdf_base64 = ""
                st.error(f"PDF creation failed: {e}")

        if st.session_state.pdf_base64:
            import base64

            pdf_bytes = base64.b64decode(st.session_state.pdf_base64)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name="swot.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    # Email Agent
    with a2:
        recipient_email = st.text_input("Recipient email", value="", key="recipient_email")
        recipient_name = st.text_input("Recipient name (optional)", value="", key="recipient_name")

        if st.button("Email Agent: Draft email", use_container_width=True):
            try:
                draft = email_agent_draft_email(
                    api_key=st.session_state.api_key,
                    swot_markdown=md_data,
                    recipient_name=recipient_name,
                )
                st.session_state.email_subject = draft.subject
                st.session_state.email_body = draft.body
            except Exception as e:
                st.error(f"Email draft failed: {e}")

        if st.session_state.email_subject or st.session_state.email_body:
            st.text_input("Subject", value=st.session_state.email_subject, key="email_subject_view")
            st.text_area("Body", value=st.session_state.email_body, height=160, key="email_body_view")

            # Optional: send via SMTP (credentials required).
            with st.expander("Send via SMTP (optional)", expanded=False):
                smtp_host = st.text_input("SMTP host", value=os.getenv("SMTP_HOST", ""), key="smtp_host")
                smtp_port = st.number_input("SMTP port", value=int(os.getenv("SMTP_PORT", "587") or 587), key="smtp_port")
                smtp_user = st.text_input("SMTP user", value=os.getenv("SMTP_USER", ""), key="smtp_user")
                smtp_password = st.text_input("SMTP password", value=os.getenv("SMTP_PASSWORD", ""), type="password", key="smtp_password")
                sender = st.text_input("From (sender)", value=os.getenv("SMTP_SENDER", smtp_user), key="smtp_sender")

                if st.button("Send email now", use_container_width=True, key="send_email_now"):
                    if not recipient_email.strip():
                        st.error("Recipient email is required.")
                    else:
                        try:
                            send_email_smtp(
                                smtp_host=smtp_host,
                                smtp_port=int(smtp_port),
                                smtp_user=smtp_user,
                                smtp_password=smtp_password,
                                sender=sender,
                                recipient=recipient_email.strip(),
                                subject=st.session_state.email_subject or "SWOT results",
                                body=st.session_state.email_body or md_data,
                            )
                            st.success("Email sent.")
                        except Exception as e:
                            st.error(f"Sending failed: {e}")

    # ----------------------------
    # Debug expander
    # ----------------------------
    with st.expander("Debug", expanded=False):
        st.write("**Inputs used**")
        st.json(st.session_state.last_request)

        st.write("**Raw model output**")
        st.code(st.session_state.raw_model_output or "", language="text")

        st.write("**Parsed / validated JSON**")
        st.code(json.dumps(sw, ensure_ascii=False, indent=2), language="json")

        if st.session_state.validation_warnings:
            st.write("**Validation warnings**")
            for w in st.session_state.validation_warnings:
                st.write(f"- {w}")

    # ----------------------------
