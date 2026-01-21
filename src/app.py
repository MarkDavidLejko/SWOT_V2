import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional
import html
import streamlit as st

from adk_extra_agents import (
    time_agent_get_time,
    pdf_agent_create_pdf_base64,
    email_agent_draft_email,
    send_email_smtp,
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
}

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_state() -> None:
    keep_api_key = st.session_state.get("api_key", "")
    for k, v in DEFAULT_STATE.items():
        st.session_state[k] = v
    st.session_state["api_key"] = keep_api_key


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

# ----------------------------
# Generation flow
# ----------------------------
if generate:
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
    a1, a2, a3 = st.columns(3)

    # Zeit Agent
    with a1:
        if st.button("Zeit Agent: Show time", use_container_width=True):
            try:
                st.session_state.time_text = time_agent_get_time(api_key=st.session_state.api_key)
            except Exception as e:
                st.session_state.time_text = f"Error: {e}"

        if st.session_state.time_text:
            st.code(st.session_state.time_text, language="text")

    # Erstelle PDF Agent
    with a2:
        if st.button("PDF Agent: Create PDF", use_container_width=True):
            try:
                pdf_obj = pdf_agent_create_pdf_base64(
                    api_key=st.session_state.api_key,
                    title=f"SWOT – {sw.get('venture_name','').strip() or 'Venture'}",
                    content=md_data,
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
    with a3:
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
