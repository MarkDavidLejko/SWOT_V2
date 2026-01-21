"""Additional ADK agents (tools-first).

This project already uses ADK for the core SWOT generation.

Your professor asked for 3 *additional* agents. In ADK, the practical way to
give agents capabilities is via *tools*.

This module defines:

1) TimeAgent
   - Tool: get_current_time
   - Purpose: show current local time (Europe/Berlin) in the UI.

2) PdfAgent
   - Tool: build_pdf_base64
   - Purpose: generate a PDF from text/markdown and return base64.

3) EmailAgent
   - Purpose: draft an email subject/body for the SWOT result.
   - Optional deterministic sender: send_email_smtp (no LLM involved).

All agents are created and run per request (no cross-request memory), matching
the existing project behavior.
"""

from __future__ import annotations

import asyncio
import base64
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from typing import Optional


def _run_sync(coro):
    """Run an async coroutine from sync code (Streamlit-friendly)."""

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            raise RuntimeError("running_loop")
    except RuntimeError:
        return asyncio.run(coro)

    result: list[str] = []
    err: list[BaseException] = []

    def _thread_main():
        try:
            result.append(asyncio.run(coro))
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=_thread_main, daemon=True)
    t.start()
    t.join()

    if err:
        raise err[0]
    return result[0] if result else ""


# -----------------
# Tools (functions)
# -----------------


def get_current_time(tz: str = "Europe/Berlin") -> dict:
    """Return the current time in the requested IANA timezone.

    Args:
        tz: IANA timezone name (default: Europe/Berlin)

    Returns:
        A dict with ISO time, pretty string, and timezone.
    """

    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo(tz))
    except Exception:
        # Fallback: local time if zoneinfo is not available
        now = datetime.now()
        tz = "local"

    return {
        "timezone": tz,
        "iso": now.isoformat(timespec="seconds"),
        "pretty": now.strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_pdf_base64(title: str, content: str) -> dict:
    """Create a simple PDF from text/markdown and return it as base64.

    The tool is intentionally minimal: it renders plain text lines.
    It does not execute HTML/markdown.
    """

    from io import BytesIO

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Missing dependency: reportlab. Add it to requirements.txt and install."
        ) from e

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 50, (title or "Document")[:120])

    # Body
    c.setFont("Helvetica", 10)
    x = 40
    y = height - 80
    line_height = 12

    # Basic wrapping: split on lines; wrap long lines at ~110 chars
    for raw_line in (content or "").splitlines() or [""]:
        line = raw_line.rstrip("\n")
        while len(line) > 110:
            part, line = line[:110], line[110:]
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 50
            c.drawString(x, y, part)
            y -= line_height
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 50
        c.drawString(x, y, line)
        y -= line_height

    c.showPage()
    c.save()

    pdf_bytes = buf.getvalue()
    return {
        "filename": "swot.pdf",
        "mime": "application/pdf",
        "base64": base64.b64encode(pdf_bytes).decode("ascii"),
        "bytes_len": len(pdf_bytes),
    }


def send_email_smtp(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    sender: str,
    recipient: str,
    subject: str,
    body: str,
) -> dict:
    """Send an email via SMTP.

    This is deterministic and does not use the LLM.
    Requires valid SMTP credentials.
    """

    import smtplib

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, int(smtp_port)) as s:
        s.starttls()
        s.login(smtp_user, smtp_password)
        s.send_message(msg)

    return {"ok": True}


@dataclass
class EmailDraft:
    subject: str
    body: str


# -----------------
# ADK agent runners
# -----------------


async def _run_agent_once(*, api_key: Optional[str], agent, prompt: str) -> str:
    """Run an ADK agent once and return final text."""

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    if api_key and api_key.strip():
        os.environ["GOOGLE_API_KEY"] = api_key.strip()

    session_service = InMemorySessionService()
    app_name = "swot_builder"
    user_id = "streamlit_user"
    session_id = f"util_{uuid.uuid4().hex}"
    session = await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)

    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    events = runner.run_async(user_id=user_id, session_id=session.id, new_message=content)
    final_text = ""
    async for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join([p.text or "" for p in event.content.parts])
            break
    return (final_text or "").strip()


def time_agent_get_time(*, api_key: Optional[str], model: str = "gemini-2.5-flash") -> str:
    """ADK TimeAgent: returns current time via tool."""

    try:
        from google.adk.agents.llm_agent import Agent
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: google-adk. Install it.") from e

    agent = Agent(
        model=model,
        name="time_agent",
        description="Shows the current time in a specific timezone.",
        tools=[get_current_time],
        instruction=(
            "Always call get_current_time(tz='Europe/Berlin') exactly once and then reply ONLY with: "
            "<pretty> (<timezone>)\nISO: <iso>"
        ),
    )

    prompt = "Show me the current time now."
    return _run_sync(_run_agent_once(api_key=api_key, agent=agent, prompt=prompt))


def pdf_agent_create_pdf_base64(
    *,
    api_key: Optional[str],
    title: str,
    content: str,
    model: str = "gemini-2.5-flash",
) -> dict:
    """ADK PdfAgent: creates PDF via tool and returns the tool JSON.

    We ask the agent to call the tool and return the JSON as-is.
    """

    try:
        from google.adk.agents.llm_agent import Agent
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: google-adk. Install it.") from e

    agent = Agent(
        model=model,
        name="pdf_agent",
        description="Creates PDFs from provided content.",
        tools=[build_pdf_base64],
        instruction=(
            "Call build_pdf_base64 exactly once with the provided title and content. "
            "Return ONLY the JSON from the tool call (no markdown, no extra text)."
        ),
    )

    prompt = f"Create a PDF. Title: {title}\nContent:\n{content}"

    text = _run_sync(_run_agent_once(api_key=api_key, agent=agent, prompt=prompt))
    # The agent is instructed to output JSON. Parse defensively.
    import json

    try:
        return json.loads(text)
    except Exception:
        # Fallback: if the model didn't comply, create directly (keeps features working).
        return build_pdf_base64(title=title, content=content)


def email_agent_draft_email(
    *,
    api_key: Optional[str],
    swot_markdown: str,
    recipient_name: str = "",
    model: str = "gemini-2.5-flash",
) -> EmailDraft:
    """ADK EmailAgent: drafts an email subject + body."""

    try:
        from google.adk.agents.llm_agent import Agent
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: google-adk. Install it.") from e

    agent = Agent(
        model=model,
        name="email_agent",
        description="Drafts concise professional emails from analysis results.",
        instruction=(
            "Draft a professional email in English. "
            "Return STRICT JSON ONLY with keys: subject, body. "
            "Body should be plain text, no markdown. "
            "Keep it concise, include a short bullet summary of the SWOT and next steps."
        ),
    )

    import json

    prompt = (
        f"Recipient name (optional): {recipient_name}\n"
        "You are emailing the SWOT results below.\n\n"
        f"SWOT (markdown):\n{swot_markdown}\n"
    )

    text = _run_sync(_run_agent_once(api_key=api_key, agent=agent, prompt=prompt))
    try:
        obj = json.loads(text)
        return EmailDraft(subject=str(obj.get("subject", "SWOT results")), body=str(obj.get("body", "")))
    except Exception:
        # Fallback if JSON not returned
        subject = "SWOT results"
        body = "SWOT results:\n\n" + swot_markdown
        return EmailDraft(subject=subject, body=body)
