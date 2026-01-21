"""LLM client wrapper.

This project originally used the `google-genai` SDK directly.
It has been migrated to Google ADK (Agent Development Kit) so the
"model call" is executed via an ADK `Agent` + `Runner`.

Public surface area is intentionally kept identical:

    generate_text(api_key, prompt, model) -> str

so the rest of the app keeps all features unchanged.
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid


def _run_sync(coro):
    """Run an async coroutine from sync code.

    Streamlit apps are typically synchronous (no running event loop).
    If a loop *is* already running (e.g., notebook), we execute the coroutine
    in a dedicated thread to avoid nested-loop issues.
    """

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            raise RuntimeError("running_loop")
    except RuntimeError:
        return asyncio.run(coro)

    # Fallback: run in a new thread with its own event loop.
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


async def _generate_text_async(api_key: str | None, prompt: str, model: str) -> str:
    """Async implementation using ADK."""

    try:
        from google.adk.agents.llm_agent import Agent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Missing dependency: google-adk (and its transitive deps). Install it in your environment."
        ) from e

    # ADK uses GOOGLE_API_KEY by default for Gemini API.
    # If api_key is provided explicitly, we set it for the current process.
    if api_key and api_key.strip():
        os.environ["GOOGLE_API_KEY"] = api_key.strip()

    agent = Agent(
        model=model,
        name="swot_json_agent",
        description="Generates strict JSON outputs for SWOT analysis.",
        instruction=(
            "You are a precise strategy consultant. "
            "You MUST output strictly valid JSON only (no markdown, no commentary)."
        ),
    )

    session_service = InMemorySessionService()
    app_name = "swot_builder"
    user_id = "streamlit_user"
    session_id = f"swot_{uuid.uuid4().hex}"  # unique per call; no cross-request memory
    session = await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    events = runner.run_async(user_id=user_id, session_id=session.id, new_message=content)

    final_text = ""
    async for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            # Join text parts defensively.
            final_text = "".join([p.text or "" for p in event.content.parts])
            break

    return (final_text or "").strip()


def generate_text(api_key: str | None, prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Returns plain text from Gemini via ADK."""
    return _run_sync(_generate_text_async(api_key=api_key, prompt=prompt, model=model))
