"""
agents/attachment_agent.py

Handles questions that come with a user-uploaded attachment (document or image).

- Text documents (PDF/DOCX/PPTX/TXT): the extracted text is made available to the
  routed domain agent as supplementary context (via the shared attachment_context
  holder). The agent still grounds its answer strictly in the Tessent manuals.
- Images (PNG/JPG/etc.): the image is sent to the Gemini vision model together
  with manual context retrieved via RAG, using the same strict grounding rules.
"""
from __future__ import annotations

from agents.attachment_context import set_attachment_context, clear_attachment_context
from agents import router_agent
from core.file_reader import MIME_MAP
from core.llm import llm
from core.rag_builder import retrieve

# Domain -> agent module's ask function (mirrors ui.py agent_map + GENERAL fallback)
AGENT_MAP = {
    "SCAN": ("agents.scan_agent", "ask_scan"),
    "ATPG": ("agents.atpg_agent", "ask_atpg"),
    "EDT": ("agents.edt_agent", "ask_edt"),
    "MBIST": ("agents.mbist_agent", "ask_mbist"),
    "JTAG": ("agents.jtag_agent", "ask_jtag"),
    "IJTAG": ("agents.ijtag_agent", "ask_ijtag"),
    "WRAPPER": ("agents.wrapper_agent", "ask_wrapper"),
    "OCC": ("agents.occ_agent", "ask_occ"),
    "STA": ("agents.sta_agent", "ask_sta"),
    "BOUNDARYSCAN": ("agents.boundaryscan_agent", "ask_boundaryscan"),
    "GLS": ("agents.gls_agent", "ask_gls"),
    "LINUX": ("agents.linux_agent", "ask_linux"),
    "TCL": ("agents.tcl_agent", "ask_tcl"),
    "GENERAL": ("agents.general_agent", "ask_general"),
}


def _route(question):
    """Route a question to a domain, falling back to GENERAL on any error."""
    try:
        topic = router_agent.route_question(question)
        if topic not in AGENT_MAP:
            topic = "GENERAL"
        return topic
    except Exception:
        return "GENERAL"


def _call_domain_agent(topic: str, question: str, history):
    """Import and call the routed domain agent's ask_<domain> function."""
    if topic not in AGENT_MAP:
        topic = "GENERAL"
    module_name, func_name = AGENT_MAP[topic]
    import importlib

    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func(question, history)


def ask_with_text_attachment(question, attachment_text, history=None, _topic=None):
    """Answer a question given a text attachment.

    Sets the attachment context, routes and runs the domain agent, then clears.
    """
    topic = _topic or _route(question)
    try:
        set_attachment_context(attachment_text)
        return _call_domain_agent(topic, question, history)
    finally:
        clear_attachment_context()


def ask_with_image(question, image_bytes, image_name, history=None, _topic=None):
    """Answer a question that includes an image (waveform, schematic, error log, slide...).

    Retrieves relevant manual context via RAG, then sends the image + question to
    the multimodal Gemini model with strict manual-only grounding.
    """
    topic = _topic or _route(question)
    ext = image_name.lower().rsplit(".", 1)[-1]
    mime = MIME_MAP.get("." + ext, "image/png")

    # Ground with manual context
    results = retrieve(question, top_k=6)
    if results:
        context_parts = []
        for r in results:
            context_parts.append(
                f"Source Document: {r['source']} (Page {r['page']})\nContent: {r['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "<NO MANUAL CONTEXT WAS RETRIEVED>"

    prompt = f"""GROUNDING RULES (MANDATORY):
1. Answer ONLY from the "Reference Context" provided below.
2. Do NOT invent commands, options, arguments, errors, or file types not present in the context.
3. For EVERY claim, append a citation [Source: <file>, Page <N>] taken from the context.
4. If the context does not contain the answer, write:
   "The retrieved manual context does not contain enough detail to answer this definitively."
   and DO NOT guess.
5. Never rely on generic background knowledge for specific command options or behaviors.
6. You have been given an image ({image_name}). Use the image (e.g. a waveform, schematic,
   error message, lab slide, or screenshot) to understand the user's question, but you must
   NOT present image-derived content as a manual citation. Cite only the Reference Context.

You are a Senior Tessent DFT Engineer.
Reference Context from Tessent Manuals & Verified Notes:
{context}

The user's question (alongside the attached image) is:
{question}

Instructions:
1. Provide a comprehensive, accurate answer using ONLY the reference context above.
2. Cite each supporting fact with [Source: <file>, Page <N>].
3. If a specific detail is not supported by the context, say so instead of guessing.
4. Structure your answer with clear headings, bullet points, and comparison tables where appropriate.
"""

    response = llm.invoke_vision(prompt, image_bytes, mime)
    return response.content.strip()