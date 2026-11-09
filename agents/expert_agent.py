from __future__ import annotations

import re

from agents.base_agent import (
    NO_CONTEXT_SENTINEL,
    retrieval_failed,
    search_verified_notes,
)
from agents.attachment_context import get_attachment_context
from core.llm import llm
from core.answer_guard import guard_or_strip

def build_expert_prompt(question, persona_title, expertise_area, history=None, extra_context=None):
    # WP-8: context retention for follow-ups lives in one tested helper; vocabulary
    # expansion is opt-in because it measured neutral (see core/query_understanding).
    from core.query_understanding import build_retrieval_query

    retrieval_query = build_retrieval_query(question, history)

    # Retrieve relevant manual context
    context = search_verified_notes(retrieval_query)
    if retrieval_failed(context):
        # WP-1: only a genuine retrieval failure becomes the sentinel. The old
        # check matched the words "error"/"not found"/"no matching" anywhere in
        # the manual text, so perfectly good evidence was discarded whenever the
        # manual happened to discuss an error message.
        context = NO_CONTEXT_SENTINEL

    attachment_block = ""
    if not extra_context:
        extra_context = get_attachment_context()
    if extra_context:
        attachment_block = (
            "\n\nUser-Attached Document Content (for understanding the question only; "
            "do NOT cite this as a manual source):\n"
            "--- BEGIN ATTACHMENT ---\n"
            f"{extra_context}\n"
            "--- END ATTACHMENT ---"
        )

    history_block = ""
    if history:
        history_lines = []
        for msg in history[-4:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            text = str(msg.get("content", "")).strip()
            if len(text) > 400:
                text = text[:400] + "..."
            if text:
                history_lines.append(f"{role}: {text}")
        if history_lines:
            history_block = "\n\nRecent Conversation History:\n" + "\n".join(history_lines)

    expertise_block = ""
    if expertise_area and expertise_area.strip():
        expertise_block = f"\nCore Domain Focus:\n{expertise_area.strip()}\n"

    prompt = f"""GROUNDING RULES (MANDATORY):
1. Answer ONLY from the "Reference Context" provided below.
2. Do NOT invent commands, options, arguments, errors, or file types not present in the context.
3. For EVERY claim, append a citation [Source: <file>, Page <N>] taken from the context.
   Copy the file name and page exactly as the Reference Context prints them. Never invent a
   file name or page number: a citation you cannot point to is worse than none. If a claim
   has no supporting line in the context, say so instead of citing.
4. If the context does not contain the answer, write:
   "The retrieved manual context does not contain enough detail to answer this definitively."
   and DO NOT guess.
5. Never rely on generic background knowledge for specific command options or behaviors.
6. If the user attached a document, you may use its text to understand the question or
   the user's specific scenario, but you must not present attachment content as if it came
   from a Tessent manual. Cite only the Reference Context.

You are an expert {persona_title} specializing in Tessent DFT workflows.{expertise_block}
Reference Context from Tessent Manuals & Verified Notes:
{context}
{attachment_block}
{history_block}

Question / Issue:
{question}

Instructions:
1. Always start your response with a dedicated "### 🧑‍🏫 Layman's Explanation" section: explain the core concept, command, or issue in simple, plain, beginner-friendly terms with an intuitive everyday analogy (e.g. plumbing, highway traffic, postal mail, airport security) so anyone without deep DFT experience can grasp it immediately.
2. Follow with a comprehensive, accurate technical answer using ONLY the reference context above.
3. Cite each supporting fact with [Source: <file>, Page <N>].
4. If a specific detail is not supported by the context, say so instead of guessing.
5. Structure your answer with clear headings, bullet points, and comparison tables where appropriate.
6. If this question is a follow-up, maintain conversational continuity while strictly grounding technical facts in the Reference Context.
"""
    return prompt

def stream_expert(question, persona_title, expertise_area, history=None, extra_context=None):
    """Generator yielding the *verified* answer as text pieces.

    The tokens are buffered until the model finishes. A citation can only be judged once
    its page text is in hand, and a fabricated pointer must never be displayed even for a
    moment, so the guard runs before the first piece is yielded. The generator interface is
    unchanged, so ``st.write_stream`` and the caller keep working.
    """
    prompt = build_expert_prompt(question, persona_title, expertise_area, history, extra_context)
    collected = []
    try:
        for chunk in llm.stream(prompt):
            if hasattr(chunk, "content"):
                content = chunk.content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            collected.append(part["text"])
                        elif isinstance(part, str):
                            collected.append(part)
                elif isinstance(content, str):
                    collected.append(content)
            else:
                collected.append(str(chunk))
    except Exception as error:  # pragma: no cover - transport failure mid-stream
        collected.append(f"\n\n[stream interrupted: {type(error).__name__}]")

    answer = guard_or_strip("".join(collected))
    for piece in re.findall(r"\S+\s*", answer):
        yield piece

def ask_expert(question, persona_title, expertise_area, history=None, extra_context=None, stream=False):
    """
    Generic expert agent helper function that retrieves Tessent manual chunks
    via RAG search and returns a detailed answer, strictly grounded in the
    retrieved manual context to prevent hallucination.
    """
    if stream:
        return stream_expert(question, persona_title, expertise_area, history, extra_context)
    prompt = build_expert_prompt(question, persona_title, expertise_area, history, extra_context)
    response = llm.invoke(prompt)
    # Nothing leaves the agent until every citation is proved against the corpus.
    return guard_or_strip(response.content.strip())
