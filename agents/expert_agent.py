from agents.base_agent import search_verified_notes
from agents.attachment_context import get_attachment_context
from core.llm import llm

def build_expert_prompt(question, persona_title, expertise_area, history=None, extra_context=None):
    # Context-aware query rewriting for follow-ups (prevents losing context on queries like 'why?' or 'what about 3 and 4')
    retrieval_query = question.strip()
    followup_words = ["it", "this", "that", "they", "these", "those", "why", "how", "what about", "what if", "explain", "elaborate", "tell me more"]
    q_lower = question.lower()
    q_words = q_lower.split()

    if history and (len(q_words) <= 6 or any(w in q_words for w in followup_words)):
        for msg in reversed(history):
            if msg.get("role") == "user":
                prev_text = msg.get("content", "").split("\n\n[📎 Attached:")[0].strip()
                if prev_text:
                    retrieval_query = f"{prev_text} {question}"
                break

    # Retrieve relevant manual context
    context = search_verified_notes(retrieval_query)
    if not context or not context.strip() or any(t in context.lower() for t in ["no verified", "no matching", "not found", "error"]):
        context = "<NO MANUAL CONTEXT WAS RETRIEVED>"

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
1. Provide a comprehensive, accurate answer using ONLY the reference context above.
2. Cite each supporting fact with [Source: <file>, Page <N>].
3. If a specific detail is not supported by the context, say so instead of guessing.
4. Structure your answer with clear headings, bullet points, and comparison tables where appropriate.
5. If this question is a follow-up, maintain conversational continuity while strictly grounding technical facts in the Reference Context.
"""
    return prompt

def stream_expert(question, persona_title, expertise_area, history=None, extra_context=None):
    """Generator yielding response tokens for real-time streaming."""
    prompt = build_expert_prompt(question, persona_title, expertise_area, history, extra_context)
    for chunk in llm.stream(prompt):
        if hasattr(chunk, "content"):
            content = chunk.content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        yield part["text"]
                    elif isinstance(part, str):
                        yield part
            elif isinstance(content, str):
                yield content
        else:
            yield str(chunk)

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
    return response.content.strip()
