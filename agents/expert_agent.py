from agents.base_agent import search_verified_notes
from agents.attachment_context import get_attachment_context
from core.llm import llm

def ask_expert(question, persona_title, expertise_area, history=None, extra_context=None):
    """
    Generic expert agent helper function that retrieves Tessent manual chunks
    via RAG search and returns a detailed answer, strictly grounded in the
    retrieved manual context to prevent hallucination.

    extra_context: optional string (e.g. text extracted from a user-attached
    document). It is provided to the model as supplementary material clearly
    labeled as a user attachment -- NOT as a manual citation source. The answer
    must still be strictly grounded in the manual context and cite it; the
    attachment is used to understand the user's specific question/context.
    """
    # Retrieve relevant manual context
    context = search_verified_notes(question)
    if not context or not context.strip() or "No verified" in context.lower():
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

You are an expert {persona_title} specializing in Tessent DFT workflows.
Reference Context from Tessent Manuals & Verified Notes:
{context}
{attachment_block}

Question / Issue:
{question}

Instructions:
1. Provide a comprehensive, accurate answer using ONLY the reference context above.
2. Cite each supporting fact with [Source: <file>, Page <N>].
3. If a specific detail is not supported by the context, say so instead of guessing.
4. Structure your answer with clear headings, bullet points, and comparison tables where appropriate.
"""

    response = llm.invoke(prompt)
    return response.content.strip()
