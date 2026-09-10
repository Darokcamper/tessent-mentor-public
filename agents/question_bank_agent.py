from core.llm import llm
from core.rag_builder import retrieve

def generate_assessment_question_bank(module: str, num_questions: int = 20) -> str:
    """
    Generates a targeted assessment question bank for a specific module (SCAN, ATPG, EDT, LINUX/TSHELL)
    strictly grounded in Tessent reference manuals and lab flow mechanics.
    """
    query = f"{module} Tessent manual lab flow commands DRC rules DRC violations failure modes patterns"
    chunks = retrieve(query, top_k=10)
    
    context_text = ""
    if chunks:
        context_parts = []
        for c in chunks:
            context_parts.append(f"Source: {c['source']} (Page {c['page']})\n{c['text']}")
        context_text = "\n\n---\n\n".join(context_parts)
    else:
        context_text = "<NO MANUAL CONTEXT WAS RETRIEVED>"

    prompt = f"""GROUNDING RULES (MANDATORY):
1. Answer ONLY from the "Reference Context from Tessent Manuals" below.
2. Do NOT invent commands, options, arguments, error messages, or file types not present in the context.
3. For EVERY model answer, append a citation [Source: <file>, Page <N>] taken from the context.
4. If the context does not cover a category, say so instead of fabricating questions.
5. Never rely on generic background knowledge for specific command options or behaviors.

You are a Lead Tessent POC Assessor.
Generate a comprehensive {num_questions}-question Assessment & Viva Bank for the module: **{module}**.

Reference Context from Tessent Manuals:
{context_text}

Requirements:
1. Generate exactly {num_questions} high-yield viva questions commonly asked by POCs during lab evaluation,
   grounded ONLY in the reference context provided.
2. Structure the questions across 4 key categories:
   - Category A: Command Rationale & Flow Placement
   - Category B: Internal DB Mechanics & Object Definitions
   - Category C: DRC Rules, Failures & Missing Command Scenarios
   - Category D: Output Artifacts, Reports & Pattern Verification
3. For EVERY question, provide:
   - **Question**: Clear, direct POC question.
   - **Why It Is Asked**: What concept the POC is testing.
   - **Benchmark Model Answer**: Grounded ONLY in the context, with a [Source: <file>, Page <N>] citation.
   - **Manual Citation**: Relevant manual/section reference from the context.

If the context does not contain enough detail for a particular benchmark answer, state that plainly
instead of inventing facts.

Output in clean Markdown format with clear headings.
"""

    response = llm.invoke(prompt)
    return response.content.strip()
