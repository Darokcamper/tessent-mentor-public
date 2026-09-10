import re
from core.llm import llm
from core.rag_builder import retrieve

def _format_chunks(chunks):
    """Formats retrieved chunks into plain text with source+page headers."""
    if not chunks:
        return ""
    parts = []
    for c in chunks:
        parts.append(f"Source: {c['source']} (Page {c['page']})\n{c['text']}")
    return "\n\n---\n\n".join(parts)

# Hard constraints injected into every prompt to prevent hallucination.
GROUNDING_RULES = """\
HARD RULES - DO NOT VIOLATE THESE:
1. Answer ONLY from the "Reference Manual Excerpts" provided below.
2. Do NOT invent command names, options, flags, arguments, errors, or file types that are
   NOT explicitly present in the excerpts. If an option is not in the excerpts, do not mention it.
3. For EVERY claim or fact, append a citation in this exact format:
   [Source: <file>, Page <N>] where <file>/<N> come from the excerpts.
4. If the excerpts do NOT contain the answer to a question, write:
   "The provided manual excerpts do not contain enough detail to answer this definitively."
   and DO NOT guess.
5. Never rely on generic Tessent background knowledge or your training data for specific
   command options or behaviors. Specific options ONLY come from the excerpts.
6. If the excerpts section is empty, output ONLY the message from rule 4 and stop.
"""

def explain_lab_command(command_or_script: str, topic: str = "GENERAL") -> str:
    """
    Provides a comprehensive, Tessent manual-grounded breakdown of a lab script or command.
    Refuses to hallucinate: every fact must trace back to a retrieved manual excerpt with a
    source+page citation.
    """
    # Formulate RAG query
    query = f"{topic} Tessent command {command_or_script} options arguments output files errors flow"
    
    # Retrieve top 12 context chunks for broader, higher-quality grounding
    chunks = retrieve(query, top_k=12)
    
    context_text = _format_chunks(chunks)
    if not context_text:
        context_text = "<NO MANUAL EXCERPTS WERE RETRIEVED>"

    prompt = f"""\
GROUNDING RULES (MANDATORY):
{GROUNDING_RULES}

You are a Principal Tessent DFT Architect and Senior Technical Assessor.
Your task is to explain the following Tessent lab command or script sequence, using ONLY the
reference manual excerpts provided.

Module Focus: {topic}
Input Command / Script Sequence:
```tcl
{command_or_script}
```

Reference Manual Excerpts:
{context_text}

Using ONLY the excerpts above, provide a structured markdown response covering:

### 1. What the command does
- What the command/script does, strictly based on the excerpts (cite source+page).

### 2. Arguments and Options
- List ONLY the arguments/options that appear in the excerpts, with their exact meaning.
- If NO options appear in the excerpts for this command, state that explicitly instead of guessing.

### 3. Script Flow / Placement
- Placement and prerequisite steps, ONLY if the excerpts mention them. Otherwise, say it is not specified.

### 4. Output Files and Reports
- ONLY files/reports mentioned in the excerpts. Otherwise state they are not specified.

### 5. What Happens If Skipped / Common Failures
- ONLY error messages or behaviors present in the excerpts. Otherwise state they are not specified.

### 6. Top 5 POC Viva Questions & Model Answers
- 5 realistic POC questions. Answers MUST be grounded only in the excerpts, each with a
  [Source: <file>, Page <N>] citation.

### 7. Manual Citations
- List each manual/PDF and page number actually used from the excerpts.

IMPORTANT: If the excerpts are empty or a topic is not covered, use rule 4's exact message and
provide no fabricated detail.
"""

    response = llm.invoke(prompt)
    return response.content.strip()
