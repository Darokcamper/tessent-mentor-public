from core.llm import llm


def ask_interview_question(
    topic,
    difficulty,
    previous_questions=None
):

    previous_questions = previous_questions or []

    prompt = f"""
You are a senior semiconductor DFT interviewer.

Topic: {topic}
Difficulty: {difficulty}

Questions already asked:
{chr(10).join(previous_questions)}

TASK:

Generate ONE completely new interview question.

Requirements:

- Do not repeat any previous question.
- Do not paraphrase a previous question.
- Choose a different concept each time.
- Make it realistic for a VLSI/DFT interview.
- Return only the question.
- No explanations.
- No numbering.
- Beginner for Freshers, so the question should be simple and straightforward.
- If the difficulty is Intermediate, the question should be moderately challenging.
- If the difficulty is Advanced, the question should be complex and require deep understanding.

IMPORTANT:

If Topic is SCAN, possible concepts include:
scan chain,
scan enable,
scan flop,
controllability,
observability,
shift mode,
capture mode,
lockup latch,
clock mixing,
scan stitching,
scan compression,
scan architecture,
scan diagnosis,
scan debugging.

If Topic is ATPG, possible concepts include:
stuck-at faults,
transition faults,
fault coverage,
test coverage,
fault simulation,
pattern generation,
test points,
X-filling,
dynamic compaction,
static compaction,
ATPG debugging.

Choose a concept NOT already used.
"""

    response = llm.invoke(prompt)

    return response.content.strip()