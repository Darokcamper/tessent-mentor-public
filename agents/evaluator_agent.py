from core.llm import llm

def evaluate_answer(question, answer):

    prompt = f"""
You are evaluating a FRESHER DFT candidate.

Question:
{question}

Candidate Answer:
{answer}

Rules:

1. Give credit for partial understanding.
2. Don't expect textbook answers.
3. If candidate identifies the main idea,
   award reasonable marks.
4. Focus on understanding rather than completeness.

Scoring:

10 = Excellent
8-9 = Strong answer
6-7 = Understands concept
4-5 = Partial understanding
0-3 = Incorrect

Output:

Score

Correct Concepts Identified

Missing Concepts

Ideal Interview Answer
"""

    response = llm.invoke(prompt)

    return response.content