from core.llm import llm

def evaluate_answer(question, answer):
    prompt = f"""
You are a Senior VLSI Design-for-Test (DFT) Technical Interviewer evaluating a candidate's answer.

Question Asked:
{question}

Candidate's Answer:
{answer}

Evaluation Guidelines:
1. Give credit for partial understanding and practical intuition.
2. Don't penalize for not using verbatim manual phrases if the technical concept is sound.
3. If the candidate identifies the core mechanism, award reasonable marks.
4. Highlight both what was right and what critical engineering details were missed.

Scoring Scale:
10 = Outstanding / Flawless industry-level answer
8-9 = Strong answer with solid understanding of mechanics
6-7 = Good grasp of basic concept, minor omissions
4-5 = Partial understanding with significant gaps or misconceptions
0-3 = Incorrect, confusing, or completely missed the concept

You MUST format your output strictly as follows:

Score: <number from 0 to 10>/10

### ✅ Correct Concepts Identified
- Bullet points listing specific correct ideas, terms, or mechanisms the candidate mentioned.

### ⚠️ Missing Concepts & Gaps
- Key engineering details, commands, options, or failure modes the candidate omitted or explained inaccurately.

### 💡 Model Interview Answer
A concise, high-scoring 2-3 paragraph model answer demonstrating the exact depth expected in a semiconductor DFT interview.

### 🎯 Key Takeaway
One punchy sentence highlighting the golden rule or core principle to remember for this topic.
"""

    response = llm.invoke(prompt)
    return response.content.strip()