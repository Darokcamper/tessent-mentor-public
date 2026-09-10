from core.llm import llm

def generate_study_plan(asked_questions, evaluations):
    """
    Generates a personalized study plan based on a user's practice interview history and evaluations.
    
    asked_questions: List of questions the user was asked.
    evaluations: List of evaluation reports (containing scores and feedback).
    """
    if not asked_questions or not evaluations:
        return """
        ### 📚 Personalized DFT Study Planner
        No interview evaluation history found yet. 
        Please practice some interview questions in the **Interview Practice** mode to generate your study plan!
        """
        
    history_text = ""
    for idx, (q, e) in enumerate(zip(asked_questions, evaluations)):
        history_text += f"--- QUESTION {idx + 1} ---\nQuestion: {q}\nEvaluation:\n{e}\n\n"
        
    prompt = f"""You are a Senior semiconductor DFT Mentor & Career Coach.
         
Analyze the candidate's recent practice interview performance below:

{history_text}

TASK:
Create a highly detailed, professional, and encouraging Study Plan and Progress Report in markdown.

Your output must include:
1. **Performance Overview**:
   - Calculate or estimate the average score.
   - Assign a proficiency tier: Novice (0-4), Competent (5-7), Proficient (8-10).
   - A short encouraging opening statement.

2. **Strengths Identified**:
   - Highlight the specific DFT concepts the user demonstrated strong understanding of.

3. **Weak Areas & Core Gaps**:
   - Identify topics or specific areas where the candidate lost points, had partial understanding, or showed misconceptions.

4. **7-Day Actionable Study Guide**:
   - Day-by-day study roadmap.
   - For each day, suggest a specific topic (e.g., Lockup latches, March C- algorithm, EDT Compactor, IEEE 1500 wrapper instructions).
   - Give 1-2 core questions they should be able to answer by the end of that day.

Return the response in clean, professional markdown with clear headings, bold text, and bullet points.
"""
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating study plan: {e}"
