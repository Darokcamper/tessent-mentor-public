"""Exam Question Bank Agent - serves real past exam questions from the Level 1 Exam.

The exam questions were OCR'd from the training platform screenshots and parsed
into structured Q&A. This agent serves them in quiz mode with answer evaluation.
"""
import json
import re
import random
from pathlib import Path

EXAM_FILE = Path("_exam_parsed.json")

# Module topic mapping for each question (based on exam order and content)
QUESTION_TOPICS = {
    1: "ATPG Flow", 2: "ATPG Flow", 3: "Scan Insertion", 4: "Scan Insertion",
    5: "Scan Insertion", 6: "ATPG Flow", 7: "ATPG Flow", 8: "Scan Insertion",
    9: "Scan Insertion", 10: "ATPG Flow", 11: "ATPG Flow", 12: "ATPG Flow",
    13: "Fault Coverage", 14: "Fault Coverage", 15: "ATPG Flow", 16: "Scan Insertion",
    17: "ATPG Flow", 18: "ATPG Flow", 19: "ATPG Flow", 20: "ATPG Flow",
    21: "Scan Insertion", 22: "ATPG Flow", 23: "ATPG Flow", 24: "ATPG Flow",
    25: "Fault Coverage", 26: "ATPG Flow", 27: "ATPG Flow", 28: "ATPG Flow",
    29: "ATPG Flow", 30: "ATPG Flow", 31: "Scan Insertion", 32: "ATPG Flow",
    33: "ATPG Flow", 34: "ATPG Flow", 35: "ATPG Flow", 36: "ATPG Flow",
    37: "ATPG Flow", 38: "ATPG Flow", 39: "ATPG Flow", 40: "ATPG Flow",
    41: "ATPG Flow", 42: "ATPG Flow", 43: "ATPG Flow", 44: "ATPG Flow",
    45: "ATPG Flow", 46: "ATPG Flow", 47: "ATPG Flow", 48: "ATPG Flow",
}


# Noise phrases to strip from question text (OCR/page-header artifacts)
NOISE_PHRASES = [
    r'^\d+\s*---\s*',
    r'The D\s*X\s*MySk\s*×?\s*DFT-\s*[bIX×]*\s*Vanta\s*×?\s*Tesse\s*X?\s*Tessei?\s*X?\s*Mail\s*Tesse\s*Tesse\s*lesse\s*',
    r'Training & support\s*',
    r'Training and support\s*',
    r'Solutions & services\s*',
    r'Solutions& services\s*',
    r'Solutions &services\s*',
    r'Training & support\s*',
    r'Industries\s*',
    r'Software & products\s*',
    r'Chat\s*',
    r'Complete 90-minute exam of 50 questions.*?badge\.?\s*',
    r'Newi?\s*',
    r'New\s*',
    r'SX\s*',
    r'ASSEs?SMENT\s*',
    r'Level 1 Exam\s*-\s*',
    r'Tessent ATPG\s*',
    r'Core Topics\s*',
    r'---\s*',
    r'^\d+\s*',  # leading question number
]


def clean_text(text):
    """Strip OCR/page-header noise from question text.
    
    The actual question starts after the last 'ASSESSMENT' or 'Core Topics'
    marker from the training portal header. Everything before that is UI noise.
    """
    # Find the last occurrence of the portal markers
    markers = ['Core Topics', 'ASSESSMENT']
    last_pos = -1
    last_marker_len = 0
    for marker in markers:
        pos = text.rfind(marker)
        if pos > last_pos:
            last_pos = pos
            last_marker_len = len(marker)
    if last_pos > 0:
        text = text[last_pos + last_marker_len:]
    # Final cleanup
    text = text.strip()
    # Remove leading question number if any
    text = re.sub(r'^\d+\s*', '', text)
    return text


def load_questions():
    """Load parsed exam questions from JSON, with cleaned text."""
    if not EXAM_FILE.exists():
        return []
    with open(EXAM_FILE, encoding='utf-8') as f:
        questions = json.load(f)
    # Clean question text
    for q in questions:
        q['text'] = clean_text(q['text'])
    return questions


def get_questions_by_topic(topic=None, count=None, shuffle=False):
    """Get exam questions, optionally filtered by topic.
    
    Args:
        topic: Filter by topic (e.g. "ATPG Flow", "Scan Insertion", "Fault Coverage")
        count: Maximum number of questions to return
        shuffle: Whether to randomize order
    
    Returns:
        List of question dicts with keys: num, text, correct, options
    """
    questions = load_questions()
    if topic:
        questions = [q for q in questions if QUESTION_TOPICS.get(q['num']) == topic]
    if shuffle:
        questions = questions.copy()
        random.shuffle(questions)
    if count:
        questions = questions[:count]
    return questions


def get_topics():
    """Return available topics."""
    return sorted(set(QUESTION_TOPICS.values()))


def format_question(q, show_answer=False):
    """Format a single question for display."""
    lines = [f"**Q{q['num']}:** {q['text']}"]
    if q.get('options'):
        lines.append("")
        for i, opt in enumerate(q['options'], 1):
            lines.append(f"  {i}. {opt}")
    if show_answer and q.get('correct'):
        lines.append("")
        lines.append(f"✅ **Answer:** {', '.join(q['correct'])}")
    return '\n'.join(lines)


def format_quiz(questions, show_answers=False):
    """Format multiple questions as a quiz."""
    parts = ["# 📝 Level 1 Exam Practice\n"]
    for q in questions:
        parts.append(format_question(q, show_answer=show_answers))
        parts.append("\n---\n")
    return '\n'.join(parts)


def evaluate_answer(question, user_answer):
    """Evaluate a user's answer against the correct answer.
    
    Returns:
        dict with keys: correct (bool), correct_answer, feedback
    """
    correct_answers = question.get('correct', [])
    user_clean = user_answer.strip().lower()
    
    # Check if user's answer matches any correct answer
    is_correct = False
    for ca in correct_answers:
        ca_clean = ca.strip().lower()
        if user_clean == ca_clean:
            is_correct = True
            break
        # Check if user selected the right option number
        if user_clean.isdigit():
            idx = int(user_clean) - 1
            if 0 <= idx < len(question.get('options', [])):
                if question['options'][idx].lower() == ca_clean:
                    is_correct = True
                    break
    
    feedback = ""
    if is_correct:
        feedback = "✅ Correct!"
    else:
        feedback = f"❌ Incorrect. The correct answer is: **{', '.join(correct_answers)}**"
    
    return {
        'correct': is_correct,
        'correct_answer': correct_answers,
        'feedback': feedback,
    }
