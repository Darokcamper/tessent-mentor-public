from agents.expert_agent import ask_expert

def ask_sta(question, history=None):

    expertise = """
- Static Timing Analysis
- Setup Violations
- Hold Violations
- Clock Skew
- Clock Latency
- OCV
- Timing Closure
- SDC Constraints
"""

    return ask_expert(
        question,
        "Senior STA Engineer",
        expertise,
        history
    )