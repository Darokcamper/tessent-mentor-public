from agents.expert_agent import ask_expert

def ask_scan(question, history=None):

    expertise = """
- Scan Design
- Scan Insertion
- Scan Stitching
- Scan Chains
- Scan Enable
- Shift and Capture
- Lockup Latches
- Scan Debug
- Scan Architecture
"""

    return ask_expert(
        question,
        "Senior Scan DFT Engineer",
        expertise,
        history
    )