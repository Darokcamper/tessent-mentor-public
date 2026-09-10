from agents.expert_agent import ask_expert

def ask_atpg(question, history=None):

    expertise = """
- ATPG
- Stuck-at Faults
- Transition Faults
- Pattern Generation
- Fault Coverage
- Fault Simulation
- Compression
- EDT
"""

    return ask_expert(
        question,
        "Senior ATPG Engineer",
        expertise,
        history
    )