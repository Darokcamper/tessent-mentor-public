from agents.base_agent import ask_expert

def ask_general(question,
    history=None):

    return ask_expert(
        question,
        "Senior VLSI Engineer",
        """
        Answer clearly and accurately.
        """,
        history
    )