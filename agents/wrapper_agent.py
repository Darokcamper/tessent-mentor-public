from agents.base_agent import ask_expert

def ask_wrapper(question,
    history=None):

    return ask_expert(
        question,
        "IEEE 1500 Wrapper Expert",
        """
        IEEE 1500
        Wrapper Cells
        Wrapper Chains
        Core Test
        SOC Testing
        Core Isolation
        Wrapper Enable
        Wrapper Boundary Register
        """,
        history
    )