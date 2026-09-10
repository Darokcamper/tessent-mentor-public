from agents.base_agent import ask_expert

def ask_occ(question,
    history=None):

    return ask_expert(
        question,
        "OCC Expert",
        """
        OCC
        On Chip Clock Controller
        Shift Clock
        Capture Clock
        At Speed Testing
        Scan Clock
        Functional Clock
        Launch Capture Testing
        """,
        history
    )