from agents.base_agent import ask_expert

def ask_jtag(question,
    history=None):
    return ask_expert(
        question,
        "You are a JTAG expert",
        """
        TAP Controller
        TMS
        TCK
        TDI
        TDO
        IR
        DR
        Boundary Scan
        """,
        history
    )