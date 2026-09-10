from agents.base_agent import ask_expert

def ask_boundaryscan(question, history=None):
    expertise = """
    - Boundary Scan (IEEE 1149.1)
    - Boundary Scan Cell (BSC)
    - TAP Controller (TAP State Diagram, TMS, TCK, TDI, TDO)
    - Instruction Register (IR) and Data Register (DR)
    - Standard Instructions (BYPASS, EXTEST, INTEST, SAMPLE, PRELOAD, IDCODE)
    - Boundary Scan Description Language (BSDL files)
    - Test Access Port (TAP) signals and timing
    """
    return ask_expert(
        question,
        "Senior Boundary Scan & JTAG Architect",
        expertise,
        history
    )
