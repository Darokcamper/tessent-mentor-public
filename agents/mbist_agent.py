from agents.base_agent import ask_expert

def ask_mbist(question,
    history=None):

    return ask_expert(
        question,
        "MBIST Expert",
        """
        MBIST
        Memory Testing
        SRAM Testing
        March Algorithms
        March C-
        March A
        BIRA
        BISR
        Redundancy Analysis
        Memory Repair
        Word Line
        Bit Line
        """,
        history
    )