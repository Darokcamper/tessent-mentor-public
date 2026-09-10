from agents.base_agent import ask_expert

def ask_edt(question,
    history=None):

    return ask_expert(
        question,
        "EDT Compression Expert",
        """
        EDT
        Decompressor
        Compactor
        MISR
        Phase Shifter
        Compression Ratio
        """,
        history
    )