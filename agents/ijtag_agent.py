from agents.base_agent import ask_expert

def ask_ijtag(question,
    history=None):

    return ask_expert(
        question,
        "IJTAG Expert",
        """
        IEEE 1687
        IJTAG
        SIB
        Segment Insertion Bit
        TDR
        Instrument Access
        Embedded Instruments
        ICL
        PDL
        Instrument Connectivity Language
        Procedural Description Language
        """,
        history
    )