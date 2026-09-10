from agents.base_agent import ask_expert

def ask_gls(question, history=None):
    expertise = """
    - Gate Level Simulation (GLS)
    - Zero-Delay and Timing Simulations
    - Standard Delay Format (SDF) backannotation
    - Setup and Hold timing violation checks in GLS
    - GLS Debugging (X-propagation debugging, logic mismatches)
    - Testbench setup for timing-accurate gate simulations
    - Unit delay simulations and timing check disabling (e.g., $no_notifier)
    """
    return ask_expert(
        question,
        "Senior Gate Level Simulation (GLS) Engineer",
        expertise,
        history
    )
