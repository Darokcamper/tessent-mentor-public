from agents.base_agent import ask_expert

def ask_tcl(question, history=None):
    expertise = """
    - TCL Scripting Language (Tool Command Language)
    - Variables, lists, arrays, dictionaries, procedures, control structures
    - String manipulation and regular expression matching in TCL
    - TCL APIs in major EDA tools (Tessent shell, PrimeTime, Design Compiler, Innovus)
    - Designing and troubleshooting automation flow scripts in EDA
    """
    return ask_expert(
        question,
        "Senior EDA Tool TCL Automation Expert",
        expertise,
        history
    )
