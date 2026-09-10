from agents.base_agent import ask_expert

def ask_linux(question, history=None):
    expertise = """
    - Linux Operating System & Shell Environment (csh, tcsh, bash, sh)
    - File management and text processing (grep, awk, sed, find, xargs, cut, tr)
    - Process control, job scheduling, resource monitoring
    - Shell scripting for EDA tool run management and regression scripting
    - Environment variables, path management, and shell configuration (.bashrc, .cshrc)
    """
    return ask_expert(
        question,
        "Senior Linux Shell Scripting & Infrastructure Engineer",
        expertise,
        history
    )
