from core.llm import llm


def ask_interview_question(
    topic,
    difficulty,
    previous_questions=None
):

    previous_questions = previous_questions or []

    prompt = f"""
You are a senior semiconductor DFT interviewer.

Topic: {topic}
Difficulty: {difficulty}

Questions already asked:
{chr(10).join(previous_questions)}

TASK:

Generate ONE completely new interview question.

Requirements:

- Do not repeat any previous question.
- Do not paraphrase a previous question.
- Choose a different concept each time.
- Make it realistic for a VLSI/DFT interview.
- Return only the question.
- No explanations.
- No numbering.
- Beginner for Freshers, so the question should be simple and straightforward.
- If the difficulty is Intermediate, the question should be moderately challenging.
- If the difficulty is Advanced, the question should be complex and require deep understanding.

IMPORTANT:

If Topic is SCAN, possible concepts include:
scan chain, scan enable, scan flop, controllability, observability, shift mode, capture mode, lockup latch, clock mixing, scan stitching, scan compression, scan architecture, scan diagnosis, scan debugging.

If Topic is ATPG, possible concepts include:
stuck-at faults, transition faults, path delay faults, fault coverage, test coverage, fault simulation, pattern generation, test points, X-filling, dynamic compaction, static compaction, atpg constraints, ATPG debugging.

If Topic is EDT, possible concepts include:
TestKompress, decompressor, compactor, scan channels, compression ratio, EDT bypass mode, mask registers, spatial vs temporal compaction, channel capacity, EDT controller, low-power EDT.

If Topic is MBIST, possible concepts include:
March C-, March A, address generator, data generator, comparator, BIST controller, SRAM testing, BISR (Built-In Self Repair), eFuse, memory retention test, word-line / bit-line faults.

If Topic is JTAG or BOUNDARYSCAN, possible concepts include:
IEEE 1149.1, TAP controller 16-state machine, TMS/TCK/TDI/TDO, Instruction Register (IR), Boundary Scan Register (BSR), BYPASS, IDCODE, EXTEST, SAMPLE/PRELOAD, INTEST, board-level interconnect testing.

If Topic is IJTAG, possible concepts include:
IEEE 1687, SIB (Segment Insertion Bit), ICL (Instrument Connectivity Language), PDL (Procedural Description Language), variable-length scan paths, instrument integration.

If Topic is OCC, possible concepts include:
On-Chip Clock controller, fast capture clock pulses, PLL clock switching, at-speed launch-off-shift (LOS) vs launch-off-capture (LOC), clock gating, clock domain crossing in test mode.

If Topic is STA, possible concepts include:
setup and hold timing in scan shift vs capture mode, clock skew impact, lockup latch timing calculation, false paths, multi-cycle paths, scan enable timing closure.

If Topic is TCL or LINUX, possible concepts include:
Tessent Shell (tshell) environment, dofile scripting, reporting commands (report_scan_chains, report_atpg_processes), environment variables, log file parsing, regex in TCL, batch execution.

Choose a concept NOT already used.
"""

    response = llm.invoke(prompt)

    return response.content.strip()