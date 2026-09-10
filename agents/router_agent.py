from core.llm import llm

def route_question(question):
    prompt = f"""Classify this VLSI / DFT question into exactly ONE category from the list below.

Categories and Descriptions:
- SCAN: Scan chains, scan insertion, lockup latches, scan stitching, scan enable, shift/capture mode.
- ATPG: Automatic Test Pattern Generation, stuck-at/transition faults, fault coverage, test pattern types.
- EDT: Embedded Deterministic Test, scan compression, decompressor, compactor, bypass mode, channels, test time reduction.
- MBIST: Memory Built-in Self Test, March algorithms, SRAM/DRAM testing, repair, memory redundancy.
- JTAG: IEEE 1149.1 boundary scan, TAP controller, instruction registers, TMS/TCK/TDI/TDO.
- IJTAG: IEEE 1687, instrument connectivity, SIB (Segment Insertion Bit), ICL/PDL.
- WRAPPER: IEEE 1500 core wrapper, wrapper chains, core isolation, WDR/WIR.
- BOUNDARYSCAN: Boundary scan cells, board-level testing, boundary scan register.
- OCC: On-Chip Clock Controller, capture clocks, PLL, at-speed clock generation, clock muxing.
- STA: Static Timing Analysis, setup/hold constraints, timing violations, clock domains, clock skew.
- GLS: Gate-Level Simulation, timing simulation, SDF annotation, unit-delay, zero-delay simulation.
- TCL: Tool Command Language, scripting for EDA tools, loops, variables, EDA commands.
- LINUX: Linux shell commands, scripting, file systems, EDA environment setup.
- GENERAL: General VLSI, digital logic, general engineering, or general discussion not covered above.

Question:
{question}

Return ONLY the uppercase category name (e.g. EDT, SCAN, STA). Do not include any explanation or extra characters."""

    response = llm.invoke(prompt)
    return response.content.strip().upper()