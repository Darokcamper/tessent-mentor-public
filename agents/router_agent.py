import re
from core.llm import llm

ROUTER_PATTERNS = [
    ("EDT", [
        r"\bedt\b", r"\btestkompress\b", r"\bdecompressor\b", r"\bcompactor\b",
        r"\bchannel capacity\b", r"\bcompression ratio\b", r"\bedt bypass\b",
        r"\bspatial compaction\b", r"\bmodular edt\b"
    ]),
    ("SCAN", [
        r"\bscan chain\b", r"\bscan chains\b", r"\bscan insertion\b", r"\bscan stitching\b",
        r"\bscan enable\b", r"\blockup latch\b", r"\blockup latches\b", r"\bshift mode\b",
        r"\bcapture mode\b", r"\bscan cell\b", r"\bscan cells\b", r"\badd_scan_chains\b",
        r"\binsert_test_logic\b"
    ]),
    ("ATPG", [
        r"\batpg\b", r"\bfastscan\b", r"\bstuck-at\b", r"\btransition fault\b",
        r"\bfault coverage\b", r"\btest pattern\b", r"\btest patterns\b",
        r"\bcreate_patterns\b", r"\brun_atpg\b", r"\bwrite_patterns\b",
        r"\btest procedure\b", r"\batpg_cycle\b", r"\batpg_sequence\b"
    ]),
    ("MBIST", [
        r"\bmbist\b", r"\bbist\b", r"\bmarch c\b", r"\bmarch [a-z]\b", r"\bsram test\b",
        r"\bmemory test\b", r"\bmemory repair\b", r"\bredundancy analysis\b"
    ]),
    ("JTAG", [
        r"\bjtag\b", r"\b1149\.1\b", r"\btap controller\b", r"\btms\b", r"\btck\b",
        r"\btdi\b", r"\btdo\b", r"\binstruction register\b", r"\bboundary scan register\b"
    ]),
    ("IJTAG", [
        r"\bijtag\b", r"\b1687\b", r"\bsib\b", r"\bsegment insertion bit\b",
        r"\bicl\b", r"\bpdl\b", r"\binstrument connectivity\b"
    ]),
    ("WRAPPER", [
        r"\b1500\b", r"\bcore wrapper\b", r"\bwrapper chain\b", r"\bwdr\b", r"\bwir\b"
    ]),
    ("OCC", [
        r"\bocc\b", r"\bon-chip clock\b", r"\bcapture clock\b", r"\bat-speed clock\b",
        r"\bclock control definition\b", r"\bccds?\b"
    ]),
    ("STA", [
        r"\bsta\b", r"\bsetup time\b", r"\bhold time\b", r"\bclock skew\b",
        r"\btiming violation\b", r"\bclock domain\b", r"\bslack\b", r"\bprimetime\b"
    ]),
    ("GLS", [
        r"\bgls\b", r"\bgate-level simulation\b", r"\bsdf\b", r"\bzero-delay\b",
        r"\bunit-delay\b", r"\btiming simulation\b"
    ]),
    ("TCL", [
        r"\btcl\b", r"\bproc\b", r"\bset_attribute\b", r"\bforeach\b"
    ]),
    ("LINUX", [
        r"\blinux\b", r"\bbash\b", r"\btshell\b", r"\bgrep\b", r"\bawk\b",
        r"\bsed\b", r"\bchmod\b"
    ]),
]

def fast_route_check(question: str):
    """Fast regex-based router. Returns category string if unambiguous, else None."""
    q = question.lower()
    scores = {}
    for cat, patterns in ROUTER_PATTERNS:
        matches = sum(1 for p in patterns if re.search(p, q))
        if matches > 0:
            scores[cat] = matches

    if not scores:
        return None

    # Check for a single clear winner
    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_cat, best_score = sorted_cats[0]
    if len(sorted_cats) == 1 or best_score > sorted_cats[1][1]:
        return best_cat
    return None

def route_question(question):
    # Fast path: instant regex matching (<0.001s)
    fast_cat = fast_route_check(question)
    if fast_cat:
        return fast_cat

    # Fallback to LLM for complex / ambiguous questions
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