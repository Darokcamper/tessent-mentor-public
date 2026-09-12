import re
import csv
import sqlite3
import importlib
from pathlib import Path
from core.llm import llm
from core.rag_builder import retrieve
from agents.attachment_context import get_attachment_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
TXT_DIR = KNOWLEDGE_DIR / "txt"
DB_PATH = KNOWLEDGE_DIR / "agent_memory.db"

def init_memory_db():
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS design_constraints (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        pass

def remember_design_constraint(key: str, value: str):
    """Saves or updates a design constraint in the SQLite memory database."""
    key = key.strip()
    value = value.strip()
    try:
        init_memory_db()
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO design_constraints (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=CURRENT_TIMESTAMP
        """, (key, value))
        conn.commit()
        conn.close()
        return f"Successfully remembered design constraint '{key}'."
    except Exception as e:
        return f"Error remembering design constraint: {e}"

def recall_design_constraint(key: str):
    """Recalls a design constraint from the SQLite memory database."""
    key = key.strip()
    try:
        init_memory_db()
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM design_constraints WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
        else:
            return f"Design constraint '{key}' not found."
    except Exception as e:
        return f"Error recalling design constraint: {e}"

def validate_tcl_syntax(script: str) -> str:
    """Validates if braces {}, brackets [], parentheses (), and quotes "" are balanced in TCL code."""
    stack = []
    line = 1
    col = 1
    i = 0
    n = len(script)
    
    while i < n:
        char = script[i]
        
        is_top_bracket_or_empty = False
        if not stack:
            is_top_bracket_or_empty = True
        elif stack[-1][0] == '[':
            is_top_bracket_or_empty = True
            
        if char == '#' and is_top_bracket_or_empty:
            is_comment = False
            if i == 0:
                is_comment = True
            else:
                j = i - 1
                while j >= 0 and script[j] in " \t":
                    j -= 1
                if j < 0 or script[j] in "\n\r;[":
                    is_comment = True
                    
            if is_comment:
                while i < n and script[i] not in "\n\r":
                    i += 1
                continue
                
        if char == '\\':
            i += 1
            if i < n:
                escaped_char = script[i]
                if escaped_char == '\n':
                    line += 1
                    col = 1
                elif escaped_char == '\r':
                    if i + 1 < n and script[i+1] == '\n':
                        i += 1
                    line += 1
                    col = 1
                else:
                    col += 2
                i += 1
            else:
                col += 1
                i += 1
            continue
            
        top = stack[-1][0] if stack else None
        
        if top == '{':
            if char == '{':
                stack.append(('{', line, col))
            elif char == '}':
                stack.pop()
        elif top == '"':
            if char == '"':
                stack.pop()
            elif char == '[':
                stack.append(('[', line, col))
            elif char == ']':
                return f"Mismatched token ']' at line {line}, column {col} (no opening token)"
        else:
            if char in ['{', '[', '(', '"']:
                stack.append((char, line, col))
            elif char in ['}', ']', ')']:
                if not stack:
                    return f"Mismatched token '{char}' at line {line}, column {col} (no opening token)"
                expected = {'}': '{', ']': '[', ')': '('}[char]
                if stack[-1][0] != expected:
                    return f"Mismatched token '{char}' at line {line}, column {col} (expected closing for '{stack[-1][0]}' from line {stack[-1][1]}, column {stack[-1][2]})"
                stack.pop()
                
        if char == '\n':
            line += 1
            col = 1
        elif char == '\r':
            if i + 1 < n and script[i+1] == '\n':
                i += 1
                char = '\n'
            line += 1
            col = 1
        else:
            col += 1
            
        i += 1
        
    if stack:
        top_char, top_line, top_col = stack[-1]
        return f"Unclosed token '{top_char}' from line {top_line}, column {top_col}"
        
    return "Success: TCL syntax is valid."

def search_web(query: str) -> str:
    """Searches the web using duckduckgo_search with error handling."""
    query = query.strip()
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No search results found."
        
        output_parts = []
        for idx, r in enumerate(results, 1):
            title = r.get("title", "No Title")
            href = r.get("href", "#")
            body = r.get("body", "")
            output_parts.append(f"{idx}. {title}\n   URL: {href}\n   Summary: {body}")
        return "\n\n".join(output_parts)
    except Exception as e:
        return f"Web search is currently unavailable or offline. Error details: {e}"

def consult_other_expert(expert_name: str, question: str) -> str:
    """Consults another expert agent dynamically by importing and executing its ask function."""
    expert_name_clean = expert_name.strip().upper()
    expert_map = {
        'SCAN': ('agents.scan_agent', 'ask_scan'),
        'STA': ('agents.sta_agent', 'ask_sta'),
        'ATPG': ('agents.atpg_agent', 'ask_atpg'),
        'EDT': ('agents.edt_agent', 'ask_edt'),
        'MBIST': ('agents.mbist_agent', 'ask_mbist'),
        'JTAG': ('agents.jtag_agent', 'ask_jtag'),
        'IJTAG': ('agents.ijtag_agent', 'ask_ijtag'),
        'WRAPPER': ('agents.wrapper_agent', 'ask_wrapper'),
        'OCC': ('agents.occ_agent', 'ask_occ'),
        'BOUNDARYSCAN': ('agents.boundaryscan_agent', 'ask_boundaryscan'),
        'GLS': ('agents.gls_agent', 'ask_gls'),
        'TCL': ('agents.tcl_agent', 'ask_tcl'),
        'LINUX': ('agents.linux_agent', 'ask_linux'),
        'GENERAL': ('agents.general_agent', 'ask_general'),
    }
    
    if expert_name_clean not in expert_map:
        expert_name_lower = expert_name_clean.lower()
        found_key = None
        for k in expert_map:
            if k.lower() == expert_name_lower:
                found_key = k
                break
        if found_key:
            module_name, func_name = expert_map[found_key]
        else:
            return f"Error: Expert '{expert_name}' is not recognized. Available experts: {', '.join(expert_map.keys())}"
    else:
        module_name, func_name = expert_map[expert_name_clean]
        
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        return func(question)
    except Exception as e:
        return f"Error consulting expert '{expert_name}': {e}"

def list_verified_sources():
    """Lists all verified source files and their topics."""
    try:
        sources = []
        csv_path = KNOWLEDGE_DIR / "sources.csv"
        if not csv_path.exists():
            return "Error: sources.csv catalog not found."
            
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("File"):
                    file_path = Path(row["File"])
                    sources.append(f"- Stem: '{file_path.stem}', Topic: '{row.get('Topic', 'Unknown')}', Source: '{row.get('Source', 'Unknown')}'")
        return "\n".join(sources)
    except Exception as e:
        return f"Error listing sources: {e}"

def search_verified_notes(query: str, source_filter: str = None):
    """Searches the vector database for matching chunks."""
    query = query.strip().strip("'\"")
    if source_filter:
        source_filter = source_filter.strip().strip("'\"")
        if source_filter.lower() in ["none", "null", "", "false"]:
            source_filter = None
            
    try:
        k = 8 if source_filter else 4
        results = retrieve(query, top_k=k, source_filter=source_filter)
        if not results:
            if source_filter:
                return (
                    f"No matching chunks found in verified notes for source_filter='{source_filter}'. "
                    "That file stem may be wrong. Call list_verified_sources() to get valid file stems, "
                    "then retry search_verified_notes with a corrected filter or without one. "
                    "IMPORTANT: Do NOT answer from your own pre-trained knowledge. If a corrected search "
                    "still finds nothing, your Final Answer MUST be: 'I do not have the exact answer in my "
                    "verified source documents.' followed by the mandated source-file list and refinement suggestions."
                )
            return (
                "No matching chunks found in verified notes. "
                "IMPORTANT: Do NOT answer from your own pre-trained knowledge. Try rephrasing the query with "
                "different key technical terms (e.g. expand acronyms), or call list_verified_sources() "
                "to see available files. If searches still find nothing relevant, your Final Answer MUST be: "
                "'I do not have the exact answer in my verified source documents.' followed by the mandated "
                "source-file list and refinement suggestions."
            )
            
        parts = []
        for r in results:
            parts.append(f"Source Document: {r['source']} (Page {r['page']})\nContent: {r['text']}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"Error searching notes: {e}"

def read_source_file_page(filename_stem: str, page_num: int):
    """Reads the literal text of a specific page from a source note file."""
    filename_stem = filename_stem.strip().strip("'\"")
    stem = Path(filename_stem).stem
    
    txt_path = TXT_DIR / f"{stem}.txt"
    if not txt_path.exists():
        matched = list(TXT_DIR.glob(f"*{stem}*"))
        if matched:
            txt_path = matched[0]
        else:
            return f"Error: File '{filename_stem}' not found in cached notes (checked in knowledge/txt)."
            
    try:
        page_num = int(page_num)
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        parts = re.split(r"--- PAGE (\d+) ---\n", content)
        for i in range(1, len(parts), 2):
            if int(parts[i]) == page_num:
                return f"Source Document: {txt_path.name} (Page {page_num})\nContent: {parts[i+1].strip()}"
        return f"Error: Page {page_num} not found in '{txt_path.name}'."
    except Exception as e:
        return f"Error reading file page: {e}"

def parse_action(action_str):
    """Parses ToolName(arg1=val1, ...) or ToolName(val1, ...) into tool_name and kwargs/args."""
    match = re.search(r"(\w+)\((.*)\)", action_str)
    if not match:
        return None, None
        
    tool_name = match.group(1).strip()
    args_str = match.group(2).strip()
    
    kwargs = {}
    kw_pattern = r"(\w+)\s*=\s*(?:['\"](.*?)['\"]|(\w+|None|\d+))"
    kw_matches = re.findall(kw_pattern, args_str)
    if kw_matches:
        for key, val_quoted, val_unquoted in kw_matches:
            val = val_quoted if val_quoted else val_unquoted
            if val == "None":
                val = None
            elif val.isdigit():
                val = int(val)
            kwargs[key] = val
        return tool_name, kwargs
        
    import csv
    from io import StringIO
    try:
        reader = csv.reader(StringIO(args_str), skipinitialspace=True)
        args_list = next(reader)
        processed_args = []
        for a in args_list:
            a = a.strip().strip("'\"")
            if a == "None":
                processed_args.append(None)
            elif a.isdigit():
                processed_args.append(int(a))
            else:
                processed_args.append(a)
        return tool_name, processed_args
    except Exception:
        args_list = [a.strip().strip("'\"") for a in args_str.split(",")]
        return tool_name, args_list

def execute_tool(tool_name, args):
    """Executes a tool by name with parsed arguments."""
    if tool_name == "list_verified_sources":
        return list_verified_sources()
        
    elif tool_name == "search_verified_notes":
        if isinstance(args, dict):
            query = args.get("query", "")
            source_filter = args.get("source_filter")
        else:
            query = args[0] if len(args) > 0 else ""
            source_filter = args[1] if len(args) > 1 else None
        return search_verified_notes(query, source_filter)
        
    elif tool_name == "read_source_file_page":
        if isinstance(args, dict):
            filename_stem = args.get("filename_stem", "")
            page_num = args.get("page_num", 1)
        else:
            filename_stem = args[0] if len(args) > 0 else ""
            page_num = args[1] if len(args) > 1 else 1
        return read_source_file_page(filename_stem, page_num)
        
    elif tool_name == "remember_design_constraint":
        if isinstance(args, dict):
            key = args.get("key", "")
            value = args.get("value", "")
        else:
            key = args[0] if len(args) > 0 else ""
            value = args[1] if len(args) > 1 else ""
        return remember_design_constraint(key, value)
        
    elif tool_name == "recall_design_constraint":
        if isinstance(args, dict):
            key = args.get("key", "")
        else:
            key = args[0] if len(args) > 0 else ""
        return recall_design_constraint(key)
        
    elif tool_name == "validate_tcl_syntax":
        if isinstance(args, dict):
            script = args.get("script", "")
        else:
            script = args[0] if len(args) > 0 else ""
        return validate_tcl_syntax(script)
        
    elif tool_name == "search_web":
        if isinstance(args, dict):
            query = args.get("query", "")
        else:
            query = args[0] if len(args) > 0 else ""
        return search_web(query)
        
    elif tool_name == "consult_other_expert":
        if isinstance(args, dict):
            expert_name = args.get("expert_name", "")
            question = args.get("question", "")
        else:
            expert_name = args[0] if len(args) > 0 else ""
            question = args[1] if len(args) > 1 else ""
        return consult_other_expert(expert_name, question)
        
    else:
        return f"Error: Tool '{tool_name}' is not recognized. Available tools: list_verified_sources, search_verified_notes, read_source_file_page, remember_design_constraint, recall_design_constraint, validate_tcl_syntax, search_web, consult_other_expert."

def ask_expert(
    question,
    role,
    expertise,
    history=None,
    extra_context=None
):
    # Context-Aware query rewriting to assist the agent if user is asking short follow-up questions
    retrieval_query = question
    if "Question:" in question:
        match = re.search(r"Question:\s*(.*?)(?:\n\nHere is|\n\nRevision Round|\n\n|$)", question, re.DOTALL)
        if match:
            retrieval_query = match.group(1).strip()

    if history:
        last_user_msg = None
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        if last_user_msg:
            words = retrieval_query.strip().split()
            is_follow_up = len(words) < 6 or any(w in retrieval_query.lower() for w in ["what about", "why", "explain", "how", "it", "that", "they", "those", "this", "first", "second", "third", "prev", "last"])
            if is_follow_up:
                clean_last = last_user_msg.split("\n\n(Strict Grounding")[0].strip()
                retrieval_query = f"{retrieval_query} {clean_last}"

    # Build the system prompt detailing role and agent guidelines
    system_prompt = f"""You are a {role}.

Expertise:
{expertise}

You are an autonomous AI agent. To solve the user's question, you must search and analyze verified DFT/VLSI sources using the tools provided below. You must NOT rely on your pre-trained memory to invent specific rules, files, scripts, commands, or parameters if they are not retrieved from the verified sources.

You have access to the following tools:
1. `list_verified_sources()`: Returns a list of all verified source filenames and their topics in the knowledge catalog. Use this if you are unsure which files contain relevant information.
2. `search_verified_notes(query: str, source_filter: str)`: Searches the vector database for matching chunks across all notes, or within a specific file if `source_filter` (file stem, e.g. "Scan_DRC_part_2") is provided.
3. `read_source_file_page(filename_stem: str, page_num: int)`: Reads the literal text of a specific page from a source note file (e.g., "Level_1_Session_1", 5).
4. `remember_design_constraint(key: str, value: str)`: Saves or updates a design constraint in the SQLite memory database.
5. `recall_design_constraint(key: str)`: Recalls a saved design constraint from the SQLite memory database.
6. `validate_tcl_syntax(script: str)`: Validates the syntax of a TCL script, checking for balanced braces, brackets, parentheses, and quotes.
7. `search_web(query: str)`: Searches the web for a query to retrieve external technical info or references.
8. `consult_other_expert(expert_name: str, question: str)`: Consults another expert agent dynamically by name (e.g. 'SCAN', 'STA', 'ATPG', 'EDT', 'MBIST', 'JTAG', 'IJTAG', 'WRAPPER', 'OCC', 'BOUNDARYSCAN', 'GLS', 'TCL', 'LINUX', 'GENERAL') and returns their answer.

Format to follow:
You MUST respond using the following structured format at each step. Do NOT output anything else before the Thought, and do NOT combine thoughts and final answers.

Thought: <Reason about what information you need to search or fetch next>
Action: <tool_name>(<arguments>)

Once you receive the tool output (Observation), you will formulate another Thought. You can repeat this process up to 4 times. Once you have all the necessary information to answer, output:

Thought: I have retrieved sufficient verified information to answer the question.
Final Answer: <Your final comprehensive answer, strictly grounded in the observations. You must cite the Source Document and Page when referencing ideas from the notes. Avoid repeating any OCR noise or corrupted formulas, and explain them in correct standard engineering English. Structure your answer with clear headings, bullet points, and comparison tables where appropriate.>

If you cannot find the answer in the verified documents after searching, your Final Answer MUST be: "I do not have the exact answer in my verified source documents." and then list the relevant source files or manuals found in the context with citations (e.g., `[Source: filename, Page N]`) and a 1-sentence summary of what topics that file covers on that page, recommending how the user can refine their question.

Let's begin!
"""

    messages = [
        ("system", system_prompt)
    ]

    if history:
        for msg in history[-3:]:
            messages.append((msg["role"], msg["content"]))

    user_msg = f"Question: {question}"
    messages.append(("user", user_msg))

    # If the user attached a document, provide its text as supplementary context.
    # Label it clearly so the agent does not mistake it for a verified manual source.
    if not extra_context:
        extra_context = get_attachment_context()
    if extra_context:
        attachment_prelude = (
            "The user has attached a document to help you understand the question. "
            "Its text is provided below as USER-ATTACHED CONTENT -- it is NOT a "
            "verified manual source, so do not cite it as such. Use it only to "
            "understand the user's specific scenario, and ground all technical "
            "claims in the verified sources you search."
        )
        messages.insert(
            -1,
            (
                "user",
                f"{attachment_prelude}\n\n"
                f"--- BEGIN ATTACHMENT ---\n{extra_context}\n--- END ATTACHMENT ---",
            ),
        )

    agent_scratchpad = []
    max_iterations = 5
    tools_called = False

    for i in range(max_iterations):
        current_messages = list(messages)
        for thought_action, observation in agent_scratchpad:
            current_messages.append(("assistant", thought_action))
            current_messages.append(("user", f"Observation: {observation}"))
            
        try:
            response = llm.invoke(current_messages)
            content = response.content.strip()
        except Exception as e:
            print(f"LLM invoke failed in agent loop: {e}")
            return f"Error executing agent reasoning: {e}"
            
        print(f"\n--- Agent Step {i+1} ---")
        print(content)
        
        if "Final Answer:" in content:
            # Tool-gate: refuse an answer produced without calling any tools,
            # so the model cannot respond purely from parametric memory.
            if not tools_called:
                agent_scratchpad.append((
                    content,
                    "Error: You output a Final Answer WITHOUT searching the verified sources. "
                    "This is forbidden. You MUST first call Action: search_verified_notes(query=\"<key technical terms from the question>\") "
                    "to retrieve grounded information. Retry now using the required Thought/Action format. "
                    "Cite the Source Document and Page from the observations in your eventual Final Answer.",
                ))
                continue
            final_part = content.split("Final Answer:", 1)[1].strip()
            # Post-processing fallback verification
            normalized = final_part.lower()
            if "do not have this information" in normalized or "not found in the retrieved context" in normalized:
                if "file://" not in final_part:
                    return "I do not have this information in my verified source documents."
            return final_part
            
        action_match = re.search(r"Action:\s*(\w+\(.*?\))", content, re.DOTALL)
        if not action_match:
            if "Thought:" in content:
                agent_scratchpad.append((content, "Error: You did not specify an Action. If you are ready, output 'Final Answer: <your answer>'. Otherwise specify Action: tool_name(args)."))
                continue
            return content
            
        action_str = action_match.group(1).strip()
        tool_name, args = parse_action(action_str)
        
        if not tool_name:
            observation = f"Error: Could not parse Action: '{action_str}'. Please use the format: ToolName(arg1=val1, ...)"
        else:
            print(f"Calling Tool: {tool_name} with args {args}")
            tools_called = True
            observation = execute_tool(tool_name, args)
            if len(observation) > 8000:
                observation = observation[:8000] + "\n... [Observation Truncated for space] ..."
            print(f"Observation: {observation[:200]}...")
            
        agent_scratchpad.append((content, observation))

    # Final wrap up if it timed out
    current_messages = list(messages)
    for thought_action, observation in agent_scratchpad:
        current_messages.append(("assistant", thought_action))
        current_messages.append(("user", f"Observation: {observation}"))
    current_messages.append(("user", "Final Loop Warning: You have reached the maximum number of reasoning steps. Please summarize the information retrieved so far and output your 'Final Answer:' now."))
    
    try:
        response = llm.invoke(current_messages)
        content = response.content.strip()
        if "Final Answer:" in content:
            return content.split("Final Answer:", 1)[1].strip()
        return content
    except Exception as e:
        return f"Error finalizing agent answer: {e}"