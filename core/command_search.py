"""
core/command_search.py - Deterministic Commands Explorer Search Engine

Implements fast, offline, deterministic search over knowledge/command_index.json:
- Exact match
- Prefix match (e.g. 'add' or 'add*')
- Substring / partial match (e.g. 'scan')
- Case-insensitive normalization (e.g. 'ADD_CLOCKS')
- Multi-attribute filtering (module, mode, flow step, context)
- Alias resolution with documented relationship types
- Explicit isolation of UNVERIFIED_IN_CORPUS candidates

Plan Reference: PHASE_1_PLAN_APPROVED.md §5.D
Stage Reference: PHASE_2_PREFLIGHT_REPORT.md Stage 2.3
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.command_indexer import load_command_index


def get_available_filter_options(index: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """Extracts unique filter values from the index for UI controls."""
    if index is None:
        index = load_command_index()

    commands = index.get("commands", {})

    modules_set = set()
    modes_set = set()
    flow_steps_set = set()

    for rec in commands.values():
        if rec.get("module"):
            modules_set.add(rec["module"])
        for m in rec.get("modules", []):
            modules_set.add(m)
        if rec.get("mode"):
            m_clean = rec["mode"].strip().lower()
            if m_clean:
                modes_set.add(m_clean)
        if rec.get("flow_step"):
            flow_steps_set.add(rec["flow_step"])

    return {
        "modules": sorted(list(modules_set)),
        "modes": sorted(list(modes_set)),
        "flow_steps": sorted(list(flow_steps_set)),
    }


def search_commands(
    query: str = "",
    module: Optional[str] = None,
    mode: Optional[str] = None,
    flow_step: Optional[str] = None,
    context: Optional[str] = None,
    include_unverified: bool = False,
    index: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Performs deterministic search and filtering over the canonical command index.

    Matching Hierarchy:
    1. Exact match on canonical command name
    2. Exact match on alias
    3. Prefix match on canonical command name
    4. Prefix match on alias
    5. Substring match on canonical command name
    6. Substring match on alias
    7. Substring match in purpose / description

    Tie-breaking: Strictly lexicographical by canonical command name.
    """
    if index is None:
        index = load_command_index()

    commands: Dict[str, Dict[str, Any]] = index.get("commands", {})
    aliases: Dict[str, Dict[str, Any]] = index.get("aliases", {})
    alias_map: Dict[str, str] = index.get("alias_map", {})
    unverified: Dict[str, Dict[str, Any]] = index.get("unverified_candidates", {})

    # Normalize inputs
    q_raw = query.strip()
    is_prefix_only = False
    # Strip trailing asterisk if provided as glob prefix (e.g. 'add*')
    if q_raw.endswith("*"):
        is_prefix_only = True
        q_norm = q_raw[:-1].strip().lower()
    else:
        q_norm = q_raw.lower()

    filter_module = module.strip() if module and module.strip() and module.strip().upper() != "ALL" else None
    filter_mode = mode.strip().lower() if mode and mode.strip() and mode.strip().upper() != "ALL" else None
    filter_flow = flow_step.strip() if flow_step and flow_step.strip() and flow_step.strip().upper() != "ALL" else None
    filter_ctx = context.strip().lower() if context and context.strip() and context.strip().upper() != "ALL" else None

    # Step 1: Filter verified canonical commands
    candidate_cmds: Dict[str, Dict[str, Any]] = {}
    for cmd_name, rec in commands.items():
        # Check module filter
        if filter_module:
            rec_mods = rec.get("modules", [])
            primary_mod = rec.get("module")
            if filter_module != primary_mod and filter_module not in rec_mods:
                continue

        # Check mode filter
        if filter_mode:
            rec_mode = (rec.get("mode") or "").lower()
            if filter_mode not in rec_mode and rec_mode != "all modes":
                continue

        # Check flow step filter
        if filter_flow:
            if rec.get("flow_step") != filter_flow:
                continue

        # Check context filter
        if filter_ctx:
            rec_ctx = (rec.get("context") or "").lower()
            if filter_ctx not in rec_ctx and rec_ctx != "all contexts":
                continue

        candidate_cmds[cmd_name] = rec

    # Step 2: If query is empty, return all matching candidates sorted alphabetically
    if not q_norm:
        sorted_results = []
        for cmd_name in sorted(candidate_cmds.keys()):
            item = dict(candidate_cmds[cmd_name])
            item["status"] = "VERIFIED_CANONICAL"
            item["match_type"] = "ALL"
            item["matched_alias"] = None
            sorted_results.append(item)

        if include_unverified and not filter_mode and not filter_flow and not filter_ctx:
            for cand_name in sorted(unverified.keys()):
                cand_rec = dict(unverified[cand_name])
                sorted_results.append(cand_rec)

        return sorted_results

    # Step 3: Match scoring across candidate commands and aliases
    ranked_buckets: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(1, 8)}

    for cmd_name, rec in candidate_cmds.items():
        name_lower = cmd_name.lower()
        purpose_lower = (rec.get("purpose") or "").lower()
        cmd_aliases = rec.get("aliases", [])

        # 1. Exact match on canonical name
        if name_lower == q_norm:
            item = dict(rec)
            item["status"] = "VERIFIED_CANONICAL"
            item["match_type"] = "EXACT"
            item["matched_alias"] = None
            ranked_buckets[1].append(item)
            continue

        # 2. Exact match on alias
        exact_alias = None
        for a in cmd_aliases:
            if a.lower() == q_norm:
                exact_alias = a
                break
        if exact_alias:
            item = dict(rec)
            item["status"] = "VERIFIED_CANONICAL"
            item["match_type"] = "ALIAS_EXACT"
            item["matched_alias"] = exact_alias
            item["alias_metadata"] = aliases.get(exact_alias)
            ranked_buckets[2].append(item)
            continue

        # 3. Prefix match on canonical name
        if name_lower.startswith(q_norm):
            item = dict(rec)
            item["status"] = "VERIFIED_CANONICAL"
            item["match_type"] = "PREFIX"
            item["matched_alias"] = None
            ranked_buckets[3].append(item)
            continue

        # 4. Prefix match on alias
        prefix_alias = None
        for a in cmd_aliases:
            if a.lower().startswith(q_norm):
                prefix_alias = a
                break
        if prefix_alias:
            item = dict(rec)
            item["status"] = "VERIFIED_CANONICAL"
            item["match_type"] = "ALIAS_PREFIX"
            item["matched_alias"] = prefix_alias
            item["alias_metadata"] = aliases.get(prefix_alias)
            ranked_buckets[4].append(item)
            continue

        # If query explicitly requested prefix-only matching (e.g. 'add*'), skip substring/purpose matches
        if is_prefix_only:
            continue

        # 5. Substring match on canonical name
        if q_norm in name_lower:
            item = dict(rec)
            item["status"] = "VERIFIED_CANONICAL"
            item["match_type"] = "SUBSTRING"
            item["matched_alias"] = None
            ranked_buckets[5].append(item)
            continue

        # 6. Substring match on alias
        sub_alias = None
        for a in cmd_aliases:
            if q_norm in a.lower():
                sub_alias = a
                break
        if sub_alias:
            item = dict(rec)
            item["status"] = "VERIFIED_CANONICAL"
            item["match_type"] = "ALIAS_SUBSTRING"
            item["matched_alias"] = sub_alias
            item["alias_metadata"] = aliases.get(sub_alias)
            ranked_buckets[6].append(item)
            continue

        # 7. Substring match in purpose / description
        if q_norm in purpose_lower:
            item = dict(rec)
            item["status"] = "VERIFIED_CANONICAL"
            item["match_type"] = "PURPOSE_KEYWORD"
            item["matched_alias"] = None
            ranked_buckets[7].append(item)
            continue

    # Flatten buckets with deterministic sort by canonical command name
    results: List[Dict[str, Any]] = []
    seen_canonical = set()

    for rank in range(1, 8):
        bucket = ranked_buckets[rank]
        # Lexicographical tie-breaker
        bucket.sort(key=lambda x: x["canonical_name"].lower())
        for item in bucket:
            cname = item["canonical_name"]
            if cname not in seen_canonical:
                seen_canonical.add(cname)
                results.append(item)

    # Step 4: Include unverified candidates if explicitly requested and matching query
    if include_unverified and not filter_mode and not filter_flow and not filter_ctx:
        matching_unverified = []
        for cand_name, cand_rec in unverified.items():
            if q_norm in cand_name.lower():
                item = dict(cand_rec)
                item["status"] = "UNVERIFIED_IN_CORPUS"
                item["match_type"] = "UNVERIFIED_CANDIDATE"
                matching_unverified.append(item)
        matching_unverified.sort(key=lambda x: x["command"].lower())
        results.extend(matching_unverified)

    return results


def search_exact(query: str, index: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Returns the exact matching canonical command or resolved alias record, if any."""
    res = search_commands(query=query, index=index)
    for r in res:
        if r.get("match_type") in ("EXACT", "ALIAS_EXACT"):
            return r
    return None


def search_prefix(prefix: str, module: Optional[str] = None, index: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Returns all commands matching the given prefix."""
    res = search_commands(query=prefix, module=module, index=index)
    return [r for r in res if r.get("match_type") in ("EXACT", "ALIAS_EXACT", "PREFIX", "ALIAS_PREFIX")]


def search_partial(substring: str, module: Optional[str] = None, index: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Returns all commands containing the substring in name or alias."""
    res = search_commands(query=substring, module=module, index=index)
    return [
        r
        for r in res
        if r.get("match_type") in ("EXACT", "ALIAS_EXACT", "PREFIX", "ALIAS_PREFIX", "SUBSTRING", "ALIAS_SUBSTRING")
    ]
