"""Deterministic, offline exports from the canonical taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable


class ExportMode(str, Enum):
    ALL = "all"
    SINGLE = "single"
    MULTIPLE = "multiple"
    THEORY = "theory"
    LAB = "lab"
    THEORY_AND_LAB = "theory_and_lab"
    CHAPTER = "chapter"


@dataclass(frozen=True)
class TaxonomyRecord:
    key: str
    node: dict[str, Any]

    @property
    def source_file(self) -> str:
        return str(self.node.get("source_evidence", {}).get("file", ""))


@dataclass
class ExportResult:
    markdown: str
    summary: dict[str, int]
    diagnostics: list[str]


class BatchExporter:
    """Exports only source text that has an unambiguous, safe boundary."""

    _CLASSIFICATION_FIELDS = ("export_classification", "classification")

    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root).resolve()
        self.knowledge_dir = (self.project_root / "knowledge").resolve()
        legacy_taxonomy = self.knowledge_dir / "canonical_taxonomy.json"
        v2_taxonomy = self.knowledge_dir / "canonical_taxonomy_v2.json"
        # WP-5: prefer the additive v2 taxonomy (existing nodes + lab nodes). The
        # original artifact stays on disk untouched and remains the fallback.
        self.taxonomy_path = v2_taxonomy if v2_taxonomy.exists() else legacy_taxonomy
        self._records = self._load_records()
        self._source_counts = self._count_sources(self._records)
        self._key_counts = self._count_keys(self._records)

    def records(self) -> list[TaxonomyRecord]:
        return list(self._records)

    def export(
        self,
        mode: ExportMode | str,
        *,
        selected_keys: Iterable[str] | None = None,
        chapter: tuple[str, str, str] | None = None,
        domain: str | None = None,
    ) -> ExportResult:
        mode = ExportMode(mode)
        diagnostics: list[str] = []
        summary = {
            "selected": 0,
            "exported": 0,
            "unclassified": 0,
            "ambiguous": 0,
            "unresolved": 0,
            "unavailable": 0,
            "invalid": 0,
        }
        selected = self._select(mode, selected_keys, chapter, summary, diagnostics)
        if domain:
            selected = [record for record in selected if record.node.get("module_domain") == domain]
        summary["selected"] = len(selected)

        blocks: list[str] = []
        unavailable: list[str] = []
        for record in selected:
            if mode == ExportMode.CHAPTER:
                summary["unresolved"] += 1
                diagnostics.append(
                    f"{record.key}: no explicit safe chapter boundary is recorded; metadata was retained without content."
                )
                continue
            if self._key_counts[record.key] > 1:
                summary["ambiguous"] += 1
                diagnostics.append(f"{record.key}: stable-key collision; item was not merged or exported.")
                continue
            # WP-5: lab nodes record an explicit heading-to-heading content boundary and
            # its own page, so their text is exportable even though several nodes share
            # one workbook file.
            excerpt = record.node.get("content_excerpt")
            if isinstance(excerpt, str) and excerpt.strip():
                blocks.append(self._render_block(record, excerpt.strip()))
                summary["exported"] += 1
                continue
            if self._source_counts.get(record.source_file, 0) != 1:
                summary["unresolved"] += 1
                diagnostics.append(
                    f"{record.key}: source file is shared by multiple taxonomy nodes; "
                    "no safe chapter boundary is available."
                )
                continue
            source_path, source_error = self._source_path(record)
            if source_error:
                summary["unavailable"] += 1
                unavailable.append(f"{record.key}: {source_error}")
                continue
            try:
                content = source_path.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError as error:
                summary["unavailable"] += 1
                unavailable.append(f"{record.key}: unable to read source ({error}).")
                continue
            if not content:
                summary["unavailable"] += 1
                unavailable.append(f"{record.key}: source file is empty.")
                continue
            blocks.append(self._render_block(record, content))
            summary["exported"] += 1

        diagnostics.extend(unavailable)
        return ExportResult(self._render_document(mode, selected, blocks, diagnostics, summary), summary, diagnostics)

    def write_export(self, destination: Path | str, markdown: str) -> Path:
        destination = Path(destination).resolve()
        if self._is_within(destination, self.knowledge_dir):
            raise ValueError("Export destination cannot be inside the protected knowledge directory.")
        if not self._is_within(destination, self.project_root):
            raise ValueError("Export destination must be inside the project root.")
        if destination.suffix.lower() != ".md":
            raise ValueError("Export destination must be a Markdown (.md) file.")
        if destination.exists():
            raise ValueError("Export destination already exists; refusing to overwrite it.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown, encoding="utf-8")
        return destination

    def _load_records(self) -> list[TaxonomyRecord]:
        payload = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("Canonical taxonomy has no nodes list.")
        records = [TaxonomyRecord(self._node_key(node), node) for node in nodes if isinstance(node, dict)]
        return sorted(records, key=lambda record: record.key)

    @staticmethod
    def _node_key(node: dict[str, Any]) -> str:
        evidence = node.get("source_evidence") or {}
        fields = (
            node.get("module_domain", ""), node.get("submodule", ""), node.get("chapter", ""),
            node.get("topic", ""), node.get("content_type", ""), evidence.get("file", ""),
        )
        return "|".join(str(value).replace("|", "\\|") for value in fields)

    @staticmethod
    def _count_sources(records: Iterable[TaxonomyRecord]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            counts[record.source_file] = counts.get(record.source_file, 0) + 1
        return counts

    @staticmethod
    def _count_keys(records: Iterable[TaxonomyRecord]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            counts[record.key] = counts.get(record.key, 0) + 1
        return counts

    def _select(self, mode, selected_keys, chapter, summary, diagnostics) -> list[TaxonomyRecord]:
        if mode == ExportMode.ALL:
            return self.records()
        if mode in (ExportMode.SINGLE, ExportMode.MULTIPLE):
            keys = list(dict.fromkeys(selected_keys or []))
            if mode == ExportMode.SINGLE and len(keys) != 1:
                diagnostics.append("Single mode requires exactly one selected node key.")
                summary["invalid"] += 1
                return []
            lookup = {record.key: record for record in self._records if self._key_counts[record.key] == 1}
            selected = []
            for key in keys:
                record = lookup.get(key)
                if record is None:
                    diagnostics.append(f"{key}: unknown or colliding node key.")
                    summary["invalid"] += 1
                else:
                    selected.append(record)
            return selected
        if mode == ExportMode.CHAPTER:
            if not chapter or len(chapter) != 3:
                diagnostics.append("Chapter mode requires an exact domain, submodule, and chapter tuple.")
                summary["invalid"] += 1
                return []
            return [record for record in self._records if self._chapter_tuple(record) == chapter]

        classifications = {record.key: self._classification(record.node) for record in self._records}
        summary["unclassified"] = sum(value is None for value in classifications.values())
        allowed = {"Theory"} if mode == ExportMode.THEORY else {"Lab"}
        if mode == ExportMode.THEORY_AND_LAB:
            allowed = {"Theory", "Lab"}
        selected = [record for record in self._records if classifications[record.key] in allowed]
        if not selected:
            diagnostics.append("No nodes have explicit Theory or Lab classification evidence under Policy A.")
        return selected

    @staticmethod
    def _chapter_tuple(record: TaxonomyRecord) -> tuple[str, str, str]:
        node = record.node
        return (str(node.get("module_domain", "")), str(node.get("submodule", "")), str(node.get("chapter", "")))

    def _classification(self, node: dict[str, Any]) -> str | None:
        for field in self._CLASSIFICATION_FIELDS:
            value = node.get(field)
            if isinstance(value, str) and value.strip().title() in {"Theory", "Lab"}:
                return value.strip().title()
        return None

    def _source_path(self, record: TaxonomyRecord) -> tuple[Path | None, str | None]:
        source = record.source_file
        if not source:
            return None, "source evidence has no file path"
        candidate = (self.project_root / source).resolve()
        if not self._is_within(candidate, self.knowledge_dir):
            return None, "source path is outside the approved knowledge directory"
        if not candidate.is_file():
            return None, "source file is missing or not a regular file"
        return candidate, None

    @staticmethod
    def _is_within(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    @staticmethod
    def _render_block(record: TaxonomyRecord, content: str) -> str:
        node = record.node
        evidence = node.get("source_evidence", {})
        page = evidence.get("page_number")
        location = f"Page {page}" if isinstance(page, int) else "No page location recorded"
        return "\n".join([
            f"## {node.get('topic', 'Untitled')}\n",
            f"- Node key: `{record.key}`",
            f"- Domain: {node.get('module_domain', '')}",
            f"- Submodule: {node.get('submodule', '')}",
            f"- Chapter: {node.get('chapter', '')}",
            f"- Content type: {node.get('content_type', '')}",
            f"- Classification: {node.get('classification') or 'unclassified'}",
            f"- Source: `{evidence.get('file', '')}` (catalog ID: {evidence.get('source_catalog_id', '')})",
            f"- Location: {location}; {evidence.get('toc_reference', 'no TOC reference recorded')}",
            "- Extraction status: verified single-node source boundary\n",
            content,
        ])

    @staticmethod
    def _render_document(mode, selected, blocks, diagnostics, summary) -> str:
        lines = [
            "# Canonical Taxonomy Batch Export",
            "",
            f"- Export mode: `{mode.value}`",
            f"- Selected scope: {len(selected)} canonical node(s)",
            "- Classification policy: Policy A (explicit metadata only)",
            "",
            "## Table of Contents",
        ]
        lines.extend(f"- {record.node.get('topic', 'Untitled')}" for record in selected)
        lines.extend(["", "## Exported Content"])
        lines.extend(blocks or ["No content was safely exportable."])
        lines.extend(["", "## Unresolved or Unavailable Items"])
        lines.extend(f"- {diagnostic}" for diagnostic in diagnostics) if diagnostics else lines.append("- None")
        lines.extend(["", "## Summary"])
        lines.extend(f"- {name}: {value}" for name, value in summary.items())
        return "\n".join(lines) + "\n"
