from __future__ import annotations

import json
import sqlite3
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .exceptions import GraphFormatError
from .graph import EDGE_TYPE_WEIGHT, Edge, MemoryNet, Node, migrate_legacy_defaults, tokenize
from .version import __version__


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """A retrieved memory and its relevance to a free-form query."""

    node: Node
    score: float
    semantic_score: float
    tag_score: float
    matched_tags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node.id,
            "text": self.node.text,
            "node_type": self.node.ntype,
            "score": round(self.score, 6),
            "semantic_score": round(self.semantic_score, 6),
            "tag_score": round(self.tag_score, 6),
            "matched_tags": list(self.matched_tags),
            "strength": round(self.node.strength, 6),
            "confidence": round(self.node.confidence, 6),
            "source_ids": list(self.node.source_ids),
            "meta": dict(self.node.meta),
        }


class AssociativeMemory(MemoryNet):
    """Public graph-memory API used by :class:`memnet_agent.MemoryAgent`."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        min_score: float = 0.04,
        node_types: Iterable[str] | None = None,
        exclude_roles: Iterable[str] | None = None,
        reinforce: bool = True,
        ts: float | None = None,
    ) -> list[MemoryHit]:
        """Retrieve memories relevant to text without inserting a temporary node."""

        query = query.strip()
        if not query or top_k <= 0 or not self.nodes:
            return []

        allowed = set(node_types) if node_types is not None else None
        blocked_roles = set(exclude_roles or ())
        candidates = [
            node for node in self.nodes.values()
            if (allowed is None or node.ntype in allowed)
            and node.meta.get("role") not in blocked_roles
        ]
        if not candidates:
            return []

        texts = [node.text for node in candidates]
        try:
            vectorizer = TfidfVectorizer(
                tokenizer=tokenize,
                token_pattern=None,
                lowercase=False,
                max_features=12000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            matrix = vectorizer.fit_transform([*texts, query])
            semantic = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        except ValueError:
            semantic = np.zeros(len(candidates), dtype=float)

        query_tags = set(self.extract_tags(query, k=8))
        hits: list[MemoryHit] = []
        for index, node in enumerate(candidates):
            node_tags = set(node.tags)
            matched = tuple(sorted(query_tags & node_tags))
            union = query_tags | node_tags
            tag_score = len(matched) / len(union) if union else 0.0
            semantic_score = float(semantic[index])
            quality = min(1.0, node.strength / 1.5) * (0.5 + 0.5 * node.confidence)
            relevance = 0.82 * semantic_score + 0.18 * tag_score
            score = relevance * (0.90 + 0.10 * quality)
            if score >= min_score:
                hits.append(
                    MemoryHit(
                        node=node,
                        score=score,
                        semantic_score=semantic_score,
                        tag_score=tag_score,
                        matched_tags=matched,
                    )
                )

        hits.sort(key=lambda hit: (-hit.score, hit.node.id))
        selected = hits[:top_k]
        if reinforce:
            now = self._now(ts)
            for hit in selected:
                self.touch(hit.node.id, ts=now, boost=0.02 + min(0.03, hit.score * 0.03))
        return selected

    def to_dict(self) -> dict[str, Any]:
        config = {
            "decay_rate": self.decay_rate,
            "decay_time_unit": self.decay_time_unit,
            "merge_threshold": self.merge_threshold,
            "semantic_threshold": self.semantic_threshold,
            "semantic_top_k": self.semantic_top_k,
            "temporal_window": self.temporal_window,
            "temporal_top_k": self.temporal_top_k,
            "prune_threshold": self.prune_threshold,
            "min_retained_nodes": self.min_retained_nodes,
            "max_tag_df_ratio": self.max_tag_df_ratio,
            "max_tag_neighbors": self.max_tag_neighbors,
        }
        return {
            "format": "memnet-agent-graph",
            "schema_version": 1,
            "library_version": __version__,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AssociativeMemory":
        if payload.get("format") not in {None, "memnet-agent-graph"}:
            raise GraphFormatError("JSON does not contain a memnet-agent graph.")
        config, migrated = migrate_legacy_defaults(dict(payload.get("config") or {}))
        memory = cls(**config)
        try:
            for raw_node in payload.get("nodes", []):
                node = Node(**raw_node)
                memory.nodes[node.id] = node
            for raw_edge in payload.get("edges", []):
                memory.edges.append(Edge(**raw_edge))
        except (TypeError, KeyError) as exc:
            raise GraphFormatError(f"Invalid graph JSON: {exc}") from exc
        memory._rebuild_edge_indexes()
        if migrated:
            now = memory._now(None)
            for node in memory.nodes.values():
                node.last_decayed = max(node.last_decayed, now)
            memory.log.append("migrated legacy 0.1.x memory defaults to safe daily decay")
        errors = memory.validate()
        if errors:
            raise GraphFormatError("Invalid graph: " + "; ".join(errors[:5]))
        return memory

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load_json(cls, path: str | Path) -> "AssociativeMemory":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GraphFormatError(f"Cannot read graph JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise GraphFormatError("Graph JSON root must be an object.")
        return cls.from_dict(payload)

    def export_graphml(self, path: str | Path) -> Path:
        """Export a portable GraphML file for Gephi, Cytoscape or NetworkX."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        namespace = "http://graphml.graphdrawing.org/xmlns"
        ET.register_namespace("", namespace)
        root = ET.Element(f"{{{namespace}}}graphml")
        keys = [
            ("n_text", "node", "text", "string"),
            ("n_type", "node", "type", "string"),
            ("n_strength", "node", "strength", "double"),
            ("n_confidence", "node", "confidence", "double"),
            ("n_tags", "node", "tags", "string"),
            ("e_type", "edge", "type", "string"),
            ("e_weight", "edge", "weight", "double"),
        ]
        for key_id, target, name, attr_type in keys:
            ET.SubElement(
                root,
                f"{{{namespace}}}key",
                id=key_id,
                **{"for": target, "attr.name": name, "attr.type": attr_type},
            )
        graph = ET.SubElement(root, f"{{{namespace}}}graph", edgedefault="undirected")
        for node in self.nodes.values():
            element = ET.SubElement(graph, f"{{{namespace}}}node", id=node.id)
            values = {
                "n_text": node.text,
                "n_type": node.ntype,
                "n_strength": str(node.strength),
                "n_confidence": str(node.confidence),
                "n_tags": json.dumps(node.tags, ensure_ascii=False),
            }
            for key, value in values.items():
                ET.SubElement(element, f"{{{namespace}}}data", key=key).text = value
        for index, edge in enumerate(self.edges):
            element = ET.SubElement(
                graph,
                f"{{{namespace}}}edge",
                id=f"e{index}",
                source=edge.src,
                target=edge.dst,
            )
            ET.SubElement(element, f"{{{namespace}}}data", key="e_type").text = edge.etype
            ET.SubElement(element, f"{{{namespace}}}data", key="e_weight").text = str(edge.weight)
        ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
        return destination

    def export_bundle(self, path: str | Path) -> Path:
        """Export SQLite, JSON and a manifest in one zip archive."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format": "memnet-agent-bundle",
            "library_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stats": self.stats(),
            "files": ["memory.sqlite", "memory.json"],
        }
        with tempfile.TemporaryDirectory(prefix="memnet-agent-") as temporary:
            temp = Path(temporary)
            self.save(temp / "memory.sqlite")
            self.save_json(temp / "memory.json")
            (temp / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for filename in ("memory.sqlite", "memory.json", "manifest.json"):
                    archive.write(temp / filename, arcname=filename)
        return destination

    def export(self, path: str | Path, *, format: str = "auto") -> Path:
        destination = Path(path)
        selected = format.lower()
        if selected == "auto":
            selected = {
                ".sqlite": "sqlite",
                ".sqlite3": "sqlite",
                ".db": "sqlite",
                ".json": "json",
                ".graphml": "graphml",
                ".zip": "bundle",
            }.get(destination.suffix.lower(), "")
        if selected == "sqlite":
            self.save(destination)
            return destination
        if selected == "json":
            return self.save_json(destination)
        if selected == "graphml":
            return self.export_graphml(destination)
        if selected in {"bundle", "zip"}:
            return self.export_bundle(destination)
        raise GraphFormatError(
            "Unsupported export format. Use sqlite, json, graphml or bundle/zip."
        )

    @classmethod
    def load_external(cls, path: str | Path) -> "AssociativeMemory":
        source = Path(path)
        suffix = source.suffix.lower()
        if suffix in {".sqlite", ".sqlite3", ".db"}:
            return cls._load_sqlite_compatible(source)
        if suffix == ".json":
            return cls.load_json(source)
        if suffix == ".zip":
            with zipfile.ZipFile(source) as archive, tempfile.TemporaryDirectory(
                prefix="memnet-agent-load-"
            ) as temporary:
                names = set(archive.namelist())
                preferred = "memory.sqlite" if "memory.sqlite" in names else "memory.json"
                if preferred not in names:
                    raise GraphFormatError(
                        "Bundle must contain memory.sqlite or memory.json."
                    )
                archive.extract(preferred, temporary)
                return cls.load_external(Path(temporary) / preferred)
        raise GraphFormatError("Supported input formats are SQLite, JSON and memnet zip bundles.")

    @classmethod
    def _load_sqlite_compatible(cls, path: str | Path) -> "AssociativeMemory":
        """Load current SQLite files and migrate the original prototype schema."""

        source = Path(path)
        try:
            with sqlite3.connect(source) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if "metadata" in tables:
                    return cls.load(source)
                if not {"nodes", "edges"}.issubset(tables):
                    raise GraphFormatError("SQLite file does not contain nodes and edges tables.")
                connection.row_factory = sqlite3.Row
                memory = cls()
                for row in connection.execute("SELECT * FROM nodes"):
                    keys = set(row.keys())
                    created_at = float(row["created_at"] or 0.0)
                    last_accessed = float(row["last_accessed"] or created_at)
                    node = Node(
                        id=str(row["id"]),
                        text=str(row["text"] or ""),
                        ntype=str(row["ntype"] or "raw"),
                        created_at=created_at,
                        last_accessed=last_accessed,
                        last_decayed=(
                            float(row["last_decayed"])
                            if "last_decayed" in keys and row["last_decayed"] is not None
                            else last_accessed
                        ),
                        strength=float(row["strength"] or 0.0),
                        access_count=int(row["access_count"] or 0),
                        tags=_decode_json_list(row["tags"]),
                        source_ids=_decode_json_list(row["source_ids"]),
                        confidence=float(row["confidence"] if row["confidence"] is not None else 1.0),
                        meta=(
                            _decode_json_object(row["meta"])
                            if "meta" in keys
                            else {"imported_schema": "legacy-sqlite-v1"}
                        ),
                    )
                    memory.nodes[node.id] = node
                for row in connection.execute("SELECT * FROM edges"):
                    memory.edges.append(
                        Edge(
                            src=str(row["src"]),
                            dst=str(row["dst"]),
                            etype=str(row["etype"]),
                            weight=float(row["weight"] or 0.0),
                            created_at=float(row["created_at"] or 0.0),
                        )
                    )
        except GraphFormatError:
            raise
        except (sqlite3.Error, OSError, KeyError, TypeError, ValueError) as exc:
            raise GraphFormatError(f"Cannot load SQLite graph: {exc}") from exc
        memory._rebuild_edge_indexes()
        errors = memory.validate()
        if errors:
            raise GraphFormatError("Invalid imported graph: " + "; ".join(errors[:5]))
        return memory


def _decode_json_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)]
    if isinstance(decoded, list):
        return [str(item) for item in decoded]
    return [str(decoded)]


def _decode_json_object(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {"legacy_meta": str(value)}
    return dict(decoded) if isinstance(decoded, dict) else {"legacy_meta": decoded}


# Backward-friendly public alias.
Memory = AssociativeMemory
