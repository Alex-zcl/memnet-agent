"""A small, deterministic graph memory with decay and consolidation.

The implementation deliberately stays local and extractive.  In particular,
"synthesis" nodes are associations between coherent memories, not verified facts.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from random import Random
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ']+")
STOPWORDS = set(
    """
    a an the and or but if of to in on at for with as by is was were are be been
    being this that these those it its it's he she they them his her their i you
    we do does did not no so than then there here what which who whom will would
    can could should shall may might must about into over under again further
    out up down off above below from between very just now
    а без более бы был была были было быть в вам вас весь во вот все всего всех
    вы где да даже для до его ее если есть ещё же за и из или им их к как ко
    когда кто ли либо мне может мы на над надо наш не него неё нет ни них но ну
    о об один она они оно от перед по под после при про раз с со так также такой
    там те тем то того тоже той только том ты у уже хотя чего чем что чтобы
    """.split()
)
EDGE_TYPE_WEIGHT = {"semantic": 1.0, "tag": 0.65, "temporal": 0.35, "inferred": 0.8}

DEFAULT_DECAY_RATE = 0.98
DEFAULT_DECAY_TIME_UNIT = 24 * 60 * 60.0
DEFAULT_MERGE_THRESHOLD = 0.90
DEFAULT_PRUNE_THRESHOLD = 0.025
DEFAULT_MIN_RETAINED_NODES = 3

LEGACY_DEFAULTS = {
    "decay_rate": 0.85,
    "decay_time_unit": 1.0,
    "merge_threshold": 0.72,
    "prune_threshold": 0.15,
}

def migrate_legacy_defaults(config: dict) -> tuple[dict, bool]:
    """Upgrade unsafe 0.1.x defaults without overriding custom tuning."""
    normalized = dict(config)
    legacy = all(
        key in normalized
        and math.isclose(float(normalized[key]), value, rel_tol=0.0, abs_tol=1e-12)
        for key, value in LEGACY_DEFAULTS.items()
    )
    if legacy:
        normalized.update(
            decay_rate=DEFAULT_DECAY_RATE,
            decay_time_unit=DEFAULT_DECAY_TIME_UNIT,
            merge_threshold=DEFAULT_MERGE_THRESHOLD,
            prune_threshold=DEFAULT_PRUNE_THRESHOLD,
        )
    return normalized, legacy


def tokenize(text: str) -> list[str]:
    return [w for w in TOKEN_RE.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS]


@dataclass(slots=True)
class Node:
    id: str
    text: str
    ntype: str
    created_at: float
    last_accessed: float
    last_decayed: float
    strength: float = 1.0
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class Edge:
    src: str
    dst: str
    etype: str
    weight: float
    created_at: float


class MemoryNet:
    """Graph memory suitable for experiments and small local agents.

    Times are always supplied in one clock domain.  Callers may use Unix time or
    synthetic ticks, but must not mix them inside one network.
    """

    def __init__(
        self,
        *,
        decay_rate: float = DEFAULT_DECAY_RATE,
        decay_time_unit: float = DEFAULT_DECAY_TIME_UNIT,
        merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
        semantic_threshold: float = 0.35,
        semantic_top_k: int = 8,
        temporal_window: float = 3.0,
        temporal_top_k: int = 6,
        prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
        min_retained_nodes: int = DEFAULT_MIN_RETAINED_NODES,
        max_tag_df_ratio: float = 0.18,
        max_tag_neighbors: int = 12,
    ) -> None:
        if not 0 < decay_rate <= 1:
            raise ValueError("decay_rate must be in (0, 1]")
        if decay_time_unit <= 0:
            raise ValueError("decay_time_unit must be positive")
        if min_retained_nodes < 0:
            raise ValueError("min_retained_nodes must be non-negative")
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._edge_index: dict[tuple[str, str, str], Edge] = {}
        self._adj: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        self.decay_rate = decay_rate
        self.decay_time_unit = decay_time_unit
        self.merge_threshold = merge_threshold
        self.semantic_threshold = semantic_threshold
        self.semantic_top_k = semantic_top_k
        self.temporal_window = temporal_window
        self.temporal_top_k = temporal_top_k
        self.prune_threshold = prune_threshold
        self.min_retained_nodes = int(min_retained_nodes)
        self.max_tag_df_ratio = max_tag_df_ratio
        self.max_tag_neighbors = max_tag_neighbors
        self.log: list[str] = []

    @staticmethod
    def _now(ts: float | None) -> float:
        return time.time() if ts is None else float(ts)

    @staticmethod
    def _edge_key(a: str, b: str, etype: str) -> tuple[str, str, str]:
        lo, hi = sorted((a, b))
        return lo, hi, etype

    def add(
        self,
        text: str,
        *,
        ts: float | None = None,
        tags: list[str] | None = None,
        ntype: str = "raw",
        source_ids: Iterable[str] | None = None,
        confidence: float = 1.0,
        meta: dict | None = None,
    ) -> Node:
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")
        if ntype not in {"raw", "fact", "synthesis"}:
            raise ValueError(f"unsupported node type: {ntype}")
        now = self._now(ts)
        nid = uuid.uuid4().hex[:12]
        node = Node(
            id=nid,
            text=text,
            ntype=ntype,
            created_at=now,
            last_accessed=now,
            last_decayed=now,
            tags=list(dict.fromkeys(tags or self.extract_tags(text))),
            source_ids=list(dict.fromkeys(source_ids or [])),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            meta=dict(meta or {}),
        )
        self.nodes[nid] = node
        return node

    def link(self, a: str, b: str, etype: str, weight: float, *, ts: float | None = None) -> None:
        if a == b or a not in self.nodes or b not in self.nodes:
            return
        if etype not in EDGE_TYPE_WEIGHT:
            raise ValueError(f"unsupported edge type: {etype}")
        key = self._edge_key(a, b, etype)
        weight = float(np.clip(weight, 0.0, 1.0))
        existing = self._edge_index.get(key)
        if existing is not None:
            existing.weight = max(existing.weight, weight)
            return
        edge = Edge(src=key[0], dst=key[1], etype=etype, weight=weight, created_at=self._now(ts))
        self.edges.append(edge)
        self._edge_index[key] = edge
        self._adj[edge.src].add(key)
        self._adj[edge.dst].add(key)

    def _rebuild_edge_indexes(self) -> None:
        coalesced: dict[tuple[str, str, str], Edge] = {}
        for edge in self.edges:
            if edge.src == edge.dst or edge.src not in self.nodes or edge.dst not in self.nodes:
                continue
            key = self._edge_key(edge.src, edge.dst, edge.etype)
            old = coalesced.get(key)
            if old is None:
                coalesced[key] = Edge(key[0], key[1], edge.etype, edge.weight, edge.created_at)
            else:
                old.weight = max(old.weight, edge.weight)
                old.created_at = min(old.created_at, edge.created_at)
        self._edge_index = coalesced
        self.edges = list(coalesced.values())
        self._adj = defaultdict(set)
        for key, edge in coalesced.items():
            self._adj[edge.src].add(key)
            self._adj[edge.dst].add(key)

    def extract_tags(self, text: str, k: int = 4) -> list[str]:
        counts = Counter(tokenize(text))
        return [word for word, _ in counts.most_common(k)]

    def build_edges_for_new(self, new_ids: list[str], *, ts: float | None = None) -> None:
        new_ids = [nid for nid in dict.fromkeys(new_ids) if nid in self.nodes]
        if not new_ids or len(self.nodes) < 2:
            return
        now = self._now(ts)
        all_ids = list(self.nodes)
        texts = [self.nodes[nid].text for nid in all_ids]
        try:
            matrix = TfidfVectorizer(
                tokenizer=tokenize,
                token_pattern=None,
                lowercase=False,
                max_features=8000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ).fit_transform(texts)
        except ValueError:
            matrix = None
        id_to_idx = {nid: idx for idx, nid in enumerate(all_ids)}

        # Semantic edges: only the strongest K neighbors for each new node.
        if matrix is not None:
            new_rows = [id_to_idx[nid] for nid in new_ids]
            sims = cosine_similarity(matrix[new_rows], matrix)
            for row_idx, nid in enumerate(new_ids):
                candidates = [
                    (float(sims[row_idx, j]), other)
                    for j, other in enumerate(all_ids)
                    if other != nid and sims[row_idx, j] >= self.semantic_threshold
                ]
                for sim, other in sorted(candidates, reverse=True)[: self.semantic_top_k]:
                    self.link(nid, other, "semantic", sim, ts=now)

        # Temporal edges: nearest nodes within the configured window.
        for nid in new_ids:
            node = self.nodes[nid]
            candidates = []
            for other in all_ids:
                if other == nid:
                    continue
                dt = abs(node.created_at - self.nodes[other].created_at)
                if dt <= self.temporal_window:
                    candidates.append((dt, other))
            for dt, other in sorted(candidates)[: self.temporal_top_k]:
                weight = 0.3 + 0.4 * (1.0 - dt / max(self.temporal_window, 1e-9))
                self.link(nid, other, "temporal", weight, ts=now)

        # Tag edges: suppress corpus-wide tags and cap each tag neighborhood.
        tag_index: dict[str, list[str]] = defaultdict(list)
        for nid in all_ids:
            for tag in set(self.nodes[nid].tags):
                tag_index[tag].append(nid)
        max_df = max(3, math.ceil(len(all_ids) * self.max_tag_df_ratio))
        for nid in new_ids:
            for tag in self.nodes[nid].tags:
                members = tag_index.get(tag, [])
                if not 1 < len(members) <= max_df:
                    continue
                idf = math.log((1 + len(all_ids)) / (1 + len(members))) + 1.0
                weight = min(0.85, 0.25 + 0.18 * idf)
                others = sorted(
                    (oid for oid in members if oid != nid),
                    key=lambda oid: abs(self.nodes[oid].created_at - self.nodes[nid].created_at),
                )[: self.max_tag_neighbors]
                for other in others:
                    self.link(nid, other, "tag", weight, ts=now)

    def touch(self, node_id: str, *, ts: float | None = None, boost: float = 0.05) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        now = self._now(ts)
        node.access_count += 1
        node.last_accessed = max(node.last_accessed, now)
        node.strength = min(2.0, node.strength + boost)

    def related(
        self,
        node_id: str,
        *,
        top_k: int = 8,
        ts: float | None = None,
        reinforce: bool = True,
    ) -> list[tuple[Node, float, list[str]]]:
        if node_id not in self.nodes:
            return []
        evidence: dict[str, dict[str, float]] = defaultdict(dict)
        for key in self._adj.get(node_id, set()):
            edge = self._edge_index[key]
            other = edge.dst if edge.src == node_id else edge.src
            evidence[other][edge.etype] = max(evidence[other].get(edge.etype, 0.0), edge.weight)
        scored = []
        for other_id, by_type in evidence.items():
            node = self.nodes[other_id]
            combined = 1.0
            for etype, weight in by_type.items():
                combined *= 1.0 - EDGE_TYPE_WEIGHT[etype] * weight
            relation_score = 1.0 - combined
            quality = node.strength * (0.5 + 0.5 * node.confidence)
            scored.append((other_id, relation_score * quality, sorted(by_type)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        selected = scored[:top_k]
        if reinforce:
            now = self._now(ts)
            self.touch(node_id, ts=now, boost=0.02)
            for other_id, _, _ in selected:
                self.touch(other_id, ts=now, boost=0.03)
        return [(self.nodes[nid], score, types) for nid, score, types in selected]

    def sleep(self, *, now: float | None = None, max_syntheses: int = 3) -> dict[str, int]:
        ts = self._now(now)
        self._decay(ts)
        merges = self._merge_similar(ts)
        splits = self._split_mixed(ts)
        pruned = self._prune()
        syntheses = self._synthesize(ts, max_new=max_syntheses)
        result = {"merges": merges, "splits": splits, "pruned": pruned, "synth": syntheses}
        self.log.append(f"sleep@{ts:.3f}: {result}, nodes_left={len(self.nodes)}")
        return result

    def _decay(self, now: float) -> None:
        for node in self.nodes.values():
            elapsed = max(0.0, now - max(node.last_decayed, node.last_accessed))
            if elapsed == 0:
                node.last_decayed = max(node.last_decayed, now)
                continue
            resilience = 1.0 + math.log1p(node.access_count)
            periods = elapsed / self.decay_time_unit
            node.strength *= self.decay_rate ** (periods / resilience)
            node.last_decayed = now

    def _text_similarities(self, ids: list[str]) -> np.ndarray | None:
        try:
            matrix = TfidfVectorizer(
                tokenizer=tokenize,
                token_pattern=None,
                lowercase=False,
                max_features=8000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ).fit_transform([self.nodes[nid].text for nid in ids])
        except ValueError:
            return None
        return cosine_similarity(matrix)

    def _merge_similar(self, now: float) -> int:
        ids = [nid for nid, n in self.nodes.items() if n.ntype in {"raw", "fact"}]
        if len(ids) < 2:
            return 0
        sims = self._text_similarities(ids)
        if sims is None:
            return 0
        pairs = [
            (float(sims[i, j]), ids[i], ids[j])
            for i in range(len(ids))
            for j in range(i + 1, len(ids))
            if sims[i, j] >= self.merge_threshold
            and self._merge_compatible(self.nodes[ids[i]], self.nodes[ids[j]])
        ]
        used: set[str] = set()
        merged = 0
        for _, a, b in sorted(pairs, reverse=True):
            if a in used or b in used or a not in self.nodes or b not in self.nodes:
                continue
            self._merge_group([a, b], now)
            used.update((a, b))
            merged += 1
        return merged

    @staticmethod
    def _same_memory_scope(first: Node, second: Node) -> bool:
        first_role = first.meta.get("role")
        second_role = second.meta.get("role")
        if first_role is not None or second_role is not None:
            return first_role == second_role
        return True

    @classmethod
    def _merge_compatible(cls, first: Node, second: Node) -> bool:
        if first.meta.get("protected") or second.meta.get("protected"):
            return False
        return cls._same_memory_scope(first, second)

    def _original_sources(self, nodes: Iterable[Node]) -> list[str]:
        out: list[str] = []
        for node in nodes:
            out.extend(node.source_ids or [node.id])
        return list(dict.fromkeys(out))

    def _merge_group(self, member_ids: list[str], now: float) -> Node:
        members = [self.nodes[nid] for nid in member_ids]
        members.sort(key=lambda n: (-(n.strength * (1 + n.access_count)), n.id))
        anchor = members[0]
        seen = set(tokenize(anchor.text))
        snippets = []
        for member in members[1:]:
            words = set(tokenize(member.text))
            novelty = len(words - seen) / max(1, len(words))
            if novelty >= 0.35:
                sentence = re.split(r"(?<=[.!?])\s+", member.text.strip())[0][:180]
                snippets.append(sentence)
                seen.update(words)
        text = anchor.text if not snippets else anchor.text + "\n" + " ".join(snippets)
        merged = self.add(
            text,
            ts=now,
            tags=list(dict.fromkeys(tag for n in members for tag in n.tags))[:8],
            ntype="fact",
            source_ids=self._original_sources(members),
            confidence=min(n.confidence for n in members),
            meta=self._merged_metadata(members, member_ids),
        )
        merged.strength = max(n.strength for n in members)
        merged.access_count = sum(n.access_count for n in members)
        merged.last_accessed = max(n.last_accessed for n in members)

        member_set = set(member_ids)
        old_edges = list(self.edges)
        self.edges = []
        for edge in old_edges:
            src = merged.id if edge.src in member_set else edge.src
            dst = merged.id if edge.dst in member_set else edge.dst
            if src != dst:
                self.edges.append(Edge(src, dst, edge.etype, edge.weight, edge.created_at))
        for nid in member_ids:
            self.nodes.pop(nid, None)
        self._rebuild_edge_indexes()
        return merged

    @staticmethod
    def _merged_metadata(members: list[Node], member_ids: list[str]) -> dict:
        metadata = {"operation": "merge", "merged_node_ids": member_ids}
        for key in ("role", "source", "category"):
            values = {node.meta.get(key) for node in members if node.meta.get(key) is not None}
            if len(values) == 1:
                metadata[key] = values.pop()
        if any(node.meta.get("protected") for node in members):
            metadata["protected"] = True
        return metadata

    def _split_mixed(self, now: float) -> int:
        candidates = [n for n in list(self.nodes.values()) if n.ntype == "fact" and len(n.text) > 500]
        splits = 0
        for node in candidates:
            if node.id not in self.nodes:
                continue
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", node.text) if len(s.strip()) > 20]
            if len(sentences) < 6:
                continue
            try:
                matrix = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, max_features=1000).fit_transform(sentences)
            except ValueError:
                continue
            sim = cosine_similarity(matrix)
            best: tuple[float, int, float] | None = None
            for boundary in range(2, len(sentences) - 1):
                left = sim[:boundary, :boundary]
                right = sim[boundary:, boundary:]
                cross = sim[:boundary, boundary:]
                left_cohesion = float((left.sum() - len(left)) / max(1, len(left) * (len(left) - 1)))
                right_cohesion = float((right.sum() - len(right)) / max(1, len(right) * (len(right) - 1)))
                cross_mean = float(cross.mean()) if cross.size else 0.0
                margin = min(left_cohesion, right_cohesion) - cross_mean
                candidate = (margin, boundary, cross_mean)
                if best is None or candidate > best:
                    best = candidate
            if best is None or best[0] < 0.08 or best[2] > 0.12:
                continue
            _, boundary, _ = best
            parts = [" ".join(sentences[:boundary]), " ".join(sentences[boundary:])]
            children = [
                self.add(
                    part,
                    ts=now,
                    ntype="fact",
                    source_ids=node.source_ids or [node.id],
                    confidence=node.confidence,
                    meta={**node.meta, "operation": "split", "parent_node_id": node.id},
                )
                for part in parts
            ]
            for child in children:
                child.strength = node.strength
                child.access_count = node.access_count
                child.last_accessed = node.last_accessed

            incident = [self._edge_index[key] for key in self._adj.get(node.id, set())]
            self.nodes.pop(node.id, None)
            self.edges = [e for e in self.edges if e.src != node.id and e.dst != node.id]
            self._rebuild_edge_indexes()
            for edge in incident:
                other_id = edge.dst if edge.src == node.id else edge.src
                if other_id not in self.nodes:
                    continue
                other_tokens = set(tokenize(self.nodes[other_id].text))
                overlaps = [len(other_tokens & set(tokenize(child.text))) for child in children]
                target_indexes = [int(np.argmax(overlaps))]
                if overlaps[0] == overlaps[1]:
                    target_indexes = [0, 1]
                for index in target_indexes:
                    self.link(children[index].id, other_id, edge.etype, edge.weight, ts=edge.created_at)
            self.link(children[0].id, children[1].id, "inferred", 0.25, ts=now)
            splits += 1
        return splits

    def _prune(self) -> int:
        candidates = []
        for nid, node in self.nodes.items():
            durable = node.meta.get("protected") or node.meta.get("role") == "knowledge"
            quality = node.strength * (0.5 + 0.5 * node.confidence)
            if not durable and quality < self.prune_threshold:
                candidates.append((quality, node.access_count, node.last_accessed, nid))
        max_prunable = max(0, len(self.nodes) - self.min_retained_nodes)
        dead = {item[3] for item in sorted(candidates)[:max_prunable]}
        if not dead:
            return 0
        for nid in dead:
            self.nodes.pop(nid, None)
        self.edges = [e for e in self.edges if e.src not in dead and e.dst not in dead]
        self._rebuild_edge_indexes()
        return len(dead)

    def _synthesize(self, now: float, *, max_new: int) -> int:
        ranked = sorted(
            (n for n in self.nodes.values() if n.ntype != "synthesis"),
            key=lambda n: (-(n.strength * (1 + math.log1p(n.access_count))), n.id),
        )
        created = 0
        used: set[str] = set()
        for node in ranked:
            if created >= max_new:
                break
            candidates = []
            for key in self._adj.get(node.id, set()):
                edge = self._edge_index[key]
                if edge.etype not in {"semantic", "tag", "inferred"} or edge.weight < 0.45:
                    continue
                other_id = edge.dst if edge.src == node.id else edge.src
                other = self.nodes.get(other_id)
                if (
                    other
                    and other.ntype != "synthesis"
                    and self._same_memory_scope(node, other)
                ):
                    candidates.append((edge.weight * other.strength, edge, other))
            candidates.sort(key=lambda x: (-x[0], x[2].id))
            partners = [item[2] for item in candidates[:2]]
            if node.id in used or len(partners) < 2 or any(p.id in used for p in partners):
                continue
            group = [node, *partners]
            sims = self._text_similarities([n.id for n in group])
            coherence = float((sims.sum() - len(group)) / (len(group) * (len(group) - 1))) if sims is not None else 0.0
            if coherence < 0.12:
                continue
            common = Counter(tag for n in group for tag in n.tags)
            theme = ", ".join(tag for tag, count in common.most_common(4) if count >= 2) or ", ".join(node.tags[:2])
            excerpts = [re.split(r"(?<=[.!?])\s+", n.text)[0][:160] for n in group]
            text = f"[association: {theme}] " + " ".join(excerpts)
            existing = self._find_similar_synthesis(text)
            confidence = float(np.clip(0.35 + 0.5 * coherence, 0.35, 0.85))
            if existing is not None:
                existing.confidence = min(1.0, existing.confidence + 0.1)
                existing.strength = min(2.0, existing.strength + 0.08)
                existing.source_ids = list(dict.fromkeys(existing.source_ids + self._original_sources(group)))
            else:
                synthesis = self.add(
                    text,
                    ts=now,
                    tags=[tag for tag, _ in common.most_common(8)],
                    ntype="synthesis",
                    source_ids=self._original_sources(group),
                    confidence=confidence,
                    meta={"operation": "association", "coherence": round(coherence, 4)},
                )
                synthesis.strength = 0.55
                for member in group:
                    self.link(synthesis.id, member.id, "inferred", 0.55, ts=now)
                created += 1
            used.update(n.id for n in group)
        return created

    def _find_similar_synthesis(self, text: str) -> Node | None:
        tokens = set(tokenize(text))
        if not tokens:
            return None
        for node in self.nodes.values():
            if node.ntype != "synthesis":
                continue
            other = set(tokenize(node.text))
            if other and len(tokens & other) / len(tokens | other) >= 0.72:
                return node
        return None

    def generate_training_examples(
        self,
        *,
        min_neighbors: int = 2,
        max_context: int = 3,
        max_examples: int = 2000,
        seed: int = 7,
        include_temporal_only: bool = False,
    ) -> list[dict]:
        candidates: list[dict] = []
        for target_id, target in self.nodes.items():
            neighbor_map: dict[str, dict[str, float]] = defaultdict(dict)
            for key in self._adj.get(target_id, set()):
                edge = self._edge_index[key]
                other_id = edge.dst if edge.src == target_id else edge.src
                neighbor_map[other_id][edge.etype] = edge.weight
            neighbors = []
            for other_id, evidence in neighbor_map.items():
                if not include_temporal_only and set(evidence) == {"temporal"}:
                    continue
                other = self.nodes[other_id]
                score = max(EDGE_TYPE_WEIGHT[t] * w for t, w in evidence.items()) * other.strength
                neighbors.append((score, other, evidence))
            neighbors.sort(key=lambda item: (-item[0], item[1].id))
            if len(neighbors) < min_neighbors:
                continue
            pool = neighbors[: min(8, len(neighbors))]
            for k in range(min_neighbors, min(max_context, len(pool)) + 1):
                for combo in combinations(pool, k):
                    contexts = []
                    for _, other, evidence in combo:
                        contexts.append(
                            {
                                "node_id": other.id,
                                "text": other.text,
                                "node_type": other.ntype,
                                "strength": round(other.strength, 3),
                                "relations": {key: round(value, 3) for key, value in sorted(evidence.items())},
                            }
                        )
                    candidates.append(
                        {
                            "context": contexts,
                            "target": target.text,
                            "target_meta": {
                                "node_id": target.id,
                                "node_type": target.ntype,
                                "strength": round(target.strength, 3),
                                "confidence": round(target.confidence, 3),
                                "source_ids": target.source_ids,
                            },
                        }
                    )
        Random(seed).shuffle(candidates)
        return candidates[:max_examples]

    def stats(self) -> dict:
        return {
            "nodes_total": len(self.nodes),
            "nodes_by_type": dict(Counter(n.ntype for n in self.nodes.values())),
            "edges_total": len(self.edges),
            "edges_by_type": dict(Counter(e.etype for e in self.edges)),
            "avg_strength": round(float(np.mean([n.strength for n in self.nodes.values()])), 3) if self.nodes else 0.0,
        }

    def validate(self) -> list[str]:
        errors = []
        seen = set()
        for edge in self.edges:
            key = self._edge_key(edge.src, edge.dst, edge.etype)
            if edge.src == edge.dst:
                errors.append(f"self-loop: {key}")
            if edge.src not in self.nodes or edge.dst not in self.nodes:
                errors.append(f"dangling edge: {key}")
            if key in seen:
                errors.append(f"duplicate edge: {key}")
            seen.add(key)
        if set(self._edge_index) != seen:
            errors.append("edge index is inconsistent")
        return errors

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS nodes(
                    id TEXT PRIMARY KEY, text TEXT NOT NULL, ntype TEXT NOT NULL,
                    created_at REAL NOT NULL, last_accessed REAL NOT NULL,
                    last_decayed REAL NOT NULL, strength REAL NOT NULL,
                    access_count INTEGER NOT NULL, tags TEXT NOT NULL,
                    source_ids TEXT NOT NULL, confidence REAL NOT NULL, meta TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS edges(
                    src TEXT NOT NULL, dst TEXT NOT NULL, etype TEXT NOT NULL,
                    weight REAL NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(src, dst, etype)
                );
                DELETE FROM metadata; DELETE FROM nodes; DELETE FROM edges;
                """
            )
            conn.execute("INSERT INTO metadata VALUES (?, ?)", ("schema_version", "2"))
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
            conn.execute("INSERT INTO metadata VALUES (?, ?)", ("config", json.dumps(config)))
            conn.executemany(
                "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        n.id, n.text, n.ntype, n.created_at, n.last_accessed, n.last_decayed,
                        n.strength, n.access_count, json.dumps(n.tags, ensure_ascii=False),
                        json.dumps(n.source_ids, ensure_ascii=False), n.confidence,
                        json.dumps(n.meta, ensure_ascii=False),
                    )
                    for n in self.nodes.values()
                ],
            )
            conn.executemany(
                "INSERT INTO edges VALUES (?,?,?,?,?)",
                [(e.src, e.dst, e.etype, e.weight, e.created_at) for e in self.edges],
            )

    @classmethod
    def load(cls, path: str | Path) -> "MemoryNet":
        with sqlite3.connect(path) as conn:
            metadata = dict(conn.execute("SELECT key, value FROM metadata"))
            config, migrated = migrate_legacy_defaults(
                json.loads(metadata.get("config", "{}"))
            )
            net = cls(**config)
            for row in conn.execute(
                "SELECT id,text,ntype,created_at,last_accessed,last_decayed,strength,"
                "access_count,tags,source_ids,confidence,meta FROM nodes"
            ):
                node = Node(
                    id=row[0], text=row[1], ntype=row[2], created_at=row[3], last_accessed=row[4],
                    last_decayed=row[5], strength=row[6], access_count=row[7], tags=json.loads(row[8]),
                    source_ids=json.loads(row[9]), confidence=row[10], meta=json.loads(row[11]),
                )
                net.nodes[node.id] = node
            for row in conn.execute("SELECT src,dst,etype,weight,created_at FROM edges"):
                net.edges.append(Edge(*row))
            net._rebuild_edge_indexes()
            if migrated:
                now = time.time()
                for node in net.nodes.values():
                    node.last_decayed = max(node.last_decayed, now)
                net.log.append("migrated legacy 0.1.x memory defaults to safe daily decay")
            return net
