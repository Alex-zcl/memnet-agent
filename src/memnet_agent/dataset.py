from __future__ import annotations

import hashlib
import json
from pathlib import Path
from random import Random
from typing import Any, Iterator, Literal

from .memory import AssociativeMemory

DatasetFormat = Literal["graph", "instruction", "chat"]


def iter_training_records(
    memory: AssociativeMemory,
    *,
    iterations: int = 1,
    max_examples_per_iteration: int = 2000,
    min_neighbors: int = 2,
    max_context: int = 3,
    seed: int = 7,
    format: DatasetFormat = "chat",
    consolidate_between_iterations: bool = False,
    max_syntheses: int = 3,
) -> Iterator[dict[str, Any]]:
    """Yield reproducible training records from the current graph.

    Multiple iterations reshuffle the graph-derived examples. When
    ``consolidate_between_iterations`` is enabled, each subsequent iteration is
    generated after one memory consolidation cycle, so the dataset evolves with
    the graph.
    """

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if format not in {"graph", "instruction", "chat"}:
        raise ValueError("format must be graph, instruction or chat")

    for iteration in range(iterations):
        examples = memory.generate_training_examples(
            min_neighbors=min_neighbors,
            max_context=max_context,
            max_examples=max_examples_per_iteration,
            seed=seed + iteration,
        )
        Random(seed + iteration).shuffle(examples)
        for position, example in enumerate(examples):
            yield _convert_record(example, iteration=iteration, position=position, format=format)
        if consolidate_between_iterations and iteration < iterations - 1:
            memory.sleep(max_syntheses=max_syntheses)


def write_training_dataset(
    memory: AssociativeMemory,
    path: str | Path,
    **kwargs: Any,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in iter_training_records(memory, **kwargs):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return destination


def _convert_record(
    example: dict[str, Any],
    *,
    iteration: int,
    position: int,
    format: DatasetFormat,
) -> dict[str, Any]:
    context = example["context"]
    target = example["target"]
    context_text = "\n\n".join(
        f"[{index}] {item['text']}" for index, item in enumerate(context, start=1)
    )
    digest = hashlib.sha256(
        f"{iteration}:{position}:{target}:{context_text}".encode("utf-8")
    ).hexdigest()[:20]
    metadata = {
        "example_id": digest,
        "iteration": iteration,
        "target_meta": example["target_meta"],
        "context_meta": [
            {
                key: value
                for key, value in item.items()
                if key != "text"
            }
            for item in context
        ],
    }
    if format == "graph":
        return {**example, "metadata": metadata}
    instruction = (
        "Восстанови целевой фрагмент памяти по связанному контексту. "
        "Не добавляй факты, которых нет в контексте."
    )
    if format == "instruction":
        return {
            "instruction": instruction,
            "input": context_text,
            "output": target,
            "metadata": metadata,
        }
    return {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": context_text},
            {"role": "assistant", "content": target},
        ],
        "metadata": metadata,
    }
