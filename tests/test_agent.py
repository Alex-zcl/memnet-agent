from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

from memnet_agent import (
    AgentSleepingError,
    AssociativeMemory,
    MemoryAgent,
    SleepConfig,
)


class RecordingModel:
    def __init__(self, answer: str = "model answer") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> dict[str, str]:
        self.prompts.append(prompt)
        return {"content": self.answer}


def test_simple_inference_uses_memory_and_updates_graph() -> None:
    model = RecordingModel("Релиз в пятницу")
    agent = MemoryAgent(model, sleep=SleepConfig.off())
    learned = agent.learn("Релиз проекта Atlas запланирован на пятницу.")
    before = agent.stats()["nodes_total"]

    result = agent.ask_with_trace("Когда релиз Atlas?")

    assert result.answer == "Релиз в пятницу"
    assert result.memories
    assert learned[0] in model.prompts[-1]
    assert "пятницу" in model.prompts[-1]
    assert agent.stats()["nodes_total"] == before + 2
    assert result.user_node_id
    assert result.assistant_node_id
    agent.close()


def test_learn_does_not_call_model_by_default() -> None:
    model = RecordingModel()
    agent = MemoryAgent(model, sleep=SleepConfig.off())
    node_ids = agent.learn("One fact. Another related fact.")
    assert node_ids
    assert model.prompts == []
    agent.close()


def test_learn_can_preprocess_silently() -> None:
    model = RecordingModel("compact knowledge")
    agent = MemoryAgent(model, sleep=SleepConfig.off())
    ids = agent.learn("Very long source material about Atlas.", preprocess_with_model=True)
    assert len(model.prompts) == 1
    assert agent.memory.nodes[ids[0]].text == "compact knowledge"
    agent.close()


def test_update_memory_can_be_disabled() -> None:
    agent = MemoryAgent(lambda prompt: "answer", update_memory=False, sleep=SleepConfig.off())
    answer = agent.ask("hello")
    assert answer == "answer"
    assert agent.stats()["nodes_total"] == 0
    agent.close()


def test_callable_alias() -> None:
    agent = MemoryAgent(lambda prompt: "answer", sleep=SleepConfig.off())
    assert agent("hello") == "answer"
    agent.close()


def test_async_model() -> None:
    async def model(prompt: str) -> str:
        return "async answer"

    async def run() -> None:
        agent = MemoryAgent(model, sleep=SleepConfig.off())
        assert await agent.aask("hello") == "async answer"
        agent.close()

    asyncio.run(run())


def test_scheduled_window_blocks_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SleepConfig.scheduled(
        start="01:00", end="03:00", timezone="UTC", allow_interrupt=False
    )
    monkeypatch.setattr(SleepConfig, "in_scheduled_window", lambda self, moment=None: True)
    agent = MemoryAgent(lambda prompt: "answer", sleep=config)
    with pytest.raises(AgentSleepingError):
        agent.ask("hello")
    # Silent graph updates remain available.
    assert agent.learn("knowledge that is long enough to store")
    agent.close()


def test_scheduled_window_calculation() -> None:
    daytime = SleepConfig.scheduled(start="10:00", end="12:00", timezone="UTC")
    assert daytime.in_scheduled_window(datetime.fromisoformat("2026-08-04T11:00:00+00:00"))
    assert not daytime.in_scheduled_window(datetime.fromisoformat("2026-08-04T13:00:00+00:00"))

    overnight = SleepConfig.scheduled(start="22:00", end="05:00", timezone="UTC")
    assert overnight.in_scheduled_window(datetime.fromisoformat("2026-08-04T23:00:00+00:00"))
    assert overnight.in_scheduled_window(datetime.fromisoformat("2026-08-04T03:00:00+00:00"))
    assert not overnight.in_scheduled_window(datetime.fromisoformat("2026-08-04T12:00:00+00:00"))


def test_graph_export_and_external_load(tmp_path: Path) -> None:
    agent = MemoryAgent(lambda prompt: "answer", sleep=SleepConfig.off())
    agent.learn(["Atlas uses PostgreSQL.", "Anna owns Atlas."])

    sqlite_path = agent.export_graph(tmp_path / "memory.sqlite")
    json_path = agent.export_graph(tmp_path / "memory.json")
    graphml_path = agent.export_graph(tmp_path / "memory.graphml")
    bundle_path = agent.export_graph(tmp_path / "memory.zip")

    assert sqlite_path.exists()
    assert json_path.exists()
    assert graphml_path.exists()
    assert bundle_path.exists()
    for path in (sqlite_path, json_path, bundle_path):
        restored = AssociativeMemory.load_external(path)
        assert restored.stats() == agent.memory.stats()
        assert not restored.validate()
    agent.close()


def test_load_graph_merge(tmp_path: Path) -> None:
    first = MemoryAgent(lambda prompt: "answer", sleep=SleepConfig.off())
    first.learn("first graph fact")
    external = first.export_graph(tmp_path / "external.json")

    second = MemoryAgent(lambda prompt: "answer", sleep=SleepConfig.off())
    second.learn("second graph fact")
    second.load_graph(external, replace=False)
    texts = [node.text for node in second.memory.nodes.values()]
    assert any("first graph" in text for text in texts)
    assert any("second graph" in text for text in texts)
    first.close()
    second.close()


def test_iterative_training_dataset(tmp_path: Path) -> None:
    memory = AssociativeMemory(semantic_threshold=0.0, temporal_window=10)
    agent = MemoryAgent(lambda prompt: "answer", memory=memory, sleep=SleepConfig.off())
    agent.learn(
        [
            "Atlas uses PostgreSQL and Redis.",
            "Atlas has an architecture review.",
            "Anna owns the Atlas review.",
            "The review is scheduled for August.",
        ]
    )
    output = agent.generate_training_dataset(
        tmp_path / "train.jsonl",
        iterations=2,
        max_examples_per_iteration=20,
        format="chat",
    )
    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert lines
    assert {line["metadata"]["iteration"] for line in lines} <= {0, 1}
    assert all("messages" in line for line in lines)
    agent.close()


def test_idle_background_maintenance_runs() -> None:
    import time

    agent = MemoryAgent(
        lambda prompt: "answer",
        sleep=SleepConfig.idle(
            after_seconds=0.01,
            check_every_seconds=0.01,
            maintenance_every_seconds=0.01,
            max_syntheses=0,
        ),
    )
    agent.learn("A persistent fact for the idle maintenance test.")
    deadline = time.monotonic() + 1.0
    while not agent.memory.log and time.monotonic() < deadline:
        time.sleep(0.01)
    assert agent.memory.log
    agent.close()


def test_worker_mode_starts_multiple_memory_workers() -> None:
    import time

    agent = MemoryAgent(
        lambda prompt: "answer",
        sleep=SleepConfig.workers(
            count=2,
            maintenance_every_seconds=0.02,
            max_syntheses=0,
        ),
    )
    assert agent.stats()["background_threads"] == 2
    deadline = time.monotonic() + 1.0
    while not agent.memory.log and time.monotonic() < deadline:
        time.sleep(0.01)
    assert agent.memory.log
    agent.close()
