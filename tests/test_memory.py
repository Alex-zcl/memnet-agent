from __future__ import annotations

from memnet_agent import AssociativeMemory


def test_free_text_search_returns_relevant_memory() -> None:
    memory = AssociativeMemory()
    rabbit = memory.add("Alice follows a white rabbit into a tunnel", tags=["alice", "rabbit"])
    memory.add("The queen plays croquet", tags=["queen", "croquet"])
    hits = memory.search("Where did Alice follow the rabbit?", top_k=1, reinforce=False)
    assert hits
    assert hits[0].node.id == rabbit.id
    assert hits[0].score > 0


def test_json_dictionary_roundtrip() -> None:
    memory = AssociativeMemory()
    first = memory.add("first memory", ts=0)
    second = memory.add("second memory", ts=1)
    memory.link(first.id, second.id, "semantic", 0.8, ts=1)
    restored = AssociativeMemory.from_dict(memory.to_dict())
    assert restored.stats() == memory.stats()
    assert not restored.validate()


def test_legacy_sqlite_is_migrated(tmp_path) -> None:
    import json
    import sqlite3

    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE nodes(
                id TEXT PRIMARY KEY, text TEXT, ntype TEXT,
                created_at REAL, last_accessed REAL, strength REAL,
                access_count INTEGER, tags TEXT, source_ids TEXT, confidence REAL
            );
            CREATE TABLE edges(
                src TEXT, dst TEXT, etype TEXT, weight REAL, created_at REAL
            );
            """
        )
        connection.execute(
            "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("a", "legacy one", "raw", 0.0, 1.0, 0.8, 1, json.dumps(["legacy"]), "[]", 1.0),
        )
        connection.execute(
            "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("b", "legacy two", "raw", 2.0, 2.0, 0.7, 0, json.dumps(["legacy"]), "[]", 0.9),
        )
        connection.execute("INSERT INTO edges VALUES (?,?,?,?,?)", ("a", "b", "tag", 0.5, 2.0))

    memory = AssociativeMemory.load_external(path)
    assert memory.stats()["nodes_total"] == 2
    assert memory.stats()["edges_total"] == 1
    assert memory.nodes["a"].last_decayed == 1.0
    assert memory.nodes["a"].meta["imported_schema"] == "legacy-sqlite-v1"
    assert not memory.validate()



def test_safe_defaults_do_not_prune_recent_memory() -> None:
    memory = AssociativeMemory()
    node = memory.add("A recent durable conversational memory", ts=100.0)
    result = memory.sleep(now=160.0, max_syntheses=0)
    assert result["pruned"] == 0
    assert node.id in memory.nodes
    assert memory.decay_time_unit == 86400.0


def test_explicit_knowledge_is_protected_from_pruning() -> None:
    memory = AssociativeMemory(prune_threshold=0.9, min_retained_nodes=0)
    node = memory.add("User name is Alexander", meta={"role": "knowledge", "protected": True})
    node.strength = 0.001
    memory.sleep(max_syntheses=0)
    assert node.id in memory.nodes


def test_different_roles_are_not_merged() -> None:
    memory = AssociativeMemory(merge_threshold=0.0)
    user = memory.add("My name is Sasha", meta={"role": "user"})
    assistant = memory.add("My name is Sasha", meta={"role": "assistant"})
    memory.sleep(max_syntheses=0)
    assert user.id in memory.nodes
    assert assistant.id in memory.nodes
