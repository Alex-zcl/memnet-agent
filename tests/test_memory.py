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
