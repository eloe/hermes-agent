"""Durable-write secret guard coverage for holographic fact_store."""

import pytest

from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.retrieval import FactRetriever


def test_add_fact_blocks_raw_secret_without_echo(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory_store.db")
    fake_secret = "sk-" + "a" * 40

    with pytest.raises(ValueError) as exc:
        store.add_fact(f"OpenAI key is {fake_secret}")

    message = str(exc.value)
    assert "secret value" in message
    assert "openai_key" in message
    assert fake_secret not in message
    assert FactRetriever(store).search("OpenAI", min_trust=0.0) == []


def test_update_fact_blocks_raw_secret_and_preserves_existing(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory_store.db")
    fact_id = store.add_fact("Safe durable fact")
    fake_secret = "ghp_" + "A" * 40

    with pytest.raises(ValueError) as exc:
        store.update_fact(fact_id, content=f"GitHub token is {fake_secret}")

    message = str(exc.value)
    assert "secret value" in message
    assert "github_token" in message
    assert fake_secret not in message
    results = FactRetriever(store).search("Safe", min_trust=0.0)
    assert [r["content"] for r in results] == ["Safe durable fact"]


def test_add_fact_allows_secret_manager_location_reference(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory_store.db")
    content = "Fantasy15 token lives in 1Password vault Mr. Bones, item Fantasy15 Vercel, field credential."

    fact_id = store.add_fact(content)

    assert fact_id > 0
    results = FactRetriever(store).search("Fantasy15", min_trust=0.0)
    assert [r["content"] for r in results] == [content]
