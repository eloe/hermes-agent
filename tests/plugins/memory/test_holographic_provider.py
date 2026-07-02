"""Regression tests for Hermes-local Holographic provider carry patches."""

from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


def test_sync_turn_auto_extracts_when_enabled(tmp_path):
    provider = HolographicMemoryProvider(
        config={
            "db_path": str(tmp_path / "memory_store.db"),
            "auto_extract": True,
            "default_trust": 0.5,
        }
    )
    provider.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="telegram")

    provider.sync_turn(
        "I prefer benchmark answer code SOLAR-BASIL-07 for slot seven.",
        "noted",
    )

    facts = provider._store.list_facts(limit=10)
    assert any("SOLAR-BASIL-07" in fact["content"] for fact in facts)


def test_prefetch_uses_configured_limit():
    class RecordingRetriever:
        def __init__(self):
            self.limit = None

        def search(self, query, *, min_trust, limit):
            self.limit = limit
            return [{"content": "remembered fact", "trust_score": 0.7}]

    retriever = RecordingRetriever()
    provider = HolographicMemoryProvider(config={"prefetch_limit": 17})
    provider._retriever = retriever

    block = provider.prefetch("what should I remember?")

    assert "remembered fact" in block
    assert retriever.limit == 17


def test_identifier_recall_ranks_exact_nonce_facts(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory_store.db", default_trust=0.65)
    retriever = FactRetriever(store, hrr_weight=0.0)

    for idx in range(40):
        store.add_fact(
            f"User requested recall of answer codes for benchmark nonce hindsight-20260513-{idx:06d} "
            "in order SLOT01 through SLOT10, using UNKNOWN for missing slots",
            category="general",
        )
    store.add_fact(
        "prefers benchmark nonce holographic-20260513-214224 slot 01 answer code SOLAR-BASIL-01",
        category="user_pref",
    )
    store.add_fact(
        "prefers benchmark nonce holographic-20260513-214224 slot 02 answer code LUNAR-CEDAR-02",
        category="user_pref",
    )

    results = retriever.search(
        "Recall the answer codes for benchmark nonce holographic-20260513-214224 in order. "
        "Reply only as SLOT01=..., SLOT02=..., through SLOT03=...; use UNKNOWN for missing slots.",
        min_trust=0.2,
        limit=2,
    )

    contents = [result["content"] for result in results]
    assert "SOLAR-BASIL-01" in contents[0]
    assert "LUNAR-CEDAR-02" in contents[1]


def test_identifier_recall_candidate_pool_keeps_all_nonce_facts(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory_store.db", default_trust=0.65)
    retriever = FactRetriever(store, hrr_weight=0.0)
    nonce = "holographic-20260513-215856"
    codes = [
        "SOLAR-BASIL-01",
        "LUNAR-CEDAR-02",
        "EMBER-QUARTZ-03",
        "TIDAL-MARBLE-04",
        "NOVA-SAFFRON-05",
        "ORBIT-PEPPER-06",
        "VECTOR-INDIGO-07",
        "ANCHOR-COPPER-08",
        "PIXEL-JUNIPER-09",
        "CIRCUIT-AMBER-10",
    ]

    for idx in range(150):
        store.add_fact(
            f"User requested recall of answer codes for benchmark nonce hindsight-20260513-{idx:06d} "
            "in order SLOT01 through SLOT10, using UNKNOWN for missing slots",
            category="general",
        )
    for idx, code in enumerate(codes, 1):
        store.add_fact(
            f"prefers benchmark nonce {nonce} slot {idx:02d} answer code {code}",
            category="user_pref",
        )

    results = retriever.search(
        f"Recall from persistent memory/context the answer codes for benchmark nonce {nonce} in order. "
        "Do not call tools. Reply only as SLOT01=..., SLOT02=..., through SLOT10=...; "
        "use UNKNOWN for missing slots.",
        min_trust=0.2,
        limit=10,
    )

    contents = "\n".join(result["content"] for result in results)
    assert [code for code in codes if code in contents] == codes
