"""Schema guidance tests for holographic fact_store."""

from plugins.memory.holographic import FACT_STORE_SCHEMA


def test_fact_store_schema_documents_promotion_tags():
    description = FACT_STORE_SCHEMA["description"]
    assert "promoted_to_gbrain" in description
    assert "transient" in description
    assert "do_not_promote" in description
    assert "schema-migrate" in description
