"""Tests for report-only GBrain promotion candidate queue."""

import json

from agent.gbrain_promotion_candidates import build_candidate, enqueue_candidate, QUEUE_PATH


def test_build_candidate_for_memory_spec_without_raw_transcript():
    c = build_candidate(
        session_id="sess-1",
        turn_id="turn-1",
        user_message="Memory spec: Fantasy15 canonical operating model should be in GBrain.",
        assistant_response="Agreed; durable memory belongs in GBrain, not pinned memory.",
        platform="telegram",
    )

    assert c is not None
    assert c.destination_slug == "projects/fantasy15"
    assert c.classification in {"project_context", "retrieval_hook"}
    assert "Fantasy15 canonical" not in c.summary
    assert c.evidence == [{"kind": "session", "id": "sess-1", "turn_id": "turn-1", "platform": "telegram"}]


def test_build_candidate_blocks_raw_secret():
    fake_secret = "sk-" + "a" * 40
    c = build_candidate(
        session_id="sess-1",
        turn_id="turn-1",
        user_message=f"Put this in GBrain: API key is {fake_secret}",
        assistant_response="Nope.",
    )

    assert c is None


def test_enqueue_candidate_dedupes_and_uses_profile_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    c = build_candidate(
        session_id="sess-1",
        turn_id="turn-1",
        user_message="Memory spec: GBrain durable memory operating model.",
        assistant_response="Queue report-only candidate.",
    )
    assert c is not None

    assert enqueue_candidate(c) is True
    assert enqueue_candidate(c) is False

    queue = tmp_path / QUEUE_PATH
    lines = queue.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["candidate_id"] == c.candidate_id
    assert payload["status"] == "pending"
    assert "Memory spec" not in payload["summary"]
