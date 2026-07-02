"""Report-only GBrain promotion candidate queue.

This module observes completed turns and appends small review candidates to a
JSONL queue. It does not write GBrain pages and does not preserve raw
transcripts; evidence references point back to session_search/session state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home
from tools.durable_write_guard import scan_text as _scan_durable_write_secret

QUEUE_PATH = "state/gbrain_promotion_candidates.jsonl"

_TRANSIENT_PATTERNS = (
    re.compile(r"\bPR\s*#?\d+\b", re.IGNORECASE),
    re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE),
    re.compile(r"\b(?:job|run|task|commit|sha)[_-]?[0-9a-f]{6,}\b", re.IGNORECASE),
    re.compile(r"\x1b\[[0-9;]*m"),
)

_DESTINATION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("projects/fantasy15", re.compile(r"\bfantasy\s*15\b|\bfantasy15\b", re.IGNORECASE)),
    ("projects/dogwalk-bones", re.compile(r"\bdogwalk\b|\bdog walk\b", re.IGNORECASE)),
    ("projects/smart-home", re.compile(r"\bhome assistant\b|\bsmart[- ]home\b|\bsonos\b|\bskylight\b", re.IGNORECASE)),
    ("meal-planning/preferences", re.compile(r"\bmeal planning\b|\bdinner\b|\bgrocery\b", re.IGNORECASE)),
    ("operations/hermes-memory-gbrain-promotion-audit", re.compile(r"\bgbrain\b|\bmemory spec\b|\bfact_store\b|\bdurable memory\b", re.IGNORECASE)),
)

_CLASSIFICATION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("safety", re.compile(r"\bsecret\b|\bcredential\b|\bapproval\b|\bprod(?:uction)?\b|\bsecurity\b", re.IGNORECASE)),
    ("procedure", re.compile(r"\bprocedure\b|\brunbook\b|\bworkflow\b|\bsteps?\b|\bhow to\b", re.IGNORECASE)),
    ("preference", re.compile(r"\bprefer\b|\bdon't\b|\bdo not\b|\balways\b|\bstyle\b", re.IGNORECASE)),
    ("retrieval_hook", re.compile(r"\bwhere\b|\bpath\b|\bslug\b|\bcanonical\b", re.IGNORECASE)),
)

_TRIGGER = re.compile(
    r"\b(we['’]?ve gone over|you should know|remember this|put this in gbrain|memory spec|canonical|durable memory|runbook|operating model)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Candidate:
    schema_version: int
    created_at: str
    session_id: str
    turn_id: str
    profile: str
    source: str
    candidate_id: str
    destination_slug: str
    classification: str
    summary: str
    evidence: list[dict[str, Any]]
    status: str = "pending"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _queue_file() -> Path:
    return get_hermes_home() / QUEUE_PATH


def _has_transient_shape(text: str) -> bool:
    return any(p.search(text) for p in _TRANSIENT_PATTERNS)


def _destination(text: str) -> str | None:
    for slug, pattern in _DESTINATION_RULES:
        if pattern.search(text):
            return slug
    return None


def _classification(text: str) -> str:
    for label, pattern in _CLASSIFICATION_RULES:
        if pattern.search(text):
            return label
    return "project_context"


def _candidate_hash(parts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update((part or "").encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            cid = data.get("candidate_id")
            if cid:
                ids.add(str(cid))
    except Exception:
        return ids
    return ids


def build_candidate(
    *,
    session_id: str | None,
    turn_id: str | int | None,
    user_message: str | None,
    assistant_response: str | None,
    platform: str | None = None,
) -> Candidate | None:
    """Return a report-only candidate for high-signal turns, else None."""

    text = "\n".join([user_message or "", assistant_response or ""])
    if not text.strip():
        return None
    if _scan_durable_write_secret(text, surface="gbrain_promotion_candidate").allowed is False:
        return None
    if _has_transient_shape(text) and not _TRIGGER.search(text):
        return None
    destination_slug = _destination(text)
    if destination_slug is None:
        return None
    if not (_TRIGGER.search(text) or destination_slug in {"operations/hermes-memory-gbrain-promotion-audit"}):
        return None

    classification = _classification(text)
    sid = session_id or ""
    tid = str(turn_id or "")
    candidate_id = _candidate_hash([sid, tid, destination_slug, classification])
    summary = (
        "Review this turn for durable GBrain promotion. "
        "Queue is report-only and stores evidence references instead of raw transcript text."
    )
    return Candidate(
        schema_version=1,
        created_at=_utc_now(),
        session_id=sid,
        turn_id=tid,
        profile="default",
        source="turn_finalizer.post_llm_call",
        candidate_id=candidate_id,
        destination_slug=destination_slug,
        classification=classification,
        summary=summary,
        evidence=[{"kind": "session", "id": sid, "turn_id": tid, "platform": platform or ""}],
    )


def enqueue_candidate(candidate: Candidate) -> bool:
    """Append candidate if not already queued. Returns True when written."""

    path = _queue_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    if candidate.candidate_id in _existing_ids(path):
        return False
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(candidate), ensure_ascii=False, sort_keys=True) + "\n")
    try:
        path.chmod(0o600)
    except Exception:
        pass
    return True


def observe_turn(**kwargs: Any) -> bool:
    """Build and enqueue a report-only promotion candidate from a completed turn."""

    candidate = build_candidate(
        session_id=kwargs.get("session_id"),
        turn_id=kwargs.get("turn_id"),
        user_message=kwargs.get("user_message"),
        assistant_response=kwargs.get("assistant_response"),
        platform=kwargs.get("platform"),
    )
    if candidate is None:
        return False
    return enqueue_candidate(candidate)
