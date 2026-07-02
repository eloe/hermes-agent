"""Shared guard for text headed into durable memory stores.

This module intentionally detects only high-confidence secret *values*.
It should allow references to secret-manager locations (for example,
"1Password vault X, item Y, field Z") while blocking raw API keys, tokens,
JWTs, private-key blocks, password assignments, and connection strings.

The guard never returns or logs the matched value. Callers may surface the
finding type and a short fingerprint only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, Optional


@dataclass(frozen=True)
class SecretFinding:
    """A high-confidence durable-write secret finding.

    ``fingerprint`` is derived from the matched value but is not reversible;
    callers can use it to correlate logs without exposing the secret.
    """

    finding_type: str
    surface: str
    fingerprint: str


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    finding: Optional[SecretFinding] = None


# Specific provider/token shapes first; generic assignment/URL patterns later.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]{16,}?-----END (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
            re.IGNORECASE,
        ),
    ),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{32,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
    (
        "connection_string",
        re.compile(
            r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:@/]+:[^\s@/]{8,}@[^\s]+",
            re.IGNORECASE,
        ),
    ),
    (
        "password_assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|client[_-]?secret)\b\s*[:=]\s*['\"]?([^'\"\s]{16,})['\"]?"
        ),
    ),
)

_ALLOWED_LOCATION_HINTS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b1Password\b.*\b(?:vault|item|field)\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\b(?:vault|item|field)\b.*\b1Password\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bsecret-manager location\b", re.IGNORECASE),
)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _matched_value(match: re.Match[str]) -> str:
    # For assignment patterns, the first capturing group is the actual value.
    if match.lastindex:
        for idx in range(1, match.lastindex + 1):
            val = match.group(idx)
            if val:
                return val
    return match.group(0)


def scan_text(text: str | None, *, surface: str = "durable_write") -> GuardResult:
    """Scan text for high-confidence secret values.

    Returns ``GuardResult(allowed=True)`` when no raw secret value is found.
    The result never contains the matched secret value.
    """

    if not text:
        return GuardResult(allowed=True)

    # A bare secret-location reference is allowed. If a raw token pattern is
    # present too, the specific token patterns below still block it.
    for finding_type, pattern in _SECRET_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        value = _matched_value(match)
        # Password-assignment-like prose can be a location pointer; allow only
        # when it does not also match one of the concrete token/value patterns.
        if finding_type == "password_assignment" and any(p.search(text) for p in _ALLOWED_LOCATION_HINTS):
            continue
        return GuardResult(
            allowed=False,
            finding=SecretFinding(
                finding_type=finding_type,
                surface=surface,
                fingerprint=_fingerprint(value),
            ),
        )
    return GuardResult(allowed=True)


def block_message(
    result: GuardResult,
    *,
    store_name: str = "Durable write",
    operation_index: int | None = None,
    update: bool = False,
) -> str:
    """Build user-visible block wording without exposing matched values."""

    finding_type = result.finding.finding_type if result.finding else "secret_value"
    if operation_index is not None:
        return (
            f"Memory batch blocked at operation {operation_index}: the proposed durable memory "
            f"appears to contain a secret value ({finding_type}). No operations were applied. "
            "Store only the secret-manager location, not the raw value."
        )
    if update:
        return (
            f"Memory update blocked: the proposed durable memory appears to contain a secret value "
            f"({finding_type}). The existing memory was left unchanged."
        )
    if store_name.lower().startswith("memory"):
        return (
            f"Memory write blocked: the proposed durable memory appears to contain a secret value "
            f"({finding_type}). No value was saved. Store only the secret-manager location, "
            "for example: 1Password vault <vault>, item <item>, field <field>."
        )
    return (
        f"Durable write blocked: proposed content appears to contain a secret value "
        f"({finding_type}). No durable store was updated."
    )


def guard_or_error(
    text: str | None,
    *,
    surface: str,
    store_name: str = "Durable write",
    operation_index: int | None = None,
    update: bool = False,
) -> Optional[str]:
    """Return a safe error string if text should be blocked, else None."""

    result = scan_text(text, surface=surface)
    if result.allowed:
        return None
    return block_message(
        result,
        store_name=store_name,
        operation_index=operation_index,
        update=update,
    )


def contains_secret(texts: Iterable[str], *, surface: str) -> GuardResult:
    """Return the first blocking finding across multiple text payloads."""

    for text in texts:
        result = scan_text(text, surface=surface)
        if not result.allowed:
            return result
    return GuardResult(allowed=True)
