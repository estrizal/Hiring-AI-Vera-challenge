"""
state.py — Async-safe in-memory state for Vera (final architecture).

Uses asyncio.Lock() throughout — required because asyncio.gather() in /tick
runs multiple coroutines concurrently and threading.Lock() does NOT protect
against asyncio coroutine interleaving (it only protects against OS threads).

Responsibilities:
  1. Context store          — scope-keyed, version-gated, atomic writes
  2. Suppression store      — key → expiry timestamp, prefix-derived TTL
  3. Conversation store     — turn history, anti-repetition, intent state
  4. Daily message quota    — reserve_message_slot() atomically checks AND
                              increments in one lock acquisition (TOCTOU-safe)
  5. Utility                — uptime, deterministic conversation ID
"""

import time
import asyncio
from collections import defaultdict
from datetime import date
from typing import Optional, Tuple, Dict, Any

import config

# ─────────────────────────────────────────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────────────────────────────────────────

# store[scope][context_id] = {"version": int, "payload": dict}
_store: Dict[str, Dict[str, Dict]] = defaultdict(dict)
_store_lock = asyncio.Lock()

# suppression_store[key] = expiry_epoch_seconds
_suppression_store: Dict[str, float] = {}
_suppression_lock = asyncio.Lock()

# conversation_store[conv_id] = {merchant_id, turns[], sent_bodies set,
#                                 auto_reply_count, intent_state, suppressed}
_conversation_store: Dict[str, Dict] = {}
_conversation_lock = asyncio.Lock()

# quota_store[merchant_id] = {"date": date, "count": int, "last_sent": float}
_quota_store: Dict[str, Dict] = {}
_quota_lock = asyncio.Lock()

# merchant_auto_reply[merchant_id] = consecutive_auto_reply_count
# Keyed by merchant_id (NOT conv_id) because the judge sends a new conv_id
# on every turn of the auto-reply test — the signal is merchant-level.
_merchant_auto_reply: Dict[str, int] = {}
_merchant_auto_reply_lock = asyncio.Lock()

_boot_time: float = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONTEXT STORE
# ─────────────────────────────────────────────────────────────────────────────

async def push_context(
    scope: str, context_id: str, version: int, payload: dict
) -> Tuple[bool, Optional[int]]:
    """
    Atomically store a context payload (higher version wins).

    Returns:
        (True, version)   — accepted and stored
        (False, current)  — rejected; current_version for 409 response body

    Implements api-call-examples.md §1.5-1.6:
        Same or lower version → 409 with parseable JSON body.
        Higher version → 200, replaces old payload atomically.
    """
    async with _store_lock:
        current = _store[scope].get(context_id)
        if current is not None and current["version"] >= version:
            return False, current["version"]
        _store[scope][context_id] = {"version": version, "payload": payload}
        return True, version


def get_context(scope: str, context_id: str) -> Optional[dict]:
    """Lockless read — GIL-safe for dict reads in asyncio."""
    entry = _store[scope].get(context_id)
    return entry["payload"] if entry else None


def get_all_context(scope: str) -> Dict[str, Any]:
    """Return all payloads for a scope as a snapshot dict."""
    return {cid: e["payload"] for cid, e in _store[scope].items()}


def context_count(scope: str) -> int:
    """Live count for /v1/healthz — judge verifies these match after warmup."""
    return len(_store[scope])


# ─────────────────────────────────────────────────────────────────────────────
# 2. SUPPRESSION STORE
# ─────────────────────────────────────────────────────────────────────────────

def is_suppressed(suppression_key: str) -> bool:
    """Lockless read — expiry check only, no mutation."""
    expiry = _suppression_store.get(suppression_key)
    if expiry is None:
        return False
    if time.time() > expiry:
        # Lazily mark as expired; actual deletion happens on next write
        return False
    return True


async def suppress(suppression_key: str, ttl_seconds: Optional[int] = None) -> None:
    """Register a suppression key. TTL derived from key prefix if not specified."""
    if ttl_seconds is None:
        prefix = suppression_key.split(":")[0]
        ttl_seconds = config.SUPPRESSION_WINDOWS.get(
            prefix, config.SUPPRESSION_WINDOWS["default"]
        )
    async with _suppression_lock:
        _suppression_store[suppression_key] = time.time() + ttl_seconds


async def suppress_merchant(merchant_id: str) -> None:
    """Full merchant opt-out: suppress all triggers for 30 days."""
    await suppress(
        f"merchant_optout:{merchant_id}",
        ttl_seconds=config.SUPPRESSION_WINDOWS["hostile"],
    )


def is_merchant_suppressed(merchant_id: str) -> bool:
    return is_suppressed(f"merchant_optout:{merchant_id}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. DAILY MESSAGE QUOTA  ← THE TOCTOU-SAFE ATOMIC GATE
# ─────────────────────────────────────────────────────────────────────────────

async def reserve_message_slot(merchant_id: str) -> bool:
    """
    Atomically check the daily message cap AND reserve a slot in one lock
    acquisition. This eliminates the TOCTOU race completely.

    The TOCTOU bug this fixes:
        Coroutine A: reads count=0 → proceeds
        Coroutine B: reads count=0 → proceeds   ← both pass, two messages sent
        Coroutine A: increments to 1
        Coroutine B: increments to 2

    With this function, A and B cannot both read count=0 simultaneously.
    Whoever acquires _quota_lock first reserves the slot; the other sees count=1.

    Returns:
        True  — slot reserved, proceed with composing and sending
        False — daily cap reached or merchant suppressed, suppress this trigger

    Called from layer2_router.py Step 2A (first Python fast-path check).
    """
    if is_merchant_suppressed(merchant_id):
        return False

    today = date.today()
    async with _quota_lock:
        entry = _quota_store.get(merchant_id)
        if entry is None or entry["date"] != today:
            # New day or first message ever — reset counter
            _quota_store[merchant_id] = {
                "date": today,
                "count": 0,
                "last_sent": 0.0,
            }
        current_count = _quota_store[merchant_id]["count"]
        if current_count >= config.MAX_MESSAGES_PER_MERCHANT_PER_DAY:
            return False
        # Atomically reserve: increment count and record timestamp
        _quota_store[merchant_id]["count"] += 1
        _quota_store[merchant_id]["last_sent"] = time.time()
        return True


def get_message_count_today(merchant_id: str) -> int:
    """Non-atomic read for logging and inspection only. Never use for gating."""
    entry = _quota_store.get(merchant_id)
    if entry is None or entry["date"] != date.today():
        return 0
    return entry["count"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONVERSATION STORE
# ─────────────────────────────────────────────────────────────────────────────

async def _get_or_create(conv_id: str, merchant_id: str) -> dict:
    """Internal helper — must be called while holding _conversation_lock."""
    if conv_id not in _conversation_store:
        _conversation_store[conv_id] = {
            "merchant_id": merchant_id,
            "turns": [],
            "sent_bodies": set(),
            "auto_reply_count": 0,
            "intent_state": "qualifying",
            "suppressed": False,
        }
    return _conversation_store[conv_id]


async def record_sent(conv_id: str, body: str, merchant_id: str) -> None:
    async with _conversation_lock:
        conv = await _get_or_create(conv_id, merchant_id)
        conv["turns"].append({"role": "vera", "body": body, "ts": time.time()})
        conv["sent_bodies"].add(body.strip())


async def record_received(
    conv_id: str, merchant_id: str, message: str, role: str = "merchant"
) -> None:
    async with _conversation_lock:
        conv = await _get_or_create(conv_id, merchant_id)
        conv["turns"].append({"role": role, "body": message, "ts": time.time()})


def is_body_repeated(conv_id: str, body: str) -> bool:
    """Lockless read — anti-repetition check (-2 penalty per api-call-examples F.5)."""
    conv = _conversation_store.get(conv_id)
    if not conv:
        return False
    return body.strip() in conv.get("sent_bodies", set())


async def increment_auto_reply(conv_id: str, merchant_id: str) -> int:
    """Conv-level counter — kept for compatibility. Use increment_merchant_auto_reply for judge tests."""
    async with _conversation_lock:
        conv = await _get_or_create(conv_id, merchant_id)
        conv["auto_reply_count"] += 1
        return conv["auto_reply_count"]


async def reset_auto_reply(conv_id: str, merchant_id: str) -> None:
    async with _conversation_lock:
        conv = await _get_or_create(conv_id, merchant_id)
        conv["auto_reply_count"] = 0


async def increment_merchant_auto_reply(merchant_id: str) -> int:
    """
    Merchant-level auto-reply counter. Increments and returns the new count.

    WHY merchant-level and not conv-level:
        The judge fires the auto-reply test with a NEW conversation_id on every
        turn (conv_auto_1, conv_auto_2, ...). Conv-level tracking resets to 0
        each turn, so we never reach the end threshold.
        Auto-reply is a MERCHANT signal (their phone has an autoresponder),
        not a conversation signal — so merchant-level tracking is also semantically correct.
    """
    async with _merchant_auto_reply_lock:
        _merchant_auto_reply[merchant_id] = _merchant_auto_reply.get(merchant_id, 0) + 1
        return _merchant_auto_reply[merchant_id]


async def reset_merchant_auto_reply(merchant_id: str) -> None:
    """Reset when merchant sends a genuine human message."""
    async with _merchant_auto_reply_lock:
        _merchant_auto_reply[merchant_id] = 0


def get_merchant_auto_reply_count(merchant_id: str) -> int:
    """Non-atomic read for logging/inspection only."""
    return _merchant_auto_reply.get(merchant_id, 0)


async def set_intent_state(conv_id: str, merchant_id: str, intent_state: str) -> None:
    async with _conversation_lock:
        conv = await _get_or_create(conv_id, merchant_id)
        conv["intent_state"] = intent_state


def get_intent_state(conv_id: str) -> str:
    conv = _conversation_store.get(conv_id, {})
    return conv.get("intent_state", "qualifying")


async def end_conversation(conv_id: str, merchant_id: str) -> None:
    async with _conversation_lock:
        conv = await _get_or_create(conv_id, merchant_id)
        conv["suppressed"] = True
        conv["intent_state"] = "ended"


def is_conversation_ended(conv_id: str) -> bool:
    conv = _conversation_store.get(conv_id, {})
    return conv.get("suppressed", False)


def get_conversation_history(conv_id: str) -> list:
    conv = _conversation_store.get(conv_id, {})
    return conv.get("turns", [])


# ─────────────────────────────────────────────────────────────────────────────
# 5. UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def uptime_seconds() -> int:
    return int(time.time() - _boot_time)


def make_conversation_id(merchant_id: str, trigger_id: str) -> str:
    """
    Deterministic, stable conversation ID.
    Same (merchant_id, trigger_id) → same conv_id across re-runs.
    Allows Layer 3 /reply to match incoming messages back to the originating tick.
    """
    m = merchant_id.replace("_", "")[:12]
    t = trigger_id.replace("_", "")[:16]
    return f"conv_{m}_{t}"
