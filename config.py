"""
config.py — Central configuration. Zero business logic.
"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# ── Models ────────────────────────────────────────────────────────────────────
# Unified to gpt-4o-mini: maximum speed on Tier 1 (500 RPM), lowest latency.
ROUTER_MODEL: str   = "gpt-4o-mini"   # Layer 2 semantic router
COMPOSER_MODEL: str = "gpt-4o"   # Layer 3 composer
INTENT_MODEL: str   = "gpt-4o-mini"   # /reply intent classifier

# ── Instructor ────────────────────────────────────────────────────────────────
# max_retries=1: one retry on schema validation failure, then fail fast.
# Never hang. A suppressed message costs 0. A 500 costs everything.
INSTRUCTOR_MAX_RETRIES: int = 1

# ── Budget ────────────────────────────────────────────────────────────────────
# Single asyncio.wait_for wraps the ENTIRE Layer 2 + Layer 3 pipeline per trigger.
# Judge hard limit: 30s. Network RT: ~0.5s each way. FastAPI overhead: ~0.5s.
# Budget = 30 - 0.5 - 0.5 - 2.0 (safety) = 27.0s.
PIPELINE_TIMEOUT: float = 27.0

# ── Daily message cap ─────────────────────────────────────────────────────────
# reserve_message_slot() atomically checks AND increments under asyncio.Lock.
# Layer 2 Python fast-path reads this non-atomically as early exit;
# the atomic gate is the FINAL check before returning the TickAction.
MAX_MESSAGES_PER_MERCHANT_PER_DAY: int = 3

# ── Auto-reply backoff ────────────────────────────────────────────────────────
# Count → (action, wait_seconds_or_None)
AUTO_REPLY_SCHEDULE = {
    1: ("send", None),        # 1st: send one bridging message for the owner
    2: ("wait", 24 * 3600),   # 2nd: wait 24h
    3: ("end",  None),        # 3rd+: close conversation
}

# ── Suppression TTLs (seconds) ────────────────────────────────────────────────
SUPPRESSION_WINDOWS = {
    "research":    7 * 24 * 3600,
    "recall":    180 * 24 * 3600,
    "perf":        3 * 24 * 3600,
    "milestone":  30 * 24 * 3600,
    "dormant":     3 * 24 * 3600,
    "ipl":         1 * 24 * 3600,
    "festival":    7 * 24 * 3600,
    "hostile":    30 * 24 * 3600,
    "objection":   7 * 24 * 3600,
    "default":         24 * 3600,
}
