"""
layer1_extractor.py — Deterministic Python pre-processing. Zero LLM calls.

Responsibilities:
  1. rank_triggers()   — sort by (base_priority + urgency) score, descending
  2. seen_merchants deduplication — ONE trigger per merchant per tick
                                    (primary fix for the Spam Cannon problem)
  3. extract_facts()   — defensive flat dict; LLM reads ONLY this, never raw JSON

The per-merchant dedup is the PRIMARY guard against the asyncio.gather
"Spam Cannon": since only one coroutine per merchant enters the pipeline per
/tick call, there is no within-tick TOCTOU race for the same merchant.
reserve_message_slot() in the router handles cross-tick races atomically.
"""

from typing import Optional, List, Tuple, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER PRIORITY TABLE
# Higher score = ranked first. Urgency (1-5) adds on top.
# ─────────────────────────────────────────────────────────────────────────────

_PRIORITY: Dict[str, int] = {
    # Tier 5 — Compliance / must-act
    "supply_alert": 50,
    "regulation_change": 50,
    # Tier 4 — Loss / dip signals
    "perf_dip": 40,
    "customer_lapsed_hard": 40,
    "seasonal_perf_dip": 38,
    # Tier 3 — Positive momentum
    "perf_spike": 30,
    "milestone_reached": 30,
    "customer_lapsed_soft": 28,
    "recall_due": 28,
    "appointment_tomorrow": 25,
    # Tier 2 — Category intelligence
    "research_digest": 20,
    "category_trend_movement": 20,
    "competitor_opened": 18,
    "review_theme_emerged": 18,
    "weather_heatwave": 15,
    "ipl_match_today": 15,
    "festival_upcoming": 15,
    "local_news_event": 12,
    # Tier 1 — Cadence / engagement
    "curious_ask_due": 10,
    "dormant_with_vera": 10,
    "scheduled_recurring": 8,
}
_DEFAULT_PRIORITY = 5


def rank_triggers(
    available_trigger_ids: List[str],
    all_triggers: Dict[str, Any],
    all_merchants: Dict[str, Any],
) -> List[Tuple[str, dict, str]]:
    """
    Sort available triggers by priority score, then deduplicate per merchant.

    DEDUPLICATION RULE (fixes the Spam Cannon):
        After sorting, iterate once through the ranked list and keep only the
        FIRST (highest-priority) trigger seen for each merchant_id.
        Lower-priority triggers for the same merchant are dropped for this tick
        — they remain in the trigger store and will be eligible next tick.

    Returns: [(trigger_id, trigger_payload, merchant_id), ...]
             At most ONE entry per merchant_id.
    """
    # ── Score every trigger ───────────────────────────────────────────────────
    scored: List[Tuple[int, str, dict, str]] = []
    for tid in available_trigger_ids:
        payload = all_triggers.get(tid)
        if not payload:
            continue
        merchant_id = (
            payload.get("merchant_id")
            or payload.get("payload", {}).get("merchant_id", "")
        )
        if not merchant_id or merchant_id not in all_merchants:
            continue
        kind = payload.get("kind", "unknown")
        urgency = _safe_int(payload.get("urgency"), default=1)
        score = _PRIORITY.get(kind, _DEFAULT_PRIORITY) + urgency
        scored.append((score, tid, payload, merchant_id))

    # Sort descending: highest-priority trigger first
    scored.sort(key=lambda x: x[0], reverse=True)

    # ── Per-merchant deduplication ────────────────────────────────────────────
    seen_merchants: set = set()
    result: List[Tuple[str, dict, str]] = []
    for _, tid, payload, mid in scored:
        if mid not in seen_merchants:
            result.append((tid, payload, mid))
            seen_merchants.add(mid)
        # else: lower-priority duplicate for same merchant — skip this tick

    return result


def _safe_int(value: Any, default: int = 1) -> int:
    """Safely convert urgency to int; return default on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# SAFE NESTED GETTER
# ─────────────────────────────────────────────────────────────────────────────

def _safe(obj: Any, *keys, default=None) -> Any:
    """
    Safely traverse nested dicts with any chain of keys.
    Returns default if any key is missing, None, or obj is not a dict.

    Example: _safe(merchant, "identity", "owner_first_name", default="there")
    Never raises. Never returns None unless that's the default.
    """
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur if cur is not None else default


# ─────────────────────────────────────────────────────────────────────────────
# FACT EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_facts(
    merchant: dict,
    category: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    turn_note: str = "",
) -> dict:
    """
    Extract ONLY verifiable facts into a flat dict.

    Rules:
      - Every value originates from the input dicts — never invented.
      - None values stripped before return (LLM never sees bare "None").
      - The LLM (Layer 3) reads ONLY this dict, never raw JSON.
      - Compute derived values (ctr_vs_peer_pct, salutation) safely.
    """
    identity = merchant.get("identity") or {}
    perf     = merchant.get("performance") or {}
    delta    = perf.get("delta_7d") or {}
    offers   = merchant.get("offers") or []
    sub      = merchant.get("subscription") or {}
    agg      = merchant.get("customer_aggregate") or {}
    signals  = merchant.get("signals") or []

    voice   = category.get("voice") or {}
    peer    = category.get("peer_stats") or {}
    digests = category.get("digest") or []
    seasonal = category.get("seasonal_beats") or []
    trends  = category.get("trend_signals") or []

    trg_inner = trigger.get("payload") or {}

    # ── Derived: CTR vs peer ─────────────────────────────────────────────────
    ctr = _safe(perf, "ctr")
    peer_ctr = _safe(peer, "avg_ctr")
    ctr_vs_peer_pct: Optional[float] = None
    if ctr is not None and peer_ctr and peer_ctr > 0:
        ctr_vs_peer_pct = round((ctr / peer_ctr - 1) * 100, 1)

    # ── Derived: salutation ──────────────────────────────────────────────────
    owner_first  = _safe(identity, "owner_first_name")
    merchant_name = _safe(identity, "name", default="the merchant")
    salutation   = owner_first or merchant_name.split()[0]

    # ── Active offers only ───────────────────────────────────────────────────
    active_offers = [
        o.get("title") for o in offers
        if o.get("status") == "active" and o.get("title")
    ] or None

    # ── Top digest item only (no multi-item hallucination risk) ─────────────
    top_digest = digests[0] if digests else {}

    # ── Customer fields ──────────────────────────────────────────────────────
    c = customer or {}
    ci = _safe(c, "identity") or {}
    cr = _safe(c, "relationship") or {}

    raw: dict = {
        # Merchant identity
        "merchant_id":           merchant.get("merchant_id"),
        "merchant_name":         merchant_name,
        "owner_first_name":      owner_first,
        "salutation":            salutation,
        "locality":              _safe(identity, "locality"),
        "city":                  _safe(identity, "city"),
        "category_slug":         merchant.get("category_slug"),
        "languages":             _safe(identity, "languages", default=["en"]),
        "is_hindi":              "hi" in (_safe(identity, "languages") or []),
        "verified":              _safe(identity, "verified"),

        # Subscription
        "sub_status":            _safe(sub, "status"),
        "sub_days_remaining":    _safe(sub, "days_remaining"),
        "sub_plan":              _safe(sub, "plan"),

        # Performance
        "views_30d":             _safe(perf, "views"),
        "calls_30d":             _safe(perf, "calls"),
        "directions_30d":        _safe(perf, "directions"),
        "ctr":                   ctr,
        "peer_avg_ctr":          peer_ctr,
        "ctr_vs_peer_pct":       ctr_vs_peer_pct,
        "views_delta_7d_pct":    _safe(delta, "views_pct"),
        "calls_delta_7d_pct":    _safe(delta, "calls_pct"),

        # Offers
        "active_offers":         active_offers,

        # Customer aggregate
        "total_customers_ytd":   _safe(agg, "total_unique_ytd"),
        "lapsed_180d":           _safe(agg, "lapsed_180d_plus"),
        "retention_6mo_pct":     _safe(agg, "retention_6mo_pct"),
        "high_risk_adult_count": _safe(agg, "high_risk_adult_count"),

        # Signals list
        "signals":               signals or None,

        # Category voice
        "voice_tone":            _safe(voice, "tone"),
        "vocab_allowed":         _safe(voice, "vocab_allowed"),
        "vocab_taboo":           _safe(voice, "vocab_taboo"),
        "peer_avg_rating":       _safe(peer, "avg_rating"),

        # Digest (top item only)
        "digest_title":          top_digest.get("title"),
        "digest_source":         top_digest.get("source"),
        "digest_summary":        top_digest.get("summary"),
        "digest_kind":           top_digest.get("kind"),

        # Seasonal / trends (capped to keep prompts short)
        "seasonal_beats":        seasonal[:2] or None,
        "trend_signals":         trends[:2] or None,

        # Trigger
        "trigger_id":            trigger.get("id"),
        "trigger_kind":          trigger.get("kind"),
        "trigger_source":        trigger.get("source"),
        "trigger_urgency":       trigger.get("urgency"),
        "trigger_expires_at":    trigger.get("expires_at"),
        "trigger_payload":       trg_inner or None,

        # Customer (optional — only present for customer-scoped triggers)
        "customer_id":           _safe(c, "customer_id") if customer else None,
        "customer_name":         ci.get("name") if customer else None,
        "customer_language_pref": ci.get("language_pref") if customer else None,
        "customer_state":        _safe(c, "state") if customer else None,
        "customer_last_visit":   cr.get("last_visit") if customer else None,
        "customer_visits_total": cr.get("visits_total") if customer else None,

        # Metadata
        "turn_note":             turn_note or None,
    }

    # Strip None values — LLM prompt stays clean, no "None" strings
    return {k: v for k, v in raw.items() if v is not None}
