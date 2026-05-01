"""
layer1_ranker.py — Deterministic pre-processing (Brain, not Mouth).

Three responsibilities:
  1. rank_triggers()           — pure Python priority sort, no LLM
  2. extract_facts()           — defensive key extraction into a flat dict
  3. dynamic_contrarian_check() — gpt-4o-mini YES/NO: does this reduce foot traffic?

The LLM (Layer 2) is a WRITER. It reads only extracted_facts — never raw JSON.
"""

import re
from typing import Optional, List, Tuple, Dict, Any

import openai
import instructor

import config
import state
from schemas import FootTrafficRisk

# ── Instructor async client (gpt-4o-mini, contrarian check only) ──────────────
_mini_client = instructor.from_openai(
    openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
)

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER PRIORITY MAP
# Higher number = ranked first in the tick batch.
# Source: challenge-brief.md §4.3 trigger kinds + urgency semantics.
# ─────────────────────────────────────────────────────────────────────────────

_TRIGGER_PRIORITY: Dict[str, int] = {
    # Tier 5 — Compliance / urgency (must-act)
    "supply_alert":             50,
    "regulation_change":        50,

    # Tier 4 — Loss / dip (strong loss-aversion hook)
    "perf_dip":                 40,
    "customer_lapsed_hard":     40,
    "seasonal_perf_dip":        38,

    # Tier 3 — Positive momentum / milestone
    "perf_spike":               30,
    "milestone_reached":        30,
    "customer_lapsed_soft":     28,
    "recall_due":               28,
    "appointment_tomorrow":     25,

    # Tier 2 — Category intelligence
    "research_digest":          20,
    "category_trend_movement":  20,
    "competitor_opened":        18,
    "weather_heatwave":         15,
    "ipl_match_today":          15,
    "festival_upcoming":        15,
    "local_news_event":         12,

    # Tier 1 — Engagement cadence
    "curious_ask_due":          10,
    "dormant_with_vera":        10,
    "scheduled_recurring":       8,
    "review_theme_emerged":     18,  # bump: social proof opportunity
}

_DEFAULT_PRIORITY = 5


def rank_triggers(
    available_trigger_ids: List[str],
    all_triggers: Dict[str, Any],
    all_merchants: Dict[str, Any],
) -> List[Tuple[str, dict, str, str]]:
    """
    Sort available triggers by priority. Filter out any that lack a merchant context.

    Returns a list of (trigger_id, trigger_payload, merchant_id, priority_note).
    """
    ranked = []
    for tid in available_trigger_ids:
        payload = all_triggers.get(tid)
        if not payload:
            continue

        merchant_id = payload.get("merchant_id") or payload.get("payload", {}).get("merchant_id", "")
        if not merchant_id:
            continue

        kind = payload.get("kind", "unknown")
        urgency = payload.get("urgency", 1)
        base_score = _TRIGGER_PRIORITY.get(kind, _DEFAULT_PRIORITY)
        # urgency (1-5) adds up to 5 bonus points
        final_score = base_score + urgency
        priority_note = f"kind={kind}, urgency={urgency}, score={final_score}"

        ranked.append((final_score, tid, payload, merchant_id, priority_note))

    # Sort descending by score; stable sort preserves order for ties
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [(tid, payload, mid, note) for _, tid, payload, mid, note in ranked]


# ─────────────────────────────────────────────────────────────────────────────
# SAFE FACT EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def _safe(obj: Any, *keys, default=None):
    """
    Safely traverse nested dict/list. Returns default if any key is missing or None.
    Example: _safe(merchant, "identity", "owner_first_name", default="there")
    """
    cur = obj
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return default
    return cur if cur is not None else default


def extract_facts(
    merchant: dict,
    category: dict,
    trigger: dict,
    customer: Optional[dict],
    priority_note: str = "",
) -> dict:
    """
    Extract ONLY verifiable facts from the four contexts into a flat dict.

    Rules:
      - Every value comes from the input dicts — never invented.
      - Missing keys return None (or a safe default string).
      - None-valued keys are stripped before returning so the LLM never sees "None".
      - The LLM (Layer 2) reads ONLY this dict — not the raw JSON.
    """
    identity = merchant.get("identity") or {}
    perf = merchant.get("performance") or {}
    delta = perf.get("delta_7d") or {}
    offers = merchant.get("offers") or []
    subscription = merchant.get("subscription") or {}
    cust_agg = merchant.get("customer_aggregate") or {}
    signals = merchant.get("signals") or []

    cat_voice = category.get("voice") or {}
    cat_peer = category.get("peer_stats") or {}
    cat_digest = category.get("digest") or []
    cat_seasonal = category.get("seasonal_beats") or []
    cat_trends = category.get("trend_signals") or []

    trg_payload = trigger.get("payload") or {}

    # ── Merchant identity ─────────────────────────────────────────────────────
    merchant_name = _safe(identity, "name", default="the merchant")
    owner_first = _safe(identity, "owner_first_name", default=None)
    # Salutation: use owner first name if available, else merchant short name
    salutation = owner_first or merchant_name.split()[0]

    # ── Performance ───────────────────────────────────────────────────────────
    ctr = _safe(perf, "ctr")
    peer_ctr = _safe(cat_peer, "avg_ctr")
    ctr_vs_peer_pct: Optional[float] = None
    if ctr is not None and peer_ctr is not None and peer_ctr > 0:
        ctr_vs_peer_pct = round((ctr / peer_ctr - 1) * 100, 1)

    # ── Offers ────────────────────────────────────────────────────────────────
    active_offers = [
        o.get("title") for o in offers
        if o.get("status") == "active" and o.get("title")
    ]

    # ── Category digest (top item only — no fabrication) ─────────────────────
    top_digest = cat_digest[0] if cat_digest else {}

    # ── Customer facts ────────────────────────────────────────────────────────
    cust_identity = _safe(customer, "identity") or {} if customer else {}
    cust_rel = _safe(customer, "relationship") or {} if customer else {}

    facts: dict = {
        # Merchant
        "merchant_id":           merchant.get("merchant_id"),
        "merchant_name":         merchant_name,
        "owner_first_name":      owner_first,
        "salutation":            salutation,
        "locality":              _safe(identity, "locality"),
        "city":                  _safe(identity, "city"),
        "category_slug":         merchant.get("category_slug"),
        "languages":             _safe(identity, "languages", default=["en"]),
        "is_hindi":              (_safe(identity, "languages") or ["en"])[0] == "hi",
        "verified":              _safe(identity, "verified", default=False),

        # Subscription
        "subscription_status":   _safe(subscription, "status"),
        "subscription_days":     _safe(subscription, "days_remaining"),
        "subscription_plan":     _safe(subscription, "plan"),

        # Performance
        "views_30d":             _safe(perf, "views"),
        "calls_30d":             _safe(perf, "calls"),
        "directions_30d":        _safe(perf, "directions"),
        "ctr":                   ctr,
        "peer_avg_ctr":          peer_ctr,
        "ctr_vs_peer_pct":       ctr_vs_peer_pct,   # negative = below peer
        "views_delta_7d_pct":    _safe(delta, "views_pct"),
        "calls_delta_7d_pct":    _safe(delta, "calls_pct"),

        # Offers
        "active_offers":         active_offers,

        # Customer aggregate
        "total_customers_ytd":   _safe(cust_agg, "total_unique_ytd"),
        "lapsed_180d":           _safe(cust_agg, "lapsed_180d_plus"),
        "retention_6mo_pct":     _safe(cust_agg, "retention_6mo_pct"),
        "high_risk_adult_count": _safe(cust_agg, "high_risk_adult_count"),

        # Signals
        "signals":               signals,

        # Category
        "voice_tone":            _safe(cat_voice, "tone"),
        "vocab_taboo":           _safe(cat_voice, "vocab_taboo", default=[]),
        "peer_avg_rating":       _safe(cat_peer, "avg_rating"),

        # Digest (top item only)
        "digest_title":          top_digest.get("title"),
        "digest_source":         top_digest.get("source"),
        "digest_summary":        top_digest.get("summary"),
        "digest_kind":           top_digest.get("kind"),

        # Seasonal / trends
        "seasonal_beats":        cat_seasonal[:2],   # cap at 2 to keep prompt short
        "trend_signals":         cat_trends[:2],

        # Trigger
        "trigger_id":            trigger.get("id"),
        "trigger_kind":          trigger.get("kind"),
        "trigger_source":        trigger.get("source"),
        "trigger_urgency":       trigger.get("urgency"),
        "trigger_expires_at":    trigger.get("expires_at"),
        "trigger_payload":       trg_payload,        # raw payload for LLM to mine

        # Customer (optional)
        "customer_id":           _safe(customer, "customer_id") if customer else None,
        "customer_name":         _safe(cust_identity, "name") if customer else None,
        "customer_language_pref": _safe(cust_identity, "language_pref") if customer else None,
        "customer_state":        _safe(customer, "state") if customer else None,
        "customer_last_visit":   _safe(cust_rel, "last_visit") if customer else None,
        "customer_visits_total": _safe(cust_rel, "visits_total") if customer else None,

        # Ranker metadata (for prompt context, not for LLM decisions)
        "priority_note":         priority_note,
    }

    # Strip None values so the LLM never sees "None" in the prompt
    return {k: v for k, v in facts.items() if v is not None}


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC CONTRARIAN CHECK (gpt-4o-mini, ~200ms)
# ─────────────────────────────────────────────────────────────────────────────

_HARDCODED_CONTRARIAN_NOTES = {
    # These are fast-path shortcuts to avoid an LLM call for known patterns
    ("ipl_match_today", "restaurants"): (
        "Saturday IPL matches typically reduce restaurant covers by ~12%. "
        "Skip match-night dine-in promo; pivot to delivery-only with existing active offer."
    ),
    ("ipl_match_today", "gyms"): (
        "IPL match evenings reduce gym footfall. Promote early-morning or next-day slots instead."
    ),
}


async def dynamic_contrarian_check(
    trigger: dict,
    category_slug: str,
    extracted_facts: dict,
) -> Optional[str]:
    """
    Layer 1 LLM call — gpt-4o-mini (max_tokens=10, ~200ms).

    Returns a contrarian note string if the trigger represents a foot-traffic
    reducing event, or None if the message should proceed normally.

    Hardcoded shortcuts are checked first to save latency for known patterns.
    The LLM handles novel scenarios (e.g., marathon roadblock, flood warning).
    """
    kind = trigger.get("kind", "")
    trigger_payload = trigger.get("payload", {})

    # ── Fast path: hardcoded known contrarian patterns ────────────────────────
    fast_key = (kind, category_slug)
    if fast_key in _HARDCODED_CONTRARIAN_NOTES:
        return _HARDCODED_CONTRARIAN_NOTES[fast_key]

    # ── Only run the LLM check for external events that could affect footfall ─
    external_event_kinds = {
        "ipl_match_today", "local_news_event", "weather_heatwave",
        "festival_upcoming", "competitor_opened",
    }
    if kind not in external_event_kinds:
        return None

    try:
        trigger_desc = str(trigger_payload)[:300]  # cap to keep prompt cheap
        result: FootTrafficRisk = await _mini_client.chat.completions.create(
            model=config.MINI_MODEL,
            max_tokens=80,
            temperature=config.MINI_TEMPERATURE,
            response_model=FootTrafficRisk,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a merchant-ops analyst. "
                        "Answer concisely and accurately."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Trigger kind: {kind}\n"
                        f"Category: {category_slug}\n"
                        f"Trigger details: {trigger_desc}\n\n"
                        "Does this event drastically reduce physical foot traffic "
                        "for this business category? "
                        "Answer reduces_foot_traffic: true or false, plus a one-sentence reason."
                    ),
                },
            ],
        )

        if result.reduces_foot_traffic:
            return (
                f"CONTRARIAN WARNING: {result.reason} "
                "Do NOT recommend a dine-in or walk-in promo. "
                "Pivot to delivery, online, or a future-date offer instead."
            )
        return None

    except Exception:
        # Fail open — if the contrarian check fails, proceed with normal composition
        return None
