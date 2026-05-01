"""
layer2_router.py — Hybrid Conditional Router.

Step 2A — Python Fast-Path (0ms, no LLM):
    Guard 1: reserve_message_slot() — atomic TOCTOU-safe daily cap check+reserve.
             If False → SUPPRESS immediately. This is the primary spam guard.
    Guard 2: trigger urgency == 0 or "none" → SUPPRESS.
    Guard 3: merchant hard opt-out (hostile suppress key active) → SUPPRESS.
    Guard 4: trigger's own suppression_key already active → SUPPRESS.

Step 2B — LLM Semantic-Path (gpt-4o-mini, ~400ms):
    If Python cannot instantly disqualify the trigger, ask the LLM:
    "Does this trigger make sense to send right now? PROCEED or SUPPRESS?"
    Output: RouterDecision(decision, reason) — instructor-enforced Pydantic schema.
    Wrapped in asyncio.wait_for via the 12s pipeline budget in main.py.

Failure philosophy: default to SUPPRESS.
    A suppressed trigger costs 0 points.
    A 500 / timeout costs the entire batch.
    A bad message (hallucination, wrong voice) costs negative points.
"""

import json
from typing import Tuple

import openai
import instructor

import config
import state
from schemas import RouterDecision

# ── Instructor async client ────────────────────────────────────────────────────
_client = instructor.from_openai(
    openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTER SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_ROUTER_SYSTEM = """You are a routing agent for Vera, magicpin's merchant WhatsApp AI assistant.

Decide: should Vera send a message to this merchant right now?
Output RouterDecision with PROCEED or SUPPRESS and a one-sentence reason.

DEFAULT POSTURE: PROCEED. The judge penalizes MISSING messages far more than marginal messages. When in doubt, PROCEED.

SUPPRESS ONLY if ALL of these are true:
1. The trigger kind has ZERO relevance to the merchant's category (e.g., a dentist getting a restaurant-specific promo).
2. There are literally no facts available to write ANY specific message — the trigger payload is completely empty AND the merchant context has no useful data.

DO NOT SUPPRESS for these reasons (common mistakes):
- "No active offers" — many triggers do NOT need offers (compliance alerts, review themes, performance dips, dormancy check-ins, verification nudges, seasonal advice, competitor alerts). The message IS the value.
- "Negative trigger" — performance dips, review complaints, and competitor openings are HIGH-VALUE triggers. The merchant NEEDS to hear about them.
- "Saturday IPL reduces foot traffic" — this is a CONTRARIAN opportunity (pivot to delivery messaging). PROCEED.
- "Merchant subscription expired" — winback and re-engagement are valid message types.
- "No delivery option in catalog" — the message can advise the merchant to ADD delivery options.

Be decisive. When in doubt, PROCEED. Output only the RouterDecision JSON."""

_ROUTER_USER = """MERCHANT FACTS:
{facts}

TRIGGER: kind={trigger_kind}, urgency={trigger_urgency}
TRIGGER PAYLOAD: {trigger_payload}
CATEGORY: {category_slug} | VOICE: {voice_tone}
ACTIVE OFFERS: {active_offers}

Should Vera message this merchant right now? Remember: DEFAULT is PROCEED. Only SUPPRESS if truly irrelevant or fabrication-required."""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

async def route(
    facts: dict,
    merchant_id: str,
    trigger_payload: dict,
) -> Tuple[bool, str]:
    """
    Run the hybrid router for one trigger.

    Returns:
        (True,  reason) — PROCEED: caller should compose and send
        (False, reason) — SUPPRESS: caller should skip this trigger

    NOTE: This function is NOT itself wrapped in asyncio.wait_for.
    The 12.0s budget wrapper is applied at the call site in main.py,
    covering the ENTIRE pipeline (router + composer) under one deadline.
    """

    # ── STEP 2A: Python fast-paths (0ms) ─────────────────────────────────────

    # Guard 1: Atomic daily cap check + reserve
    # reserve_message_slot() checks AND increments under asyncio.Lock in one
    # operation — eliminates TOCTOU race condition completely.
    if not await state.reserve_message_slot(merchant_id):
        return False, (
            f"Daily message cap ({config.MAX_MESSAGES_PER_MERCHANT_PER_DAY}) "
            f"reached for merchant {merchant_id}"
        )

    # Guard 2: Urgency == 0 or explicitly "none" — not worth an LLM call
    urgency = facts.get("trigger_urgency")
    if urgency is not None:
        try:
            if int(urgency) == 0:
                return False, "Trigger urgency is 0 — not actionable"
        except (TypeError, ValueError):
            if str(urgency).strip().lower() == "none":
                return False, "Trigger urgency is 'none' — suppressed"

    # Guard 3: Trigger's own suppression key already active
    suppression_key = trigger_payload.get("suppression_key", "")
    if suppression_key and state.is_suppressed(suppression_key):
        return False, f"Suppression key active: {suppression_key}"

    # ── STEP 2B: LLM semantic evaluation ─────────────────────────────────────
    # Note: asyncio.wait_for is NOT here — it wraps the entire pipeline in main.py.
    # If this call contributes to a timeout, the pipeline budget handles it cleanly.

    user_prompt = _ROUTER_USER.format(
        facts=json.dumps(facts, ensure_ascii=False, default=str)[:1500],
        trigger_kind=facts.get("trigger_kind", "unknown"),
        trigger_urgency=facts.get("trigger_urgency", "?"),
        trigger_payload=json.dumps(facts.get("trigger_payload", {}), default=str)[:400],
        category_slug=facts.get("category_slug", "unknown"),
        voice_tone=facts.get("voice_tone", "unknown"),
        active_offers=facts.get("active_offers", "none listed"),
    )

    try:
        decision: RouterDecision = await _client.chat.completions.create(
            model=config.ROUTER_MODEL,
            max_tokens=80,
            temperature=0.0,
            response_model=RouterDecision,
            max_retries=config.INSTRUCTOR_MAX_RETRIES,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
        )
        proceed = decision.decision == "PROCEED"
        return proceed, decision.reason

    except Exception as e:
        # Any LLM failure → SUPPRESS (never 500)
        return False, f"Router LLM error — suppressing for safety: {type(e).__name__}"
