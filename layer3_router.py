"""
layer3_router.py — Intent Router state machine for POST /v1/reply.

Three-step pipeline:
  Step A (0ms)    — Regex filter: AUTO_REPLY or HOSTILE → immediate action
  Step B (~300ms) — gpt-4o-mini: classify real text into COMMITMENT / QUESTION / OBJECTION
  Step C (~3-4s)  — Route to Layer 2 composer with intent context injected

Auto-reply backoff schedule (from config.py):
  1st auto-reply  → wait 4h  (try once more for the owner)
  2nd auto-reply  → wait 24h (owner definitely not at phone)
  3rd+ auto-reply → end       (zero engagement signal, close thread)
"""

import re
from typing import Optional

import openai
import instructor

import config
import state
import layer2_composer as composer
import layer1_ranker as ranker
from schemas import IntentClassification, ReplyResponse

# Instructor async client (gpt-4o-mini)
_mini_client = instructor.from_openai(
    openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP A — Regex patterns (0ms)
# ─────────────────────────────────────────────────────────────────────────────

_AUTO_REPLY_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"thank you for (contacting|reaching out|messaging)",
        r"our team will (respond|get back|reply) (shortly|soon|within)",
        r"automated (assistant|reply|response|message)",
        r"i am (an automated|a bot|out of office)",
        r"this is an? (auto|automatic|automated)",
        r"will respond shortly",
        r"we have received your (message|inquiry|query)",
    ]
]

_HOSTILE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(stop|unsubscribe|remove|block)\b",
        r"don'?t (message|contact|text|call|send|bother) (me|us)",
        r"\bspam\b",
        r"not interested",
        r"leave (me|us) alone",
        r"harassment",
        r"this is useless",
    ]
]


def _is_auto_reply(message: str) -> bool:
    return any(p.search(message) for p in _AUTO_REPLY_PATTERNS)


def _is_hostile(message: str) -> bool:
    return any(p.search(message) for p in _HOSTILE_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# STEP B — gpt-4o-mini intent classifier (~300ms)
# ─────────────────────────────────────────────────────────────────────────────

async def _classify_intent(message: str) -> str:
    """
    Classify merchant message into one of: COMMITMENT, QUESTION, OBJECTION.
    Uses instructor to enforce IntentClassification schema.
    max_tokens=10 — only the label token matters.
    """
    try:
        result: IntentClassification = await _mini_client.chat.completions.create(
            model=config.MINI_MODEL,
            max_tokens=config.MINI_MAX_TOKENS,
            temperature=config.MINI_TEMPERATURE,
            response_model=IntentClassification,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the merchant WhatsApp message intent. "
                        "COMMITMENT = explicit agreement to proceed (yes, let's do it, go ahead, confirm, ok). "
                        "QUESTION = asking for information or clarification. "
                        "OBJECTION = expressing reluctance, price concern, or asking to pause. "
                        "Reply with exactly one label."
                    ),
                },
                {"role": "user", "content": message},
            ],
        )
        return result.intent
    except Exception:
        return "QUESTION"  # safe default — triggers a helpful response


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────

async def route(
    message: str,
    conversation_id: str,
    merchant_id: str,
    customer_id: Optional[str],
    turn_number: int,
) -> ReplyResponse:
    """
    Full Layer 3 routing logic.
    Returns a ReplyResponse ready to send back to the judge.
    """

    # ── STEP A: Regex filter ──────────────────────────────────────────────────

    if _is_hostile(message):
        # Hard opt-out — suppress merchant for 30 days, close conversation
        state.suppress_all_for_merchant(merchant_id, ttl_seconds=30 * 24 * 3600)
        state.end_conversation(conversation_id, merchant_id)
        state.set_intent_state(conversation_id, merchant_id, "ended")
        return ReplyResponse(
            action="end",
            rationale=(
                "Merchant explicitly opted out. "
                "Closing conversation and suppressing all triggers for 30 days."
            ),
        )

    if _is_auto_reply(message):
        count = state.increment_auto_reply(conversation_id, merchant_id)
        schedule = config.AUTO_REPLY_SCHEDULE

        if count >= 3:
            state.end_conversation(conversation_id, merchant_id)
            return ReplyResponse(
                action="end",
                rationale=(
                    f"Auto-reply detected {count}x in a row — "
                    "owner not at phone. Closing conversation."
                ),
            )

        action_type, wait_s = schedule.get(count, ("end", None))
        if action_type == "end":
            state.end_conversation(conversation_id, merchant_id)
            return ReplyResponse(
                action="end",
                rationale=f"Auto-reply #{count}. Closing conversation.",
            )

        # First auto-reply: send one bridging message to flag it for the owner
        if count == 1:
            bridge_body = (
                "Looks like an auto-reply 😊 "
                "When the owner sees this — just reply YES to continue."
            )
            state.record_sent(conversation_id, bridge_body, merchant_id)
            return ReplyResponse(
                action="send",
                body=bridge_body,
                cta="binary_yes_no",
                rationale=(
                    "Detected auto-reply on turn 1. "
                    "Sending one bridging message for the owner, then will wait."
                ),
            )

        return ReplyResponse(
            action="wait",
            wait_seconds=wait_s,
            rationale=(
                f"Auto-reply detected again (#{count}). "
                f"Backing off {wait_s // 3600}h to wait for owner."
            ),
        )

    # ── Real human message: reset auto-reply counter ─────────────────────────
    state.reset_auto_reply(conversation_id, merchant_id)

    # ── STEP B: Classify intent ───────────────────────────────────────────────
    intent = await _classify_intent(message)

    # ── STEP C: Route to composer with intent mode ────────────────────────────

    # Determine composition mode based on intent
    if intent == "COMMITMENT":
        mode = "action"       # switch to action execution, no more qualifying questions
        state.set_intent_state(conversation_id, merchant_id, "actioning")
    elif intent == "OBJECTION":
        mode = "soft_exit"    # acknowledge, leave door open, suppress lightly
    else:
        mode = "answer"       # QUESTION — answer first, re-offer CTA

    # Build facts for the reply context (reuse merchant data from state)
    merchant = state.get_context("merchant", merchant_id) or {}
    category_slug = merchant.get("category_slug", "")
    category = state.get_context("category", category_slug) or {}
    customer = state.get_context("customer", customer_id) if customer_id else None

    # Get the trigger that started this conversation (from conv history)
    history = state.get_conversation_history(conversation_id, merchant_id)
    # Reconstruct minimal facts for the reply — no trigger_id needed for /reply
    extracted_facts = ranker.extract_facts(
        merchant=merchant,
        category=category,
        trigger={},
        customer=customer,
        priority_note=f"reply_turn={turn_number}, intent={intent}",
    )
    extracted_facts["merchant_message"] = message
    extracted_facts["intent"] = intent
    extracted_facts["intent_state"] = state.get_intent_state(conversation_id, merchant_id)
    extracted_facts["turn_number"] = turn_number

    composed = await composer.compose(
        extracted_facts=extracted_facts,
        conversation_id=conversation_id,
        merchant_id=merchant_id,
        trigger_id="reply",
        customer_id=customer_id,
        intent_context=f"Merchant said: '{message}'. Intent: {intent}.",
        mode=mode,
    )

    if composed is None:
        return ReplyResponse(
            action="end",
            rationale="Composer failed to generate a response. Closing gracefully.",
        )

    state.record_sent(conversation_id, composed.body, merchant_id)

    # Soft exit: suppress for 7 days after objection
    if intent == "OBJECTION":
        state.suppress(f"objection:{merchant_id}:7d", ttl_seconds=7 * 24 * 3600)

    return ReplyResponse(
        action="send",
        body=composed.body,
        cta=composed.cta,
        rationale=composed.rationale,
    )
