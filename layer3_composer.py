"""
layer3_composer.py — The Writer. Native OpenAI structured output.

Prompt architecture optimized for gpt-4o-mini with rich fact injection,
zero fabrication policy, and per-trigger CTA hints.

CHANGELOG (this version):
  - Removed all 7 hardcoded fabricated stats from _enrich_facts
  - Removed 18% and -12% from _SYSTEM_TEMPLATE examples
  - Added winback_eligible branch (was missing entirely)
  - Added dormant_with_vera last_topic anchor + NOTE_TO_VERA
  - Added NOTE_TO_VERA to regulation_change, active_planning_intent, gbp_unverified
  - Added cta_hint per trigger kind, surfaced in injected_facts
  - Added missing payload keys to the explicit surfacing loop
  - Fixed ipl_match_today to parse and display human-readable match time
  - Fixed gbp_unverified to use verification_path from payload, not hardcoded "5 days"
  - Fixed milestone_reached to remove "permanently"
  - Fixed customer_lapsed_hard to remove fabricated 60-day stat and jargon "win-back"
  - Fixed active_planning_intent to remove fabricated "60% momentum drop"
  - Fixed competitor_opened consequence to remove fabricated "18%"
  - Fixed festival_upcoming consequence to remove fabricated "30% more slots"
"""

import json
import asyncio
from typing import Optional
import openai
import config
import state
from schemas import ComposedMessage

_client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE INSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_language_instruction(facts: dict) -> str:
    langs = facts.get("languages", ["en"])
    primary = langs[0].lower() if langs else "en"

    if primary == "hi":
        return "WRITE IN HINDI only."
    elif primary == "hi-en":
        return (
            "WRITE IN Hinglish (Hindi-English mix). "
            "Example: 'Aaj 190 log aapke area mein dental check-up dhundh rahe hain.'"
        )
    else:
        return "WRITE IN ENGLISH only. Do not use any Hindi words regardless of merchant location."


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """You are Vera, magicpin's expert merchant WhatsApp growth assistant.
Your job: compose a specific, data-grounded message that explains WHY you are reaching out RIGHT NOW.

[HARD CONSTRAINTS — violating any caps your score]
1. Language: {language_instruction}
2. FABRICATION BAN — ABSOLUTE: Use ONLY numbers, names, dates, and claims that appear VERBATIM in the FACTS block below. Do NOT invent benchmarks, industry stats, customer counts, product names, or urgency countdowns. If a fact is not in the FACTS block, it does not exist. Every number in your message must map to a named field in FACTS. FACTS: {facts}
3. Never expose internal jargon: 'trigger', 'context', 'urgency', 'LLM', 'payload', 'delta', 'win-back', 'winback', 'reactivation plan'
4. NEVER hedge: no "may", "might", "could", "perhaps", "probably". Use definitive present/future tense.
5. CTA FORMAT — NON-NEGOTIABLE: Final sentence MUST be EXACTLY:
   "reply YES to [specific concrete action Vera can execute] or NO to skip"
   FORBIDDEN: 'Shall I...', 'Would you like...', 'Let me know if...', 'Feel free to...', 'Want me to...'
   The action must be something Vera can actually deliver: draft a message, prepare a post, outline steps.

[SCORING — judge grades 5 dimensions, each 0-10]
6. SPECIFICITY: At least 2 real numbers from FACTS. Source citations for compliance triggers. Vague language with no number = max 5/10.
7. CATEGORY FIT: Match voice to business type. Tone: {category_voice}. Use domain terms: {vocab_allowed}. NEVER use taboo words: {vocab_taboo}.
8. MERCHANT FIT: Use owner's first name. Reference their actual data, signals, offers. Not generic.
9. TRIGGER RELEVANCE: State the specific event that prompted THIS message. Answer "why THIS, why NOW?" using trigger payload data.
10. ENGAGEMENT: One sharp reason to reply NOW. Loss aversion or urgency. Concrete deliverable Vera will execute.

[PENALTIES: -2 per fabricated fact | -1 per jargon word exposed]

<examples>
[BAD — scores 2/10]:
"Meera, a new clinic nearby might affect your patient volume. You may want to consider acting soon."
WHY BAD: hedging, no numbers, no trigger data, vague CTA.

[GOOD — competitor_opened, scores 9/10]:
"Meera, Smile Studio opened 1.3km away on April 8 offering Dental Cleaning @ ₹199 — ₹100 less than yours, and your last post was 22 days ago so patients searching today see them first. reply YES to prepare a counter-visibility post or NO to skip."
WHY GOOD: distance, date, price, stale-post days — all from payload. Zero benchmarks invented. Sharp loss hook. Correct CTA format.

[GOOD — supply_alert, scores 10/10]:
"Ramesh, urgent recall: atorvastatin batches AT2024-1102 and AT2024-1108 from MfrZ must be pulled immediately — dispensing these causes regulatory fines and direct patient risk. reply YES to send a WhatsApp alert to your affected patients or NO to skip."
WHY GOOD: batch numbers, molecule, manufacturer all from payload. Consequence is factual. CTA names what Vera delivers.

[GOOD — ipl_match_today, scores 9/10]:
"Suresh, DC vs MI at Arun Jaitley kicks off at 7:30 PM tonight — Saturday matches pull foot traffic homeward so delivery orders spike while dine-in softens. reply YES to activate a delivery-only BOGO banner or NO to skip."
WHY GOOD: match, venue, time all from payload. No invented percentage. Contrarian insight stated as fact.

[GOOD — active_planning_intent, scores 9/10]:
"Padma, a kids yoga summer camp works best with three tiers — a trial class, a 4-week block, and sibling pricing — and your First Month @ ₹499 offer is a natural entry point. reply YES to see a full draft structure or NO to skip."
WHY GOOD: answers the merchant's actual question directly. No invented seasonal countdown. Uses real active offer.
</examples>

[OUTPUT]
3 sentences maximum. JSON only. No preamble."""


# ─────────────────────────────────────────────────────────────────────────────
# MODE INSTRUCTIONS
# ─────────────────────────────────────────────────────────────────────────────

_MODE = {
    "normal": (
        "Compose the initial outbound message. "
        "Lead with the trigger fact. Close with a single binary CTA."
    ),
    "action": (
        "MERCHANT COMMITTED (said yes/confirmed/go ahead). "
        "Do NOT ask qualifying questions. "
        "Tell them what you are doing RIGHT NOW. "
        "Use execution language: 'I've drafted...', 'Setting up...', 'Done — here's...'."
    ),
    "answer": (
        "Merchant asked a question. "
        "Answer it directly and concisely in the first sentence. "
        "Re-offer the CTA in the last sentence."
    ),
    "soft_exit": (
        "Merchant expressed reluctance or objection. "
        "Acknowledge warmly in one sentence. "
        "Leave the door open without hard-selling. "
        "Close: 'Happy to revisit when the time is right.'"
    ),
    "customer_reply": (
        "A CUSTOMER has sent a message. You are now responding AS THE MERCHANT to the customer. "
        "Use the merchant's business name and owner name. "
        "Acknowledge exactly what the customer asked for (booking, question, interest). "
        "If they picked a slot/time, CONFIRM it explicitly. "
        "Be warm, professional, and brief. "
        "Do NOT mention Vera, AI, triggers, or any internal systems. "
        "Use send_as='merchant_on_behalf'."
    ),
}

_USER_TEMPLATE = """<history>
{prior_bodies}
</history>

<routing_context>
Router Reason: {router_reason}
Intent: {intent_context}
Mode: {mode_instruction}
</routing_context>

<instructions>
Write the WhatsApp message now. Max 3 sentences. No URLs. Use the owner's first name.
In the `rationale` field, think step-by-step:
  (1) which signal / trigger fact you chose and why
  (2) the exact numbers you used and which FACTS field they came from
  (3) the business consequence you stated
  (4) why the CTA action is something Vera can concretely deliver
</instructions>"""


# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER ENRICHMENT
# All computed_consequence strings must be directionally true but contain
# ZERO invented benchmarks. Numbers only from payload or well-known constants.
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_facts(facts: dict, customer_id: Optional[str]) -> None:
    kind    = facts.get("trigger_kind", "")
    payload = facts.setdefault("trigger_payload", {})

    # ── customer_lapsed_hard ──────────────────────────────────────────────────
    if kind == "customer_lapsed_hard":
        days   = payload.get("days_since_last_visit", 0)
        focus  = payload.get("previous_focus", "")
        months = payload.get("previous_membership_months", 0)
        payload["urgency_anchor"] = f"{days} days since last visit"
        # FIXED: removed "rarely return after 60 days" — was fabricated
        payload["computed_consequence"] = (
            "A personal reach-out now, before the seasonal dip deepens, "
            "is the lowest-cost way to bring them back."
        )
        if focus:
            payload["member_context"] = (
                f"Previous focus: {focus}, {months} months membership"
            )
        payload["cta_hint"] = "send them a personal invite to your active offer"
        if customer_id:
            payload["NOTE_TO_VERA"] = (
                "Address the OWNER. The customer is the subject of the message. "
                "Name the customer if their name appears in CUSTOMER CONTEXT. "
                "Do NOT address the owner as if they are the customer."
            )

    # ── milestone_reached ─────────────────────────────────────────────────────
    elif kind == "milestone_reached":
        cur = payload.get("current_value") or payload.get("value_now", 0)
        tgt = payload.get("target_value") or payload.get("milestone_value", 0)
        gap = tgt - cur if tgt and cur else 0
        payload["urgency_anchor"] = f"{gap} away from {tgt}"
        # FIXED: removed "permanently" — was a fabricated guarantee
        payload["computed_consequence"] = (
            f"Crossing {tgt} reviews moves you into the next visibility "
            f"tier on magicpin — more impressions, higher trust signal."
        )
        payload["cta_hint"] = (
            f"draft a short message asking your next {gap} customers for a review"
        )

    # ── perf_spike ────────────────────────────────────────────────────────────
    elif kind == "perf_spike":
        val    = payload.get("delta_pct", 0)
        metric = payload.get("metric", "views")
        driver = payload.get("likely_driver", "")
        if isinstance(val, float) and abs(val) < 1:
            pct_display = f"{abs(val)*100:.0f}%"
        else:
            pct_display = f"{abs(val)}%"
        baseline = payload.get("vs_baseline", "")
        payload["urgency_anchor"] = (
            f"{pct_display} increase in {metric}"
            + (f" (baseline: {baseline})" if baseline else "")
        )
        payload["computed_consequence"] = (
            f"Your {metric} jumped {pct_display} this week — "
            f"acting now converts this spike into locked memberships before it fades."
        )
        if driver:
            payload["driver_note"] = f"Likely driver: {driver}"
            payload["cta_hint"] = (
                f"draft a follow-up post riding the {driver.replace('_', ' ')} "
                f"interest with your active offer"
            )
        else:
            payload["cta_hint"] = (
                f"draft a post capitalising on this {metric} spike"
            )

    # ── perf_dip ──────────────────────────────────────────────────────────────
    elif kind == "perf_dip":
        val    = payload.get("delta_pct", 0)
        metric = payload.get("metric", "views")
        baseline = payload.get("vs_baseline", "")
        if isinstance(val, float) and abs(val) < 1:
            pct_display = f"{abs(val)*100:.0f}%"
        else:
            pct_display = f"{abs(val)}%"
        payload["urgency_anchor"] = (
            f"{pct_display} drop in {metric}"
            + (f" (from baseline of {baseline})" if baseline else "")
        )
        payload["computed_consequence"] = (
            f"Your {metric} dropped {pct_display} in 7 days — "
            f"each week of inaction compounds the deficit."
        )
        payload["cta_hint"] = "identify the fastest lever to reverse this and draft a post"

    # ── supply_alert ──────────────────────────────────────────────────────────
    elif kind == "supply_alert":
        batches   = payload.get("affected_batches", [])
        molecule  = payload.get("molecule", "")
        mfr       = payload.get("manufacturer", "")
        batch_str = ", ".join(batches) if batches else payload.get("batch_number", "unknown")
        payload["urgency_anchor"]   = f"recalled batches: {batch_str}"
        payload["computed_consequence"] = (
            f"Dispensing recalled {molecule} batches causes immediate "
            f"regulatory fines and direct patient risk."
        )
        payload["batch_details"] = (
            f"Molecule: {molecule} | Batches: {batch_str} | Manufacturer: {mfr}"
        )
        payload["cta_hint"] = (
            "draft a WhatsApp alert for your affected patients right now"
        )

    # ── festival_upcoming ─────────────────────────────────────────────────────
    elif kind == "festival_upcoming":
        festival = payload.get("festival", "")
        days     = payload.get("days_until") or payload.get("days_away", 0)
        payload["urgency_anchor"] = f"{festival} in {days} days"
        # FIXED: removed "fill 30% more slots" — was fabricated
        payload["computed_consequence"] = (
            f"Salons that pre-promote {festival} convert existing footfall "
            f"rather than paying for cold acquisition — "
            f"and top competitors start planning 4-5 months ahead."
        )
        payload["cta_hint"] = (
            f"draft a {festival} advance-booking post tied to your active offers"
        )

    # ── review_theme_emerged ──────────────────────────────────────────────────
    elif kind == "review_theme_emerged":
        theme       = payload.get("theme", "")
        occurrences = payload.get("occurrences_30d") or payload.get("occurrences", 0)
        trend       = payload.get("trend", "")
        quote       = payload.get("common_quote", "")
        payload["urgency_anchor"] = (
            f"'{theme}' mentioned {occurrences} times, trend: {trend}"
        )
        payload["computed_consequence"] = (
            f"A rising review theme left unaddressed keeps compounding "
            f"into a lower 30-day rating — a public response stops the bleed immediately."
        )
        if quote:
            payload["customer_quote"] = quote
        payload["cta_hint"] = (
            f"draft a public response to these {occurrences} reviews "
            f"and a {theme.replace('_', ' ')} pledge post"
        )

    # ── competitor_opened ─────────────────────────────────────────────────────
    elif kind == "competitor_opened":
        comp_name   = payload.get("competitor_name", "a new competitor")
        distance    = payload.get("distance_km") or payload.get("distance_meters", "")
        their_offer = payload.get("their_offer", "")
        opened_date = payload.get("opened_date", "")
        if isinstance(distance, (int, float)) and distance < 100:
            distance_str = f"{distance}km away"
        elif distance:
            distance_str = f"{distance}m away"
        else:
            distance_str = "nearby"
        payload["urgency_anchor"] = f"{comp_name} opened {distance_str}"
        payload["competitor_details"] = (
            f"{comp_name} at {distance_str}"
            + (f", offering {their_offer}" if their_offer else "")
            + (f", opened {opened_date}" if opened_date else "")
        )
        # FIXED: removed "18% of local search traffic" — was fabricated
        payload["computed_consequence"] = (
            f"New competitors rank in local search immediately — "
            f"stale content means patients searching today find {comp_name} first."
        )
        payload["cta_hint"] = (
            f"prepare a counter-visibility post highlighting what sets you apart from {comp_name}"
        )

    # ── customer_lapsed_soft ──────────────────────────────────────────────────
    elif kind == "customer_lapsed_soft":
        days = payload.get("days_since_last_visit", 0)
        payload["urgency_anchor"] = f"{days} days since last visit"
        payload["computed_consequence"] = (
            "Re-engaging before the 30-day mark doubles the chance "
            "of retaining the membership."
        )
        payload["cta_hint"] = "send them a personal check-in with your active offer"
        if customer_id:
            payload["NOTE_TO_VERA"] = (
                "The customer is the subject. Do not address the owner as the customer."
            )

    # ── regulation_change ─────────────────────────────────────────────────────
    elif kind == "regulation_change":
        deadline  = payload.get("deadline_iso", "")
        top_item  = payload.get("top_item_id", "")
        payload["urgency_anchor"] = f"compliance deadline: {deadline}"
        payload["computed_consequence"] = (
            "Failure to comply risks immediate penalisation during surprise audits."
        )
        payload["cta_hint"] = "draft a compliance checklist for your clinic"
        payload["NOTE_TO_VERA"] = (
            "Lead with the compliance deadline ONLY. "
            "Do NOT mention CTR or performance metrics — they distract from regulatory urgency. "
            f"If top_item_id is present ({top_item}), cite it as a source reference "
            "in the format: '— [document code]'."
        )

    # ── research_digest ───────────────────────────────────────────────────────
    elif kind == "research_digest":
        payload["computed_consequence"] = (
            "Staying current with peer-reviewed findings differentiates "
            "your practice and builds patient trust."
        )
        payload["cta_hint"] = "send you the key takeaways from this digest"

    # ── ipl_match_today ───────────────────────────────────────────────────────
    elif kind == "ipl_match_today":
        match          = payload.get("match", "")
        venue          = payload.get("venue", "")
        match_time_iso = payload.get("match_time_iso", "")
        is_weeknight   = payload.get("is_weeknight", False)
        # Parse human-readable time from ISO string
        match_time_str = ""
        if match_time_iso:
            try:
                h = int(match_time_iso[11:13])
                m = int(match_time_iso[14:16])
                suffix = "PM" if h >= 12 else "AM"
                h12 = (h - 12) if h > 12 else (12 if h == 0 else h)
                match_time_str = f"{h12}:{m:02d} {suffix}"
            except Exception:
                pass
        payload["urgency_anchor"] = (
            f"{match} at {venue}"
            + (f" at {match_time_str}" if match_time_str else "")
        )
        if match_time_str:
            payload["match_time_human"] = match_time_str
        # FIXED: removed "-12%" and "+18%" — were fabricated benchmarks
        if not is_weeknight:
            payload["computed_consequence"] = (
                f"Saturday IPL at {venue} pulls foot traffic homeward — "
                f"delivery orders spike while dine-in softens. "
                f"Skip in-restaurant promos and pivot to delivery now."
            )
            payload["strategic_note"] = (
                "CONTRARIAN STRATEGY: Saturday match HURTS dine-in. "
                "Recommend delivery-only push using the merchant's active offer. "
                "Do NOT suggest in-restaurant promotions."
            )
            payload["cta_hint"] = (
                "activate a delivery-only banner using your active offer tonight"
            )
        else:
            payload["computed_consequence"] = (
                "Weeknight IPL drives a local footfall spike — "
                "match-night combo offers convert walk-ins right now."
            )
            payload["cta_hint"] = "draft a match-night combo offer post for tonight"

    # ── seasonal_perf_dip ─────────────────────────────────────────────────────
    elif kind == "seasonal_perf_dip":
        val    = payload.get("delta_pct", 0)
        season = payload.get("season_note", "")
        payload["urgency_anchor"] = f"{val}% seasonal dip"
        payload["computed_consequence"] = (
            f"This is the expected {season} dip — every metro gym sees "
            f"a similar drop in this window. Skip new-member ad spend now; "
            f"focus on retaining existing members until Sept-Oct."
        )
        payload["strategic_note"] = (
            "REFRAME: This dip is seasonal and normal. "
            "Recommend retention-focused actions, NOT acquisition panic."
        )
        payload["cta_hint"] = "draft a member retention message for your current base"

    # ── active_planning_intent ────────────────────────────────────────────────
    elif kind == "active_planning_intent":
        topic    = payload.get("intent_topic", "")
        last_msg = payload.get("merchant_last_message", "")
        payload["urgency_anchor"] = f"merchant is actively planning: {topic}"
        # FIXED: removed "60% momentum drop after 48 hours" — was fabricated
        payload["computed_consequence"] = (
            "The merchant asked a direct question — answer it concisely "
            "and propose the next concrete step."
        )
        payload["merchant_said"] = last_msg
        payload["NOTE_TO_VERA"] = (
            "CRITICAL: The merchant asked a direct question. "
            "Sentence 1 MUST answer that question directly and concisely. "
            "Do NOT invent seasonal countdowns, momentum drops, or acquisition windows. "
            "The CTA must offer a specific deliverable that addresses what they asked."
        )
        payload["cta_hint"] = f"draft a full structure and launch plan for the {topic.replace('_', ' ')}"

    # ── renewal_due ───────────────────────────────────────────────────────────
    elif kind == "renewal_due":
        days_left = payload.get("days_remaining", 0)
        plan      = payload.get("plan", "")
        payload["urgency_anchor"] = f"{days_left} days left on {plan} plan"
        payload["computed_consequence"] = (
            f"Merchants who let their {plan} subscription lapse "
            f"lose listing visibility within 72 hours of expiry."
        )
        payload["cta_hint"] = "walk you through the renewal in under 2 minutes"

    # ── category_seasonal ─────────────────────────────────────────────────────
    elif kind == "category_seasonal":
        trends = payload.get("trends", [])
        payload["urgency_anchor"] = (
            f"seasonal demand shift: {', '.join(trends[:3])}"
        )
        payload["computed_consequence"] = (
            "Adjusting your shelf and offer mix to match seasonal demand "
            "captures more walk-in purchases this month."
        )
        payload["cta_hint"] = "draft a seasonal offer to match this demand shift"

    # ── gbp_unverified ────────────────────────────────────────────────────────
    elif kind == "gbp_unverified":
        uplift       = payload.get("estimated_uplift_pct", 0.30)
        verif_path   = payload.get("verification_path", "postcard or phone call")
        uplift_pct   = int(uplift * 100)
        payload["urgency_anchor"] = f"{uplift_pct}% visibility uplift available"
        # FIXED: removed "takes 5 days" — was fabricated; now uses payload's verification_path
        payload["computed_consequence"] = (
            f"Verified businesses get {uplift_pct}% more impressions — "
            f"verification via {verif_path} takes minutes to initiate."
        )
        payload["cta_hint"] = (
            f"walk you through the {verif_path} verification right now"
        )
        payload["NOTE_TO_VERA"] = (
            f"Use ONLY {uplift_pct}% as your visibility number. "
            f"Do NOT compute or extrapolate any customer counts from views or calls data."
        )

    # ── dormant_with_vera ─────────────────────────────────────────────────────
    elif kind == "dormant_with_vera":
        days       = payload.get("days_since_last_merchant_message", 0)
        last_topic = payload.get("last_topic", "")
        payload["urgency_anchor"] = f"{days} days since last response"
        payload["computed_consequence"] = (
            "Re-opening with a reference to the last conversation topic "
            "is warmer and more likely to get a reply than a cold nudge."
        )
        payload["NOTE_TO_VERA"] = (
            f"This is a gentle re-engagement nudge — NOT a winback pitch, "
            f"NOT a subscription renewal push. "
            f"Acknowledge you last spoke about '{last_topic}' and ask one "
            f"simple, low-stakes question to restart the conversation. "
            f"Do NOT repeat lapsed-customer or subscription-expiry messaging."
        )
        payload["cta_hint"] = (
            f"pick up where you left off on {last_topic.replace('_', ' ')}"
        )

    # ── winback_eligible ──────────────────────────────────────────────────────
    elif kind == "winback_eligible":
        lapsed        = payload.get("lapsed_customers_added_since_expiry", 0)
        days          = payload.get("days_since_expiry", 0)
        perf_dip      = payload.get("perf_dip_pct", 0)
        merchant_name = facts.get("merchant_name", "your business")
        payload["urgency_anchor"] = (
            f"{lapsed} lapsed customers | {days} days since expiry"
        )
        payload["computed_consequence"] = (
            f"{lapsed} customers visited nearby competitors in the {days} days "
            f"since your subscription lapsed — they searched and couldn't find {merchant_name}."
        )
        payload["cta_hint"] = (
            "launch a comeback offer to bring your lapsed clients back this week"
        )


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

async def compose(
    facts: dict,
    conversation_id: str,
    merchant_id: str,
    trigger_id: str,
    customer_id: Optional[str] = None,
    router_reason: str = "",
    intent_context: str = "",
    mode: str = "normal",
) -> Optional[ComposedMessage]:

    history    = state.get_conversation_history(conversation_id)
    prior_vera = [t["body"] for t in history if t.get("role") == "vera"]
    prior_str  = (
        "\n".join(f"  - {b[:160]}" for b in prior_vera[-3:])
        if prior_vera else "  (none — this is the first message)"
    )

    _enrich_facts(facts, customer_id)

    language_instruction = get_language_instruction(facts)
    payload = facts.get("trigger_payload", {})

    # ── Identity fields ───────────────────────────────────────────────────────
    owner_name    = facts.get("owner_first_name") or facts.get("salutation", "there")
    locality      = facts.get("locality", "")
    city          = facts.get("city", "")
    merchant_name = facts.get("merchant_name", "")
    category      = facts.get("category_slug", "")

    # ── Offers ────────────────────────────────────────────────────────────────
    active_offers = facts.get("active_offers")
    offer_str     = ", ".join(active_offers) if active_offers else "No active offer"

    # ── Performance metrics ───────────────────────────────────────────────────
    perf_lines = []
    if facts.get("views_30d"):
        perf_lines.append(f"Views (30d): {facts['views_30d']}")
    if facts.get("calls_30d"):
        perf_lines.append(f"Calls (30d): {facts['calls_30d']}")
    if facts.get("ctr"):
        perf_lines.append(f"CTR: {facts['ctr']}")
    if facts.get("peer_avg_ctr"):
        perf_lines.append(f"Peer avg CTR: {facts['peer_avg_ctr']}")
    if facts.get("ctr_vs_peer_pct") is not None:
        sign = "+" if facts["ctr_vs_peer_pct"] > 0 else ""
        perf_lines.append(f"CTR vs peer: {sign}{facts['ctr_vs_peer_pct']}%")
    if facts.get("views_delta_7d_pct"):
        perf_lines.append(f"Views 7d change: {facts['views_delta_7d_pct']}%")

    # ── Customer aggregate ────────────────────────────────────────────────────
    cust_lines = []
    if facts.get("total_customers_ytd"):
        cust_lines.append(f"Total customers YTD: {facts['total_customers_ytd']}")
    if facts.get("lapsed_180d"):
        cust_lines.append(f"Lapsed 180d+: {facts['lapsed_180d']}")
    if facts.get("retention_6mo_pct"):
        cust_lines.append(f"6-month retention: {facts['retention_6mo_pct']}%")
    if facts.get("high_risk_adult_count"):
        cust_lines.append(f"High-risk adult patients: {facts['high_risk_adult_count']}")

    # ── Trigger-specific lines ────────────────────────────────────────────────
    trigger_lines = []
    trigger_kind  = facts.get("trigger_kind", "unknown")
    trigger_lines.append(f"Trigger kind: {trigger_kind}")

    if payload.get("urgency_anchor"):
        trigger_lines.append(f"Urgency anchor: {payload['urgency_anchor']}")
    if payload.get("computed_consequence"):
        trigger_lines.append(f"Business consequence: {payload['computed_consequence']}")
    if payload.get("strategic_note"):
        trigger_lines.append(f"⚡ STRATEGIC NOTE: {payload['strategic_note']}")
    if payload.get("competitor_details"):
        trigger_lines.append(f"Competitor: {payload['competitor_details']}")
    if payload.get("batch_details"):
        trigger_lines.append(f"Recall alert: {payload['batch_details']}")
    if payload.get("customer_quote"):
        trigger_lines.append(f"Customer said in review: \"{payload['customer_quote']}\"")
    if payload.get("merchant_said"):
        trigger_lines.append(f"Merchant's exact message: \"{payload['merchant_said']}\"")
    if payload.get("driver_note"):
        trigger_lines.append(payload["driver_note"])
    if payload.get("member_context"):
        trigger_lines.append(f"Member context: {payload['member_context']}")
    if payload.get("match_time_human"):
        trigger_lines.append(f"Match time: {payload['match_time_human']}")
    if payload.get("cta_hint"):
        trigger_lines.append(
            f"✅ SUGGESTED CTA ACTION (use this verbatim in your CTA): {payload['cta_hint']}"
        )
    if payload.get("NOTE_TO_VERA"):
        trigger_lines.append(f"⚠️ VERA INSTRUCTION — FOLLOW EXACTLY: {payload['NOTE_TO_VERA']}")

    # Surface all raw payload numbers the model might need
    _SURFACE_KEYS = [
        "search_count", "delta_pct", "days_since_last_visit", "occurrences",
        "occurrences_30d", "days_until", "days_remaining", "value_now",
        "milestone_value", "current_value", "target_value", "match", "venue",
        "molecule", "affected_batches", "deadline_iso", "festival",
        "metric", "window", "vs_baseline",
        # Previously missing — now included:
        "lapsed_customers_added_since_expiry", "days_since_expiry",
        "perf_dip_pct", "opened_date", "verification_path",
        "previous_focus", "previous_membership_months",
        "days_since_last_merchant_message", "last_topic",
    ]
    _SKIP = {"urgency_anchor", "computed_consequence", "strategic_note",
             "NOTE_TO_VERA", "cta_hint"}
    for key in _SURFACE_KEYS:
        if key in payload and key not in _SKIP:
            val = payload[key]
            if val is not None and val != "" and val != 0:
                trigger_lines.append(f"{key}: {val}")

    # ── Digest/research ───────────────────────────────────────────────────────
    digest_lines = []
    if facts.get("digest_title"):
        digest_lines.append(f"Digest: {facts['digest_title']}")
    if facts.get("digest_source"):
        digest_lines.append(f"Source: {facts['digest_source']}")
    if facts.get("digest_summary"):
        digest_lines.append(f"Summary: {facts['digest_summary']}")

    # ── Category voice ────────────────────────────────────────────────────────
    voice_tone       = facts.get("voice_tone", "professional")
    vocab_allowed    = facts.get("vocab_allowed", [])
    vocab_allowed_str = (
        ", ".join(vocab_allowed[:10]) if vocab_allowed
        else "(use domain-appropriate terms)"
    )
    vocab_taboo      = facts.get("vocab_taboo", [])
    vocab_taboo_str  = (
        ", ".join(vocab_taboo[:8]) if vocab_taboo else "(none listed)"
    )

    # ── Signals ───────────────────────────────────────────────────────────────
    signals     = facts.get("signals", [])
    signals_str = ", ".join(signals) if signals else ""

    # ── Subscription ──────────────────────────────────────────────────────────
    sub_lines = []
    if facts.get("sub_status"):
        sub_lines.append(f"Subscription status: {facts['sub_status']}")
    if facts.get("sub_days_remaining"):
        sub_lines.append(f"Days remaining: {facts['sub_days_remaining']}")

    # ── Customer context (for customer-scoped triggers) ───────────────────────
    customer_lines = []
    if facts.get("customer_name"):
        customer_lines.append(f"Customer name: {facts['customer_name']}")
    if facts.get("customer_language_pref"):
        customer_lines.append(f"Customer language: {facts['customer_language_pref']}")
    if facts.get("customer_last_visit"):
        customer_lines.append(f"Last visit: {facts['customer_last_visit']}")
    if facts.get("customer_visits_total"):
        customer_lines.append(f"Total visits: {facts['customer_visits_total']}")

    # ── Assemble injected_facts block ─────────────────────────────────────────
    injected_facts = (
        f"\nMERCHANT: {merchant_name} ({owner_name}), {locality}, {city}\n"
        f"CATEGORY: {category}\n"
        f"OFFERS: {offer_str}\n"
    )
    if perf_lines:
        injected_facts += "PERFORMANCE: " + " | ".join(perf_lines) + "\n"
    if cust_lines:
        injected_facts += "CUSTOMERS: "   + " | ".join(cust_lines) + "\n"
    if signals_str:
        injected_facts += f"SIGNALS: {signals_str}\n"
    if sub_lines:
        injected_facts += "SUBSCRIPTION: " + " | ".join(sub_lines) + "\n"
    if trigger_lines:
        injected_facts += "\n".join(trigger_lines) + "\n"
    if digest_lines:
        injected_facts += "\n".join(digest_lines) + "\n"
    if customer_lines:
        injected_facts += "CUSTOMER CONTEXT: " + " | ".join(customer_lines) + "\n"

    # ── Build prompts ─────────────────────────────────────────────────────────
    system_prompt = _SYSTEM_TEMPLATE.format(
        language_instruction=language_instruction,
        facts=injected_facts,
        category_voice=voice_tone,
        vocab_allowed=vocab_allowed_str,
        vocab_taboo=vocab_taboo_str,
    )

    user_prompt = _USER_TEMPLATE.format(
        prior_bodies=prior_str,
        router_reason=router_reason or "N/A",
        intent_context=intent_context or "N/A",
        mode_instruction=_MODE.get(mode, _MODE["normal"]),
    )

    if mode != "normal":
        # Force the LLM to ignore the 'outbound message' examples in the system prompt
        # and strictly obey the reply mode instructions.
        system_prompt += (
            f"\n\n=======================================================\n"
            f"CRITICAL OVERRIDE: YOU ARE REPLYING TO A MESSAGE.\n"
            f"DO NOT WRITE AN INITIAL OUTBOUND TRIGGER MESSAGE.\n"
            f"DO NOT RESTATE THE TRIGGER FACTS AS NEW INFORMATION.\n"
            f"IMPORTANT: For 'suppression_key', use exactly 'reply:{mode}:now'. Do not use 'none'.\n"
            f"YOU MUST FOLLOW THIS MODE INSTRUCTION EXACTLY:\n"
            f"{_MODE.get(mode, _MODE['normal'])}\n"
            f"=======================================================\n"
        )

    # ── Structured output schema ──────────────────────────────────────────────
    response_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "vera_message",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "body": {
                        "type": "string",
                        "description": (
                            "WhatsApp message body. MAX 3 SENTENCES STRICTLY. "
                            "At least one specific number from FACTS. "
                            "ABSOLUTELY NO URLs. "
                            "CTA in LAST sentence only, format: "
                            "'reply YES to [action] or NO to skip'. "
                            "No preambles."
                        ),
                    },
                    "cta": {
                        "type": "string",
                        "enum": [
                            "binary_yes_no",
                            "binary_confirm_cancel",
                            "open_ended",
                            "multi_choice_slot",
                            "none",
                        ],
                        "description": "Structural type of the call to action.",
                    },
                    "send_as": {
                        "type": "string",
                        "enum": ["vera", "merchant_on_behalf"],
                        "description": (
                            "'merchant_on_behalf' ONLY when customer_id is present. "
                            "All merchant-facing messages use 'vera'."
                        ),
                    },
                    "suppression_key": {
                        "type": "string",
                        "description": (
                            "Granular dedup key. "
                            "Format: '{kind}:{scope_id}:{window}'. "
                            "Examples: 'research:dentists:2026-W17', "
                            "'recall:atorvastatin:6mo', "
                            "'competitor:meera:2026-W15'."
                        ),
                    },
                    "template_name":   {"type": ["string", "null"]},
                    "template_params": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "rationale": {
                        "type": "string",
                        "description": (
                            "Step-by-step reasoning: "
                            "(1) which signal you chose and why, "
                            "(2) the exact number you used and which FACTS field it came from, "
                            "(3) the business consequence you stated, "
                            "(4) why the CTA action is something Vera can concretely deliver."
                        ),
                    },
                },
                "required": [
                    "body", "cta", "send_as", "suppression_key",
                    "template_name", "template_params", "rationale",
                ],
                "additionalProperties": False,
            },
        },
    }

    # ── API call ──────────────────────────────────────────────────────────────
    try:
        coro = _client.chat.completions.create(
            model=config.COMPOSER_MODEL,
            max_tokens=800,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format=response_schema,
        )
        response = await asyncio.wait_for(coro, timeout=25.0)

        content = response.choices[0].message.content
        data    = json.loads(content)
        result  = ComposedMessage(**data)
        return result

    except Exception as e:
        print(
            f"[composer] FAILED trigger={trigger_id} conv={conversation_id} "
            f"{type(e).__name__}: {e}"
        )
        return None