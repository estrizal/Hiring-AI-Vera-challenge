"""
schemas.py — All Pydantic models for Vera (final production architecture).

Groups:
  1. LLM structured outputs   — RouterDecision, ComposedMessage
  2. FastAPI endpoint I/O     — exact shapes the judge sends and receives

Validators:
  - body: URL rejection (raises ValueError → instructor forces full rewrite, not strip)
  - body: 10-char minimum, 500-char maximum (proxy for 3-sentence limit)
  - rationale: non-empty
  - suppression_key: non-generic format enforced
"""

import re
from datetime import datetime
from typing import Literal, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

_URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# LLM STRUCTURED OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

class RouterDecision(BaseModel):
    """Layer 2 semantic-path output. gpt-4o-mini classifies PROCEED or SUPPRESS."""
    decision: Literal["PROCEED", "SUPPRESS"]
    reason: str = Field(description="One sentence explaining the routing decision.")


class ComposedMessage(BaseModel):
    """
    Layer 3 Composer output. instructor enforces this on every call.
    max_retries=1 in the create() call → fail fast, do not hang.
    """

    rationale: str = Field(
        description="CRITICAL FIRST STEP: Explain your reasoning before writing. Name the exact number you will use for Specificity. State the exact business consequence you will use for Trigger Relevance. Name the exact compulsion hook you will use for Engagement."
    )

    body: str = Field(
        description=(
            "WhatsApp message body. MAX 3 SENTENCES STRICTLY. "
            "You MUST incorporate the 'Business Consequence' and 'Number Anchor' if provided in the prompt. "
            "ABSOLUTELY NO URLs (http/https/www) — -3 judge penalty per URL. "
            "NO internal jargon ('trigger', 'context', 'urgency', 'LLM'). "
            "CTA goes in the LAST sentence only. No preambles."
        )
    )

    cta: Literal[
        "binary_yes_no",          # action triggers (perf_dip, dormant, competitor)
        "binary_confirm_cancel",  # commitment follow-through
        "open_ended",             # research_digest, curious_ask, trend
        "multi_choice_slot",      # booking flows with real slot times
        "none",                   # pure-information (regulation_change)
    ] = Field(description="The structural type of the call to action.")

    send_as: Literal["vera", "merchant_on_behalf"] = Field(
        description=(
            "'merchant_on_behalf' ONLY when customer_id is present. "
            "All merchant-facing messages use 'vera'."
        )
    )

    suppression_key: str = Field(
        description=(
            "Granular dedup key — never generic. "
            "Format: '{kind}:{scope_id}:{window}'. "
            "Examples: 'research:dentists:2026-W17', 'recall:c_001:m_001:6mo'."
        )
    )



    template_name: Optional[str] = Field(
        default=None,
        description="Pre-approved WhatsApp template name for first outbound messages.",
    )

    template_params: Optional[List[str]] = Field(
        default=None,
        description="Ordered params for {{1}}, {{2}}, ... template placeholders.",
    )

    # ── Validators ─────────────────────────────────────────────────────────

    @field_validator("body", mode="before")
    @classmethod
    def reject_urls(cls, v: str) -> str:
        """
        Hard rejection — raises ValueError so instructor forces a FULL rewrite.

        Why NOT stripping:
          Stripping 'Book here: https://link' → 'Book here: ' = broken grammar
          that tanks Engagement score regardless of the penalty.
          Raising ValueError feeds the error message back to the LLM as the
          retry prompt, producing grammatically correct copy without URLs.

        Judge penalty: -3 per URL (exceeds hallucination penalty of -2).
        """
        if _URL_PATTERN.search(v):
            raise ValueError(
                "URLs are strictly forbidden in the message body (-3 judge penalty per URL). "
                "Rewrite the ENTIRE message without any links. "
                "Reference the source or information by its name or citation instead of a URL."
            )
        if len(v.strip()) < 10:
            raise ValueError("body is too short — write a complete WhatsApp message")
        if len(v) > 500:
            raise ValueError(
                "body exceeds 500 characters. Rewrite in MAX 3 SENTENCES. "
                "Cut everything that isn't a fact or a CTA."
            )
        return v

    @field_validator("rationale", mode="after")
    @classmethod
    def rationale_not_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 10:
            raise ValueError("rationale must be a non-empty explanation of the message decision")
        return v

    @field_validator("suppression_key", mode="after")
    @classmethod
    def suppression_key_structured(cls, v: str) -> str:
        if v.lower() in ("24h", "none", "default", "merchant", "trigger", ""):
            raise ValueError(
                f"suppression_key '{v}' is too generic. "
                "Use structured format: 'research:dentists:2026-W17'"
            )
        if ":" not in v:
            raise ValueError(
                f"suppression_key '{v}' must contain ':' separators. "
                "Format: 'kind:scope:window'"
            )
        return v


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI ENDPOINT SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ContextPushRequest(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int
    payload: dict
    delivered_at: datetime


class ContextPushResponse(BaseModel):
    accepted: bool
    ack_id: Optional[str] = None
    stored_at: Optional[str] = None
    reason: Optional[str] = None
    current_version: Optional[int] = None


class TickRequest(BaseModel):
    now: datetime
    available_triggers: List[str]


class TickAction(BaseModel):
    """All fields required — missing fields = -2 judge penalty per api-call-examples.md F.2."""
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    send_as: Literal["vera", "merchant_on_behalf"]
    trigger_id: str
    template_name: Optional[str] = None
    template_params: Optional[List[str]] = None
    body: str
    cta: str
    suppression_key: str
    rationale: str


class TickResponse(BaseModel):
    actions: List[TickAction]


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    from_role: Literal["merchant", "customer"]
    message: str
    received_at: datetime
    turn_number: int


class ReplyResponse(BaseModel):
    action: Literal["send", "wait", "end"]
    body: str = ""          # NEVER null in JSON — judge calls body.lower() unconditionally
    cta: Optional[str] = None
    wait_seconds: Optional[int] = None
    rationale: str


class ContextCounts(BaseModel):
    category: int = 0
    merchant: int = 0
    customer: int = 0
    trigger: int = 0


class HealthzResponse(BaseModel):
    status: str
    uptime_seconds: int
    contexts_loaded: ContextCounts


class MetadataResponse(BaseModel):
    team_name: str
    model: str
    approach: str
    version: str
    submitted_at: str


class FootTrafficRisk(BaseModel):
    reduces_foot_traffic: bool
    reason: str
