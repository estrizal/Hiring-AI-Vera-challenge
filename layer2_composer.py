"""
layer2_composer.py — The Mouth (one structured LLM call per tick action).

The LLM is a WRITER, not a decision-maker.
  - Receives only extracted_facts (never raw JSON).
  - instructor enforces the ComposedMessage Pydantic schema.
  - temperature=0 for determinism (challenge requirement).

Full system prompts live in prompts.py (next phase).
"""

from typing import Optional

import openai
import instructor

import config
import state
from schemas import ComposedMessage

# Instructor async client (gpt-4o)
_client = instructor.from_openai(
    openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
)


async def compose(
    extracted_facts: dict,
    conversation_id: str,
    merchant_id: str,
    trigger_id: str,
    customer_id: Optional[str] = None,
    intent_context: Optional[str] = None,
    mode: str = "normal",
) -> Optional[ComposedMessage]:
    """
    Compose a WhatsApp message from extracted_facts.
    Returns ComposedMessage on success, None if all retries fail.
    instructor handles validation retries automatically.

    STUB — full prompt wired in prompts.py next phase.
    """
    try:
        import prompts
        system_prompt = prompts.build_system_prompt(extracted_facts, mode)
        user_prompt = prompts.build_user_prompt(
            extracted_facts, conversation_id, intent_context
        )
    except ImportError:
        system_prompt = "You are Vera, a merchant AI assistant. Compose a WhatsApp message."
        user_prompt = f"Facts: {extracted_facts}\nConversation ID: {conversation_id}"

    # Inject prior sent bodies for anti-repetition
    history = state.get_conversation_history(conversation_id, merchant_id)
    prior_bodies = [t["body"] for t in history if t["role"] == "vera"]
    if prior_bodies:
        user_prompt += (
            "\n\nALREADY SENT in this conversation (do NOT repeat):\n"
            + "\n".join(f"- {b[:120]}" for b in prior_bodies[-3:])
        )

    try:
        result: ComposedMessage = await _client.chat.completions.create(
            model=config.COMPOSER_MODEL,
            max_tokens=config.COMPOSER_MAX_TOKENS,
            temperature=config.COMPOSER_TEMPERATURE,
            response_model=ComposedMessage,
            max_retries=2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return result
    except Exception as e:
        print(f"[composer] ERROR for trigger={trigger_id}: {e}")
        return None
