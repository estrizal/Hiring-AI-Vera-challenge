"""
test_language_and_customer.py — Tests Hindi language support and Customer-scoped replies.
"""

import urllib.request
import urllib.error
import json
from datetime import datetime
import uuid

BOT_URL = "http://localhost:8080"

PASS = "[PASS]"
FAIL = "[FAIL]"

def _post(path, payload):
    req = urllib.request.Request(
        f"{BOT_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return json.loads(body), e.code
        except Exception:
            return {"raw": body}, e.code
    except Exception as e:
        return {"error": str(e)}, -1

def section(title):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}\n")

def check(label, condition, got=None):
    status = PASS if condition else FAIL
    print(f"  {status} {label}")
    if not condition and got is not None:
        print(f"         Got: {got}")

def push_context(scope, cid, payload):
    _post("/v1/context", {
        "scope": scope, "context_id": cid, "version": 1,
        "payload": payload,
        "delivered_at": datetime.utcnow().isoformat() + "Z",
    })

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: HINDI/HINGLISH SUPPORT
# ─────────────────────────────────────────────────────────────────────────────
def test_hindi_support():
    section("TEST 1: HINDI LANGUAGE SUPPORT")
    mid = f"m_hindi_{uuid.uuid4().hex[:6]}"
    
    # Push the category first
    push_context("category", "restaurants", {
        "slug": "restaurants", "name": "Restaurants", "voice_tone": "operator-to-operator"
    })
    
    # Push a merchant with 'hi' in languages
    push_context("merchant", mid, {
        "merchant_id": mid, "category_slug": "restaurants",
        "business_name": "Delhi Chaat", "owner_first_name": "Rahul",
        "languages": ["en", "hi"],  # This triggers is_hindi=True in extractor
        "performance": {"views": 100, "calls": 10}
    })
    
    # Push a trigger for them
    tid = f"trg_hindi_{uuid.uuid4().hex[:6]}"
    push_context("trigger", tid, {
        "id": tid, "scope": "merchant", "kind": "perf_dip",
        "merchant_id": mid, "urgency": 4,
        "payload": {"metric": "views", "delta_pct": -0.2},
        "suppression_key": f"hindi_perf_{mid}"
    })
    
    # Tick
    data, _ = _post("/v1/tick", {
        "now": datetime.utcnow().isoformat() + "Z",
        "available_triggers": [tid],
    })
    
    actions = data.get("actions", [])
    check("Trigger fired", len(actions) == 1, got=len(actions))
    
    if actions:
        body = actions[0].get("body", "").lower()
        print(f"  Body: \"{body}\"")
        
        # Check for common Hinglish words
        hindi_words = ["hai", "kya", "aap", "ko", "se", "bhi", "ki", "chahte", "karna", "ye", "dekha", "views", "kam"]
        has_hindi = any(word in body.split() or word + "," in body.split() for word in hindi_words)
        
        check("Body contains Hindi/Hinglish vocabulary", has_hindi, got="No Hinglish words detected")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: CUSTOMER-SCOPED REPLY
# ─────────────────────────────────────────────────────────────────────────────
def test_customer_scoped_reply():
    section("TEST 2: CUSTOMER-SCOPED REPLY (MERCHANT ON BEHALF)")
    mid = f"m_cust_{uuid.uuid4().hex[:6]}"
    cid = f"c_user_{uuid.uuid4().hex[:6]}"
    
    # Push merchant and customer contexts
    push_context("merchant", mid, {
        "merchant_id": mid, "category_slug": "dentists",
        "business_name": "Smile Care", "owner_first_name": "Dr. Smith",
        "languages": ["en"]
    })
    push_context("customer", cid, {
        "customer_id": cid, "identity": {"name": "Priya", "type": "lapsed"},
        "history": {"last_visit_days_ago": 180}
    })
    
    # The official judge tests this by sending a reply from the customer
    # after a recall_due trigger.
    conv_id = f"conv_cust_{uuid.uuid4().hex[:6]}"
    
    # Simulate the customer replying to a previous message Vera sent on behalf of the merchant
    res, code = _post("/v1/reply", {
        "conversation_id": conv_id, 
        "merchant_id": mid, 
        "customer_id": cid,          # Judge provides the customer_id
        "from_role": "customer",     # Judge sets role to customer
        "message": "Hi, yes I'd like to book an appointment for my cleaning next Tuesday.",
        "received_at": datetime.utcnow().isoformat() + "Z", 
        "turn_number": 2,
    })
    
    action = res.get("action", "")
    body = res.get("body", "")
    
    print(f"  Action: {action}")
    print(f"  Body: \"{body}\"")
    
    check("Action is 'send'", action == "send", got=action)
    check("Body is not empty", bool(body.strip()), got="Empty body")
    
    body_lower = body.lower()
    is_merchant_persona = "vera" not in body_lower and "ai" not in body_lower
    check("Bot speaks AS the merchant (no mention of Vera)", is_merchant_persona, got="Mentioned Vera/AI")

if __name__ == "__main__":
    test_hindi_support()
    test_customer_scoped_reply()
    print("\n" + "=" * 55)
    print("  ALL TESTS COMPLETE")
    print("=" * 55 + "\n")
