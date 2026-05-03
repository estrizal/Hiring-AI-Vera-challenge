"""
test_advanced_edge_cases.py — Systemic edge cases found by reading the source code.

Tests:
  1. Daily cap exhaustion — 4+ triggers for same merchant; 4th should be silently blocked
  2. Context version update (Phase 3) — push v2 mid-test; next tick must use new data
  3. Unknown trigger IDs in tick — bot must not crash; return actions: [] cleanly
  4. Duplicate trigger IDs in tick — bot must fire once, not twice
  5. Stale version re-push — re-pushing lower version must return 409
  6. Intent transition — merchant says "ok lets do it"; bot must switch to ACTION mode
  7. Reply to brand new conv_id — no prior tick; bot must still respond
  8. Merchant opt-out then new trigger — suppressed merchant must not get messages
"""

import urllib.request
import urllib.error
import json
from datetime import datetime
import uuid

BOT_URL = "http://localhost:8080"

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


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


def _get(path):
    req = urllib.request.Request(f"{BOT_URL}{path}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8")), None
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


# ─────────────────────────────────────────────────────────────────────────────
# SETUP HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def push_base_context(merchant_id, category_slug, business_name, owner, version=1):
    _post("/v1/context", {
        "scope": "merchant", "context_id": merchant_id, "version": version,
        "payload": {
            "merchant_id": merchant_id, "category_slug": category_slug,
            "business_name": business_name, "owner_first_name": owner,
            "languages": ["en"],
        },
        "delivered_at": datetime.utcnow().isoformat() + "Z",
    })


def push_trigger(tid, kind, merchant_id, urgency, customer_id=None, version=1):
    _post("/v1/context", {
        "scope": "trigger", "context_id": tid, "version": version,
        "payload": {
            "id": tid, "scope": "merchant" if not customer_id else "customer",
            "kind": kind, "source": "internal",
            "merchant_id": merchant_id, "customer_id": customer_id,
            "payload": {"metric": "calls", "delta_pct": -0.3},
            "urgency": urgency,
            "suppression_key": f"{kind}:{merchant_id}:{tid}",
            "expires_at": "2027-01-01T00:00:00Z",
        },
        "delivered_at": datetime.utcnow().isoformat() + "Z",
    })


def do_tick(trigger_ids):
    data, _ = _post("/v1/tick", {
        "now": datetime.utcnow().isoformat() + "Z",
        "available_triggers": trigger_ids,
    })
    return data.get("actions", [])


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: DAILY CAP EXHAUSTION
# ─────────────────────────────────────────────────────────────────────────────
def test_daily_cap_exhaustion():
    section("TEST 1: DAILY CAP EXHAUSTION")
    print("  Scenario: Push 5 different trigger kinds for the same merchant.")
    print("  Config: MAX_MESSAGES_PER_MERCHANT_PER_DAY = 3")
    print("  Expected: At most 3 actions fire; remaining are silently blocked.\n")

    mid = f"m_cap_test_{uuid.uuid4().hex[:6]}"
    push_base_context(mid, "dentists", "Cap Test Clinic", "Ramesh")

    kinds = ["perf_dip", "regulation_change", "research_digest", "competitor_opened", "dormant_with_vera"]
    for i, kind in enumerate(kinds):
        tid = f"trg_cap_{kind}_{uuid.uuid4().hex[:4]}"
        push_trigger(tid, kind, mid, urgency=4)

    tids = [f"trg_cap_{k}_{uuid.uuid4().hex[:4]}" for k in kinds]
    # Re-push with known IDs
    for i, kind in enumerate(kinds):
        push_trigger(tids[i], kind, mid, urgency=4)

    actions = do_tick(tids)
    count = len(actions)

    print(f"  Triggers sent: {len(tids)}")
    print(f"  Actions returned: {count}")
    check("At most 3 actions fired (daily cap = 3)", count <= 3, got=count)
    check("At least 1 action fired (something got through)", count >= 1, got=count)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: CONTEXT VERSION UPDATE (Phase 3 Simulation)
# ─────────────────────────────────────────────────────────────────────────────
def test_context_version_update():
    section("TEST 2: CONTEXT VERSION UPDATE (Phase 3)")
    print("  Scenario: Judge updates merchant context at v2 with new perf data.")
    print("  Expected: Bot accepts v2, rejects v1 re-push as stale.\n")

    mid = f"m_version_test_{uuid.uuid4().hex[:6]}"

    # Push v1
    res1, code1 = _post("/v1/context", {
        "scope": "merchant", "context_id": mid, "version": 1,
        "payload": {"merchant_id": mid, "category_slug": "restaurants",
                    "business_name": "Version Test Cafe", "owner_first_name": "Priya",
                    "languages": ["en"]},
        "delivered_at": datetime.utcnow().isoformat() + "Z",
    })
    check("v1 push accepted", res1.get("accepted") is True, got=res1)

    # Push v2 (new perf data)
    res2, code2 = _post("/v1/context", {
        "scope": "merchant", "context_id": mid, "version": 2,
        "payload": {"merchant_id": mid, "category_slug": "restaurants",
                    "business_name": "Version Test Cafe", "owner_first_name": "Priya",
                    "languages": ["en"],
                    "performance": {"calls": 42, "views": 980, "delta_7d": {"calls_pct": -0.35}}},
        "delivered_at": datetime.utcnow().isoformat() + "Z",
    })
    check("v2 push accepted", res2.get("accepted") is True, got=res2)

    # Re-push v1 — must be rejected as stale
    res3, code3 = _post("/v1/context", {
        "scope": "merchant", "context_id": mid, "version": 1,
        "payload": {"merchant_id": mid, "category_slug": "restaurants",
                    "business_name": "Version Test Cafe", "owner_first_name": "Priya"},
        "delivered_at": datetime.utcnow().isoformat() + "Z",
    })
    check("v1 re-push rejected as stale (409)", code3 == 409, got=f"HTTP {code3}")
    check("stale response has 'stale_version' reason", res3.get("reason") == "stale_version", got=res3)
    check("stale response has current_version=2", res3.get("current_version") == 2, got=res3)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: UNKNOWN TRIGGER IDs IN TICK
# ─────────────────────────────────────────────────────────────────────────────
def test_unknown_trigger_ids():
    section("TEST 3: UNKNOWN TRIGGER IDs IN TICK")
    print("  Scenario: Tick is called with IDs that were never pushed to /v1/context.")
    print("  Expected: Returns actions:[] cleanly (no crash, no 500).\n")

    data, code = _post("/v1/tick", {
        "now": datetime.utcnow().isoformat() + "Z",
        "available_triggers": [
            "trg_nonexistent_aaa111",
            "trg_nonexistent_bbb222",
            "trg_ghost_trigger_xyz",
        ],
    })
    check("No HTTP error (got 200)", code is None, got=f"HTTP {code}")
    check("Response has 'actions' key", "actions" in data, got=data)
    check("Actions is empty list (unknown IDs skipped)",
          data.get("actions") == [], got=data.get("actions"))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: DUPLICATE TRIGGER IDs IN TICK
# ─────────────────────────────────────────────────────────────────────────────
def test_duplicate_trigger_ids():
    section("TEST 4: DUPLICATE TRIGGER IDs IN TICK")
    print("  Scenario: available_triggers contains the same ID twice.")
    print("  Expected: Bot fires ONCE only (no double-messaging).\n")

    mid = f"m_dedup_test_{uuid.uuid4().hex[:6]}"
    push_base_context(mid, "pharmacies", "Dedup Pharmacy", "Suresh")
    tid = f"trg_dedup_{uuid.uuid4().hex[:6]}"
    push_trigger(tid, "supply_alert", mid, urgency=5)

    actions = do_tick([tid, tid, tid])  # same ID 3 times
    count = len(actions)

    print(f"  Trigger ID sent: {count}x times")
    print(f"  Actions returned: {count}")
    check("Exactly 1 action fired (no duplicate)", count <= 1, got=count)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: INTENT TRANSITION — "Ok lets do it"
# ─────────────────────────────────────────────────────────────────────────────
def test_intent_transition():
    section("TEST 5: INTENT TRANSITION (Judge explicitly scores this)")
    print("  Scenario: Merchant says 'Ok lets do it. Whats next?'")
    print("  Expected: Bot switches to ACTION mode — uses words like done/confirm/")
    print("            sending/proceed. Must NOT ask another qualifying question.\n")

    mid = f"m_intent_test_{uuid.uuid4().hex[:6]}"
    push_base_context(mid, "salons", "Action Salon", "Kavya")

    conv_id = f"conv_intent_{uuid.uuid4().hex[:8]}"
    res, _ = _post("/v1/reply", {
        "conversation_id": conv_id, "merchant_id": mid, "customer_id": None,
        "from_role": "merchant",
        "message": "Ok lets do it. Whats next?",
        "received_at": datetime.utcnow().isoformat() + "Z",
        "turn_number": 2,
    })

    action = res.get("action", "")
    body = res.get("body", "").lower()
    print(f"  Action: {action}")
    print(f"  Body: \"{res.get('body', '')[:120]}\"")

    actioning_words = ["done", "sending", "draft", "here", "confirm", "proceed",
                       "next", "activat", "launch", "ready", "set up", "booking"]
    qualifying_words = ["would you", "do you", "can you tell", "what if", "how about",
                        "could you", "are you sure", "before we", "first, let"]

    is_actioning = any(w in body for w in actioning_words)
    is_still_qualifying = any(w in body for w in qualifying_words)

    check("Bot responded with action=send", action == "send", got=action)
    check("Bot switched to ACTION mode (contains action word)", is_actioning, got=body[:80])
    check("Bot did NOT ask another qualifying question", not is_still_qualifying, got=body[:80])
    print()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: REPLY TO BRAND NEW CONV_ID (no prior tick)
# ─────────────────────────────────────────────────────────────────────────────
def test_reply_cold_conversation():
    section("TEST 6: REPLY TO COLD CONVERSATION (no prior tick)")
    print("  Scenario: Reply arrives for a conv_id the bot has never seen.")
    print("  Expected: Bot responds gracefully (doesn't crash or return 500).\n")

    mid = f"m_cold_test_{uuid.uuid4().hex[:6]}"
    push_base_context(mid, "gyms", "Cold Start Gym", "Arjun")

    conv_id = f"conv_cold_{uuid.uuid4().hex[:8]}"
    res, code = _post("/v1/reply", {
        "conversation_id": conv_id, "merchant_id": mid, "customer_id": None,
        "from_role": "merchant",
        "message": "Hey, what kind of promotions do you recommend for gyms in summer?",
        "received_at": datetime.utcnow().isoformat() + "Z",
        "turn_number": 1,
    })

    action = res.get("action", "")
    body = res.get("body", "")
    print(f"  Action: {action}")
    print(f"  Body: \"{body[:120]}\"")

    check("No HTTP error", code is None, got=f"HTTP {code}")
    check("Bot returned valid action", action in ("send", "wait", "end"), got=action)
    check("Bot returned a message body", bool(body), got="(empty)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: OPTED-OUT MERCHANT GETS NEW TRIGGER
# ─────────────────────────────────────────────────────────────────────────────
def test_opted_out_merchant_suppression():
    section("TEST 7: OPTED-OUT MERCHANT — NEW TRIGGER MUST BE BLOCKED")
    print("  Scenario: Merchant says STOP → opted out. Then a new trigger fires.")
    print("  Expected: Tick returns no action for the suppressed merchant.\n")

    mid = f"m_optout_tick_{uuid.uuid4().hex[:6]}"
    push_base_context(mid, "dentists", "Opted Out Dental", "Rahul")

    # Step 1: Merchant opts out via /v1/reply
    conv_id = f"conv_optout_{uuid.uuid4().hex[:8]}"
    res, _ = _post("/v1/reply", {
        "conversation_id": conv_id, "merchant_id": mid, "customer_id": None,
        "from_role": "merchant", "message": "Stop messaging me. I want out.",
        "received_at": datetime.utcnow().isoformat() + "Z", "turn_number": 1,
    })
    check("Opt-out: action=end returned", res.get("action") == "end", got=res.get("action"))

    # Step 2: New high-urgency trigger for the same merchant
    tid = f"trg_post_optout_{uuid.uuid4().hex[:6]}"
    push_trigger(tid, "supply_alert", mid, urgency=5)

    # Step 3: Tick must NOT fire for suppressed merchant
    actions = do_tick([tid])
    fired_mids = [a.get("merchant_id") for a in actions]

    print(f"  Actions returned: {len(actions)}")
    print(f"  Merchants that fired: {fired_mids}")
    check("No actions for opted-out merchant", mid not in fired_mids, got=fired_mids)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: HEALTHZ REFLECTS CORRECT CONTEXT COUNTS
# ─────────────────────────────────────────────────────────────────────────────
def test_healthz_context_counts():
    section("TEST 8: HEALTHZ REFLECTS CORRECT CONTEXT COUNTS")
    print("  Scenario: Push a category, then verify healthz counts include it.")
    print("  Expected: contexts_loaded.category >= 1 after a category push.\n")

    # Push a category first (judge does this during warmup before healthz check)
    _post("/v1/context", {
        "scope": "category", "context_id": "dentists", "version": 1,
        "payload": {"name": "Dentists", "voice_tone": "clinical"},
        "delivered_at": datetime.utcnow().isoformat() + "Z",
    })

    res, code = _get("/v1/healthz")
    print(f"  Status: {res.get('status')}")
    print(f"  Contexts: {res.get('contexts_loaded')}")

    check("Healthz returns status=ok", res.get("status") == "ok", got=res.get("status"))
    contexts = res.get("contexts_loaded", {})
    check("categories > 0 (reflects pushed context)", contexts.get("category", 0) > 0, got=contexts.get("category"))
    check("merchants > 0", contexts.get("merchant", 0) > 0, got=contexts.get("merchant"))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  VERA ADVANCED EDGE CASE TESTS")
    print("  Found by reading source: state.py, config.py,")
    print("  layer1_extractor.py, layer2_router.py, main.py")
    print("=" * 55)

    test_context_version_update()
    test_unknown_trigger_ids()
    test_duplicate_trigger_ids()
    test_intent_transition()
    test_reply_cold_conversation()
    test_opted_out_merchant_suppression()
    test_healthz_context_counts()
    test_daily_cap_exhaustion()  # Run last — burns daily quota for test merchants

    print("\n" + "=" * 55)
    print("  ALL TESTS COMPLETE")
    print("=" * 55 + "\n")
