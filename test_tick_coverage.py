"""
test_tick_coverage.py — Tests that /v1/tick fires actions for ALL trigger kinds.

This is the #1 remaining risk area. The official judge scored 2/6 trigger kinds
last time because the router was suppressing most of them.
"""

import urllib.request
import urllib.error
import json
from datetime import datetime

BOT_URL = "http://localhost:8080"

def _post(path, payload):
    req = urllib.request.Request(
        f"{BOT_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"HTTP {e.code} Error: {body}")
        return {"error": body}
    except Exception as e:
        print(f"Connection Error: {e}")
        return {"error": str(e)}


# ── The exact same dataset the official judge uses ──────────────────────────
CATEGORIES = {
    "dentists": {"name": "Dentists", "voice_tone": "clinical, peer-to-peer"},
    "salons": {"name": "Salons", "voice_tone": "warm, friendly"},
    "restaurants": {"name": "Restaurants", "voice_tone": "operator-to-operator"},
    "gyms": {"name": "Gyms", "voice_tone": "coaching, motivational"},
    "pharmacies": {"name": "Pharmacies", "voice_tone": "trustworthy, precise"},
}

MERCHANTS = {
    "m_001_drmeera_dentist_delhi": {
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "category_slug": "dentists",
        "business_name": "Dr. Meera's Dental Clinic",
        "owner_first_name": "Meera",
        "languages": ["en"],
    },
    "m_002_bharat_dentist_mumbai": {
        "merchant_id": "m_002_bharat_dentist_mumbai",
        "category_slug": "dentists",
        "business_name": "Bharat Dental Care",
        "owner_first_name": "Bharat",
        "languages": ["en"],
    },
    "m_005_pizzajunction_restaurant_delhi": {
        "merchant_id": "m_005_pizzajunction_restaurant_delhi",
        "category_slug": "restaurants",
        "business_name": "Pizza Junction",
        "owner_first_name": "Suresh",
        "languages": ["en"],
    },
    "m_009_apollo_pharmacy_jaipur": {
        "merchant_id": "m_009_apollo_pharmacy_jaipur",
        "category_slug": "pharmacies",
        "business_name": "Apollo Pharmacy",
        "owner_first_name": "Vikas",
        "languages": ["en"],
    },
}

CUSTOMERS = {
    "c_001_priya_for_m001": {
        "customer_id": "c_001_priya_for_m001",
        "name": "Priya Sharma",
        "lapsed_days": 15,
    },
}

# Representative triggers — one per KIND that the judge is likely to test
TRIGGERS = {
    "trg_regulation": {
        "id": "trg_regulation", "scope": "merchant", "kind": "regulation_change", "source": "external",
        "merchant_id": "m_001_drmeera_dentist_delhi", "customer_id": None,
        "payload": {"category": "dentists", "top_item_id": "d_2026W17_dci_radiograph", "deadline_iso": "2026-12-15"},
        "urgency": 4, "suppression_key": "compliance:dci_radiograph:2026", "expires_at": "2026-12-15T00:00:00Z"
    },
    "trg_recall": {
        "id": "trg_recall", "scope": "customer", "kind": "recall_due", "source": "internal",
        "merchant_id": "m_001_drmeera_dentist_delhi", "customer_id": "c_001_priya_for_m001",
        "payload": {"service_due": "6_month_cleaning", "last_service_date": "2026-05-12", "due_date": "2026-11-12",
                    "available_slots": [{"iso": "2026-11-05T18:00:00+05:30", "label": "Wed 5 Nov, 6pm"}]},
        "urgency": 3, "suppression_key": "recall:c_001:6mo", "expires_at": "2026-11-30T00:00:00Z"
    },
    "trg_perf_dip": {
        "id": "trg_perf_dip", "scope": "merchant", "kind": "perf_dip", "source": "internal",
        "merchant_id": "m_002_bharat_dentist_mumbai", "customer_id": None,
        "payload": {"metric": "calls", "delta_pct": -0.50, "window": "7d", "vs_baseline": 12},
        "urgency": 4, "suppression_key": "perf_dip:m_002:calls:2026-W17", "expires_at": "2026-05-10T00:00:00Z"
    },
    "trg_ipl": {
        "id": "trg_ipl", "scope": "merchant", "kind": "ipl_match_today", "source": "external",
        "merchant_id": "m_005_pizzajunction_restaurant_delhi", "customer_id": None,
        "payload": {"match": "DC vs MI", "venue": "Arun Jaitley Stadium", "city": "Delhi",
                    "match_time_iso": "2026-04-26T19:30:00+05:30", "is_weeknight": False},
        "urgency": 3, "suppression_key": "ipl:m_005:2026-04-26", "expires_at": "2026-04-26T23:59:59+05:30"
    },
    "trg_supply_alert": {
        "id": "trg_supply_alert", "scope": "merchant", "kind": "supply_alert", "source": "external",
        "merchant_id": "m_009_apollo_pharmacy_jaipur", "customer_id": None,
        "payload": {"alert_id": "d_2026W17_atorvastatin_recall", "molecule": "atorvastatin",
                    "affected_batches": ["AT2024-1102", "AT2024-1108"], "manufacturer": "MfrZ"},
        "urgency": 5, "suppression_key": "alert:atorvastatin:2026-04", "expires_at": "2026-05-30T00:00:00Z"
    },
    "trg_research": {
        "id": "trg_research", "scope": "merchant", "kind": "research_digest", "source": "external",
        "merchant_id": "m_001_drmeera_dentist_delhi", "customer_id": None,
        "payload": {"category": "dentists", "top_item_id": "d_2026W17_jida_fluoride"},
        "urgency": 2, "suppression_key": "research:dentists:2026-W17", "expires_at": "2026-05-03T00:00:00Z"
    },
}


def run_tick_coverage_test():
    print("=====================================================")
    print("       TICK TRIGGER COVERAGE TEST")
    print("=====================================================\n")

    # 1. Push all categories
    print("--- PUSHING CATEGORIES ---")
    for slug, cat in CATEGORIES.items():
        res = _post("/v1/context", {
            "scope": "category", "context_id": slug, "version": 1,
            "payload": cat, "delivered_at": datetime.utcnow().isoformat() + "Z"
        })
        accepted = res.get("accepted", False)
        print(f"  [{'PASS' if accepted else 'FAIL'}] category/{slug}")

    # 2. Push all merchants
    print("\n--- PUSHING MERCHANTS ---")
    for mid, m in MERCHANTS.items():
        res = _post("/v1/context", {
            "scope": "merchant", "context_id": mid, "version": 1,
            "payload": m, "delivered_at": datetime.utcnow().isoformat() + "Z"
        })
        accepted = res.get("accepted", False)
        print(f"  [{'PASS' if accepted else 'FAIL'}] merchant/{mid[:20]}")

    # 3. Push customers
    print("\n--- PUSHING CUSTOMERS ---")
    for cid, c in CUSTOMERS.items():
        res = _post("/v1/context", {
            "scope": "customer", "context_id": cid, "version": 1,
            "payload": c, "delivered_at": datetime.utcnow().isoformat() + "Z"
        })
        accepted = res.get("accepted", False)
        print(f"  [{'PASS' if accepted else 'FAIL'}] customer/{cid[:20]}")

    # 4. Push all triggers
    print("\n--- PUSHING TRIGGERS ---")
    for tid, t in TRIGGERS.items():
        res = _post("/v1/context", {
            "scope": "trigger", "context_id": tid, "version": 1,
            "payload": t, "delivered_at": datetime.utcnow().isoformat() + "Z"
        })
        accepted = res.get("accepted", False)
        print(f"  [{'PASS' if accepted else 'FAIL'}] trigger/{tid}")

    # 5. TICK — fire ALL triggers at once
    print("\n--- TICK: FIRING ALL TRIGGERS ---")
    trigger_ids = list(TRIGGERS.keys())
    res = _post("/v1/tick", {
        "now": datetime.utcnow().isoformat() + "Z",
        "available_triggers": trigger_ids
    })

    if "error" in res:
        print(f"[FAIL] Tick returned error: {res['error']}")
        return

    actions = res.get("actions", [])
    print(f"\nBot returned {len(actions)} action(s) out of {len(trigger_ids)} triggers.\n")

    # 6. Check which trigger kinds fired
    fired_kinds = set()
    expected_kinds = set()
    for tid, t in TRIGGERS.items():
        expected_kinds.add(t["kind"])

    for action in actions:
        tid = action.get("trigger_id", "")
        body = action.get("body", "")[:80]
        merchant = action.get("merchant_id", "")[:25]
        kind = TRIGGERS.get(tid, {}).get("kind", "unknown")
        fired_kinds.add(kind)

        print(f"  [FIRED] {tid}")
        print(f"          Kind: {kind}")
        print(f"          Merchant: {merchant}")
        print(f"          Body: \"{body}...\"")
        print()

    # 7. Summary
    missing_kinds = expected_kinds - fired_kinds
    print("=" * 50)
    print(f"\nTRIGGER COVERAGE: {len(fired_kinds)}/{len(expected_kinds)} kinds fired")
    print(f"Fired:   {sorted(fired_kinds)}")
    if missing_kinds:
        print(f"MISSING: {sorted(missing_kinds)}")
        print("\n[FAIL] Not all trigger kinds fired!")
    else:
        print("\n[PASS] All trigger kinds fired!")


if __name__ == "__main__":
    run_tick_coverage_test()
