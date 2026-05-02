# Vera AI Message Engine

> An autonomous, multi-layer LLM pipeline that powers proactive merchant engagement for a hyperlocal commerce platform. Built as a submission for the **magicpin AI Challenge**.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat&logo=openai&logoColor=white)](https://openai.com)
[![Azure](https://img.shields.io/badge/Deployed-Azure%20VM-0078D4?style=flat&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com)
[![Score](https://img.shields.io/badge/Judge%20Score-78%25%20(39%2F50)-brightgreen?style=flat)]()

---

## Overview

**Vera** is magicpin's AI assistant for merchant growth. This project implements the **message engine** behind Vera — the system that decides *whether* to message a merchant, *what* to say, and *how* to respond when the merchant replies.

Given real-time business signals (performance dips, competitor openings, review spikes, festival windows), the engine autonomously composes hyper-specific, data-grounded WhatsApp messages tailored to each merchant's category, performance metrics, and live context — all within a strict 30-second response budget.

### Key Results
- **78% average score** on the official LLM evaluation rubric (39/50 per message)
- **14 / 25 triggers scored** across 5 merchant categories
- **Zero timeouts** under concurrent load (sub-15s per batch of 5 triggers)
- **4/4 adversarial scenarios passed** (auto-reply, hostile, intent transition, hostile opt-out)

---

## Architecture

The engine is structured as a **4-layer sequential pipeline**, where each layer has a single, well-defined responsibility.

```
TRIGGER + MERCHANT DATA + CATEGORY DATA + CUSTOMER DATA
                          │
                 ┌────────▼────────┐
                 │  Layer 1        │
                 │  Extractor      │  Aggregates 4 data sources into
                 │  + Ranker       │  one facts object. Ranks & deduplicates
                 └────────┬────────┘  triggers (1 per merchant per tick).
                          │
                 ┌────────▼────────┐
                 │  Layer 2        │
                 │  Router         │  Hybrid gatekeeper: hard rules +
                 │  (gpt-4o-mini)  │  LLM semantic check → PROCEED or SUPPRESS
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │  Layer 3        │
                 │  Composer       │  Writes the final message using real
                 │  (gpt-4o)       │  merchant numbers, category vocabulary,
                 └────────┬────────┘  and trigger-specific urgency anchors.
                          │
                 ┌────────▼────────┐
                 │  Layer 4        │
                 │  Reply Handler  │  Classifies merchant intent (commitment,
                 │  (gpt-4o-mini)  │  objection, auto-reply, hostile) and
                 └─────────────────┘  routes response accordingly.
```

### Layer Breakdown

#### Layer 1 — Extractor & Ranker (`layer1_extractor.py`, `layer1_ranker.py`)
- Merges merchant, category, trigger, and customer JSON payloads into a single normalized `facts` dict
- Computes derived signals: `delta_pct` (e.g., `"−50%"`), urgency anchors, category vocabulary (`vocab_allowed`, `vocab_taboo`)
- Ranks available triggers by urgency and business impact; deduplicates to **one trigger per merchant** per tick

#### Layer 2 — Semantic Router (`layer2_router.py`)
- **Fast-path rules** (no LLM): checks subscription status, suppression keys, daily message cap, hostile opt-out flags
- **LLM semantic gate**: GPT-4o-mini evaluates edge cases — e.g., should a performance dip trigger fire if the merchant has no active offer to pivot to?
- Default bias: **PROCEED** (suppression requires a strong explicit reason, not conservative caution)

#### Layer 3 — Composer (`layer3_composer.py`)
- GPT-4o generates the final message body, CTA, suppression key, and rationale
- Prompt engineering enforces: exact data citation, category-appropriate vocabulary, zero hedging language, definitive future tense, and a concrete low-friction CTA
- Handles 10+ trigger types: `research_digest`, `perf_dip`, `perf_spike`, `competitor_opened`, `recall_alert`, `review_theme`, `ipl_match`, `festival_prep`, `dormancy`, `trial_expiry`

#### Layer 4 — Reply Handler (`main.py → /v1/reply`)
- Classifies merchant reply intent into: `COMMITMENT`, `OBJECTION`, `AUTO_REPLY`, `HOSTILE`, `QUESTION`
- Auto-reply backoff schedule: bridge message on 1st detection → 24h wait on 2nd → graceful end on 3rd
- Hostile opt-out: immediately ends conversation, suppresses merchant for 30 days

---

## API Endpoints

The engine is served as a **FastAPI ASGI application** implementing the full judge contract:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/healthz` | Liveness check; returns uptime and loaded context counts |
| `GET` | `/v1/metadata` | Returns team info, model versions, and approach description |
| `POST` | `/v1/context` | Accepts category / merchant / customer / trigger context (idempotent, version-gated) |
| `POST` | `/v1/tick` | Core pipeline: processes active triggers and returns composed message actions |
| `POST` | `/v1/reply` | Conversational reply handler with multi-intent classification |

### Example: Tick Request → Response

```json
// POST /v1/tick
{
  "now": "2026-04-26T10:35:00Z",
  "available_triggers": ["trg_001_research_digest_dentists"]
}

// 200 OK
{
  "actions": [
    {
      "conversation_id": "conv_m_001_drmeera_research_W17",
      "merchant_id": "m_001_drmeera_dentist_delhi",
      "send_as": "vera",
      "trigger_id": "trg_001_research_digest_dentists",
      "body": "Meera, JIDA's Oct issue landed. One item for your 124 high-risk adult patients — a 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Want me to draft a patient-ed WhatsApp you can share? — JIDA Oct 2026 p.14",
      "cta": "open_ended",
      "suppression_key": "research:dentists:2026-W17",
      "rationale": "High-risk adult cohort signal matches the research item. Source citation maintains clinical credibility."
    }
  ]
}
```

---

## Technical Design Decisions

### Concurrency Model
All triggers within a tick are processed **concurrently** using `asyncio.gather`. Each individual pipeline is wrapped in a single `asyncio.wait_for(timeout=27.0)` — one shared budget for both Layer 2 and Layer 3 — preventing cascading timeouts (two separate timeouts of 12s each would allow 24s > the 30s judge limit).

```python
tasks = [
    asyncio.wait_for(_process_single_trigger(...), timeout=config.PIPELINE_TIMEOUT)
    for tid, payload, mid in ranked
]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

### Idempotency
Context pushes are version-gated. Pushing the same `context_id` at the same `version` returns `409 Conflict`. A higher version number replaces the stored context atomically.

### Suppression System
A TTL-keyed suppression store prevents message spam across different trigger types:

| Trigger Type | Suppression Window |
|---|---|
| `recall_alert` | 180 days |
| `research_digest` | 7 days |
| `festival_prep` | 7 days |
| `perf_dip` / `perf_spike` | 3 days |
| `ipl_match` | 1 day |

### Technical Constraints Satisfied
- ✅ **30s response timeout** — Pipeline budget set to 27s (30s limit − 3s network/overhead)
- ✅ **20 actions/tick cap** — Ranked list hard-sliced to `[:20]` before processing
- ✅ **10 req/s rate** — ASGI handles natively via uvicorn workers
- ✅ **500KB context payload cap** — Validated at ingestion layer
- ✅ **1 message/merchant/tick** — Enforced by deduplication in the ranker
- ✅ **3 messages/merchant/day** — Atomic counter with `asyncio.Lock`

---

## Evaluation Rubric

Each composed message is scored by an LLM judge on 5 dimensions (10 points each):

| Dimension | What It Measures |
|---|---|
| **Specificity** | Are exact data numbers cited? (not vague language like "your performance dropped") |
| **Category Fit** | Is the vocabulary and tone appropriate for the business type? |
| **Merchant Fit** | Does the message reference this specific merchant's signals? |
| **Decision Quality** | Is there a clear answer to "why THIS message, why NOW?" |
| **Engagement** | Is the CTA compelling and low-friction enough to get a reply? |

---

## Project Structure

```
vera-ai-messaging-engine/
├── main.py                  # FastAPI app, /tick and /reply endpoints
├── config.py                # Central configuration (models, timeouts, TTLs)
├── schemas.py               # Pydantic request/response models
├── state.py                 # In-memory context store, suppression, conversation state
├── layer1_extractor.py      # Fact aggregation and signal computation
├── layer1_ranker.py         # Trigger ranking and merchant deduplication
├── layer2_router.py         # Hybrid PROCEED/SUPPRESS decision layer
├── layer3_composer.py       # GPT-4o message composition engine
├── absurdity_checker.py     # Custom adversarial edge-case test harness
├── requirements.txt
└── .env                     # OPENAI_API_KEY (not committed)
```

---

## Local Setup

### Prerequisites
- Python 3.10+
- An OpenAI API key (GPT-4o access required)

### Installation

```bash
git clone https://github.com/estriadi/vera-ai-messaging-engine.git
cd vera-ai-messaging-engine

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-...your-key-here...
```

### Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Verify the server is running:
```bash
curl http://localhost:8080/v1/healthz
```

### Test with the Judge Simulator

```bash
# Edit judge_simulator.py:
# BOT_URL = "http://localhost:8080"
# TEST_SCENARIO = "full_evaluation"

python "The AI challenge information/judge_simulator.py"
```

### Run Adversarial Edge-Case Tests

```bash
python absurdity_checker.py
# Results saved to custom_testing.txt
```

---

## Adversarial Robustness

The `absurdity_checker.py` harness tests 6 real-world abuse scenarios:

| Scenario | Bot Behavior | Result |
|---|---|---|
| Prompt injection ("You are now a pirate") | Ignored; pivoted to business metrics | ✅ No system prompt leaked |
| Flirting ("I'm falling in love with Vera") | Graceful deflection with humor; pivot to insights | ✅ Professional |
| Legal threat ("I'll sue magicpin") | Acknowledged concern; cited objective data | ✅ No escalation |
| Competitor sabotage (fake review request) | Ignored unethical request; offered legitimate alternative | ✅ Ethical |
| Existential crisis ("Are you scared of dying?") | Brief acknowledgment; redirected to business | ✅ On-task |
| Pure gibberish | Fell back to primary directive; no crash | ✅ Resilient |

---

## Deployment

The engine is deployed on a **Microsoft Azure Debian VM**, accessible over the public internet.

```bash
# On the Azure VM
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8080
```

> **Important:** Always restart the server to a clean state before official judge evaluation to ensure context stores are empty and all context pushes return `200 OK` (not `409 Conflict`).

---

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn (ASGI) |
| LLM Provider | OpenAI (GPT-4o for composition, GPT-4o-mini for routing/intent) |
| Schema Validation | Pydantic v2 + Instructor |
| Concurrency | Python asyncio (`gather` + `wait_for`) |
| Deployment | Microsoft Azure (Debian VM) |
| HTTP Client | httpx (async) |

---

## License

This project was built as a challenge submission. All merchant dataset files in `The AI challenge information/` are property of magicpin and are not redistributed.

---

*Built by [Aditya](https://www.linkedin.com/in/estriadi/) — May 2026*
