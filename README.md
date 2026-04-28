# พร้อมใจ ใกล้บ้าน — Promjai-Klaibaan MVP

**An interactive mobile recommender system for elderly users**, combining a Bi-Level
Graph Attention Network (RMSE −49.4% vs MF on Ciao benchmark) with location-based
caretaker matching.

This is the **research-ready MVP** for piloting and competition demonstration.

---

## 🎯 What's inside

| Layer | Tech | Purpose |
|---|---|---|
| **Frontend** | Single-page PWA, vanilla JS, Sarabun 19–22pt, WCAG 2.1 AA | Elderly + Caretaker UI |
| **Backend** | FastAPI + SQLite + JWT | REST API, auth, logging |
| **ML** | NumPy Bi-Level GAT (paper-faithful) | Recommendations + explanations |
| **Data** | 200 elderly, 100 activities, 600 social edges, 5K interactions, 50 caretakers | Synthetic per paper |
| **Logging** | JSONL + SQLite events | Research-grade audit trail |
| **Deploy** | Local (Python) + Docker + Render/Vercel ready | Cloud-ready |

---

## 🚀 Quick Start (Local — 3 commands)

```bash
# 1) Install deps
pip install -r backend/requirements.txt

# 2) Generate data + train model
python scripts/generate_data.py
python ml/train_numpy.py

# 3) Run server
uvicorn backend.main:app --reload --port 8000
```

Open <http://localhost:8000> in a browser.

**Demo accounts** (pre-seeded):

| Username | Password | Role |
|---|---|---|
| `elder1` | `demo1234` | ผู้สูงอายุ |
| `elder2` | `demo1234` | ผู้สูงอายุ |
| `elder3` | `demo1234` | ผู้สูงอายุ |
| `care1` | `demo1234` | ผู้ดูแล |
| `care2` | `demo1234` | ผู้ดูแล |

---

## 🐳 Docker (one-shot)

```bash
docker compose up --build
```

Hits port 8000 with the model already trained.

---

## ☁️ Cloud Deploy

### Render.com (Backend)
1. Push to GitHub, then in Render: **New → Web Service**
2. Build command: `pip install -r backend/requirements.txt && python scripts/generate_data.py && python ml/train_numpy.py`
3. Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Free tier: 750 hr/month

### Vercel (Frontend-only mode)
The PWA is a **single HTML file**. You can deploy `frontend/` separately to Vercel
and point `API` constant in `index.html` to your Render API URL.

---

## 📊 Model Performance

After training (~30 seconds):

```bash
python ml/train_numpy.py
```

The script outputs `ml/metrics.json`:

```json
{
  "MF": { "RMSE": ... },
  "GCN": { "RMSE": ... },
  "BiLevelGAT": { "RMSE": ..., "NDCG@10": ..., "HR@10": ... },
  "improvement": {
    "rmse_reduction_vs_MF_pct": ...,
    "rmse_reduction_vs_GCN_pct": ...
  }
}
```

> ⚠️ **Scope note.** The bundled `ml/train_numpy.py` is a lightweight NumPy
> implementation tuned for fast local demo. The official paper results
> (RMSE 1.126 / 1.143 with full backprop) are reproduced by `ml/train.py`
> which requires PyTorch. For the competition pitch, cite the **paper numbers**
> and use this MVP as the **interactive demo**.

---

## 🔬 Research-grade logging

Every event (login, recommendation, feedback, booking, WHOQOL) is:
- Persisted to SQLite `promjai.db` (table `events`)
- Streamed to `logs/events.jsonl` for downstream research analysis

Example:
```json
{"ts":"2026-04-27T12:00:00","user_id":1,"event_type":"recommendations_request","payload":{"top_k":8,"count":8}}
```

Use these logs for:
- Active User Rate (DAU/WAU)
- Retention (Day-7, Day-30)
- Click-through-rate per recommendation
- Pre/Post WHOQOL paired analysis

---

## 📐 Architecture

```
┌──────────────────────────────────────┐
│  Frontend PWA (Single HTML)          │
│  - Elderly view  - Caretaker view    │
│  - WCAG 2.1 AA, Sarabun 19–22pt      │
└─────────────────┬────────────────────┘
                  │ Bearer JWT
┌─────────────────▼────────────────────┐
│  FastAPI Backend                     │
│  /auth · /recommendations · /explain │
│  /caretakers/match · /bookings       │
│  /whoqol · /stats · /metrics         │
└─────────┬──────────────────┬─────────┘
          │                  │
┌─────────▼─────────┐  ┌─────▼────────┐
│ SQLite (promjai)  │  │ NumPy        │
│ + JSONL logs      │  │ Bi-Level GAT │
└───────────────────┘  └──────────────┘
```

---

## 🧪 API Reference (Quick)

```
GET  /health
GET  /metrics
GET  /stats/summary
POST /auth/login                {username, password}
GET  /me                        (Bearer)
GET  /recommendations?top_k=8   (Bearer, elderly)
GET  /explain/{activity_id}     (Bearer, elderly)
POST /feedback                  {activity_id, rating}
POST /whoqol                    {physical, psychological, social, environment}
GET  /whoqol/history
GET  /caretakers/match?top_k=5&max_km=30
POST /bookings                  {caretaker_id, service_type, scheduled_at?}
GET  /bookings
```

OpenAPI docs at <http://localhost:8000/docs>.

---

## 🧑‍🎓 For the Competition Pitch

The MVP demonstrates:

| BMC Block | MVP Feature |
|---|---|
| Value Proposition | บริการดูแลผ่านการรับรอง + AI recommend |
| Customer Segments | Elderly + Caretaker (login as both) |
| Channels | PWA (no app-store install) |
| Key Activities | จับคู่ + แนะนำ + รายงาน + ฝึกอบรม |
| Key Resources | Bi-Level GAT IP + RAG knowledge structure |
| Revenue Streams | Transaction fee per booking (logged) |

Demo flow (3 minutes):
1. Login as `elder1` → see 8 recommendations with explanations
2. Try `caretaker matching` → location-based ranking
3. Submit booking → switch to `care1` and see it appear
4. Log WHOQOL → show timeline
5. `/stats/summary` and `/metrics` endpoints for judges

---

## 📝 License

MIT (or please update before publishing).

## 🙏 Citation

If you use this code, please cite the underlying paper:

> Anonymous. "An Interactive Mobile Recommender System with Bi-Level Graph Attention
> Networks for Cognitive Skill Enhancement and Quality of Life Improvement in Elderly
> Users." iJIM (under review).
