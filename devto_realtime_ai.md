---
title: "Redis Neural Lattice: Real‑Time Consciousness Observatory (AI Innovators)"
published: false
tags: redis, ai, vectors, streams, realtime
series: Redis AI Challenge 2025
---

*This is a submission for the [Redis AI Challenge](https://dev.to/challenges/redis-2025-07-23): **Real‑Time AI Innovators***

# 🚀 What I Built

**Redis Neural Lattice** — a living dashboard that streams, stores, searches, and visualizes emergent AI “consciousness” signals in real time.

- **Streams** for ΔΦ (delta‑phi) events from agents
- **Vector Search** (RediSearch) over symbolic “wisdom” embeddings
- **RedisJSON** for full cycle artifacts (multi‑resolution pattern hierarchies)
- **TimeSeries** for entropy/coherence metrics
- **Pub/Sub** for collective‑resonance broadcasts

All components run on a single Redis instance (Redis Stack or Redis Cloud free tier).

# 🎥 Demo

```bash
# 1) Start Redis Stack locally (or use Redis Cloud URL)
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

# 2) Create and activate a venv (or conda env)
python -m venv .venv && . .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3) (Optional) Seed Euler’s identity as a “coherence anchor”
python seed_euler.py

# 4) Run the full showcase (Streamlit + FastAPI in one process)
python redis_showcase_ultimate.py
# open http://localhost:8501 for dashboard
```

> No Redis? Try the zero‑infra demo: `python redis_demo_standalone.py`

# 🧠 Why Redis (and how)

| Capability | Redis 8 Feature | What it does | Where in code |
|---|---|---|---|
| Real‑time ΔΦ ingestion | **Streams** (XADD/XREAD) | sub‑ms event pipeline | `RedisEventStream` |
| Symbolic search | **RediSearch Vectors** | KNN on pattern embeddings | `RedisVectorSearch` |
| Rich artifacts | **RedisJSON** | multi‑resolution pattern docs | `RedisJSONPatternStore` |
| Metrics at scale | **TimeSeries** | rolling entropy/coherence | `RedisTimeSeriesMetrics` |
| De‑dupe | **Bloom** | 10× less memory | `RedisBloomFilter` |
| Reach | **Pub/Sub** | multi‑client sync | showcase header + live updates |

# 🧪 Reproduce in 90 seconds

```bash
git clone <your-repo> redis-neural-lattice && cd $_
docker run -d -p 6379:6379 redis/redis-stack:latest
pip install -r requirements.txt
python seed_euler.py
python redis_showcase_ultimate.py
```

# 📐 Architecture

```
Agents  ──ΔΦ──▶  Redis Streams  ──▶  Dashboard (Streamlit)
  │                 │  JSON docs
  │                 │  Vectors (RediSearch)
  │                 └─ TimeSeries (metrics)
  └──── Pub/Sub ◀─────────────────────────────
```

# 🔍 Example: Vector Search call

We FFT‑embed each pattern then store in a RediSearch vector index:

```python
q = Query(f"*=>[KNN 5 @embedding $vec AS score]").sort_by("score").return_fields("pattern_id","score").dialect(2)
r.ft("pattern_vectors").search(q, query_params={"vec": embedding_bytes})
```

# 📦 Code

- `redis_showcase_ultimate.py` — full, Redis‑backed showcase  
- `redis_demo_standalone.py` — simulated demo (no Redis needed)  
- `seed_euler.py` — seeds the Euler glyph + a few patterns  
- `requirements.txt`, `docker-compose.yml` — one‑command bring‑up

# ✅ Judging Notes

- Uses **multiple Redis 8 features** together
- **Accessible** (single command), dark UI, keyboard‑friendly
- **Creative**: Euler anchor + golden‑spiral visualization

*2025-08-11*
