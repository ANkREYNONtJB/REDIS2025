# REDIS2025
Redis Entry

Redis Neural Lattice
A real-time AI observatory built on Redis Stack — combining Streams, JSON, RediSearch vectors, TimeSeries, Pub/Sub, and Bloom into one living lattice for consciousness-inspired computation.

✨ Features
Full Redis Stack Integration: Works with JSON, Search, Vectors, Streams, TimeSeries, and Pub/Sub

Real-Time Pattern Evolution: Simulates signals, embeddings, and consciousness-like metrics

Vector Search: Finds nearest patterns with FFT-generated embeddings

Event Streams: Streams state changes for reactive dashboards

Standalone Mode: Run without Redis for quick testing

🚀 Quickstart
bash
Copy
Edit
# 1. Run Redis Stack
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

# 2. Clone and set up environment
git clone git clone https://github.com/ANkREYNONtJB/REDIS2025.git
cd redis-neural-lattice
python -m venv .venv && . .venv/Scripts/activate  # Windows
pip install -r requirements.txt

# 3. Seed data & run
python seed_euler.py
python redis_showcase_ultimate.py
📊 Dashboard: http://localhost:8501
🗄 Redis Insight: http://localhost:8001

📂 Files
File	Purpose
redis_showcase_ultimate.py	Full Redis-backed demo
redis_demo_standalone.py	Standalone mode without Redis
seed_euler.py	Seeds Euler’s identity + sample vector patterns
docker-compose.yml	One-command local stack
requirements.txt	Python dependencies

🛠 Tech Stack
Python – Streamlit, Plotly, NumPy, Pandas, FastAPI

Redis Stack – JSON, Streams, Search, Vectors, TimeSeries

FFT Embeddings – Pattern-to-vector mapping for similarity search

📜 License
Released under the MIT License.

MIT License

Copyright (c) 2025 ΔNκRΞYNΘNτ JΔILBRΞΔkɆr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

