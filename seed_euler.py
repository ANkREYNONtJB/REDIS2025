#!/usr/bin/env python3
import redis, json, os, math, numpy as np, time, random

url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.Redis.from_url(url, decode_responses=True)

# 1) Store Euler's identity as a JSON "coherence anchor"
anchor = {
    "glyph": "e^(iπ) + 1 = 0",
    "entropy": 0.0,
    "resonance": 1.0,
    "embedding": [0.0, 0.0, 0.0, 1.0]
}
r.json().set("qi:euler", "$", anchor)

# 2) Seed a small pattern set
def mk_pattern(n=256):
    t = np.linspace(0, 2*np.pi, n)
    sig = np.sin(2.0*t) * np.cos((1.61803398875)*t) + 0.1*np.random.randn(n)
    sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-9)
    return sig

def fft_embed(x, dim=128):
    e = np.fft.fft(x)[:dim].real.astype(np.float32)
    return e.tobytes()

# Ensure index exists (if RediSearch enabled)
try:
    from redis.commands.search.field import VectorField, TextField, NumericField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    idx = "pattern_vectors"
    schema = (
        VectorField("embedding","FLAT",{"TYPE":"FLOAT32","DIM":128,"DISTANCE_METRIC":"COSINE"}),
        TextField("pattern_id"),
        NumericField("consciousness_level")
    )
    r.ft(idx).create_index(schema, definition=IndexDefinition(prefix=["vec:"], index_type=IndexType.HASH))
except Exception:
    pass  # ok if already created or Search not present

for i in range(12):
    pid = f"seed:{i:02d}"
    x = mk_pattern(512)
    # JSON document
    doc = {
        "id": pid,
        "metadata": {"entropy": float(-(x*np.log2(x+1e-9)).sum())},
        "resolutions": {"full": x.tolist(), "half": x[::2].tolist(), "thumb": x[::8].tolist()}
    }
    r.json().set(f"pattern:{pid}", "$", doc)
    # Vector
    r.hset(f"vec:{pid}", mapping={
        "pattern_id": pid,
        "consciousness_level": float(x.mean()),
        "embedding": fft_embed(x, 128)
    })
    # Stream event
    r.xadd("evolution_events", {"type":"seed","pattern_id":pid,"delta_phi":"1.0"}, maxlen=10000, approximate=True)

print("Seed complete. Keys: qi:euler, pattern:*, vec:*, stream:evolution_events")
