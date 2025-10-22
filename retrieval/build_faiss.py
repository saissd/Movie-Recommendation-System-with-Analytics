import os, numpy as np, pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

DATA = Path("data")
items = pd.read_parquet(DATA/"items.parquet") if (DATA/"items.parquet").exists() else pd.read_csv(DATA/"items.csv")

if "text" not in items.columns:
    cols = [c for c in ["title", "genres", "overview", "category"] if c in items.columns]
    items["text"] = items[cols].fillna("").agg(" ".join, axis=1) if cols else ""

model = SentenceTransformer("all-MiniLM-L6-v2")
vecs = model.encode(items["text"].tolist(), convert_to_numpy=True, show_progress_bar=False).astype("float32")
faiss.normalize_L2(vecs)
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)

Path("data/model").mkdir(parents=True, exist_ok=True)
faiss.write_index(index, "data/model/faiss.index")
print(f"vecs shape: {vecs.shape}")
print("wrote: data/model/faiss.index exists?", os.path.exists("data/model/faiss.index"))
