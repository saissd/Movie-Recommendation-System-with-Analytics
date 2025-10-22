import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.getenv("REC_MODEL_NAME", "all-MiniLM-L6-v2")
DEFAULT_REGION = os.getenv("REGION", "us")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
ITEMS_CSV = DATA_DIR / "items.csv"

class ModelIndex:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.st_model = SentenceTransformer(model_name)
        self.dim = self.st_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.items_df = self._load_or_create_items()

        if "text" not in self.items_df.columns:
            cols = [c for c in ["title", "genres", "overview", "category"] if c in self.items_df.columns]
            if cols:
                self.items_df["text"] = self.items_df[cols].fillna("").agg(" ".join, axis=1)
            else:
                self.items_df["text"] = ""

        self.item_vecs = self._embed_texts(self.items_df["text"].tolist())
        faiss.normalize_L2(self.item_vecs)
        self.index.add(self.item_vecs.astype(np.float32))

    def _load_or_create_items(self) -> pd.DataFrame:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if ITEMS_CSV.exists():
            return pd.read_csv(ITEMS_CSV)
        # Synthetic fallback dataset with titles
        rows = []
        genres = ["Romance", "Thriller", "Comedy", "Action", "Drama", "Sci-Fi"]
        for i in range(300):
            g = genres[i % len(genres)]
            rows.append({
                "item_id": i,
                "title": f"Movie {i:05d}",
                "genres": g,
                "lang": "en",
                "year": 1990 + (i % 30),
                "overview": f"{g} movie about relationships, conflict, and discovery #{i}"
            })
        df = pd.DataFrame(rows)
        df.to_csv(ITEMS_CSV, index=False)
        return df

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = self.st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vecs.astype(np.float32)

    def embed_user(self, query: str) -> np.ndarray:
        v = self._embed_texts([query])[0]
        v = v / (np.linalg.norm(v) + 1e-9)
        return v.astype(np.float32)

    def recommend(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        q = self.embed_user(query).reshape(1, -1)
        scores, idx = self.index.search(q, min(k, len(self.items_df)))
        out = []
        for s, i in zip(scores[0].tolist(), idx[0].tolist()):
            row = self.items_df.iloc[i].to_dict()
            out.append({"score": float(s), **row})
        return out

    def drift_score(self, recent_user_vecs: List[np.ndarray]) -> float:
        if not recent_user_vecs:
            return 0.0
        recent = np.vstack(recent_user_vecs).mean(axis=0)
        recent /= (np.linalg.norm(recent) + 1e-9)
        catalog_mean = self.item_vecs.mean(axis=0)
        catalog_mean /= (np.linalg.norm(catalog_mean) + 1e-9)
        return float(max(0.0, 0.5 * (1 - float(np.dot(recent, catalog_mean)))))
