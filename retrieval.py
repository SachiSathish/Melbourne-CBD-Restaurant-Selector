# retrieval.py
# Hybrid retrieval (BM25 + FAISS) with optional geo-aware boosting.

import os
import pickle
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

# ========= Config =========
# Use forward slashes or a raw string on Windows to avoid unicodeescape issues.
DATA_PATH  = "C:/RMIT/Semester 2/Case Studies in DS/WIL Project/Start/Updated_with_kbtext.csv"
BM25_PATH  = "bm25_index.pkl"
EMB_PATH   = "faiss_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"

# ========= Load data =========
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

if "kb_text" not in df.columns:
    raise KeyError("Column 'kb_text' not found in dataset. Regenerate kb_text first.")

# Support either lat/lon or latitude/longitude column names (geo is optional)
LAT_COL = "latitude" if "latitude" in df.columns else ("lat" if "lat" in df.columns else None)
LON_COL = "longitude" if "longitude" in df.columns else ("lon" if "lon" in df.columns else None)

# ========= BM25 =========
if os.path.exists(BM25_PATH):
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
else:
    tokenized_corpus = [str(doc).lower().split() for doc in df["kb_text"].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

# ========= Embeddings + FAISS =========
if os.path.exists(EMB_PATH):
    embeddings = np.load(EMB_PATH)
else:
    _model_tmp = SentenceTransformer(MODEL_NAME)
    embeddings = _model_tmp.encode(
        df["kb_text"].astype(str).tolist(),
        convert_to_numpy=True,
        show_progress_bar=True
    )
    np.save(EMB_PATH, embeddings)

# Keep a model instance for queries
model = SentenceTransformer(MODEL_NAME)

# In-memory FAISS index
dim = int(embeddings.shape[1])
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# ========= Geo helpers (NEW) =========
def _haversine_np(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km."""
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def _geo_scores(near, radius_km):
    """
    near: (lat, lon) or (lat, lon, name) or None
    Returns: (geo_norm_all_docs, distance_all_docs, landmark_name)
    geo_norm ∈ [0,1], where 1 at landmark, 0 at >= radius_km.
    If no lat/lon columns exist, returns zeros.
    """
    if near is None or LAT_COL is None or LON_COL is None:
        z = np.zeros(len(df), dtype=float)
        return z, None, None
    try:
        if len(near) == 3:
            near_lat, near_lon, near_name = near
        else:
            near_lat, near_lon = near
            near_name = None
        lat_arr = df[LAT_COL].to_numpy(dtype=float)
        lon_arr = df[LON_COL].to_numpy(dtype=float)
        dist_km = _haversine_np(near_lat, near_lon, lat_arr, lon_arr)
        r = max(radius_km or 1.0, 1e-6)
        geo_norm = 1.0 - np.clip(dist_km / r, 0.0, 1.0)  # 1 close, 0 far
        return geo_norm, dist_km, near_name
    except Exception:
        z = np.zeros(len(df), dtype=float)
        return z, None, None

# ========= Public API =========
def retrieve_with_scores(query: str,
                         top_k: int = 5,
                         near=None,
                         radius_km: float = 0.8,
                         geo_weight: float = 0.8) -> pd.DataFrame:
    """
    Hybrid retrieval with optional geo boosting.

    Args:
      query: user text
      top_k: number of results to return
      near: (lat, lon) or (lat, lon, name) or None
      radius_km: radius used to compute geo score falloff
      geo_weight: weight for geo score in hybrid combination

    Returns:
      DataFrame with:
        ['osm_id','name','amenity','kb_text',
         'bm25_score','dense_score','geo_score','hybrid_score',
         'distance_km'(opt), 'near'(opt)]
    """
    # --- BM25 ---
    tokens = str(query).lower().split()
    bm25_scores = bm25.get_scores(tokens)  # shape [N]

    # --- Dense / FAISS ---
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, min(top_k, len(df)))
    dense_norm = np.zeros_like(bm25_scores, dtype=float)
    if I.size > 0:
        d = D[0]
        dense_norm[I[0]] = 1.0 - (d / (d.max() + 1e-8))  # higher = better

    # --- Normalise BM25 to [0,1] ---
    bm_min, bm_max = bm25_scores.min(), bm25_scores.max()
    bm25_norm = (bm25_scores - bm_min) / (bm_max - bm_min + 1e-8)

    # --- Geo (NEW) ---
    geo_norm, dist_all, near_name = _geo_scores(near, radius_km)

    # --- Combine ---
    text_score = bm25_norm + dense_norm
    hybrid_scores = text_score + (geo_weight * geo_norm)

    # --- Select top_k ---
    top_idx = np.argsort(hybrid_scores)[::-1][:top_k]
    rows = df.iloc[top_idx][["osm_id", "name", "amenity", "kb_text"]].copy()
    rows["bm25_score"]   = bm25_scores[top_idx]
    rows["dense_score"]  = dense_norm[top_idx]
    rows["geo_score"]    = geo_norm[top_idx]         # NEW
    rows["hybrid_score"] = hybrid_scores[top_idx]

    if dist_all is not None:
        rows["distance_km"] = dist_all[top_idx]      # NEW
        if near_name:
            rows["near"] = near_name                 # NEW

    return rows.reset_index(drop=True)

def retrieve(query: str, top_k: int = 5, **kwargs):
    """Convenience wrapper returning list[dict] (no scores)."""
    rows = retrieve_with_scores(query, top_k=top_k, **kwargs)
    return rows[["name", "amenity", "kb_text"]].to_dict(orient="records")
