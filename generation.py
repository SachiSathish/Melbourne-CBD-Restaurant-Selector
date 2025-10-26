# generation.py
import requests
from prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from retrieval import retrieve_with_scores
from geo import resolve_location
from retrieval import retrieve_with_scores


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "llama3.2:3b"
BASES = ["http://127.0.0.1:11434", "http://localhost:11434"]

def format_context_block(rows):
    import pandas as pd
    if isinstance(rows, pd.DataFrame):
        recs = rows.to_dict(orient="records")
    elif isinstance(rows, list):
        recs = rows
    else:
        recs = []

    blocks = []
    for i, r in enumerate(recs):
        name = r.get("name", "Unknown")
        kb   = r.get("kb_text", "")
        # Keep it simple: only name + kb_text
        blocks.append(f"{name}: {kb}")
    return "\n\n".join(blocks)


def call_ollama(prompt: str, system_prompt: str) -> str:
    """
    Call Ollama via /api/chat only. Tries 127.0.0.1 then localhost.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt}
        ],
        "stream": False,
        "options": {"num_ctx": 2048, "temperature": 0.2}
    }
    last_err = None
    for base in BASES:
        try:
            r = requests.post(f"{base}/api/chat", json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "message" in data:
                return (data["message"].get("content") or "").strip()
            return (data.get("response") or "").strip()
        except Exception as e:
            last_err = f"{base}/api/chat error: {e}"
    raise RuntimeError(f"Ollama chat endpoint failed. Last error: {last_err}")

    
def answer(question, k=5, min_hybrid=0.2):
    # ------------
    # Retrieve context docs (with constraint filtering)
    # ------------
    import re
    import pandas as pd
    from retrieval import df as KB

    # Parse query constraints
    def _extract_budget(q: str):
        m = re.search(r"(?:under|less than|<=?)\s*\$?\s*(\d+)", q.lower())
        return int(m.group(1)) if m else None

    def _extract_cuisine(q: str, known_cuisines: set):
        ql = q.lower()
        for c in sorted(known_cuisines, key=len, reverse=True):
            if re.search(rf"\b{re.escape(c.lower().replace('_',' '))}\b", ql):
                return c
        return None

    # Collect known cuisines from KB
    known_cuisines = set()
    if "cuisine" in KB.columns:
        for v in KB["cuisine"].astype(str):
            for token in re.split(r"[\/,&;]| and |,|\|", v.lower()):
                t = token.strip()
                if t and t not in {"food", "restaurant", "restaurants", "cafe", "cafes"}:
                    known_cuisines.add(t)

    max_pp = _extract_budget(question)
    want_cuisine = _extract_cuisine(question, known_cuisines)
    near = resolve_location(question)  # (lat, lon, name) or None

    # Retrieve a generous pool of candidates
    pool_k = max(k * 6, 30)
    rows = retrieve_with_scores(question, top_k=pool_k, near=near)

    # Convert to DataFrame
    cand = pd.DataFrame(rows)

    # ---- Normalise price ----
    def _num_price(val):
        s = re.sub(r"[^0-9.\-]", "", str(val or ""))
        if not s:
            return None
        try:
            if "-" in s:
                parts = [p for p in s.split("-") if p.strip()]
                nums = [float(p) for p in parts]
                return sum(nums) / len(nums) if nums else None
            return float(s)
        except:
            return None

    # --- Normalise price into a single numeric column we can rely on ---
    PRICE_COLS = ["avg_price_pp", "price_per_person", "price", "avg_price", "cost_pp"]
    price_src = next((c for c in PRICE_COLS if c in cand.columns), None)

    if price_src:
        cand["price_num"] = cand[price_src].apply(_num_price)
    else:
        # create an empty price column so downstream code doesn't KeyError
        import numpy as np
        cand["price_num"] = np.nan


    # ---- Apply filters ----
    f = cand.copy()

    # a) cuisine
    if want_cuisine and "cuisine" in f.columns:
        wc = want_cuisine.lower().replace("_", " ")
        f = f[f["cuisine"].astype(str).str.lower().str.contains(rf"\b{re.escape(wc)}\b", regex=True)]

    # b) budget
    if max_pp is not None and "price_num" in f.columns:
        f = f[(f["price_num"].notna()) & (f["price_num"] <= max_pp + 0.49)]
        
    # c) geo (<= 0.8 km)
    if near is not None and "distance_km" in f.columns:
        f = f[(f["distance_km"].notna()) & (f["distance_km"] <= 0.8)]

    # Relax filters if needed
    if f.empty and near is not None and "distance_km" in cand.columns:
        f = cand[cand["distance_km"].notna() & (cand["distance_km"] <= 1.2)]
    if f.empty and max_pp is not None:
        f = cand[(cand["price_num"].notna()) & (cand["price_num"] <= max_pp + 5)]
    if f.empty and want_cuisine and "cuisine" in cand.columns:
        wc = want_cuisine.lower().replace("_", " ")
        f = cand[cand["cuisine"].astype(str).str.lower().str.contains(rf"\b{re.escape(wc)}\b", regex=True)]
    if f.empty:
        f = cand

    # Take top-k filtered results
    filtered_rows = f.head(k).to_dict(orient="records")



##new addition
    import re

    def _extract_budget(q: str):
        m = re.search(r"(?:under|less than|<=?)\s*\$?\s*(\d+)", q.lower())
        return int(m.group(1)) if m else None

    def _extract_cuisine(q: str, known_cuisines: set):
        ql = q.lower()
        # match any cuisine token that appears as a whole word in the question
        for c in sorted(known_cuisines, key=len, reverse=True):
            if re.search(rf"\b{re.escape(c.lower().replace('_',' '))}\b", ql):
                return c
        return None

    # build cuisine lexicon once from your KB (names already in retrieval.df)
    from retrieval import df as KB
    known_cuisines = set()
    if "cuisine" in KB.columns:
        for v in KB["cuisine"].astype(str):
            for token in re.split(r"[\/,&;]| and |,|\|", v.lower()):
                t = token.strip()
                if t and t not in {"food","restaurant","restaurants","cafe","cafes"}:
                    known_cuisines.add(t)

    max_pp = _extract_budget(question)
    want_cuisine = _extract_cuisine(question, known_cuisines)
## new additionend

    # Simple confidence gate: if top doc hybrid score is too low, refuse
    rows = pd.DataFrame(filtered_rows)   # << use filtered data here!

    context_lines = []
    for _, r in rows.iterrows():
        has_dist = ("distance_km" in r) and pd.notna(r["distance_km"])
        dist = f" · Distance: {float(r['distance_km']):.1f} km" if has_dist else ""
        context_lines.append(f"{r['name']}: {r['kb_text']}{dist}")

    context_block = "\n\n".join(context_lines)


    safe_template = (
        USER_PROMPT_TEMPLATE
        .replace("{", "{{")
        .replace("}", "}}")
        .replace("{{question}}", "{question}")
        .replace("{{context_block}}", "{context_block}")
    )

    user_prompt = safe_template.format(
        question=question,
        context_block=context_block
    )

    return call_ollama(user_prompt, SYSTEM_PROMPT)






    


