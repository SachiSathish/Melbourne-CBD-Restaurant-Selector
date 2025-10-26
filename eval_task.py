# eval/eval_task.py
# Metrics: CSR (1), Success@R (4), Price MAE (5), Rating MAE (6), Unanswered Rate (7)

import re, json, time, math
import pandas as pd
from pathlib import Path

# ---- your pipeline imports ----
from generation import answer
from retrieval import retrieve_with_scores
from retrieval import df as KB  # your loaded dataset (contains names, price, rating, lat/lon)

# ====== Config ======
GOLD_PATH = "eval/gold.jsonl"   # from Step 1
TOP_K = 5
MIN_HYBRID = 0.3
RADIUS_KM = 0.8   # geo radius for Success@R & CSR geo checks

# Landmarks used for eval (feel free to extend)
LANDMARKS = {
    "flinders street station": (-37.8183, 144.9671),
    "southern cross station":  (-37.8183, 144.9526),
    "melbourne central":       (-37.8105, 144.9631),
    "parliament station":      (-37.8110, 144.9730),
    "state library":           (-37.8099, 144.9656),
    "federation square":       (-37.8179, 144.9691),
    "bourke street mall":      (-37.8142, 144.9632),
}

# Column detection for lat/lon (whatever your CSV used)
LAT_COL = "latitude" if "latitude" in KB.columns else ("lat" if "lat" in KB.columns else None)
LON_COL = "longitude" if "longitude" in KB.columns else ("lon" if "lon" in KB.columns else None)

# ====== Regex helpers ======
CIT_RE     = re.compile(r"\[([^\[\]\|]+?)\]")         # [Name]
PRICE_RE_1 = re.compile(r"\$?\s*(\d+(?:\.\d+)?)")     # $18 or 18
RATING_RE  = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*5")   # 4.3/5

def name_norm(s: str) -> str:
    return str(s or "").strip().lower()

def get_row_by_name(name: str):
    m = KB[KB["name"].astype(str).str.strip().str.lower() == name_norm(name)]
    return m.iloc[0] if len(m) else None

def haversine_km(p1, p2):
    R = 6371.0
    (lat1, lon1), (lat2, lon2) = p1, p2
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1,lon1,lat2,lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def parse_prices_by_line(answer_text: str):
    """line_idx -> list of numeric prices found in that line"""
    prices = {}
    for i, line in enumerate(answer_text.splitlines()):
        vals = [float(m.group(1)) for m in PRICE_RE_1.finditer(line)]
        if vals:
            prices[i] = vals
    return prices

def parse_rating_by_line(answer_text: str):
    ratings = {}
    for i, line in enumerate(answer_text.splitlines()):
        m = RATING_RE.search(line)
        if m:
            ratings[i] = float(m.group(1))
    return ratings

def locate_line_indices_for_names(answer_text: str, cited_names):
    """map each cited name -> first line index that mentions it"""
    idx_map = {}
    lines = answer_text.splitlines()
    for name in cited_names:
        for i, line in enumerate(lines):
            if name.lower() in line.lower():
                idx_map[name] = i
                break
    return idx_map

def satisfies_constraints(row, cons, landmark_latlon=None, R=RADIUS_KM):
    # cuisine (best-effort, strict if missing)
    if cons.get("cuisine"):
        text = ""
        for col in ("cuisine","cuisines"):
            if col in row.index and pd.notna(row[col]):
                text += f" {str(row[col]).lower()} "
        want = {c.lower() for c in cons["cuisine"]}
        if text and not any(w in text for w in want):
            return False
        if not text:
            return False

    # budget
    if cons.get("max_pp") is not None:
        try:
            pp = float(row.get("avg_price_pp", row.get("price_per_person")))
            if pd.isna(pp) or pp > float(cons["max_pp"]):
                return False
        except Exception:
            return False

    # geo
    if landmark_latlon is not None and LAT_COL and LON_COL:
        try:
            lat = float(row[LAT_COL]); lon = float(row[LON_COL])
            if haversine_km(landmark_latlon, (lat, lon)) > R:
                return False
        except Exception:
            return False
    elif cons.get("near"):
        # Asked for near-X but we can't check distance
        return False

    return True

def eval_one(ex, k=TOP_K, min_hybrid=MIN_HYBRID, R=RADIUS_KM):
    qid = ex.get("qid"); q = ex.get("question"); cons = ex.get("constraints", {})

    # Optional landmark target
    landmark_latlon = None
    near_key = (cons.get("near") or "").lower().strip()
    if near_key:
        landmark_latlon = LANDMARKS.get(near_key)

    # Run pipeline (your answer() already calls retriever + LLM)
    t0 = time.time()
    a = answer(q, k=k, min_hybrid=min_hybrid)
    latency = time.time() - t0

    # (7) Unanswered?
    low = a.strip().lower()
    unanswered = low.startswith(("i don’t know","i don't know"))

    # Extract citations
    cited = set(CIT_RE.findall(a))

    # (1) CSR: all cited satisfy constraints
    csr_ok = True if cited else False
    if cited:
        for name in cited:
            row = get_row_by_name(name)
            if row is None or not satisfies_constraints(row, cons, landmark_latlon, R):
                csr_ok = False; break

    # (4) Success@R: at least one cited within R km of landmark
    success_at_R = False
    if landmark_latlon and cited and LAT_COL and LON_COL:
        for name in cited:
            row = get_row_by_name(name)
            if row is not None and pd.notna(row.get(LAT_COL)) and pd.notna(row.get(LON_COL)):
                d = haversine_km(landmark_latlon, (float(row[LAT_COL]), float(row[LON_COL])))
                if d <= R: success_at_R = True; break

    # (5) Price MAE: compare per-person price on the same line as the venue
    line_prices = parse_prices_by_line(a)
    name_line_map = locate_line_indices_for_names(a, cited)
    price_errs = []
    price_errs = []

    def _to_num(val):
        """Convert things like '$1-20' or '15.5 AUD' to a mid-point float."""
        if isinstance(val, (int, float)):
            return float(val)
        if not isinstance(val, str):
            return None
        import re
        s = re.sub(r"[^0-9.\-]", "", val)  # remove non-digits
        if "-" in s:  # handle ranges like '1-20'
            parts = [p for p in s.split("-") if p]
            try:
                nums = [float(p) for p in parts]
                return sum(nums) / len(nums)
            except Exception:
                return None
        try:
            return float(s)
        except Exception:
            return None

    for name, line_idx in name_line_map.items():
        row = get_row_by_name(name)
        if row is None:
            continue
        truth = row.get("avg_price_pp", row.get("price_per_person"))
        if pd.isna(truth):
            continue
        vals = line_prices.get(line_idx, [])
        if vals:
            stated = sum(vals) / len(vals)   # average if range like "15 20"
            st = _to_num(stated)
            tr = _to_num(truth)
            if st is not None and tr is not None:
                price_errs.append(abs(st - tr))

    price_mae = (sum(price_errs) / len(price_errs)) if price_errs else None


    # (6) Rating MAE: compare rating on same line
    line_ratings = parse_rating_by_line(a)
    rating_errs = []
    for name, line_idx in name_line_map.items():
        row = get_row_by_name(name)
        if row is None or pd.isna(row.get("rating")): continue
        stated = line_ratings.get(line_idx)
        if stated is not None:
            rating_errs.append(abs(float(stated) - float(row["rating"])))
    rating_mae = (sum(rating_errs)/len(rating_errs)) if rating_errs else None

    return {
        "qid": qid, "question": q,
        "unanswered": unanswered,     # (7)
        "csr_ok": csr_ok,             # (1)
        "success_at_R": success_at_R, # (4)
        "price_mae": price_mae,       # (5)
        "rating_mae": rating_mae,     # (6)
        "latency_sec": round(latency, 2),
        "n_cited": len(cited),
        "answer": a
    }

def evaluate(gold_path=GOLD_PATH):
    examples = [json.loads(l) for l in Path(gold_path).read_text(encoding="utf-8").splitlines()]
    rows = [eval_one(ex) for ex in examples]
    df = pd.DataFrame(rows)
    
    raw_lines = Path(gold_path).read_text(encoding="utf-8").splitlines()
    examples = []
    for i, l in enumerate(raw_lines, start=1):
        l = l.lstrip("\ufeff").strip()   # <-- strip BOM + whitespace
        if not l:
            continue                     # skip blank lines
        try:
            examples.append(json.loads(l))
        except Exception as e:
            raise ValueError(f"Bad JSON on line {i}: {e}\nLine content: {l!r}")
        
    summary = pd.DataFrame({
        "Unanswered Rate (UR)": [df["unanswered"].mean()],
        "Constraint Satisfaction Rate (CSR)": [df["csr_ok"].mean()],
        f"Success@{RADIUS_KM}km": [df["success_at_R"].mean()],
        "Price MAE (pp)": [df["price_mae"].mean(skipna=True)],
        "Rating MAE": [df["rating_mae"].mean(skipna=True)],
        "Latency p50 (s)": [df["latency_sec"].median()],
        "Latency p95 (s)": [df["latency_sec"].quantile(0.95)],
    })

    df.to_csv("eval/per_query_results.csv", index=False)
    summary.to_csv("eval/summary_results.csv", index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print("\nSaved: eval/per_query_results.csv and eval/summary_results.csv")

if __name__ == "__main__":
    evaluate()
