import json
from pathlib import Path

path = Path("C:\RMIT\Semester 2\Case Studies in DS\WIL Project\Start\eval\gold.jsonl")  # adjust path if needed

if not path.exists():
    print("❌ File not found:", path.resolve())
    raise SystemExit

with open(path, "rb") as f:
    raw = f.read()

print("File size (bytes):", len(raw))
print("First bytes:", raw[:20])

# Decode with BOM-safe UTF-8
text = raw.decode("utf-8-sig")

lines = text.splitlines()
print("Line count:", len(lines))

for i, line in enumerate(lines, 1):
    if not line.strip():
        print(f"⚠️ Line {i} is empty/whitespace")
        continue
    try:
        obj = json.loads(line)
        print(f"✅ Line {i} valid JSON:", obj.get("qid"))
    except json.JSONDecodeError as e:
        print(f"❌ Line {i} invalid JSON: {e}")
