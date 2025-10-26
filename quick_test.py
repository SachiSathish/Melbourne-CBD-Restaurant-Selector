# quick_test.py
from generation import answer

queries = [
    "Italian restaurants in CBD"   
]

for q in queries:
    print("Q:", q)
    print("A:", answer(q, k=5))
    print("-" * 60)