"""
scripts/search_demo.py

Milestone 4 — quick manual test of semantic_search() from the terminal.

Run:
    python scripts/search_demo.py "running shoes under 4000 that are eco friendly"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vector_search.semantic_search import semantic_search


def main():
    query = " ".join(sys.argv[1:]) or "a lightweight eco-friendly water bottle"
    results = semantic_search(query, top_k=5)

    print(f"\nQuery: {query}\n")
    for i, r in enumerate(results, start=1):
        m = r["metadata"]
        print(
            f"{i}. {m.get('product_name', r['product_id'])}  "
            f"(similarity={r['similarity']}, price=Rs.{m.get('price')}, "
            f"sustainability={m.get('sustainability_score')})"
        )


if __name__ == "__main__":
    main()
