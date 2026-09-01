import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
scripts/recommend_demo.py -- readable demo/smoke test for Milestone 5.
Run: python scripts/recommend_demo.py
"""

from recommendation.hybrid import recommend


def print_results(title: str, results: list):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

    if not results:
        print("  (no results)")
        return

    if results[0].get("out_of_domain_query"):
        print("  ⚠  This query doesn't look like a good match for this catalog.")
        print("     (best semantic similarity found was below the confidence threshold --")
        print("      showing the closest available items, but treat these as low-confidence.)\n")

    weights = results[0].get("weights_used", {})
    weight_str = ", ".join(f"{k}={v:.0%}" for k, v in weights.items())
    cold = results[0].get("cold_start")
    print(f"  weights used: {weight_str}" + ("  (cold start -- collaborative excluded)" if cold else ""))

    for r in results:
        print(f"\n  #{r['rank']}  {r.get('product_name', '?')}")
        print(f"       {r.get('category', '?')} · {r.get('brand', '?')} · Rs.{r.get('price', '?')}")
        print(f"       final score: {r['final_score']:.3f}")

        raw = r.get("raw_signals", {})
        bd = r.get("score_breakdown", {})

        content_raw = raw.get("content")
        content_contrib = bd.get("content")
        print(f"         content        : raw={content_raw:.3f}  -> contributes {content_contrib:.3f}")

        cf_raw = raw.get("collaborative")
        cf_contrib = bd.get("collaborative")
        if cf_raw is None:
            print(f"         collaborative  : n/a (no interaction history for this user)")
        else:
            print(f"         collaborative  : raw={cf_raw:.3f}  -> contributes {cf_contrib:.3f}")

        sus_raw = raw.get("sustainability")
        sus_contrib = bd.get("sustainability")
        print(f"         sustainability : raw={sus_raw:.3f}  -> contributes {sus_contrib:.3f}")

    print()


if __name__ == "__main__":
    QUERY = "lightweight running shoes"

    print_results(
        f'QUERY: "{QUERY}"  |  user: anonymous (cold start)',
        recommend(QUERY, user_id=None, top_k=5),
    )

    print_results(
        f'QUERY: "{QUERY}"  |  user: U0096 (has real interaction history)',
        recommend(QUERY, user_id="U0096", top_k=5),
    )

    print_results(
        f'QUERY: "{QUERY}"  |  user: anonymous, custom weights (content=0.9, sustainability=0.1)',
        recommend(QUERY, user_id=None, top_k=5, weights={"content": 0.9, "collaborative": 0.0, "sustainability": 0.1}),
    )

    print_results(
        f'QUERY: "{QUERY}"  |  user: anonymous, sustainability-first weights (sustainability=0.8)',
        recommend(QUERY, user_id=None, top_k=5, weights={"content": 0.2, "collaborative": 0.0, "sustainability": 0.8}),
    )

    print_results(
        'QUERY: "wireless bluetooth headphones"  |  user: anonymous (out-of-catalog test)',
        recommend("wireless bluetooth headphones", user_id=None, top_k=5),
    )

    print_results(
        'QUERY: "recycled cotton tote bag under 3000 rupees"  |  user: anonymous',
        recommend("recycled cotton tote bag under 3000 rupees", user_id=None, top_k=5),
    )
