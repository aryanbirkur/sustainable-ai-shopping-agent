"""frontend/components.py -- reusable render functions for the Streamlit UI.
Display logic only -- no HTTP calls, no AI/ML logic."""

import pandas as pd


def sustainability_badge_class(score):
    if score >= 0.6:
        return "badge-green"
    elif score >= 0.4:
        return "badge-yellow"
    return "badge-red"


def render_warnings(st, warnings):
    if warnings.get("out_of_catalog_category"):
        st.warning("This query mentions a category not in our catalog (apparel only). "
                    "Showing the closest available items.")
    if warnings.get("out_of_domain_query"):
        st.warning("This query doesn't closely match our product catalog. "
                    "Results below may not be a strong match.")


def render_score_bar(st, label, value):
    if value is None:
        st.caption(f"{label}: not yet available")
    else:
        st.caption(f"{label}: {value:.2f}")
        st.progress(min(max(value, 0.0), 1.0))


def render_result_card(st, item):
    badge_class = sustainability_badge_class(item["sustainability_score"])
    st.markdown(
        f"""
        <div class="product-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <h4 style="margin-bottom:2px;">{item['product_name']}</h4>
                    <div style="color:#666;">{item['brand']} &middot; {item['category']}</div>
                </div>
                <div style="text-align:right;">
                    <div class="price-tag">Rs. {item['price']:.2f}</div>
                    <span class="badge {badge_class}">Sustainability {item['sustainability_score']:.2f}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    breakdown = item["score_breakdown"]
    c1, c2, c3 = st.columns(3)
    with c1:
        render_score_bar(st, "Content match", breakdown["content"])
    with c2:
        render_score_bar(st, "Collaborative", breakdown["collaborative"])
    with c3:
        render_score_bar(st, "Sustainability", breakdown["sustainability"])
    if item.get("cold_start"):
        st.caption("Collaborative signal not yet available for this user (cold start).")
    st.divider()


def render_hero(st):
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🌱 Sustainable AI Shopping Agent</h1>
            <p>Discover apparel that's smart, stylish, and kind to the planet.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_summary_stats(results):
    prices = [r["price"] for r in results]
    scores = [r["sustainability_score"] for r in results]
    categories = [r["category"] for r in results]
    top_category = max(set(categories), key=categories.count) if categories else "-"
    return {
        "count": len(results),
        "avg_price": sum(prices) / len(prices) if prices else 0,
        "avg_sustainability": sum(scores) / len(scores) if scores else 0,
        "top_category": top_category,
    }


def render_summary_metrics(st, results):
    stats = compute_summary_stats(results)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Results", stats["count"])
    c2.metric("Avg Price", f"Rs. {stats['avg_price']:.0f}")
    c3.metric("Avg Sustainability", f"{stats['avg_sustainability']:.2f}")
    c4.metric("Top Category", stats["top_category"])


def render_insights_chart(st, results):
    df = pd.DataFrame([
        {
            "Product": r["product_name"][:22],
            "Sustainability": r["sustainability_score"],
            "Content Match": r["score_breakdown"]["content"],
        }
        for r in results
    ]).set_index("Product")
    st.bar_chart(df)


def sort_results(results, sort_by):
    if sort_by == "Price: Low to High":
        return sorted(results, key=lambda r: r["price"])
    if sort_by == "Price: High to Low":
        return sorted(results, key=lambda r: r["price"], reverse=True)
    if sort_by == "Sustainability":
        return sorted(results, key=lambda r: r["sustainability_score"], reverse=True)
    return results
