"""frontend/components.py -- reusable render functions for the Streamlit UI.
Display logic only -- no HTTP calls, no AI/ML logic."""

import pandas as pd


def sustainability_tier(score):
    """Returns ('high'|'mid'|'low', label) -- tier drives both card color and text."""
    if score >= 0.6:
        return "high", "High"
    elif score >= 0.4:
        return "mid", "Moderate"
    return "low", "Low"


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
    tier, tier_label = sustainability_tier(item["sustainability_score"])
    is_real_product = str(item.get("product_id", "")).startswith("HM")
    source_tag = '<span class="data-source-tag">Real product · H&amp;M</span>' if is_real_product else ""

    price_display = f"Rs. {item['price']:.2f}" if item.get("price") is not None else "Price not available"

    if item.get("image_path"):
        st.image(item["image_path"], use_container_width=True)

    st.markdown(
        f"""
        <div class="product-card tier-{tier}">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <h4>{item['product_name']}{source_tag}</h4>
                    <div class="product-meta">{item['brand']}</div>
                    <div class="product-meta">{item['category']}</div>
                </div>
                <div style="text-align:right;">
                    <div class="price-tag">{price_display}</div>
                </div>
            </div>
            <div class="sustainability-line tier-{tier}">
                Sustainability: {tier_label} ({item['sustainability_score']:.2f})
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
        <div class="page-header">
            <h1>Sustainable AI Shopping Agent</h1>
            <p>Search apparel with sustainability signal built into every ranking.</p>
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
