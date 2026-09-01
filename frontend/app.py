"""
frontend/app.py -- Streamlit UI for the Sustainable AI Shopping Agent.
Milestone 10. Pure display/HTTP layer -- no AI/ML logic lives here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from frontend.api_client import check_health, get_recommendations, get_recommendations_manual
from frontend.styles import apply_custom_css
from frontend.components import render_warnings, render_result_card, render_hero, render_summary_metrics, render_insights_chart, sort_results
from config import settings

st.set_page_config(page_title="Sustainable AI Shopping Agent", page_icon="🌱", layout="wide")
apply_custom_css(st)

render_hero(st)

if not check_health():
    st.error(f"Cannot reach the API at {settings.API_BASE_URL}. Make sure it's running "
             "(uvicorn app.main:app --reload) in another terminal tab.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    use_manual = st.checkbox("Skip intent extraction, set filters manually")
    with st.expander("Manual filter controls", expanded=use_manual):
        price_min = st.number_input("Min price (Rs.)", min_value=0.0, value=0.0, step=100.0)
        price_max = st.number_input("Max price (Rs.)", min_value=0.0, value=8000.0, step=100.0)
        category = st.selectbox("Category", ["Any", "Shoes", "Bags", "T-Shirts", "Jeans", "Jackets", "Shirts", "Dresses"])
        sustainability_tilt = st.checkbox("Emphasize sustainability")
    top_k = st.slider("Number of results", min_value=1, max_value=50, value=9)
    layout_columns = st.radio("Layout", [1, 2, 3], index=2, horizontal=True, format_func=lambda n: f"{n}-column")
    sort_by = st.selectbox("Sort by", ["Relevance", "Price: Low to High", "Price: High to Low", "Sustainability"])

query = st.text_input("What are you looking for?", placeholder="e.g. running shoes under 4000, eco-friendly")
search_clicked = st.button("Search", type="primary")

if search_clicked:
    if not query or not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            if use_manual:
                data, error = get_recommendations_manual(
                    query=query,
                    price_min=price_min if price_min > 0 else None,
                    price_max=price_max if price_max > 0 else None,
                    category=None if category == "Any" else category,
                    sustainability_tilt=sustainability_tilt,
                    top_k=top_k,
                )
            else:
                data, error = get_recommendations(query=query, top_k=top_k)

        if error:
            st.error(error)
        elif not data["results"]:
            st.info("No results found for this query.")
        else:
            render_warnings(st, data.get("warnings", {}))
            results = sort_results(data["results"], sort_by)
            render_summary_metrics(st, results)
            with st.expander("Insights", expanded=False):
                render_insights_chart(st, results)
            st.write(f"Showing {len(results)} result(s)")

            cols = st.columns(layout_columns)
            for i, item in enumerate(results):
                with cols[i % layout_columns]:
                    render_result_card(st, item)
