"""frontend/styles.py -- custom CSS for the Streamlit UI. Display only."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,500;0,600;1,500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --paper: #EFEDE4;
    --card: #FBFAF5;
    --ink: #1F2A24;
    --moss: #47614F;
    --ochre: #C98A2C;
    --clay: #B25A42;
}

.stApp {
    background: var(--paper);
}
.stApp, .stApp p, .stApp span, .stApp label, .stApp div {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}
h1, h2, h3, h4 {
    font-family: 'Fraunces', serif;
    color: var(--ink);
    font-weight: 500;
}

/* Sidebar: a real dark panel, not a pale echo of the main page */
section[data-testid="stSidebar"] {
    background-color: var(--ink);
    border-right: none;
}
section[data-testid="stSidebar"] * {
    color: #EFEDE4 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Fraunces', serif;
    font-weight: 500;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(239,237,228,0.15);
}

/* Page header: plain, no boxed banner, no gradient */
.page-header {
    padding: 8px 0 20px 0;
    border-bottom: 1px solid rgba(31,42,36,0.12);
    margin-bottom: 24px;
}
.page-header h1 {
    font-size: 2.2rem;
    margin: 0 0 6px 0;
    letter-spacing: -0.01em;
}
.page-header p {
    font-size: 1rem;
    color: var(--moss);
    margin: 0;
    max-width: 60ch;
}

/* Buttons: flat, deliberate -- no gradient, no lift */
.stButton > button {
    background: var(--ochre);
    color: var(--card);
    border: none;
    border-radius: 6px;
    padding: 8px 22px;
    font-weight: 500;
    transition: background 0.15s ease;
}
.stButton > button:hover {
    background: #B47A22;
    color: var(--card);
}

/* Product card: structure carries meaning -- left edge color = sustainability tier */
.product-card {
    border: 1px solid rgba(31,42,36,0.10);
    border-left: 3px solid var(--moss);
    border-radius: 4px;
    padding: 18px 20px;
    margin-bottom: 14px;
    background: var(--card);
    transition: border-color 0.15s ease;
}
.product-card:hover {
    border-color: rgba(31,42,36,0.22);
}
.product-card.tier-high { border-left-color: var(--moss); }
.product-card.tier-mid  { border-left-color: var(--ochre); }
.product-card.tier-low  { border-left-color: var(--clay); }

.product-card h4 {
    font-size: 1.15rem;
    margin: 0 0 2px 0;
}
.product-meta {
    color: var(--moss);
    font-size: 0.9rem;
}
.price-tag {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--ink);
}
.sustainability-line {
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 6px;
}
.sustainability-line.tier-high { color: var(--moss); }
.sustainability-line.tier-mid  { color: var(--ochre); }
.sustainability-line.tier-low  { color: var(--clay); }

.data-source-tag {
    display: inline-block;
    font-size: 0.72rem;
    color: var(--moss);
    border: 1px solid rgba(71,97,79,0.35);
    border-radius: 3px;
    padding: 1px 6px;
    margin-left: 8px;
}

/* Progress bars used for score breakdown -- recolor from Streamlit default blue */
div[data-testid="stProgress"] > div > div > div {
    background-color: var(--moss) !important;
}

div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid rgba(31,42,36,0.10);
    border-radius: 4px;
    padding: 12px 8px;
}
div[data-testid="stMetricLabel"] {
    color: var(--moss) !important;
}
</style>
"""


def apply_custom_css(st):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
