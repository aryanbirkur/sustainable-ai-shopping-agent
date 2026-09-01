"""frontend/styles.py -- custom CSS for the Streamlit UI. Display only."""

CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #f4f9f4 0%, #eef7f0 45%, #e8f5e9 100%);
}
section[data-testid="stSidebar"] {
    background-color: #f1f8f4;
    border-right: 1px solid #dcece0;
}
h1, h2, h3 { color: #1b4332; }

.hero-banner {
    background: linear-gradient(120deg, #1b4332 0%, #2d6a4f 55%, #40916c 100%);
    padding: 36px 32px;
    border-radius: 18px;
    margin-bottom: 26px;
    box-shadow: 0 8px 24px rgba(27,67,50,0.25);
}
.hero-banner h1 { color: white; margin: 0 0 6px 0; font-size: 2.1rem; }
.hero-banner p { color: #d8f3dc; margin: 0; font-size: 1.05rem; }

.stButton > button {
    background: linear-gradient(90deg, #2d6a4f, #40916c);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 22px;
    font-weight: 600;
    transition: opacity 0.15s ease;
}
.stButton > button:hover { opacity: 0.88; color: white; }

.product-card {
    border: 1px solid #e3e8e3;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 16px;
    background: rgba(255,255,255,0.92);
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.product-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 26px rgba(0,0,0,0.10);
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    color: white;
}
.badge-green { background: linear-gradient(90deg, #2d6a4f, #40916c); }
.badge-yellow { background: linear-gradient(90deg, #b08900, #d4a017); }
.badge-red { background: linear-gradient(90deg, #a4262c, #c0392b); }
.price-tag { font-size: 1.15rem; font-weight: 700; color: #1b4332; }

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.85);
    border-radius: 12px;
    padding: 10px 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
</style>
"""


def apply_custom_css(st):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
