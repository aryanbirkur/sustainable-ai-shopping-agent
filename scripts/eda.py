"""
scripts/eda.py

Small, purposeful exploratory data analysis on the cleaned product data.
Saves PNG charts to data/eda/. Deliberately limited to plots that inform
the recommendation problem (not a dump of every possible chart).

Module type: Data analysis / visualization. No ML/AI here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import pandas as pd

from config.settings import CLEAN_PRODUCTS_PATH, EDA_DIR


def _save(fig, name):
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    path = EDA_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_products_by_category(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    df["category"].value_counts().plot(kind="bar", ax=ax, color="#3b7a57")
    ax.set_title("Products by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    _save(fig, "01_products_by_category.png")


def plot_price_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    df["price"].plot(kind="hist", bins=30, ax=ax, color="#4c72b0")
    ax.set_title("Price Distribution (₹)")
    ax.set_xlabel("Price")
    _save(fig, "02_price_distribution.png")


def plot_rating_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    df["rating"].plot(kind="hist", bins=20, ax=ax, color="#dd8452")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating (0-5)")
    _save(fig, "03_rating_distribution.png")


def plot_carbon_footprint_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    df["carbon_footprint_kg"].plot(kind="hist", bins=30, ax=ax, color="#55a868")
    ax.set_title("Carbon Footprint Distribution (kg CO2e, estimated)")
    ax.set_xlabel("kg CO2e per unit")
    _save(fig, "04_carbon_footprint_distribution.png")


def plot_sustainability_feature_distributions(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df["recycled_material_percentage"].plot(kind="hist", bins=20, ax=axes[0], color="#8172b2")
    axes[0].set_title("Recycled Material %")
    df["recyclability_score"].plot(kind="hist", bins=20, ax=axes[1], color="#937860")
    axes[1].set_title("Recyclability Score")
    _save(fig, "05_sustainability_feature_distributions.png")


def plot_price_vs_sustainability(df):
    # Simple proxy sustainability indicator for this exploratory chart only:
    # average of recyclability_score and recycled_material_percentage/100.
    proxy = (df["recyclability_score"] + df["recycled_material_percentage"] / 100) / 2
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["price"], proxy, alpha=0.4, color="#4c72b0")
    ax.set_title("Price vs. Sustainability Proxy")
    ax.set_xlabel("Price (₹)")
    ax.set_ylabel("Sustainability proxy (0-1)")
    _save(fig, "06_price_vs_sustainability.png")


def plot_rating_vs_sustainability(df):
    proxy = (df["recyclability_score"] + df["recycled_material_percentage"] / 100) / 2
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["rating"], proxy, alpha=0.4, color="#dd8452")
    ax.set_title("Rating vs. Sustainability Proxy")
    ax.set_xlabel("Rating (0-5)")
    ax.set_ylabel("Sustainability proxy (0-1)")
    _save(fig, "07_rating_vs_sustainability.png")


def main():
    df = pd.read_csv(CLEAN_PRODUCTS_PATH)
    print(f"Running EDA on {len(df)} clean products...")

    plot_products_by_category(df)
    plot_price_distribution(df)
    plot_rating_distribution(df)
    plot_carbon_footprint_distribution(df)
    plot_sustainability_feature_distributions(df)
    plot_price_vs_sustainability(df)
    plot_rating_vs_sustainability(df)

    print(f"\nDone. {7} charts saved to {EDA_DIR}/")
    print("NOTE: the 'sustainability proxy' used in charts 06/07 is a simple "
          "illustrative average for EDA only — it is NOT the real sustainability "
          "score, which will be built properly in Milestone 3.")


if __name__ == "__main__":
    main()
