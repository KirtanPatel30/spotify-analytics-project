"""
Phase 2 - Data Analytics
=========================
Answers 6 key business questions using SQL + pandas.
Generates and saves charts to the data/ folder.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
DB_PATH     = os.path.join(DATA_DIR, "spotify.db")
CHARTS_DIR  = os.path.join(DATA_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

sns.set_style("whitegrid")
PALETTE = "mako"


def get_conn():
    if not os.path.exists(DB_PATH):
        print("❌ spotify.db not found. Run phase1_engineering/etl_pipeline.py first.")
        exit(1)
    return sqlite3.connect(DB_PATH)


# ── Q1: Top genres by average popularity ──────────────────────────────────────
def q1_top_genres(conn):
    print("\n📊 Q1: Top 15 genres by average popularity")
    query = """
        SELECT track_genre, 
               ROUND(AVG(popularity), 2) AS avg_popularity,
               COUNT(*) AS total_tracks
        FROM tracks
        GROUP BY track_genre
        HAVING total_tracks >= 100
        ORDER BY avg_popularity DESC
        LIMIT 15
    """
    df = pd.read_sql(query, conn)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(df["track_genre"], df["avg_popularity"], color=sns.color_palette(PALETTE, len(df)))
    ax.set_xlabel("Average Popularity Score")
    ax.set_title("Top 15 Genres by Average Popularity", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, df["avg_popularity"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "q1_top_genres.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Chart saved → {path}")
    return df


# ── Q2: Explicit vs non-explicit popularity ────────────────────────────────────
def q2_explicit_vs_clean(conn):
    print("\n📊 Q2: Do explicit songs perform better?")
    query = """
        SELECT explicit,
               ROUND(AVG(popularity), 2) AS avg_popularity,
               COUNT(*) AS total_tracks,
               ROUND(AVG(danceability), 3) AS avg_danceability,
               ROUND(AVG(energy), 3) AS avg_energy
        FROM tracks
        GROUP BY explicit
    """
    df = pd.read_sql(query, conn)
    df["label"] = df["explicit"].map({0: "Clean", 1: "Explicit"})
    print(df[["label", "avg_popularity", "total_tracks", "avg_danceability", "avg_energy"]].to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = [("avg_popularity", "Avg Popularity"), ("avg_danceability", "Avg Danceability"), ("avg_energy", "Avg Energy")]
    colors = ["#2d6a4f", "#e63946"]
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(df["label"], df[col], color=colors, width=0.5)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, df[col].max() * 1.2)
        for i, v in enumerate(df[col]):
            ax.text(i, v + df[col].max() * 0.02, f"{v:.2f}", ha="center", fontsize=10)
    fig.suptitle("Explicit vs Clean Songs — Key Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "q2_explicit_vs_clean.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Chart saved → {path}")
    return df


# ── Q3: Audio features of hits vs non-hits ─────────────────────────────────────
def q3_hits_vs_nonhits(conn):
    print("\n📊 Q3: What audio features make a hit song?")
    query = """
        SELECT is_hit,
               ROUND(AVG(danceability), 3) AS danceability,
               ROUND(AVG(energy), 3) AS energy,
               ROUND(AVG(valence), 3) AS valence,
               ROUND(AVG(acousticness), 3) AS acousticness,
               ROUND(AVG(speechiness), 3) AS speechiness,
               ROUND(AVG(instrumentalness), 3) AS instrumentalness,
               ROUND(AVG(liveness), 3) AS liveness
        FROM tracks
        GROUP BY is_hit
    """
    df = pd.read_sql(query, conn)
    df["label"] = df["is_hit"].map({0: "Non-Hit (popularity < 70)", 1: "Hit (popularity ≥ 70)"})
    features = ["danceability", "energy", "valence", "acousticness", "speechiness", "instrumentalness", "liveness"]
    print(df[["label"] + features].to_string(index=False))

    melted = df.melt(id_vars="label", value_vars=features, var_name="Feature", value_name="Score")
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=melted, x="Feature", y="Score", hue="label",
                palette=["#457b9d", "#e63946"], ax=ax)
    ax.set_title("Audio Features: Hit Songs vs Non-Hit Songs", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.legend(title="")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "q3_hits_vs_nonhits.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Chart saved → {path}")
    return df


# ── Q4: Top artists by track count ────────────────────────────────────────────
def q4_top_artists(conn):
    print("\n📊 Q4: Which artists have the most tracks?")
    query = """
        SELECT artists,
               COUNT(*) AS total_tracks,
               ROUND(AVG(popularity), 1) AS avg_popularity
        FROM tracks
        GROUP BY artists
        ORDER BY total_tracks DESC
        LIMIT 15
    """
    df = pd.read_sql(query, conn)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("rocket", len(df))
    bars = ax.barh(df["artists"], df["total_tracks"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Tracks")
    ax.set_title("Top 15 Artists by Track Count", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, df["total_tracks"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "q4_top_artists.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Chart saved → {path}")
    return df


# ── Q5: Indian/Punjabi genres vs global comparison ────────────────────────────
def q5_indian_vs_global(conn):
    print("\n📊 Q5: How do Indian genres compare to global genres?")
    indian_genres = ("indian", "punjabi", "desi", "bollywood", "bhangra")
    query = f"""
        SELECT track_genre,
               ROUND(AVG(popularity), 2) AS avg_popularity,
               ROUND(AVG(danceability), 3) AS avg_danceability,
               ROUND(AVG(energy), 3) AS avg_energy,
               ROUND(AVG(valence), 3) AS avg_valence,
               COUNT(*) AS total_tracks
        FROM tracks
        GROUP BY track_genre
        HAVING total_tracks >= 50
        ORDER BY avg_popularity DESC
    """
    all_df = pd.read_sql(query, conn)
    all_df["region"] = all_df["track_genre"].apply(
        lambda g: "Indian/South Asian" if any(k in g for k in indian_genres) else "Global"
    )

    top_global  = all_df[all_df["region"] == "Global"].head(10)
    indian      = all_df[all_df["region"] == "Indian/South Asian"]
    compare_df  = pd.concat([top_global, indian]).drop_duplicates()

    print("\n  Indian/South Asian genres found:")
    print(indian[["track_genre", "avg_popularity", "total_tracks"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors_map = {"Indian/South Asian": "#e9c46a", "Global": "#264653"}
    for ax, metric, label in zip(axes,
                                  ["avg_popularity", "avg_danceability"],
                                  ["Average Popularity", "Average Danceability"]):
        data = compare_df.sort_values(metric, ascending=True).tail(20)
        bar_colors = [colors_map[r] for r in data["region"]]
        ax.barh(data["track_genre"], data[metric], color=bar_colors)
        ax.set_xlabel(label)
        ax.set_title(f"{label} by Genre", fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#e9c46a", label="Indian/South Asian"),
                       Patch(facecolor="#264653", label="Global")]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Indian/South Asian Genres vs Global Genres", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "q5_indian_vs_global.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Chart saved → {path}")
    return compare_df


# ── Q6: Popularity distribution ───────────────────────────────────────────────
def q6_popularity_distribution(conn):
    print("\n📊 Q6: Popularity distribution overview")
    df = pd.read_sql("SELECT popularity FROM tracks", conn)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(df["popularity"], bins=50, color="#457b9d", edgecolor="white")
    axes[0].set_title("Popularity Distribution", fontweight="bold")
    axes[0].set_xlabel("Popularity Score")
    axes[0].set_ylabel("Number of Songs")

    bins   = [0, 20, 40, 60, 70, 80, 100]
    labels = ["0-20", "21-40", "41-60", "61-70", "71-80", "81-100"]
    df["pop_bucket"] = pd.cut(df["popularity"], bins=bins, labels=labels, include_lowest=True)
    bucket_counts = df["pop_bucket"].value_counts().sort_index()
    axes[1].bar(bucket_counts.index.astype(str), bucket_counts.values,
                color=sns.color_palette("Blues_d", len(bucket_counts)))
    axes[1].set_title("Songs per Popularity Bucket", fontweight="bold")
    axes[1].set_xlabel("Popularity Range")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "q6_popularity_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Chart saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  PHASE 2 — DATA ANALYTICS")
    print("=" * 55)

    conn = get_conn()

    q1_top_genres(conn)
    q2_explicit_vs_clean(conn)
    q3_hits_vs_nonhits(conn)
    q4_top_artists(conn)
    q5_indian_vs_global(conn)
    q6_popularity_distribution(conn)

    conn.close()
    print("\n✅ Phase 2 complete! Charts saved to data/charts/")
    print("   Run phase3_datascience/ml_model.py next.")
    print("=" * 55)
