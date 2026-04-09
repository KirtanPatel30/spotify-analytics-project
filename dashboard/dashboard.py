"""
Dashboard - Interactive Plotly Dash App
========================================
Run this file and open http://127.0.0.1:8050 in your browser.

Features:
  - Genre popularity explorer
  - Audio features radar chart
  - Hit vs Non-hit comparison
  - Top artists leaderboard
  - Indian vs Global music comparison
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, dash_table

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(DATA_DIR, "spotify.db")

if not os.path.exists(DB_PATH):
    print("❌ spotify.db not found. Run phase1_engineering/etl_pipeline.py first.")
    exit(1)

# ── Load data ──────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql("SELECT * FROM tracks", conn)
conn.close()
print(f"✅ Loaded {len(df):,} tracks for dashboard")

# Pre-compute genre stats
genre_stats = (
    df.groupby("track_genre")
    .agg(
        avg_popularity   = ("popularity", "mean"),
        avg_danceability = ("danceability", "mean"),
        avg_energy       = ("energy", "mean"),
        avg_valence      = ("valence", "mean"),
        avg_acousticness = ("acousticness", "mean"),
        total_tracks     = ("track_id", "count"),
        hit_rate         = ("is_hit", "mean"),
    )
    .round(3)
    .reset_index()
    .query("total_tracks >= 50")
    .sort_values("avg_popularity", ascending=False)
)

all_genres = sorted(df["track_genre"].unique().tolist())
audio_features = ["danceability", "energy", "valence", "acousticness",
                  "speechiness", "instrumentalness", "liveness"]

# ── KPI helper ─────────────────────────────────────────────────────────────────
def kpi_card(label, value):
    return html.Div([
        html.P(label, style={"margin": "0 0 4px", "fontSize": "12px", "color": "#aaa"}),
        html.H3(value, style={"margin": "0", "color": "#1DB954", "fontSize": "22px"}),
    ], style={"background": "#282828", "padding": "16px 20px",
              "borderRadius": "8px", "minWidth": "130px"})


# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Spotify Analytics Dashboard")

app.layout = html.Div([

    # ── Header ─────────────────────────────────────────────────────────────────
    html.Div([
        html.H1("🎵 Spotify Tracks Analytics", style={"margin": "0", "color": "#1DB954"}),
        html.P(f"Exploring {len(df):,} songs across {df['track_genre'].nunique()} genres",
               style={"color": "#aaa", "margin": "4px 0 0"}),
    ], style={"background": "#191414", "padding": "24px 32px", "borderBottom": "1px solid #333"}),

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    html.Div([
        kpi_card("Total Songs",    f"{len(df):,}"),
        kpi_card("Genres",         f"{df['track_genre'].nunique():,}"),
        kpi_card("Artists",        f"{df['artists'].nunique():,}"),
        kpi_card("Avg Popularity", f"{df['popularity'].mean():.1f}/100"),
        kpi_card("Hit Songs (≥70)",f"{df['is_hit'].sum():,}"),
        kpi_card("Hit Rate",       f"{df['is_hit'].mean()*100:.1f}%"),
    ], style={"display": "flex", "gap": "16px", "padding": "20px 32px",
              "background": "#191414", "flexWrap": "wrap"}),

    # ── Tab content ────────────────────────────────────────────────────────────
    html.Div([
        dcc.Tabs(id="tabs", value="genres", children=[
            dcc.Tab(label="📊 Genre Explorer",    value="genres"),
            dcc.Tab(label="🎸 Audio Features",    value="audio"),
            dcc.Tab(label="🏆 Top Artists",       value="artists"),
            dcc.Tab(label="🇮🇳 Indian vs Global", value="india"),
            dcc.Tab(label="📈 Hit Prediction",    value="hits"),
        ], style={"fontFamily": "sans-serif"}),
        html.Div(id="tab-content", style={"padding": "24px 32px"}),
    ], style={"background": "#fff", "minHeight": "600px"}),

], style={"fontFamily": "'Segoe UI', sans-serif", "background": "#f5f5f5"})


# ── Tab routing ────────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "genres":   return tab_genres()
    if tab == "audio":    return tab_audio()
    if tab == "artists":  return tab_artists()
    if tab == "india":    return tab_india()
    if tab == "hits":     return tab_hits()


# ── Tab 1: Genre Explorer ──────────────────────────────────────────────────────
def tab_genres():
    top20 = genre_stats.head(20)
    fig1 = px.bar(top20, x="avg_popularity", y="track_genre",
                  orientation="h", color="avg_popularity",
                  color_continuous_scale="Greens",
                  title="Top 20 Genres by Average Popularity",
                  labels={"track_genre": "Genre", "avg_popularity": "Avg Popularity"})
    fig1.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)

    fig2 = px.scatter(genre_stats, x="avg_danceability", y="avg_popularity",
                      size="total_tracks", color="avg_energy",
                      hover_name="track_genre", color_continuous_scale="Viridis",
                      title="Genre: Danceability vs Popularity (size = track count, color = energy)",
                      labels={"avg_danceability": "Avg Danceability", "avg_popularity": "Avg Popularity"})

    return html.Div([
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
    ])


# ── Tab 2: Audio Features ──────────────────────────────────────────────────────
def tab_audio():
    # Correlation heatmap
    corr_cols = audio_features + ["popularity", "tempo", "loudness"]
    corr = df[corr_cols].corr().round(2)
    fig1 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                     title="Correlation Heatmap — Audio Features vs Popularity",
                     zmin=-1, zmax=1)

    # Radar chart — hits vs non-hits
    hits    = df[df["is_hit"] == 1][audio_features].mean()
    nonhits = df[df["is_hit"] == 0][audio_features].mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(r=hits.values.tolist() + [hits.values[0]],
                                   theta=audio_features + [audio_features[0]],
                                   fill="toself", name="Hit Songs", line_color="#1DB954"))
    fig2.add_trace(go.Scatterpolar(r=nonhits.values.tolist() + [nonhits.values[0]],
                                   theta=audio_features + [audio_features[0]],
                                   fill="toself", name="Non-Hit Songs", line_color="#e63946"))
    fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                       title="Audio Feature Radar: Hit vs Non-Hit Songs")

    return html.Div([
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
    ])


# ── Tab 3: Top Artists ─────────────────────────────────────────────────────────
def tab_artists():
    top_artists = (
        df.groupby("artists")
        .agg(total_tracks=("track_id", "count"), avg_popularity=("popularity", "mean"))
        .round(2)
        .reset_index()
        .sort_values("total_tracks", ascending=False)
        .head(25)
    )
    fig = px.bar(top_artists, x="total_tracks", y="artists", orientation="h",
                 color="avg_popularity", color_continuous_scale="Teal",
                 title="Top 25 Artists by Number of Tracks (color = avg popularity)",
                 labels={"artists": "Artist", "total_tracks": "Tracks",
                         "avg_popularity": "Avg Popularity"})
    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    table = dash_table.DataTable(
        data=top_artists.to_dict("records"),
        columns=[{"name": c, "id": c} for c in top_artists.columns],
        style_cell={"textAlign": "left", "padding": "8px"},
        style_header={"fontWeight": "bold", "background": "#f0f0f0"},
        page_size=15,
        sort_action="native",
    )
    return html.Div([dcc.Graph(figure=fig), html.Br(), table])


# ── Tab 4: Indian vs Global ────────────────────────────────────────────────────
def tab_india():
    indian_kw = ("indian", "punjabi", "desi", "bollywood", "bhangra")
    genre_stats["region"] = genre_stats["track_genre"].apply(
        lambda g: "Indian/South Asian" if any(k in g for k in indian_kw) else "Global"
    )
    top_global = genre_stats[genre_stats["region"] == "Global"].head(15)
    indian     = genre_stats[genre_stats["region"] == "Indian/South Asian"]
    compare    = pd.concat([top_global, indian]).drop_duplicates()

    fig = px.scatter(compare, x="avg_danceability", y="avg_popularity",
                     color="region", size="total_tracks", hover_name="track_genre",
                     color_discrete_map={"Indian/South Asian": "#e9c46a", "Global": "#264653"},
                     title="Indian/South Asian Genres vs Global — Danceability vs Popularity",
                     labels={"avg_danceability": "Avg Danceability",
                             "avg_popularity": "Avg Popularity"})

    metrics = ["avg_popularity", "avg_danceability", "avg_energy", "avg_valence"]
    region_avg = compare.groupby("region")[metrics].mean().round(3).reset_index()
    melted = region_avg.melt(id_vars="region", var_name="metric", value_name="score")
    fig2 = px.bar(melted, x="metric", y="score", color="region", barmode="group",
                  color_discrete_map={"Indian/South Asian": "#e9c46a", "Global": "#264653"},
                  title="Average Metrics: Indian/South Asian vs Global Genres",
                  labels={"metric": "Metric", "score": "Score"})

    return html.Div([dcc.Graph(figure=fig), dcc.Graph(figure=fig2)])


# ── Tab 5: Hit Prediction Insights ────────────────────────────────────────────
def tab_hits():
    hit_dist = df["is_hit"].value_counts().reset_index()
    hit_dist.columns = ["is_hit", "count"]
    hit_dist["label"] = hit_dist["is_hit"].map({0: "Not a Hit", 1: "Hit (≥70)"})
    fig1 = px.pie(hit_dist, values="count", names="label",
                  color_discrete_sequence=["#457b9d", "#1DB954"],
                  title="Hit vs Non-Hit Song Distribution")

    fig2 = px.box(df, x="is_hit", y="popularity", color="is_hit",
                  color_discrete_map={0: "#457b9d", 1: "#1DB954"},
                  title="Popularity Distribution: Hits vs Non-Hits",
                  labels={"is_hit": "Is Hit", "popularity": "Popularity Score"})

    # Feature boxplots for top 4 features
    figs = []
    for feat in ["danceability", "energy", "valence", "loudness"]:
        f = px.box(df, x="is_hit", y=feat, color="is_hit",
                   color_discrete_map={0: "#457b9d", 1: "#1DB954"},
                   title=f"{feat.capitalize()}: Hit vs Non-Hit",
                   labels={"is_hit": "Is Hit"})
        figs.append(dcc.Graph(figure=f, style={"width": "48%", "display": "inline-block"}))

    return html.Div([
        dcc.Graph(figure=fig1, style={"width": "48%", "display": "inline-block"}),
        dcc.Graph(figure=fig2, style={"width": "48%", "display": "inline-block"}),
        html.Div(figs),
    ])


if __name__ == "__main__":
    print("\n🚀 Starting dashboard at http://127.0.0.1:8050")
    print("   Press Ctrl+C to stop.\n")
    app.run(debug=False)