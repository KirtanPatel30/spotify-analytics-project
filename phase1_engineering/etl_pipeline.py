"""
Phase 1 - Data Engineering: ETL Pipeline
=========================================
This script:
1. Loads the raw Spotify dataset CSV
2. Cleans and validates the data
3. Generates a data quality report
4. Loads clean data into a SQLite database
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
CSV_PATH   = os.path.join(DATA_DIR, "dataset.csv")
DB_PATH    = os.path.join(DATA_DIR, "spotify.db")
REPORT_PATH = os.path.join(DATA_DIR, "data_quality_report.json")


# ── Step 1: Extract ────────────────────────────────────────────────────────────
def extract(path):
    print("\n📥 [EXTRACT] Loading raw CSV...")
    df = pd.read_csv(path)
    print(f"   Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    return df


# ── Step 2: Transform / Clean ──────────────────────────────────────────────────
def transform(df):
    print("\n🔧 [TRANSFORM] Cleaning data...")
    report = {"start_rows": len(df), "issues_fixed": []}

    # Drop unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
        report["issues_fixed"].append("Dropped unnamed index column")

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(subset=["track_id"], inplace=True)
    dupes = before - len(df)
    if dupes:
        report["issues_fixed"].append(f"Removed {dupes} duplicate track_ids")
    print(f"   Duplicates removed: {dupes}")

    # Drop rows with null values in critical columns
    critical_cols = ["track_name", "artists", "track_genre", "popularity"]
    before = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    nulls = before - len(df)
    if nulls:
        report["issues_fixed"].append(f"Removed {nulls} rows with null critical fields")
    print(f"   Null rows removed: {nulls}")

    # Remove songs with 0 duration (clearly bad data)
    before = len(df)
    df = df[df["duration_ms"] > 0]
    bad_dur = before - len(df)
    if bad_dur:
        report["issues_fixed"].append(f"Removed {bad_dur} songs with 0ms duration")
    print(f"   Zero-duration rows removed: {bad_dur}")

    # Remove extreme outliers — songs longer than 15 minutes (900000 ms)
    before = len(df)
    df = df[df["duration_ms"] <= 900000]
    long_songs = before - len(df)
    if long_songs:
        report["issues_fixed"].append(f"Removed {long_songs} songs longer than 15 minutes")
    print(f"   Extreme-length songs removed: {long_songs}")

    # Clip audio features to valid [0, 1] range
    features = ["danceability", "energy", "speechiness",
                "acousticness", "instrumentalness", "liveness", "valence"]
    for col in features:
        df[col] = df[col].clip(0, 1)

    # Clip popularity to [0, 100]
    df["popularity"] = df["popularity"].clip(0, 100)

    # Add derived columns
    df["duration_min"]   = (df["duration_ms"] / 60000).round(2)
    df["is_hit"]         = (df["popularity"] >= 70).astype(int)   # 1 = hit, 0 = not
    df["explicit"]       = df["explicit"].astype(int)

    # Standardise genre to lowercase
    df["track_genre"] = df["track_genre"].str.lower().str.strip()

    # Reset index
    df.reset_index(drop=True, inplace=True)

    report["end_rows"]      = len(df)
    report["rows_removed"]  = report["start_rows"] - report["end_rows"]
    report["columns"]       = list(df.columns)
    report["timestamp"]     = datetime.now().isoformat()

    print(f"\n   ✅ Clean dataset: {len(df):,} rows  |  {len(df.columns)} columns")
    return df, report


# ── Step 3: Load into SQLite ───────────────────────────────────────────────────
def load(df, db_path):
    print(f"\n💾 [LOAD] Writing to SQLite → {db_path}")
    conn = sqlite3.connect(db_path)

    # Main tracks table
    df.to_sql("tracks", conn, if_exists="replace", index=False)

    # Genre summary table
    genre_summary = (
        df.groupby("track_genre")
        .agg(
            total_tracks   = ("track_id", "count"),
            avg_popularity = ("popularity", "mean"),
            avg_danceability = ("danceability", "mean"),
            avg_energy     = ("energy", "mean"),
            hit_rate       = ("is_hit", "mean"),
        )
        .round(3)
        .reset_index()
    )
    genre_summary.to_sql("genre_summary", conn, if_exists="replace", index=False)

    # Artist summary table
    artist_summary = (
        df.groupby("artists")
        .agg(
            total_tracks   = ("track_id", "count"),
            avg_popularity = ("popularity", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("total_tracks", ascending=False)
        .head(500)
    )
    artist_summary.to_sql("artist_summary", conn, if_exists="replace", index=False)

    conn.close()
    print("   ✅ Tables created: tracks, genre_summary, artist_summary")


# ── Step 4: Save quality report ────────────────────────────────────────────────
def save_report(report, path):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📋 Data quality report saved → {path}")
    print(json.dumps({k: v for k, v in report.items() if k != "columns"}, indent=2))


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  PHASE 1 — ETL PIPELINE")
    print("=" * 55)

    if not os.path.exists(CSV_PATH):
        print(f"\n❌ dataset.csv not found at: {CSV_PATH}")
        print("   Please download it from Kaggle and place it in the data/ folder.")
        exit(1)

    raw_df          = extract(CSV_PATH)
    clean_df, report = transform(raw_df)
    load(clean_df, DB_PATH)
    save_report(report, REPORT_PATH)

    print("\n✅ Phase 1 complete! Run phase2_analytics/analytics.py next.")
    print("=" * 55)
