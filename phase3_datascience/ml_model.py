"""
Phase 3 - Data Science: ML Model
==================================
Predicts whether a song will be a HIT (popularity >= 70)
using audio features.

Models used:
  - Logistic Regression (baseline)
  - Random Forest (main model)

Output:
  - Accuracy, Precision, Recall, F1
  - Confusion matrix chart
  - Feature importance chart
  - Model comparison chart
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
DB_PATH    = os.path.join(DATA_DIR, "spotify.db")
CHARTS_DIR = os.path.join(DATA_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

sns.set_style("whitegrid")

FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_min", "explicit", "key", "mode", "time_signature"
]
TARGET = "is_hit"


# ── Load data ──────────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(DB_PATH):
        print("❌ spotify.db not found. Run phase1_engineering/etl_pipeline.py first.")
        exit(1)
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT {', '.join(FEATURES + [TARGET])} FROM tracks", conn)
    conn.close()
    df.dropna(inplace=True)
    print(f"   Loaded {len(df):,} rows for modelling")
    print(f"   Hit rate: {df[TARGET].mean()*100:.1f}% of songs are hits")
    return df


# ── Train models ───────────────────────────────────────────────────────────────
def train_models(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # ── Logistic Regression (baseline) ────────────────────────────────────────
    print("\n🔵 Training Logistic Regression (baseline)...")
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_prob = lr.predict_proba(X_test_sc)[:, 1]
    results["Logistic Regression"] = {
        "model": lr,
        "pred":  lr_pred,
        "prob":  lr_prob,
        "acc":   accuracy_score(y_test, lr_pred),
        "auc":   roc_auc_score(y_test, lr_prob),
        "report": classification_report(y_test, lr_pred, output_dict=True),
        "cm":    confusion_matrix(y_test, lr_pred),
    }
    print(f"   Accuracy: {results['Logistic Regression']['acc']*100:.2f}%  |  AUC: {results['Logistic Regression']['auc']:.3f}")

    # ── Random Forest (main model) ─────────────────────────────────────────────
    print("\n🌲 Training Random Forest (main model)... (may take ~30 seconds)")
    rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                 min_samples_split=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    results["Random Forest"] = {
        "model": rf,
        "pred":  rf_pred,
        "prob":  rf_prob,
        "acc":   accuracy_score(y_test, rf_pred),
        "auc":   roc_auc_score(y_test, rf_prob),
        "report": classification_report(y_test, rf_pred, output_dict=True),
        "cm":    confusion_matrix(y_test, rf_pred),
        "feature_importance": pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False),
    }
    print(f"   Accuracy: {results['Random Forest']['acc']*100:.2f}%  |  AUC: {results['Random Forest']['auc']:.3f}")

    return results, X_test, y_test


# ── Plot confusion matrix ──────────────────────────────────────────────────────
def plot_confusion_matrix(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, res) in zip(axes, results.items()):
        cm = res["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Not Hit", "Hit"], yticklabels=["Not Hit", "Hit"])
        acc = res["acc"] * 100
        ax.set_title(f"{name}\nAccuracy: {acc:.2f}%", fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    fig.suptitle("Confusion Matrices — Hit Song Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "ml_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Confusion matrix saved → {path}")


# ── Plot feature importance ────────────────────────────────────────────────────
def plot_feature_importance(results):
    fi = results["Random Forest"]["feature_importance"]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(fi))
    bars = ax.barh(fi.index, fi.values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance — Random Forest Model\n(What makes a song a hit?)",
                 fontsize=13, fontweight="bold")
    for bar, val in zip(bars, fi.values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "ml_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Feature importance chart saved → {path}")


# ── Plot ROC curves ────────────────────────────────────────────────────────────
def plot_roc(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#457b9d", "#e63946"]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Hit Song Prediction", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "ml_roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   ROC curve saved → {path}")


# ── Print summary ──────────────────────────────────────────────────────────────
def print_summary(results):
    print("\n" + "=" * 55)
    print("  MODEL RESULTS SUMMARY")
    print("=" * 55)
    for name, res in results.items():
        r = res["report"]
        print(f"\n  {name}")
        print(f"    Accuracy  : {res['acc']*100:.2f}%")
        print(f"    AUC Score : {res['auc']:.3f}")
        print(f"    Precision : {r['weighted avg']['precision']:.3f}")
        print(f"    Recall    : {r['weighted avg']['recall']:.3f}")
        print(f"    F1 Score  : {r['weighted avg']['f1-score']:.3f}")

    print("\n  Top 5 most important features (Random Forest):")
    fi = results["Random Forest"]["feature_importance"]
    for feat, score in fi.head(5).items():
        print(f"    {feat:<25} {score:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  PHASE 3 — DATA SCIENCE / ML MODEL")
    print("=" * 55)

    print("\n📥 Loading data...")
    df = load_data()

    print("\n🤖 Training models...")
    results, X_test, y_test = train_models(df)

    print("\n📊 Generating charts...")
    plot_confusion_matrix(results)
    plot_feature_importance(results)
    plot_roc(results, y_test)

    print_summary(results)

    print("\n✅ Phase 3 complete! Run dashboard/dashboard.py for the interactive view.")
    print("=" * 55)
