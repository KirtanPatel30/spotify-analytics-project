# 🎵 Spotify Tracks Analytics Project

> An end-to-end Data Engineering · Data Analytics · Data Science project built on 114,000 Spotify tracks across 113 genres.

![Dashboard Preview](dashboard_genre.png)

---

## 📌 Project Overview

This project explores what makes a song popular on Spotify. Using a dataset of 114,000 tracks, I built a complete data pipeline — from raw CSV to a cleaned database, SQL-driven insights, a machine learning model, and an interactive dashboard.

The project was intentionally designed to cover **three roles in one**:

| Role | What I built |
|------|-------------|
| 🔧 Data Engineer | ETL pipeline that cleans, validates and loads data into SQLite |
| 📊 Data Analyst | SQL queries + 6 charts answering real business questions |
| 🤖 Data Scientist | Random Forest ML model predicting whether a song will be a hit |

---

## 🖥️ Dashboard Screenshots

### Genre Explorer
![Genre Explorer](dashboard_genre.png)
*Top 20 genres by average popularity + danceability vs popularity bubble chart*

### Audio Features Analysis
![Audio Features](dashboard_audio.png)
*Correlation heatmap + radar chart comparing Hit vs Non-Hit songs across 7 audio features*

### Top Artists Leaderboard
![Top Artists](dashboard_artists.png)
*Top 25 artists by track count with sortable data table*

### Indian vs Global Music
![Indian vs Global](dashboard_india.png)
*Unique angle — comparing Indian/South Asian genres against global genres on danceability and popularity*

---

## 📁 Project Structure

```
spotify_project/
│
├── data/                          ← Place dataset.csv here
│   ├── spotify.db                 ← Generated SQLite database
│   └── charts/                    ← Generated chart images
│
├── phase1_engineering/
│   └── etl_pipeline.py            ← Data cleaning + ETL pipeline
│
├── phase2_analytics/
│   └── analytics.py               ← SQL queries + chart generation
│
├── phase3_datascience/
│   └── ml_model.py                ← ML model (Random Forest + Logistic Regression)
│
├── dashboard/
│   └── dashboard.py               ← Interactive Plotly Dash app
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Source:** [Spotify Tracks Dataset — Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

- 114,000 tracks across 113 genres
- 21 columns including audio features, artist info and popularity scores
- Key columns: `popularity`, `danceability`, `energy`, `tempo`, `valence`, `acousticness`, `speechiness`, `instrumentalness`, `liveness`, `loudness`, `track_genre`, `explicit`

---

## 🔧 Phase 1 — Data Engineering (ETL Pipeline)

**File:** `phase1_engineering/etl_pipeline.py`

- Loads raw CSV (114,000 rows)
- Removes duplicate `track_id` entries
- Drops rows with nulls in critical columns
- Removes songs with 0ms or >15 minute duration (bad data)
- Clips audio features to valid ranges
- Adds derived columns: `duration_min`, `is_hit` (popularity ≥ 70)
- Loads clean data into **SQLite** with 3 tables: `tracks`, `genre_summary`, `artist_summary`
- Outputs a **data quality report** as JSON

```bash
python phase1_engineering/etl_pipeline.py
```

---

## 📈 Phase 2 — Data Analytics

**File:** `phase2_analytics/analytics.py`

Six business questions answered using **SQL + pandas + matplotlib/seaborn**:

1. Which genres have the highest average popularity?
2. Do explicit songs perform better than clean songs?
3. What audio features separate hit songs from non-hits?
4. Which artists have the most tracks in the dataset?
5. How do Indian/South Asian genres compare to global genres?
6. What does the overall popularity distribution look like?

All charts are saved automatically to `data/charts/`.

```bash
python phase2_analytics/analytics.py
```

---

## 🤖 Phase 3 — Data Science / ML Model

**File:** `phase3_datascience/ml_model.py`

Predicts whether a song will be a **hit (popularity ≥ 70)** using audio features.

**Features used:**
`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_min`, `explicit`, `key`, `mode`, `time_signature`

**Models trained:**
- Logistic Regression (baseline)
- Random Forest (main model, 200 estimators)

**Outputs:**
- Accuracy, Precision, Recall, F1, AUC score
- Confusion matrix chart
- Feature importance chart
- ROC curve comparison

```bash
python phase3_datascience/ml_model.py
```

---

## 🖥️ Interactive Dashboard

**File:** `dashboard/dashboard.py`

Built with **Plotly Dash** — runs in your browser with 5 interactive tabs:

| Tab | Content |
|-----|---------|
| 📊 Genre Explorer | Top genres bar chart + danceability vs popularity bubble chart |
| 🎸 Audio Features | Correlation heatmap + radar chart (hits vs non-hits) |
| 🏆 Top Artists | Leaderboard bar chart + sortable data table |
| 🇮🇳 Indian vs Global | South Asian genres vs global genres comparison |
| 📈 Hit Prediction | Hit/non-hit distribution + box plots by audio feature |

```bash
python dashboard/dashboard.py
# Open http://127.0.0.1:8050 in your browser
```

---

## 🚀 How to Run

**Step 1 — Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/spotify-analytics-project.git
cd spotify-analytics-project
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Download the dataset**

Download `dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) and place it in the `data/` folder.

**Step 4 — Run in order**
```bash
python phase1_engineering/etl_pipeline.py    # ETL pipeline
python phase2_analytics/analytics.py         # Analytics + charts
python phase3_datascience/ml_model.py        # ML model
python dashboard/dashboard.py                # Dashboard → http://127.0.0.1:8050
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat&logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly_Dash-5.x-3F4F75?style=flat&logo=plotly)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=flat&logo=sqlite)

- **Python** — core language
- **Pandas & NumPy** — data manipulation
- **SQLite** — database storage
- **Matplotlib & Seaborn** — static chart generation
- **Scikit-learn** — machine learning (Logistic Regression, Random Forest)
- **Plotly Dash** — interactive web dashboard

---

## 💡 Key Findings

- **Pop-film** is the highest popularity genre on average, followed by **chill** and **sad**
- **Indian** genre ranks in the top 5 for popularity globally
- Explicit songs have slightly **higher danceability and energy** than clean songs
- The top feature predicting a hit song is **loudness**, followed by **danceability** and **energy**
- Hit songs tend to be more **danceable and energetic** but less **acoustic and instrumental**
- **BTS** has one of the highest average popularity scores (67.89) among top artists

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

*Built as a portfolio project covering Data Engineering, Data Analytics and Data Science.*
