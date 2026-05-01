# 🎵 Music Era Discovery using PySpark

> Can a machine rediscover 100 years of music history using only audio mathematics — with no knowledge of culture, history, or genre?

This project processes **577,018 Spotify tracks spanning 1922 to 2020** through a fully distributed PySpark ML pipeline to discover hidden patterns in how music has evolved. The algorithm — given nothing but audio numbers — independently found the exact decade where music history changed forever.

---

## 🏆 The Main Finding

Without being told anything about music history, the algorithm identified a **clean cluster shift between the 1960s and 1970s** — the same boundary historians point to as the moment electric instruments, new recording technology, and genres like rock and soul transformed the global sound.

The machine found it from math alone.

| Decade | What the Algorithm Found |
|--------|--------------------------|
| 1920s – 1960s | 🔴 Acoustic / Vintage Style — dominant |
| **1970s – 2020s** | **⚡ Modern Pop & Dance — takes over and never lets go** |

---

## 📦 Dataset

[Spotify Dataset 1921–2020](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks) — Kaggle

- **577,018 songs** after cleaning · **11 decades** · **9 audio features per song**
- Features: `danceability` `energy` `loudness` `speechiness` `acousticness` `instrumentalness` `liveness` `valence` `tempo`

---

## 🛠️ Tech Stack

| | |
|---|---|
| Distributed Processing | PySpark 4.0.2 |
| Machine Learning | MLlib — VectorAssembler, StandardScaler, PCA, KMeans, ClusteringEvaluator |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab |

---

## 🔧 Pipeline
tracks.csv
→ PySpark Read CSV
→ Year extraction from release_date (regexp_extract)
→ Type casting with try_cast + dropna()
→ Decade column added (floor(year/10)*10)
→ VectorAssembler → 9-feature vector
→ StandardScaler → normalized features (μ=0, σ=1)
→ PCA (9D → 2D) for visualization only
→ KMeans tested K=3 to K=7 via Silhouette Score
→ Final model: K=4 (score: 0.3635)
→ Cluster analysis across 11 decades

---

## 📊 Results

### 4 Clusters Discovered

| Cluster | Name | Key Feature | Songs | % |
|---------|------|------------|-------|---|
| 0 | 🔴 Acoustic / Vintage Style | Acousticness = 0.71 | 187,742 | 32.5% |
| 1 | ⚡ Modern Pop & Dance | Energy = 0.72, Loudness = -7.3dB | 308,442 | 53.5% |
| 2 | 🎙️ Rap & Spoken Word | Speechiness = 0.852 | 26,514 | 4.6% |
| 3 | 🎻 Classical / Instrumental | Instrumentalness = 0.801 | 54,320 | 9.4% |

### Model Performance

| K | Silhouette Score |
|---|-----------------|
| 3 | 0.3393 |
| **4** | **0.3635 ← chosen** |
| 5 | 0.3140 |
| 6 | 0.2625 |
| 7 | 0.2827 |

### Validated by Real Songs

**🔴 Acoustic / Vintage Style**
Heartbreak Anniversary — Giveon · Someone You Loved — Lewis Capaldi

**⚡ Modern Pop & Dance**
Save Your Tears — The Weeknd · telepatía — Kali Uchis

**🎙️ Rap & Spoken Word**
Yes Indeed — Lil Baby & Drake · Carry On — XXXTENTACION

**🎻 Classical / Instrumental**
Experience — Ludovico Einaudi · everything i wanted — Billie Eilish

---

## ▶️ How to Run

### Google Colab (recommended)

1. Upload `Music_Era_Discovery.ipynb` to Colab
2. Download `tracks.csv` from [Kaggle](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks)
3. Upload the CSV when Cell 3 asks for it
4. Run all cells — PySpark 4.0.2 is pre-installed on Colab

### Local

```bash
git clone https://github.com/DaveRucha/music-era-discovery-pyspark.git
cd music-era-discovery-pyspark
pip install pyspark matplotlib seaborn pandas numpy
# Place tracks.csv in the project folder
jupyter notebook Music_Era_Discovery.ipynb
```

> 8GB RAM recommended for local execution.

---

## 📁 Structure
music-era-discovery-pyspark/
├── Music_Era_Discovery.ipynb       # Full notebook 
├── BDT_Project_Presentation.pptx  # Project slides
└── README.md

---

## 💡 What I Learned

- Real-world music data is messy — `release_date` comes in multiple formats, most columns load as strings, and about 9,000 rows had to be dropped before the pipeline was stable
- Silhouette scoring is far more reliable than the elbow method for heterogeneous data — K=4 was clear and consistent
- PCA for visualization vs PCA for clustering are two completely different decisions — clustering on 2D PCA features would have lost half the signal
- Unsupervised ML can surface historically meaningful patterns when the features are the right ones — the 1970s shift wasn't a lucky result, it shows up because acousticness and energy genuinely crossed over in that decade

---

**Rucha Dave** ·

*577,018 songs. Zero history labels. One clean answer.*

