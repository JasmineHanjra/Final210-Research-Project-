# 🎵 Spotify Popularity Prediction & Analysis

This project explores a dataset of 6,300 Spotify tracks and builds models to understand and predict the popularity of songs based on features like genre, duration, and explicit content.

## 📁 Dataset Overview

- **File:** `spotify_tracks.csv`
- **Entries:** 6,300 songs
- **Features:** genre, artists, album, popularity score, duration, explicit content flag

## 🔍 Analysis Highlights

- Converted track duration from milliseconds to minutes.
- Visualized:
  - Genre vs. Popularity
  - Explicit vs. Non-explicit track popularity
  - Top 10 artists and albums by average popularity
  - Popularity distribution
- Removed outliers in popularity for cleaner modeling.
- Encoded genres using one-hot encoding for ML models.

## 📊 Model Performance

Three regression models were compared to predict song popularity:

| Model               | MSE   | MAE   | R² Score |
|--------------------|-------|-------|----------|
| Linear Regression  | 260.7 | 12.95 | 0.308    |
| Random Forest      | 297.8 | 13.33 | 0.209    |
| Gradient Boosting  | 286.9 | 13.80 | 0.238    |
| 🔍 Best RF (Tuned) | 275.4 | -     | -        |

- **Best Random Forest Params:**  
  `max_depth=30`, `min_samples_split=10`, `n_estimators=300`

## 🧠 Key Insights from Coefficients

Genres with highest positive impact on popularity:
- `dance`, `pop`, `rock`, `country`, `party`, `summer`, `soul`

Genres with highest negative impact:
- `breakbeat`, `metal-misc`, `progressive-house`, `grindcore`

## 🧪 Example Prediction

```python
predict_popularity(3.5, 'Pop', spotify_data, model_scaled)
# Output: Predicted Popularity ≈ 29.04
