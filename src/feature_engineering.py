import pandas as pd

# ==============================
# LOAD CLEAN DATA
# ==============================
df = pd.read_csv("data/processed/clean_data.csv")

# Convert date column to datetime (safety)
df["date"] = pd.to_datetime(df["date"])

# ==============================
# SORT BY DATE (MANDATORY)
# ==============================
df = df.sort_values("date").reset_index(drop=True)

# ==============================
# CREATE LAG FEATURES
# ==============================
df["lag_1"] = df["load"].shift(1)
df["lag_7"] = df["load"].shift(7)

# ==============================
# CREATE ROLLING FEATURES
# ==============================
df["rolling_mean_7"] = df["load"].rolling(window=7).mean()

# ==============================
# CREATE CALENDAR FEATURES
# ==============================
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek

# ==============================
# DROP ROWS WITH NaN (FROM LAGS)
# ==============================
df = df.dropna().reset_index(drop=True)

# ==============================
# SAVE FEATURED DATA
# ==============================
df.to_csv("data/processed/final_features.csv", index=False)

print("✅ Feature engineering complete")
print("✅ Final feature dataset saved to data/processed/final_features.csv")
print("Final columns:", df.columns.tolist())
