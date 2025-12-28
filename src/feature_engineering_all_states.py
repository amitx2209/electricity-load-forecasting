import pandas as pd
import os

# ===============================
# PATHS
# ===============================
INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/processed_features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# STATE NAME NORMALIZATION
# ===============================
STATE_NAME_MAPPING = {
    "UP": "Uttar Pradesh",
    "MP": "Madhya Pradesh",
    "HP": "Himachal Pradesh",
    "J&K": "Jammu and Kashmir",
    "DNH": "Dadra and Nagar Haveli",
    "Pondy": "Puducherry",
    "WB": "West Bengal",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "UK": "Uttarakhand",
}

# ===============================
# FEATURE ENGINEERING FUNCTION
# ===============================
def create_features(df):
    df = df.copy()
    df.sort_values("date", inplace=True)

    # Lag features
    df["lag_1"] = df["load"].shift(1)
    df["lag_7"] = df["load"].shift(7)

    # Rolling features
    df["rolling_mean_7"] = df["load"].rolling(window=7).mean()

    # Calendar features
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday

    # Drop rows with NaNs created by lags
    df.dropna(inplace=True)

    return df

# ===============================
# PROCESS EACH STATE FILE
# ===============================
for file in os.listdir(INPUT_DIR):
    if not file.endswith("_data.csv"):
        continue

    state_raw_name = file.replace("_data.csv", "")
    state_name = STATE_NAME_MAPPING.get(state_raw_name, state_raw_name)

    file_path = os.path.join(INPUT_DIR, file)
    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"])

    feature_df = create_features(df)

    output_path = os.path.join(
        OUTPUT_DIR, f"{state_name}_features.csv"
    )

    feature_df.to_csv(output_path, index=False)

    print(f"Processed features for: {state_name}")

print("✅ Feature engineering completed for all states.")
