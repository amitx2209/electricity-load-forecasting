import pandas as pd
import os

# ===============================
# PATHS
# ===============================
RAW_DATA_PATH = "data/raw/electricity_load.csv"
OUTPUT_DIR = "data/processed"

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# LOAD RAW DATA
# ===============================
df = pd.read_csv(RAW_DATA_PATH)

# Rename first column to date if needed
if df.columns[0] != "date":
    df.rename(columns={df.columns[0]: "date"}, inplace=True)

# Convert date column
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# ===============================
# IDENTIFY STATE COLUMNS
# ===============================
state_columns = [col for col in df.columns if col != "date"]

print(f"Found {len(state_columns)} states.")

# ===============================
# PROCESS EACH STATE
# ===============================
for state in state_columns:
    state_df = df[["date", state]].copy()
    state_df.rename(columns={state: "load"}, inplace=True)

    # Drop missing values
    state_df.dropna(inplace=True)

    # Sort by date
    state_df = state_df.sort_values("date")

    # Save per-state file
    output_path = os.path.join(OUTPUT_DIR, f"{state}_data.csv")
    state_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

print("✅ Preprocessing completed for all states.")
