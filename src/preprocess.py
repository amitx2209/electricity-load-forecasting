import pandas as pd

# ==============================
# PROJECT CONFIGURATION
# ==============================
SELECTED_STATE = "Bihar"

# ==============================
# LOAD RAW DATA
# ==============================
df = pd.read_csv("data/raw/electricity_load.csv")

print("Original columns:")
print(df.columns.tolist())

# ==============================
# FIX DATE COLUMN (Unnamed: 0)
# ==============================
if "Unnamed: 0" not in df.columns:
    raise ValueError("Expected 'Unnamed: 0' column not found")

df = df.rename(columns={"Unnamed: 0": "date"})
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")


# ==============================
# EXTRACT SELECTED STATE
# ==============================
if SELECTED_STATE not in df.columns:
    raise ValueError(f"State '{SELECTED_STATE}' not found in dataset")

df = df[["date", SELECTED_STATE]]
df = df.rename(columns={SELECTED_STATE: "load"})

# ==============================
# SORT & CLEAN DATA
# ==============================
df = df.sort_values("date").reset_index(drop=True)

df["load"] = pd.to_numeric(df["load"], errors="coerce")
df["load"] = df["load"].fillna(method="ffill")

df = df.drop_duplicates(subset="date")

# ==============================
# SAVE CLEAN DATA
# ==============================
df.to_csv("data/processed/clean_data.csv", index=False)

print("✅ Preprocessing complete")
print("✅ State selected:", SELECTED_STATE)
print("✅ Clean data saved to data/processed/clean_data.csv")
print("Final columns:", df.columns.tolist())
