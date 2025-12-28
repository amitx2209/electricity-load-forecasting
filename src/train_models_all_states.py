import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# ===============================
# PATHS
# ===============================
FEATURE_DIR = "data/processed_features"
MODEL_DIR = "models"
RESULTS_PATH = "results/model_performance.csv"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

results = []

# ===============================
# TRAIN MODEL FOR EACH STATE
# ===============================
for file in os.listdir(FEATURE_DIR):
    if not file.endswith("_features.csv"):
        continue

    state_name = file.replace("_features.csv", "")
    file_path = os.path.join(FEATURE_DIR, file)

    df = pd.read_csv(file_path)

    # -------------------------------
    # Features and target
    # -------------------------------
    X = df.drop(columns=["date", "load"])
    y = df["load"]

    # -------------------------------
    # Time-based train-test split
    # -------------------------------
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # -------------------------------
    # Train model
    # -------------------------------
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # -------------------------------
    # Evaluate
    # -------------------------------
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # -------------------------------
    # Save model
    # -------------------------------
    model_path = os.path.join(MODEL_DIR, f"{state_name}_model.pkl")
    joblib.dump(model, model_path)

    results.append({
        "State": state_name,
        "RMSE": rmse
    })

    print(f"✅ Trained model for {state_name} | RMSE: {rmse:.2f}")

# ===============================
# SAVE PERFORMANCE SUMMARY
# ===============================
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_PATH, index=False)

print("🎯 Training completed for all states.")
print(f"📊 Performance summary saved to {RESULTS_PATH}")
