import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# LOAD FEATURE DATA
# ==============================
df = pd.read_csv("data/processed/final_features.csv")

# Separate features and target
X = df.drop(columns=["date", "load"])
y = df["load"]

# ==============================
# TIME-SERIES TRAIN TEST SPLIT
# ==============================
train_size = int(len(df) * 0.8)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

# ==============================
# DEFINE MODELS
# ==============================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        random_state=42
    ),
    "SVR": SVR(kernel="rbf")
}

# ==============================
# TRAIN & EVALUATE
# ==============================
results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2_Score": r2
    })

# ==============================
# SAVE RESULTS
# ==============================
results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)

print("\n✅ Model training & evaluation complete")
print(results_df)
