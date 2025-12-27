import streamlit as st
import pandas as pd
import joblib

# ==============================
# LOAD MODEL & DATA
# ==============================
model = joblib.load("models/best_model.pkl")
df = pd.read_csv("data/processed/final_features.csv")

st.set_page_config(page_title="Electricity Load Forecasting", layout="centered")

st.title("⚡ Electricity Load Forecasting")
st.write("Predict next-day electricity demand using machine learning.")

# ==============================
# PREPARE INPUT
# ==============================
latest_features = df.drop(columns=["date", "load"]).iloc[-1:]

# ==============================
# PREDICTION
# ==============================
if st.button("Predict Next Day Load"):
    prediction = model.predict(latest_features)[0]
    st.success(f"🔮 Predicted Electricity Load: **{prediction:.2f}**")
