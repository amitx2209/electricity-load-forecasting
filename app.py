import streamlit as st
import pandas as pd
import joblib
from datetime import timedelta, date

# =================================
# PAGE CONFIG
# =================================
st.set_page_config(
    page_title="Electricity Load Forecasting",
    page_icon="⚡",
    layout="centered"
)

# =================================
# LOAD MODEL & DATA (CACHED)
# =================================
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/final_features.csv")

model = load_model()
df = load_data()

# =================================
# PREPARE DATA
# =================================
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

FEATURE_COLUMNS = df.drop(columns=["date", "load"]).columns.tolist()

latest_row = df.iloc[-1]
last_date = latest_row["date"].date()
last_actual_load = latest_row["load"]

# =================================
# HEADER
# =================================
st.title("⚡ Electricity Load Forecasting")
st.subheader("Bihar Electricity Consumption Prediction")

st.markdown(
    """
    This application predicts **future electricity consumption for Bihar**
    using a trained machine learning model.

    Select a future date to generate the prediction.
    """
)

st.divider()

# =================================
# DATA SUMMARY (WITH ICONS & UNIT)
# =================================
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="📅 Last Available Date",
        value=last_date.strftime("%d/%m/%Y")
    )

with col2:
    st.metric(
        label="⚡ Last Actual Load (Bihar)",
        value=f"{last_actual_load:.2f} kWh"
    )

st.divider()

# =================================
# DATE INPUT (FORMAT: DD/MM/YYYY)
# =================================
st.subheader("📅 Select Future Date")

today = date.today()
default_date = today if today > last_date else last_date + timedelta(days=1)

selected_date = st.date_input(
    "Future Date",
    min_value=last_date + timedelta(days=1),
    value=default_date,
    format="DD/MM/YYYY"
)

st.divider()

# =================================
# PREDICTION LOGIC (UNLIMITED)
# =================================
if st.button("🔮 Predict Load"):
    future_days = (selected_date - last_date).days

    with st.spinner(f"Generating forecast ({future_days} days ahead)..."):
        temp_df = df.copy()

        for _ in range(future_days):
            latest_features = temp_df[FEATURE_COLUMNS].iloc[-1:]
            next_load = model.predict(latest_features)[0]

            new_row = temp_df.iloc[-1:].copy()
            new_row["date"] = temp_df["date"].iloc[-1] + timedelta(days=1)
            new_row["load"] = next_load
            new_row["lag_1"] = next_load
            new_row["lag_7"] = temp_df["load"].tail(7).mean()
            new_row["rolling_mean_7"] = temp_df["load"].tail(7).mean()

            temp_df = pd.concat([temp_df, new_row], ignore_index=True)

        final_prediction = temp_df.iloc[-1]["load"]

        st.success(
            f"""
            ⚡ **Electricity Consumption Prediction (Bihar)**

            **Date:** {selected_date.strftime('%d/%m/%Y')}  
            **Predicted Consumption:** **{final_prediction:.2f} kWh**
            """
        )

st.divider()

# =================================
# MODEL INFORMATION
# =================================
st.subheader("🧠 Model Information")

st.markdown(
    """
    - **State:** Bihar  
    - **Target Variable:** Daily electricity consumption (kWh)  
    - **Model:** Best-performing regression model (selected using RMSE)  
    - **Features:** Lag values, rolling mean, calendar features  
    - **Forecasting Method:** Recursive time-series forecasting  

    *Note:* Prediction uncertainty increases for longer forecast horizons,
    which is a standard characteristic of time-series models.
    """
)

# =================================
# CENTERED FOOTER
# =================================
st.markdown(
    """
    <div style="text-align:center; padding-top:20px;">
        <hr>
        <b>Developed by Amit Sharma</b><br>
        B.Tech Final Year Project – Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
