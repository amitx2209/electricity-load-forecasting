import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Electricity Load Forecasting",
    page_icon="⚡",
    layout="centered"
)

# =====================================
# CUSTOM CSS (UI POLISH)
# =====================================
st.markdown(
    """
    <style>
    /* Center main content */
    .block-container {
        padding-top: 2rem;
    }

    /* Button hover glow */
    div.stButton > button {
        background-color: #0E1117;
        color: white;
        border: 1px solid #2f80ed;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        transition: all 0.3s ease-in-out;
    }

    div.stButton > button:hover {
        background-color: #2f80ed;
        color: white;
        box-shadow: 0 0 15px rgba(47, 128, 237, 0.8);
        transform: scale(1.03);
    }

    /* Metric cards spacing */
    div[data-testid="metric-container"] {
        background-color: #0E1117;
        border: 1px solid #262730;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    }

    /* Footer */
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================
# TITLE
# =====================================
st.title("⚡ Electricity Load Forecasting")
st.markdown(
    "Predict **daily electricity consumption** for Indian states "
    "using machine learning."
)

# =====================================
# PATHS
# =====================================
FEATURE_DIR = "data/processed_features"
MODEL_DIR = "models"

# =====================================
# LOAD AVAILABLE STATES
# =====================================
@st.cache_data
def get_available_states():
    return sorted(
        f.replace("_features.csv", "")
        for f in os.listdir(FEATURE_DIR)
        if f.endswith("_features.csv")
    )

states = get_available_states()

if not states:
    st.error("No processed state feature files found.")
    st.stop()

# =====================================
# STATE SELECTION
# =====================================
st.subheader("📍 Select State")

selected_state = st.selectbox(
    "State",
    states,
    index=states.index("Bihar") if "Bihar" in states else 0
)

# =====================================
# LOAD STATE DATA
# =====================================
@st.cache_data
def load_state_data(state):
    df = pd.read_csv(os.path.join(FEATURE_DIR, f"{state}_features.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_state_data(selected_state)

# =====================================
# LAST ACTUAL VALUES
# =====================================
last_available_date = df["date"].max()
last_actual_load = df.loc[
    df["date"] == last_available_date, "load"
].values[0]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "📅 Last Available Date",
        last_available_date.strftime("%d/%m/%Y")
    )

with col2:
    st.metric(
        "⚡ Last Actual Load",
        f"{last_actual_load:.2f} MU"
    )

# =====================================
# LOAD MODEL
# =====================================
model_path = os.path.join(MODEL_DIR, f"{selected_state}_model.pkl")

if not os.path.exists(model_path):
    st.error(f"Trained model not found for {selected_state}.")
    st.stop()

model = joblib.load(model_path)

# =====================================
# DATE INPUT
# =====================================
st.subheader("📅 Select Prediction Date")

selected_date = st.date_input(
    "Prediction Date",
    value=datetime.today().date()
)

# =====================================
# PREDICTION
# =====================================
if st.button("🔮 Predict Load"):

    if pd.to_datetime(selected_date) <= last_available_date:
        st.warning(
            "Please select a **future date** beyond the last available data."
        )
        st.stop()

    days_ahead = (pd.to_datetime(selected_date) - last_available_date).days

    last_row = df.iloc[-1].copy()
    current_features = last_row.drop(labels=["date", "load"])
    current_date = last_available_date

    with st.spinner("Predicting future electricity load..."):

        for _ in range(days_ahead):
            prediction = model.predict(
                pd.DataFrame([current_features])
            )[0]

            current_features["lag_7"] = current_features["lag_1"]
            current_features["lag_1"] = prediction
            current_features["rolling_mean_7"] = (
                current_features["rolling_mean_7"] * 6 + prediction
            ) / 7

            next_date = current_date + pd.Timedelta(days=1)
            current_features["day"] = next_date.day
            current_features["month"] = next_date.month
            current_features["weekday"] = next_date.weekday()

            current_date = next_date

    st.success("Prediction completed successfully ✅")

    st.metric(
        "⚡ Predicted Load",
        f"{prediction:.2f} MU"
    )

# =====================================
# CUSTOM FOOTER
# =====================================
st.markdown(
    "<hr>"
    "<p style='text-align:center; color:gray;'>"
    "Electricity Load Forecasting • Multi-State ML Project"
    "</p>",
    unsafe_allow_html=True
)
