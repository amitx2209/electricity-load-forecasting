import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import altair as alt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Electricity Load Forecasting",
    page_icon="⚡",
    layout="centered"
)

# =========================================================
# CUSTOM CSS (POLISH)
# =========================================================
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #0E1117;
        color: white;
        border: 1px solid #2f80ed;
        border-radius: 8px;
        padding: 0.55rem 1.3rem;
        transition: all 0.25s ease-in-out;
    }

    div.stButton > button:hover {
        background-color: #2f80ed;
        box-shadow: 0 0 14px rgba(47, 128, 237, 0.8);
        transform: scale(1.03);
    }

    div[data-testid="metric-container"] {
        background-color: #0E1117;
        border: 1px solid #262730;
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TITLE
# =========================================================
st.title("⚡ Electricity Load Forecasting")
st.markdown(
    "State-wise **daily electricity demand forecasting** using machine learning."
)

# =========================================================
# PATHS
# =========================================================
FEATURE_DIR = "data/processed_features"
MODEL_DIR = "models"

# =========================================================
# LOAD STATES
# =========================================================
@st.cache_data
def get_states():
    states = sorted(
        f.replace("_features.csv", "")
        for f in os.listdir(FEATURE_DIR)
        if f.endswith("_features.csv")
    )
    return ["Select State"] + states

states = get_states()

# =========================================================
# STATE SELECTION
# =========================================================
st.subheader("📍 Select State")
selected_state = st.selectbox("State", states, index=0)

if selected_state == "Select State":
    st.info("Please select a state to continue.")
    st.stop()

# =========================================================
# LOAD STATE DATA
# =========================================================
@st.cache_data(show_spinner=False)
def load_state_data(state):
    df = pd.read_csv(os.path.join(FEATURE_DIR, f"{state}_features.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df.copy()

df = load_state_data(selected_state)

# =========================================================
# LAST & AVERAGE LOAD
# =========================================================
last_available_date = df["date"].max()
last_actual_load = df.loc[df["date"] == last_available_date, "load"].iloc[0]
avg_30_day_load = df.tail(30)["load"].mean()

col1, col2 = st.columns(2)

with col1:
    st.metric("📅 Last Available Date", last_available_date.strftime("%d/%m/%Y"))

with col2:
    st.metric("⚡ Last Actual Load", f"{last_actual_load:.2f} GWh")

# =========================================================
# RECENT TREND (CONTEXT ONLY)
# =========================================================
st.subheader("📈 Recent Electricity Consumption Trend (Last 30 Days)")

trend_df = df.tail(30)[["date", "load"]]

trend_chart = (
    alt.Chart(trend_df)
    .mark_line(color="#2F80ED", strokeWidth=2)
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("load:Q", title="Electricity Load (GWh)")
    )
    .properties(height=280)
)

st.altair_chart(trend_chart, use_container_width=True)

# =========================================================
# LOAD MODEL
# =========================================================
model_path = os.path.join(MODEL_DIR, f"{selected_state}_model.pkl")
model = joblib.load(model_path)

# =========================================================
# MODEL DETAILS
# =========================================================
st.subheader("🤖 Model Details")

st.markdown(
    f"""
    • **State:** {selected_state}  
    • **Model Type:** Random Forest Regressor  
    • **Training Strategy:** Independent state-wise model  
    """
)

# =========================================================
# DATE INPUT
# =========================================================
st.subheader("📅 Select Prediction Date")

selected_date = st.date_input(
    "Prediction Date (DD/MM/YYYY)",
    value=datetime.today().date(),
    format="DD/MM/YYYY"
)

# =========================================================
# PREDICTION
# =========================================================
if st.button("🔮 Predict Load"):

    if pd.to_datetime(selected_date) <= last_available_date:
        st.error("Please select a future date beyond the last available data.")
        st.stop()

    days_ahead = (pd.to_datetime(selected_date) - last_available_date).days
    horizon_text = "day ahead" if days_ahead == 1 else "days ahead"

    st.info(
        f"Predicting electricity load for **{selected_state}** on "
        f"**{selected_date.strftime('%d/%m/%Y')}** "
        f"({days_ahead} {horizon_text})."
    )

    last_row = df.iloc[-1].copy()
    current_features = last_row.drop(["date", "load"])
    current_date = last_available_date

    with st.spinner("Predicting future electricity load..."):
        for _ in range(days_ahead):
            prediction = model.predict(pd.DataFrame([current_features]))[0]

            current_features["lag_7"] = current_features["lag_1"]
            current_features["lag_1"] = prediction
            current_features["rolling_mean_7"] = (
                current_features["rolling_mean_7"] * 6 + prediction
            ) / 7

            next_date = current_date + timedelta(days=1)
            current_features["day"] = next_date.day
            current_features["month"] = next_date.month
            current_features["weekday"] = next_date.weekday()

            current_date = next_date

    # =====================================================
    # COMPARISON CARDS (PROFESSIONAL)
    # =====================================================
    st.subheader("📊 Load Comparison")

    diff_avg = prediction - avg_30_day_load
    diff_last = prediction - last_actual_load

    pct_avg = (diff_avg / avg_30_day_load) * 100
    pct_last = (diff_last / last_actual_load) * 100

    arrow_avg = "▲" if diff_avg >= 0 else "▼"
    color_avg = "#27AE60" if diff_avg >= 0 else "#EB5757"

    arrow_last = "▲" if diff_last >= 0 else "▼"
    color_last = "#27AE60" if diff_last >= 0 else "#EB5757"

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("📊 30-Day Average Load", f"{avg_30_day_load:.2f} GWh")

    with c2:
        st.metric("🔮 Predicted Load", f"{prediction:.2f} GWh")

    with c3:
        st.markdown(
            f"""
            <div style="text-align:center; line-height:1.4;">
                <span style="font-size:22px; font-weight:600; color:{color_avg};">
                    {arrow_avg} {abs(diff_avg):.2f} GWh
                </span><br>
                <span style="font-size:14px; color:#aaaaaa;">
                    ({pct_avg:+.2f}% vs 30-day average)
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div style="text-align:center; margin-top:10px;">
            <span style="font-size:14px; color:{color_last};">
                {arrow_last} {abs(diff_last):.2f} GWh
                ({pct_last:+.2f}% vs last actual load)
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("Note: GWh and MU (Million Units) are equivalent energy units.")

# =========================================================
# PROJECT INFO
# =========================================================
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:16px; color:#cccccc;">
    This application demonstrates a scalable, multi-state electricity load
    forecasting system using machine learning. Independent models are trained
    for each Indian state using historical electricity consumption data.
    </p>
    """,
    unsafe_allow_html=True
)

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <p style="text-align:center; color:gray; font-size:14px;">
    ⚡ Electricity Load Forecasting using Machine Learning <br>
    Developed by <b>Amit Kumar</b> <br>
    🔗 <a href="https://github.com/amitx2209/electricity-load-forecasting" target="_blank">
    GitHub Repository</a>
    </p>
    """,
    unsafe_allow_html=True
)
