import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle

from auth import load_authenticator
from logger import setup_logger
from utils import plot_feature, plot_shap
from ui import render_shap_tab

# === App Config ===
st.set_page_config(layout="wide")

# === Authentication ===
authenticator = load_authenticator()
name, auth_status, username = authenticator.login('Login', 'main')
logger = setup_logger()

if auth_status:
    # === Password Reset Check ===
    if authenticator.credentials["usernames"][username].get("password_reset", False):
        st.warning("🔒 You are required to reset your password.")
        if st.button("Change Password"):
            authenticator.reset_password(username)
            st.success("✅ Password updated. Please log in again.")
            st.stop()
    
    authenticator.logout('Logout', 'main')

    st.title("📈 Time Series Anomaly Detection with SHAP")
    st.write(f"Welcome *{name}* 👋")
    logger.info(f"User {username} logged in successfully")

    # === Load trained pipeline from root ===
    try:
        model_path = os.path.join(os.path.dirname(__file__), "iforest_shap_timeseries.pkl")
        st.write("📄 Model path:", model_path)
        st.write("✅ File exists?", os.path.exists(model_path))

        with open(model_path, "rb") as f:
            assets = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model file not found. Please make sure 'iforest_shap_timeseries.pkl' is placed in the root of your repo.")
        st.stop()

    model = assets['model']
    scaler = assets['scaler']
    explainer = assets['explainer']
    features = assets['features']

    # === Upload CSV ===
    uploaded = st.file_uploader("📂 Upload live CSV", type="csv")
    if uploaded:
        logger.info(f"{username} uploaded file: {uploaded.name}")
        live_df = pd.read_csv(uploaded, parse_dates=['Datetime'])
        live_df.set_index('Datetime', inplace=True)

        X_live = live_df[features]
        X_scaled = scaler.transform(X_live)
        preds = model.predict(X_scaled)
        scores = model.decision_function(X_scaled)
        shap_vals = explainer.shap_values(X_scaled)

        live_df['anomaly'] = preds
        live_df['anomaly_score'] = scores
        live_df['row_index'] = np.arange(len(live_df))

        # 🗓️ Let user pick a datetime from calendar
        selected_datetime = st.datetime_input("📅 Select Datetime", value=live_df.index[0])

        # 🔢 Safely map to row index (if it exists)
        if selected_datetime in live_df.index:
            index = live_df.index.get_loc(selected_datetime)
        else:
            st.warning("Selected datetime not found in dataset.")
            st.stop()

        #index = st.number_input("🔢 Select row index", min_value=0, max_value=len(live_df)-1, value=0)
        selected_feature = st.selectbox("📊 Select feature to visualize", features)
        timestamp = live_df.index[index]

        tab1, tab2 = st.tabs(["📈 Feature Plot", "🧠 SHAP Explanation"])
        tab1.plotly_chart(plot_feature(live_df, selected_feature), use_container_width=True)
        render_shap_tab(shap_vals, features, index, live_df, tab2)

elif auth_status == False:
    st.error("Username/password is incorrect ❌")

elif auth_status == None:
    st.warning("Please enter your username and password 🔐")
