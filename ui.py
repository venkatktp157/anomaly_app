import streamlit as st
from utils import plot_shap

def render_shap_tab(shap_values, features, index, timestamp, tab):
    fig = plot_shap(shap_values, features, index, timestamp)
    tab.plotly_chart(fig, use_container_width=True)

    with tab.expander("ℹ️ SHAP Interpretation Guide"):
        st.markdown("""
        <div style='padding:0.5em 1em; border-left: 4px solid #4CAF50; background-color:#1e1e1e; color:#f0f0f0;'>
        <strong>🔍 How to Read SHAP Values (Isolation Forest):</strong><br><br>
        • The model output is an <strong>anomaly score</strong> (lower = more anomalous)<br>
        • SHAP values explain how each feature contributed<br>
        • Negative → pushes toward anomaly<br>
        • Positive → pushes toward normal<br>
        • Magnitude → strength of contribution
        </div>
        """, unsafe_allow_html=True)