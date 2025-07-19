import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_feature(df, feature, anomaly_col='anomaly'):
    fig = go.Figure()

    # Plot normal line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[feature],
        mode='lines',
        name='Normal'
    ))

    # Plot anomaly markers
    anomaly_pts = df[df[anomaly_col] == -1]
    fig.add_trace(go.Scatter(
        x=anomaly_pts.index,
        y=anomaly_pts[feature],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=8, symbol='x')
    ))

    fig.update_layout(
        title=f"{feature} over time",
        template="plotly_white",
        xaxis_title="Datetime",
        yaxis_title=feature
    )
    return fig

def plot_shap(shap_values, features, index, df):
    timestamp = df.index[index]

    shap_df = pd.DataFrame({
        'Feature': features,
        'SHAP Value': shap_values[index],
        'Row Index': index,         # Tooltip reference
        'Timestamp': timestamp      # Visible on axis or hover
    })

    shap_df['SHAP Value'] = shap_df['SHAP Value'].round(3)

    fig = px.bar(
        shap_df,
        x='SHAP Value',
        y='Feature',
        orientation='h',
        color='SHAP Value',
        color_continuous_scale='RdBu',
        title=f"SHAP Explanation (Datetime: {timestamp})",
        hover_data=['Row Index']
    )

    fig.update_layout(
        xaxis_title="SHAP Value",
        yaxis_title="Feature"
    )

    return fig

