import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
from datetime import timedelta
import numpy as np

# Load trained model
model = joblib.load("random_forest_model.pkl")

# Streamlit settings
st.set_page_config(page_title="Intermittent Demand Forecasting App", layout="wide")
st.title("Intermittent Demand Prediction - Deployment")

# File upload
uploaded_file = st.file_uploader("Upload Monthly Feature-Engineered Excel File", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Debug: show column names
    st.write("Uploaded Columns:", df.columns.tolist())

    # Convert date columns
    df["Monthly_Date"] = pd.to_datetime(df["Monthly_Date"])
    df["Month"] = df["Monthly_Date"].dt.month

    # Feature columns used in model
    feature_columns = [
        "Quarter", "Is_Peak_Season", "Lag_Month_2", "Lag_Month_3", "Lag_Month_4", "Lag_Month_5", "Lag_Month_6", "Lag_Month_7",
        "Rolling_Mean_7", "Rolling_Std_7", "EWMA_7", "Cumulative_Demand", "Demand_CV", "Mean_Demand", "Median_Demand", 
        "Sell_Rate", "Log_Monthly_Sold_Qty", "High_Demand_Spike", "ABC_Category_Encoded",
        "PCA_1", "PCA_2", "PCA_3", "Part_Cluster"
    ]

    if all(col in df.columns for col in feature_columns):
        df["Predicted_Sold_Qty"] = np.round(model.predict(df[feature_columns])).astype(int)
    else:
        st.error("Uploaded file is missing one or more required feature columns for prediction.")
        st.stop()

    # --- Extend Forecast for T+1 to T+7 ---
    last_date = df["Monthly_Date"].max()
    future_months = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=7, freq='MS')

    future_rows = []
    for part in df["Part No"].unique():
        latest_row = df[df["Part No"] == part].sort_values("Monthly_Date").iloc[-1]
        for date in future_months:
            row = latest_row.copy()
            row["Monthly_Date"] = date
            row["Month"] = date.month
            row["Quarter"] = date.quarter
            row["Is_Peak_Season"] = 1 if date.month in [3, 4, 11, 12] else 0
            future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    future_df["Predicted_Sold_Qty"] = np.round(model.predict(future_df[feature_columns])).astype(int)
    df = pd.concat([df, future_df], ignore_index=True)

    # --- Deployment Overview ---
    st.markdown("## Deployment Highlights")
    st.markdown("""
    - This interface allows stakeholders to upload monthly data and get predictions using the trained Random Forest model.
    - It includes model accuracy metrics, forecast visualizations, download options, and deviation analysis.
    """)

    # --- Model Accuracy ---
    st.markdown("### Model Accuracy Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAPE", "2.90%")
    col2.metric("RMSE", "42.53")
    col3.metric("MAE", "4.86")

    st.markdown("**Avg TSCV Metrics (5 folds):**")
    col4, col5, col6 = st.columns(3)
    col4.metric("Avg MAPE", "~10.4%")
    col5.metric("Avg RMSE", "~281")
    col6.metric("Avg MAE", "~96.5")

    # --- Forecast Summary ---
    st.markdown("### Forecast Summary")
    total_parts = df["Part No"].nunique()
    total_forecast = df["Predicted_Sold_Qty"].sum()
    if "Monthly_Sold_Qty" in df.columns:
        df["Deviation_%"] = ((df["Predicted_Sold_Qty"] - df["Monthly_Sold_Qty"]).abs() / df["Monthly_Sold_Qty"].replace(0, 1)) * 100
        avg_deviation = df["Deviation_%"].mean().round(2)
    else:
        df["Monthly_Sold_Qty"] = 0
        df["Deviation_%"] = 0
        avg_deviation = 0.0

    col7, col8, col9 = st.columns(3)
    col7.metric("Total Parts Forecasted", f"{total_parts}")
    col8.metric("Total Forecast Quantity", f"{int(total_forecast)} units")
    col9.metric("Avg Deviation (%)", f"{avg_deviation}%")

    # --- Actual vs Predicted Line Chart (Latest Month) ---
    st.markdown("### Actual vs Predicted - Latest Month (Line Chart)")
    latest_month = df["Monthly_Date"].max()
    latest_df = df[df["Monthly_Date"] == latest_month].sort_values(by="Part No")

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(latest_df["Part No"], latest_df["Monthly_Sold_Qty"], label="Actual", marker='o')
    ax1.plot(latest_df["Part No"], latest_df["Predicted_Sold_Qty"], label="Predicted", marker='x')
    ax1.set_title(f"Actual vs Predicted Demand - {latest_month.strftime('%B %Y')}")
    ax1.set_ylabel("Sold Quantity")
    ax1.set_xlabel("Part No")
    ax1.legend()
    plt.xticks(rotation=90)
    st.pyplot(fig1)

    # --- Top 10 Predicted Parts ---
    st.markdown("### Top 10 Parts with Highest Predicted Demand")
    top_10 = latest_df.sort_values(by="Predicted_Sold_Qty", ascending=False).head(10)
    st.bar_chart(top_10.set_index("Part No")["Predicted_Sold_Qty"])

    # --- Forecast Table After August 2024 ---
    st.markdown("### Forecast Results After August 2024")
    future_only_df = df[df["Monthly_Date"] > pd.to_datetime("2024-08-31")]
    st.dataframe(future_only_df[["Monthly_Date", "Part No", "Predicted_Sold_Qty"]].sort_values(by="Monthly_Date"))

    # --- Heatmap ---
    st.markdown("### Forecasted Demand Heatmap (Part No vs Month)")
    heatmap_data = df.pivot_table(index="Part No", columns="Month", values="Predicted_Sold_Qty", aggfunc='sum', fill_value=0)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax2)
    st.pyplot(fig2)

    # --- Deviation Chart ---
    st.markdown("### Parts with Highest Deviation (Actual vs Predicted)")
    deviation_top10 = df[df["Monthly_Date"] == latest_month].sort_values(by="Deviation_%", ascending=False).head(10)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.bar(deviation_top10["Part No"], deviation_top10["Deviation_%"], color='orange')
    ax3.set_title("Top 10 Parts by Forecast Deviation (%)")
    ax3.set_ylabel("Deviation %")
    plt.xticks(rotation=90)
    st.pyplot(fig3)

    # --- Download Button ---
    st.markdown("### Download Forecast Results")
    to_download = df[["Monthly_Date", "Part No", "Monthly_Sold_Qty", "Predicted_Sold_Qty", "Deviation_%"]]
    buffer = io.BytesIO()
    to_download.to_excel(buffer, index=False)
    st.download_button(
        label="Download Forecast as Excel",
        data=buffer,
        file_name="Forecast_Results_With_Deviation.xlsx",
        mime="application/vnd.ms-excel"
    )