# Intermittent Demand Forecasting

This capstone project aims to forecast intermittent demand for automotive spare parts using machine learning. The project focuses on predicting monthly demand for parts that exhibit high variability and zero-inflated patterns.

## Project Overview
A Random Forest Regressor model was selected based on its performance (MAPE: 2.9%, RMSE: 41). The predictions are integrated into a Streamlit-based app that allows users to upload monthly feature-engineered data and view forecasts with downloadable results.

## Files Included
- `app.py` – Streamlit frontend
- `random_forest_model.pkl` – Trained model
- `Intermittent_Demand_Prediction_Using_ML_Capstone.ipynb` – python notebook
- Notebooks and Excel files – For data processing, feature engineering, demand classification, and forecast evaluation
- Aggregated & Engineered: `Monthly_Aggregated_Data.xlsx, Monthly_Feature_Engineered_Data.xlsx, filtered_intermittent_data.xlsx, sorted_combined_data.zip (raw data)`
- Classified + Forecast Results: `Monthly_Demand_Classified.xlsx, Monthly_Forecast_Tplus2_to_Tplus8.xlsx, TS_CV_Results.xlsx, Prediction_vs_7MonthMA_Comparison.xlsx`


## How to Run
1. Install dependencies: `streamlit`, `pandas`, `joblib`, `matplotlib`, `seaborn`, etc.
2. Run the app:

streamlit run app.py

3. Upload the Excel file with monthly features to get predictions.

## Demo video
[watch the demo] 
https://drive.google.com/file/d/11GuhjRXJ0_Iqop8_bF1aeufY_ZQTHA0r/view?usp=drive_link

This video shows how to use the Streamlit app for forecasting Intermittent Demand. 

## Done by
Sushmitha Dani  
MS Business Analytics 
Capstone Project 1 - Intermittent Demand Prediction using Machine Learning
REVA University, 2025
