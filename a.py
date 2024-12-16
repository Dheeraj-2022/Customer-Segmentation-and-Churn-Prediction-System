import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Load pre-trained models and transformers
try:
    rf_model = joblib.load('rf_churn_model.pkl')
    kmeans_model = joblib.load('customer_clusters.pkl')
    pca_transform = joblib.load('pca_transform.pkl')
except Exception as e:
    st.error(f"Error loading models: {e}")

# Streamlit App Header
st.title("Customer Churn and Segmentation App")
st.write("A machine learning application for customer churn prediction and segmentation.")

# File uploader
data_file = st.file_uploader("Upload your CSV file", type=["csv"])

if data_file:
    # Load and display the dataset
    try:
        data = pd.read_csv(data_file)
        st.write("Uploaded Data:", data.head())
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")

    # Data Preprocessing
    st.subheader("Data Preprocessing")
    try:
        # Convert TotalCharges to numeric, coercing errors to NaN, and fill missing values
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data.fillna(data.mean(), inplace=True)

        # Normalize numeric features
        scaler_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
        for col in scaler_cols:
            data[col] = (data[col] - data[col].mean()) / data[col].std()

        # Encode categorical variables
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            data[col] = data[col].astype('category').cat.codes

        st.write("Preprocessed Data:", data.head())
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")

    # Customer Segmentation (Clustering)
    st.subheader("Customer Segmentation")
    with st.spinner('Running clustering...'):
        try:
            # Apply PCA transformation and predict clusters
            pca_features = pca_transform.transform(data[['MonthlyCharges', 'tenure', 'Contract']])
            data['Cluster'] = kmeans_model.predict(pca_features)

            st.write("Cluster Assignments:", data[['Cluster']].value_counts())
        except Exception as e:
            st.error(f"Error during clustering: {e}")

    # Churn Prediction
    st.subheader("Churn Prediction")
    with st.spinner('Predicting churn probabilities...'):
        try:
            # Drop unnecessary columns
            prediction_features = data.drop(columns=['customerID', 'Churn'], errors='ignore')
            churn_probabilities = rf_model.predict_proba(prediction_features)[:, 1]
            data['Churn Probability'] = churn_probabilities

            st.write("Churn Predictions:", data[['customerID', 'Churn Probability']])

            # Visualization
            st.bar_chart(data[['Cluster', 'Churn Probability']].groupby('Cluster').mean())
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # Download results
    st.subheader("Download Processed Data")
    processed_data = data.copy()
    processed_csv = processed_data.to_csv(index=False)
    st.download_button(
        label="Download Processed Data",
        data=processed_csv,
        file_name="processed_customer_data.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload a CSV file to begin.")
