import streamlit as st
import pandas as pd
from joblib import load

rf_model = load('random_forest_model.joblib')
scaler = load('rf_scaler.joblib')
le = load('rf_label_encoder.joblib')

top5_features = [
    'Stress Level (1-10)',
    'Caffeine Intake (mg/day)',
    'Sleep Hours',
    'Physical Activity (hrs/week)',
    'Heart Rate (bpm)'
]

st.title('Anxiety Level Prediction (Random Forest Classifier)')
st.header("Enter Your Details:")

input_data = {}
for feature in top5_features:
    input_data[feature] = st.number_input(f"{feature}", value=0)

if st.button('Predict'):
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=top5_features, fill_value=0)
    input_scaled = scaler.transform(input_df)
    prediction = rf_model.predict(input_scaled)
    predicted_label = le.inverse_transform(prediction)
    st.success(f"Predicted Anxiety Level: **{predicted_label[0]}**")