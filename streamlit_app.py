import streamlit as st
import numpy as np
import pickle
import os

# Load the model
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ model.pkl not found. Please train the model first.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))

# Streamlit UI
st.set_page_config(page_title="Crop Prediction", page_icon="🌾", layout="centered")

st.title("🌱 Crop Prediction Model")
st.markdown("Enter the values below to predict the best suitable crop.")

# Input form
with st.form("prediction_form"):
    n = st.number_input("Nitrogen (N)", min_value=0.0)
    p = st.number_input("Phosphorus (P)", min_value=0.0)
    k = st.number_input("Potassium (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0)
    ph = st.number_input("pH", min_value=0.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

    submit = st.form_submit_button("Predict")

# Predict
if submit:
    input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Predicted Crop: **{prediction}**")
