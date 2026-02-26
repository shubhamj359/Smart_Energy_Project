
import streamlit as st
import numpy as np
import joblib

model = joblib.load('energy_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🌱 Smart Energy Predictor")

gap = st.number_input("Global Active Power", 0.0, 10.0)
grp = st.number_input("Global Reactive Power", 0.0, 2.0)
voltage = st.number_input("Voltage", 200.0, 260.0)
gi = st.number_input("Global Intensity", 0.0, 40.0)
sm1 = st.number_input("Sub Metering 1", 0, 50)
sm2 = st.number_input("Sub Metering 2", 0, 50)
sm3 = st.number_input("Sub Metering 3", 0, 50)

if st.button("Predict"):
    data = np.array([[gap, grp, voltage, gi, sm1, sm2, sm3]])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    
    if pred == 1:
        st.error("High Energy Consumption")
    else:
        st.success("Low Energy Consumption")
