import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("diabetes_rf_model.pkl")

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details to predict diabetes risk")

# Input fields
preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 25)
insulin = st.number_input("Insulin", 0, 900, 100)
bmi = st.number_input("BMI", 0.0, 80.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Patient is likely DIABETIC")
    else:
        st.success("✅ Low Risk: Patient is likely NON-DIABETIC")
