import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("Heart Disease Prediction App 💓")

st.write("Enter the patient details to predict the risk of heart disease:")

age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Serum Cholestoral in mg/dl", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", value=1.0)
slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3) colored by fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)", [1, 2, 3])

input_data = pd.DataFrame([[
    age,
    1 if sex == "Male" else 0,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]], columns=[
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("High risk of heart disease 💔")
    else:
        st.success("Low risk of heart disease 💖")
