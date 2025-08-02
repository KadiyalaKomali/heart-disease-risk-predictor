import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
le_sex = pickle.load(open('le_sex.pkl', 'rb'))
le_cp = pickle.load(open('le_cp.pkl', 'rb'))
le_ecg = pickle.load(open('le_ecg.pkl', 'rb'))
le_angina = pickle.load(open('le_angina.pkl', 'rb'))
le_slope = pickle.load(open('le_slope.pkl', 'rb'))

st.set_page_config(page_title="Heart Disease Risk Predictor")
st.title("💓 Heart Disease Risk Predictor")

with st.form("predict_form"):
    st.header("Enter Health Parameters")

    age = st.slider("Age", 1, 120, 45)
    sex = st.selectbox("Sex", ["M", "F"])
    cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    rbp = st.number_input("Resting Blood Pressure", 50, 200, 120)
    chol = st.number_input("Cholesterol", 0, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    maxhr = st.number_input("Max Heart Rate", 60, 220, 150)
    angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": le_sex.transform([sex]),
        "ChestPainType": le_cp.transform([cp]),
        "RestingBP": [rbp],
        "Cholesterol": [chol],
        "FastingBS": [fbs],
        "RestingECG": le_ecg.transform([ecg]),
        "MaxHR": [maxhr],
        "ExerciseAngina": le_angina.transform([angina]),
        "Oldpeak": [oldpeak],
        "ST_Slope": le_slope.transform([slope])
    })

    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("🚨 High Risk of Heart Disease!")
    else:
        st.success("✅ Low Risk of Heart Disease.")
