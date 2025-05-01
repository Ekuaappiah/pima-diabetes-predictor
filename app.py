import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mstats

# Load the trained model
model = joblib.load('models/pima_diabetes_predictor.pkl')

def preprocess_input_data(input_data):
    features_to_winsorize = ['BloodPressure', 'Insulin', 'DiabetesPedigreeFunction']
    for feature in features_to_winsorize:
        input_data[feature] = mstats.winsorize(input_data[feature], limits=[0.05, 0.05])

    input_data['HighGlucose'] = (input_data['Glucose'] > 140).astype(int)
    input_data['Age_BMI'] = input_data['Age'] * input_data['BMI']

    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age', 'HighGlucose', 'Age_BMI']
    return input_data[features]

def predict_diabetes(inputs):
    input_data = pd.DataFrame([inputs], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    preprocessed_data = preprocess_input_data(input_data)
    prediction = model.predict(preprocessed_data)
    return prediction[0]

# --- Page Config ---
st.set_page_config(page_title="Diabetes Predictor", page_icon="🧬", layout="centered")

# --- Title and Intro ---
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🧬 Pima Diabetes Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Fill in the health parameters below to assess your risk of developing Type 2 Diabetes.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Input Section ---
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("👶 Number of Pregnancies", 0, 20, step=1, value=3)
        glucose = st.number_input("🍬 Plasma Glucose Concentration", 0, 200, step=1, value=120)
        blood_pressure = st.number_input("💓 Diastolic Blood Pressure (mm Hg)", 0, 200, step=1, value=70)
        skin_thickness = st.number_input("📏 Triceps Skinfold Thickness (mm)", 0, 100, step=1, value=20)

    with col2:
        insulin = st.number_input("💉 2-Hour Serum Insulin (mu U/ml)", 0, 500, step=1, value=80)
        bmi = st.number_input("⚖️ Body Mass Index (kg/m²)", 0.0, 50.0, step=0.1, value=30.0)
        diabetes_pedigree = st.number_input("🧬 Diabetes Pedigree Function", 0.0, 2.5, step=0.1, value=0.5)
        age = st.number_input("🎂 Age (years)", 18, 120, step=1, value=40)

# --- Prediction ---
inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

st.markdown("---")
if st.button("🔍 Predict"):
    prediction = predict_diabetes(inputs)
    if prediction == 1:
        st.markdown("""
            <div style="padding:20px; border-left:6px solid #e74c3c; border-radius:8px;">
                <h3 style="color:#e74c3c;">⚠️ High Risk Detected</h3>
                 <p>You may have <strong>Type 2 Diabetes</strong>.</p>
                <p style="margin-top:10px;">We recommend consulting a healthcare provider for further tests and professional guidance.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="padding:20px; border-left:6px solid #2ecc71; border-radius:8px;">
                <h3 style="color:#27ae60;">✅ No Immediate Risk</h3>
                <p>You are unlikely to have <strong>Type 2 Diabetes</strong> based on the input data.</p>
                <p style="margin-top:10px;">Maintain a healthy lifestyle and regular check-ups to stay on track.</p>
            </div>
        """, unsafe_allow_html=True)

# --- Optional Sidebar ---
st.sidebar.markdown("## ℹ️ About This App")
st.sidebar.markdown("""
This app helps you assess your risk for Type 2 Diabetes based on some common health measurements.
By entering your details (such as age, glucose level, BMI, and more), the app will estimate whether you might be at risk for diabetes.

**How it works:**
- The app uses a trained model that analyzes your health data.
- It considers important factors like glucose levels, body mass index (BMI), and family history to give you an estimate of your diabetes risk.
- Based on your inputs, it provides one of two outcomes:
    - **High risk**: If the model suggests you may have Type 2 Diabetes.
    - **Low risk**: If you're less likely to have diabetes based on your data.

**Next steps:**
If you receive a high-risk result, it's important to consult a healthcare provider for further tests and professional advice.

**Dataset:** Pima Indians Diabetes Dataset (a large set of health data used for training the model)
""")
