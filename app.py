import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mstats

# Load the trained model
model = joblib.load('models/pima_diabetes_predictor.pkl')

def preprocess_input_data(input_data):
    # Add the same preprocessing steps as done during training

    # Winsorize specified columns to handle outliers
    features_to_winsorize = ['BloodPressure', 'Insulin', 'DiabetesPedigreeFunction']
    for feature in features_to_winsorize:
        input_data[feature] = mstats.winsorize(input_data[feature], limits=[0.05, 0.05])

    # Feature Engineering: Add HighGlucose and Age_BMI
    input_data['HighGlucose'] = (input_data['Glucose'] > 140).astype(int)
    input_data['Age_BMI'] = input_data['Age'] * input_data['BMI']

    # Ensure that the features match the ones used during model training
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age', 'HighGlucose', 'Age_BMI']
    return input_data[features]


# Prediction function
def predict_diabetes(inputs):
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([inputs], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # Preprocess the input data
    preprocessed_data = preprocess_input_data(input_data)

    # Predict with the loaded model
    prediction = model.predict(preprocessed_data)

    return prediction[0]


# Streamlit app UI
st.title("Pima Diabetes Predictor")
st.write("Enter the following details to predict the likelihood of having Type 2 Diabetes:")

# Input fields with default values
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1, value=3)
glucose = st.number_input("Plasma Glucose Concentration", min_value=0, max_value=200, step=1, value=120)
blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0, max_value=200, step=1, value=70)
skin_thickness = st.number_input("Triceps Skinfold Thickness (mm)", min_value=0, max_value=100, step=1, value=20)
insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0, max_value=500, step=1, value=80)
bmi = st.number_input("Body Mass Index (kg/m²)", min_value=0.0, max_value=50.0, step=0.1, value=30.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.1, value=0.5)
age = st.number_input("Age (years)", min_value=18, max_value=120, step=1, value=40)

# Collect all inputs into a list
inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

# Make prediction on button click
if st.button("Predict"):
    prediction = predict_diabetes(inputs)
    if prediction == 1:
        st.error("You have Type 2 Diabetes. Please consult a doctor.")
    else:
        st.success("You are not at risk of Type 2 Diabetes.")
