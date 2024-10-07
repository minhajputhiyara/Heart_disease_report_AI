import streamlit as st
import pickle
import numpy as np
import shap
from langchain_huggingface import HuggingFaceEndpoint
import xgboost as xgb

# Load the model
filename = 'heart_model1.pkl'
with open(filename, 'rb') as file:
    classifier = pickle.load(file)

# Load feature names
feature_names_exp = [
    'Age of the individual', 'Gender of the individual', 'Chest pain type experienced by the individual',
    'Resting blood pressure upon admission to the hospital', 'Serum cholesterol level', 'Fasting blood sugar',
    'Resting electrocardiographic results', 'Maximum heart rate achieved during the Thallium stress test',
    'Exercise-induced angina', 'ST depression induced by exercise relative to rest',
    'Slope of the peak exercise ST segment', 'Number of major vessels colored by fluoroscopy', 'Thalassemia type'
]

# Streamlit UI
st.title("Heart Disease Prediction and Explanation")
st.write("Please input the following features:")

# Input fields
age = st.number_input('Age:', min_value=0)
gender = st.selectbox('Gender:', ['Male', 'Female'])
chest_pain_type = st.selectbox('Chest Pain Type:', [0, 1, 2, 3])  # Replace with actual values
resting_blood_pressure = st.number_input('Resting Blood Pressure:', min_value=0)
serum_cholesterol = st.number_input('Serum Cholesterol Level:', min_value=0)
fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl:', [0, 1])  # 0: No, 1: Yes
rest_ecg = st.selectbox('Resting Electrocardiographic Results:', [0, 1, 2])
max_heart_rate = st.number_input('Maximum Heart Rate:', min_value=0)
exercise_angina = st.selectbox('Exercise-Induced Angina:', [0, 1])
st_depression = st.number_input('ST Depression:', min_value=0.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment:', [0, 1, 2])
num_vessels = st.number_input('Number of Major Vessels Colored by Fluoroscopy:', min_value=0, max_value=3)
thalassemia = st.selectbox('Thalassemia Type:', [0, 1, 2, 3])  # Replace with actual values

# Button to make prediction
import matplotlib.pyplot as plt  # Add this import
if st.button("Predict"):
    # Prepare input data
    gender = 1 if gender == 'Male' else 0  # Encode gender as numeric
    input_data = np.array([[float(age), float(gender), float(chest_pain_type), float(resting_blood_pressure),
                             float(serum_cholesterol), float(fasting_blood_sugar), float(rest_ecg),
                             float(max_heart_rate), float(exercise_angina), float(st_depression),
                             float(slope), float(num_vessels), float(thalassemia)]])

    # Make prediction directly using the input data
    prediction = classifier.predict(input_data)
    st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")

    # Explain prediction with SHAP
    explainer = shap.Explainer(classifier, input_data)
    shap_values = explainer(input_data)

    # Create prompt for Hugging Face LLM
    abs_shap_values = np.abs(shap_values.values[0])
    sorted_indices = np.argsort(abs_shap_values)[::-1]
    sorted_abs_shap_values = abs_shap_values[sorted_indices]
    sorted_feature_names_exp = np.array(feature_names_exp)[sorted_indices]

    # Construct the prompt
    top_features = [
        f"{feature}: {round(value, 2)}"
        for feature, value in zip(sorted_feature_names_exp, sorted_abs_shap_values)
    ][:5]
    prompt = ("Based on the model's analysis, the prediction was primarily influenced by the following factors: "
              + ", ".join(top_features) +
              ". Could you provide a detailed explanation of why these factors are significant?")

    # Call Hugging Face LLM
    huggingfacehub_api_token = "YOUR API HERE"
    llm = HuggingFaceEndpoint(repo_id='tiiuae/falcon-7b-instruct',
    huggingfacehub_api_token=huggingfacehub_api_token)
    bot_response = llm.invoke(prompt,temperature=0.2, top_k=10)

    # Display the LLM response
    st.subheader("LLM Explanation")
    st.write(bot_response)
