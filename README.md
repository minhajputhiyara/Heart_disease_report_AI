# Heart Disease Prediction Model with XGBoost and SHAP Explainability

## Overview

This repository contains a project focused on predicting heart disease using the XGBoost algorithm and explaining the model's predictions using SHAP (SHapley Additive exPlanations). The model is fine-tuned using GridSearchCV and performs feature importance analysis to interpret the results. Additionally, the project includes a **Streamlit** web application that allows users to interact with the model and visualize its predictions.

## Table of Contents

1. [Project Description](#project-description)
2. [Dataset Information](#dataset-information)
3. [Installation](#installation)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Model Explainability using SHAP](#model-explainability-using-shap)
7. [Saving and Loading the Model](#saving-and-loading-the-model)
8. [Streamlit Application](#streamlit-application)
9. [Chatbot Integration with HuggingFace](#chatbot-integration-with-huggingface)
10. [Results](#results)
11. [Usage](#usage)
12. [Contributing](#contributing)
13. [License](#license)

---

## Project Description

The goal of this project is to predict the likelihood of heart disease in patients using a dataset containing various medical features. The model is built using the XGBoost algorithm for classification, and SHAP is used for explainability to interpret the importance of different features. The project also integrates with HuggingFace's language model to generate natural language explanations of the model's predictions. A **Streamlit** app is also provided to make the model accessible via a web-based interface.

## Dataset Information

- **Dataset Name**: [Heart Disease Dataset](https://www.kaggle.com/ronitf/heart-disease-uci)
- **Columns**: 
  - Age, Gender, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise-Induced Angina, ST Depression, etc.
  - Target: Indicates the presence (1) or absence (0) of heart disease.
  
## Installation

To get this project up and running on your local machine, follow these steps:

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

### 2. Install the required dependencies:
bash
Copy code
pip install -r requirements.txt

### 3. Required Libraries

- `xgboost`
- `pandas`
- `numpy`
- `sklearn`
- `shap`
- `matplotlib`
- `seaborn`
- `pickle`
- `langchain_huggingface`
- `streamlit`

### 4. Installation

To install the required libraries, you can use pip. Run the following command in your terminal:

```bash
pip install xgboost pandas numpy sklearn shap matplotlib seaborn langchain_huggingface streamlit

### 5. Running Streamlit app

  streamlit run main.py

