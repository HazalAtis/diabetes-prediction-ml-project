
import gradio as gr
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib

# Load model and scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Top selected features based on importance/EDA
FEATURES = [
    "HighBP", "HighChol", "BMI", "Smoker", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump", "GenHlth", "Age"
]

def predict_diabetes(*inputs):
    data = pd.DataFrame([inputs], columns=FEATURES)
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]
    label = "Diabetic" if prediction == 1 else "Not Diabetic"
    return f"{label} (Risk Score: {probability:.2f})"

inputs = [
    gr.Radio(["0", "1"], label="High Blood Pressure"),
    gr.Radio(["0", "1"], label="High Cholesterol"),
    gr.Number(label="BMI"),
    gr.Radio(["0", "1"], label="Smoker"),
    gr.Radio(["0", "1"], label="Physically Active"),
    gr.Radio(["0", "1"], label="Eats Fruit Regularly"),
    gr.Radio(["0", "1"], label="Eats Vegetables Regularly"),
    gr.Radio(["0", "1"], label="Heavy Alcohol Consumption"),
    gr.Slider(1, 5, step=1, label="General Health (1=Excellent, 5=Poor)"),
    gr.Slider(1, 13, step=1, label="Age Group (1=18-24, ..., 13=80+)")
]

demo = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction"),
    title="🩺 Diabetes Risk Predictor",
    description="Enter your health indicators to predict diabetes risk.",
)

demo.launch()
