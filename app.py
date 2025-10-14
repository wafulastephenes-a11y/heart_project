import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")
st.title("Heart Check Predictor")
st.caption("Enter metrics")

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("svm_pca_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    return model, scaler, pca

model, scaler, pca = load_artifacts()

# Collect inputs in the same order and types as training
st.subheader("Health Check Form")

age = st.number_input("Age", min_value=10, max_value=120, value=56, step=1)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda v: "Male" if v == 1 else "Female")
cp = st.selectbox("Chest pain type (cp)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting blood pressure (trestbps)", min_value=80, max_value=250, value=130, step=1)
chol = st.number_input("Serum cholesterol (chol)", min_value=100, max_value=700, value=243, step=1)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda v: "Yes" if v == 1 else "No")
restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2])
thalach = st.number_input("Max heart rate (thalach)", min_value=50, max_value=250, value=153, step=1)
exang = st.selectbox("Exercise induced angina (exang)", options=[0, 1], format_func=lambda v: "Yes" if v == 1 else "No")
oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
slope = st.selectbox("Slope of ST segment (slope)", options=[0, 1, 2])
ca = st.selectbox("Number of vessels (ca)", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2])

# Assemble features in exact training order (exclude condition)
feature_order = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

row = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
X_df = pd.DataFrame([row], columns=feature_order)

st.write("Model-ready features (ordered):")
st.dataframe(X_df)

if st.button("Predict"):
    try:
        # Transform: scale -> PCA
        X_scaled = scaler.transform(X_df.values)
        X_pca = pca.transform(X_scaled)

        # Predict
        y_pred = model.predict(X_pca)[0]
        st.success("Prediction: " + ("Heart disease" if int(y_pred) == 1 else "No heart disease"))

        # Optional: probability if supported
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_pca)[0]
            proba_series = pd.Series(proba, index=["Class 0", "Class 1"])
            st.write("Class probabilities")
            st.bar_chart(proba_series)
    except Exception as e:
        st.error("Error during prediction. Check inputs and artifact compatibility.")
        st.exception(e)