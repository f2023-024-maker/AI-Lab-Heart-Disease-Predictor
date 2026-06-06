import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
st.set_page_config(page_title="Multi-Disease AI Predictor", page_icon="⚕️", layout="wide")

with st.sidebar:
    st.title("⚕️ AI Diagnostic Tools")
    st.write("Developed by Izzat Sinan Shahidi (f2023-024)")
    st.divider()
    selected_system = st.radio("Select Screening Module:", ["❤️ Heart Disease", "🩸 Diabetes"])

@st.cache_resource
def train_heart_model():
    # Using your local dataset
    df = pd.read_csv("heart.csv")
    X, y = df.drop(columns=['target']), df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def train_diabetes_model():
    df = pd.read_csv("diabetes.csv")
    X, y = df.drop(columns=['Outcome']), df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

heart_classifier = train_heart_model()
diabetes_classifier = train_diabetes_model()

if selected_system == "❤️ Heart Disease":
    st.title("❤️ Cardiovascular Risk Assessment")
    st.info("Analyzes patient vitals to determine the likelihood of heart disease.")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male (1)", "Female (0)"])
        sex_val = 1 if "Male" in sex else 0
        cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
    with col2:
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 400, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120?", ["No (0)", "Yes (1)"])
        fbs_val = 1 if "Yes" in fbs else 0

    if st.button("Run Heart Analysis", type="primary"):
        input_data = [[age, sex_val, cp, trestbps, chol, fbs_val, 0, 150, 0, 1.0, 1, 0, 2]]
        prediction = heart_classifier.predict(input_data)
        if prediction[0] == 1:
            st.error("⚠️ High Cardiovascular Risk Detected.")
        else:
            st.success("✅ Low Cardiovascular Risk.")

elif selected_system == "🩸 Diabetes":
    st.title("🩸 Diabetes Onset Prediction")
    st.info("Analyzes metabolic metrics to predict the likelihood of diabetes.")
    
    col1, col2 = st.columns(2)
    with col1:
        age_d = st.number_input("Patient Age", 1, 100, 30)
        glucose = st.number_input("Glucose Level", 0, 300, 110)
        blood_pressure_d = st.number_input("Blood Pressure (Diastolic)", 0, 150, 70)
    with col2:
        bmi = st.number_input("BMI (Body Mass Index)", 0.0, 70.0, 25.0)
        insulin = st.number_input("Insulin Level", 0, 900, 80)
    
    if st.button("Run Diabetes Analysis", type="primary"):
        input_data = [[1, glucose, blood_pressure_d, 20, insulin, bmi, 0.5, age_d]]
        prediction = diabetes_classifier.predict(input_data)
        if prediction[0] == 1:
            st.error("⚠️ High Risk of Diabetes Detected.")
        else:
            st.success("✅ Low Risk of Diabetes.")