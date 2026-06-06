import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="CardioCare AI", page_icon="🫀", layout="centered")

st.title("🫀 CardioCare: AI Heart Disease Screening")
st.markdown("**Developed by Izzat Sinan Shahidi (f2023-024)** | AI Lab Project")
st.info("Aligned with UN SDG 3: Good Health & Well-being. This tool provides a rapid, preliminary cardiovascular risk assessment using Machine Learning.")

@st.cache_resource # Makes the app load instantly after the first run
def train_model():
    health_data = pd.read_csv("heart.csv")
    
    X_features = health_data.drop(columns=['target'])
    y_target = health_data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

classifier = train_model()

st.sidebar.header("📋 Patient Vitals Input")

with st.sidebar.expander("Basic Demographics", expanded=True):
    patient_age = st.number_input("Age", min_value=20, max_value=100, value=50)
    patient_sex = st.selectbox("Biological Sex", options=["Male (1)", "Female (0)"])
    sex_val = 1 if "Male" in patient_sex else 0

with st.sidebar.expander("Clinical Metrics", expanded=True):
    chest_pain = st.slider("Chest Pain Type (0=None, 3=Severe)", 0, 3, 1)
    blood_pressure = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["No (0)", "Yes (1)"])
    fbs_val = 1 if "Yes" in fasting_bs else 0
baseline_metrics = [0, 150, 0, 1.0, 1, 0, 2]

st.markdown("### 🔍 Risk Analysis Engine")
st.write("Enter the patient's vitals in the sidebar and click the button below to run the Random Forest classification algorithm.")

if st.button("Generate Diagnostic Report", type="primary"): 
    patient_profile = [[patient_age, sex_val, chest_pain, blood_pressure, cholesterol, fbs_val] + baseline_metrics] 
    
    with st.spinner('Analyzing patient data...'):
        risk_prediction = classifier.predict(patient_profile)
    
    st.divider()
    if risk_prediction[0] == 1:
        st.error("#### ⚠️ High Cardiovascular Risk Detected")
        st.write("The model indicates a high probability of heart disease based on the provided metrics. Immediate clinical consultation is recommended.")
    else:
        st.success("#### ✅ Low Cardiovascular Risk")
        st.write("The model indicates a low probability of heart disease. Maintain a healthy lifestyle and continue routine checkups.")