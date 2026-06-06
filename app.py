import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("❤️ Heart Disease Risk Predictor")
st.write("Aligns with UN SDG 3: Good Health & Well-being")

df = pd.read_csv("heart.csv")


X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)


st.sidebar.header("Patient Health Metrics")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", value=120)
chol = st.sidebar.number_input("Serum Cholestoral (mg/dl)", value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])


if st.button("Analyze Risk"): 
    input_data = [[age, sex, cp, trestbps, chol, fbs, 0, 150, 0, 1.0, 1, 0, 2]] 
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease detected. Please consult a doctor.")
    else:
        st.success("✅ Low Risk of Heart Disease detected.")