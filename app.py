import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/student_model.pkl")

st.title("📊 Student Performance Predictor")

st.write("Enter student details to predict exam score:")

hours = st.number_input("Hours Studied", 0, 12, 1)
attendance = st.number_input("Attendance", 0, 100, 75)
previous = st.number_input("Previous Score", 0, 100, 60)

if st.button("Predict Score"):
    input_data = pd.DataFrame(
        [[hours, attendance, previous]],
        columns=["Hours_Studied", "Attendance", "Previous_Score"]
    )

    prediction = model.predict(input_data)

    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")