import streamlit as st
import pandas as pd
import joblib
import os

# Path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'titanic_survival_model.pkl')

# Load the model
model = joblib.load(model_path)

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction System")

st.write("Enter passenger details to see if they would have survived the disaster.")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3], help="1 = First, 2 = Second, 3 = Third")
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 100, 30)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=600.0, value=32.0)

if st.button("Predict Survival"):
    # Create DataFrame for prediction
    input_df = pd.DataFrame([[pclass, sex, age, sibsp, fare]], 
                            columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Fare'])
    
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        st.success("### Result: Survived ✨")
        st.balloons()
    else:
        st.error("### Result: Did Not Survive ❌")