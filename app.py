import streamlit as st
import numpy as np
import pickle

# Load trained logistic regression model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Title of the app
st.title("Titanic Survival Prediction App")
st.write("Enter passenger details below to predict survival probability:")

# User input fields
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=30.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Preprocessing for model input
sex_encoded = 1 if sex == "male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_map[embarked]

# Prepare input for prediction
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# Predict button
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🎉 The passenger is likely to **Survive** (Probability: {probability:.2f})")
    else:
        st.error(f"❌ The passenger is likely **Not to Survive** (Probability: {probability:.2f})")
