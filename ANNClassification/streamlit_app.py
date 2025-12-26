import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pathlib

## Load the trained regression model
BASE_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "regression_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)


## Load the encoders and scalers
with open('label_encoder_gender.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)


with open('ohe_geo.pkl', 'rb') as file:
    geo_encoder = pickle.load(file)

with open('scaler_regression.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit App Initialization
st.title("Estimated Salary Prediction")

## Input Data

geography = st.selectbox("Select Geography", geo_encoder.categories_[0])
gender = st.selectbox("Gender", gender_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure (years)', 0, 10)
number_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
# exited = st.selectbox('Exited', [0, 1])

input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    # 'Exited': [exited],
    })

geo_encodd = geo_encoder.transform([[geography]])
geo_df = pd.DataFrame(geo_encodd, columns=geo_encoder.get_feature_names_out(['Geography']))

input_df = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)

input_scaled = scaler.transform(input_df)

## Predict salary based on given input
prediction = model.predict(input_scaled)
predicted_salary = prediction[0][0]

st.write(f"Estimated Salary Prediction: ${predicted_salary:,.2f}")