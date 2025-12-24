import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import tensorflow as tf
import pathlib

## Load the trained model 
BASE_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

#load encoders and scalers
with open('./label_encoder_gender.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)

with open('./ohe_geo.pkl', 'rb') as file:
    geo_encoder = pickle.load(file)

with open('./scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit App Initialization

st.title('Customer Churn Prediction')

## Input Fields
geography = st.selectbox('Geography', geo_encoder.categories_[0])
gender = st.selectbox('Gender', gender_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure (years)', 0, 10)
number_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])



input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    })

encode_geo = geo_encoder.transform([[geography]])
geo_df = pd.DataFrame(encode_geo, columns=geo_encoder.get_feature_names_out(['Geography']))

input_df = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)

scaled_features = scaler.transform(input_df)

## Make prediction for the input data

prediction = model.predict(scaled_features)
churn_probability = prediction[0][0]

if churn_probability and churn_probability > 0.5:
    st.write(f'The customer is likely to churn. Churn probability: {churn_probability:.2f}')
else:
    st.write(f'The customer is unlikely to churn. Churn probability: {churn_probability:.2f}')