# Estimated Salary Prediction App

A Streamlit web application that predicts customer estimated salary using an Artificial Neural Network (ANN) regression model trained on banking customer data.

## Overview

This application uses machine learning to predict a customer's estimated annual salary based on their banking profile and demographic information. The model was trained on the Churn_Modelling dataset and uses features like credit score, age, balance, geography, and banking behavior patterns.

## Web Application
- The trained model is deployed with user interface on Streamlit

Site: https://cbdwcaevwqfyak2zhgsxgl.streamlit.app

## Model Details

- **Algorithm**: Artificial Neural Network (ANN) with regression output
- **Architecture**:
  - Input : 11 features
  - Hidden layer 1: 64 neurons with ReLU activation
  - Hidden layer 2: 32 neurons with ReLU activation
  - Output layer: 1 neuron (linear activation for regression)
- **Loss Function**: Mean Absolute Error (MAE)
- **Optimizer**: Adam
- **Training**: Early stopping with patience of 10 epochs

## Performance

The model was evaluated using:
- Mean Absolute Error (MAE) as the primary metric
- Validation loss monitoring with early stopping
- Test set evaluation for final performance assessment
