import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Title of the app
st.title('Experience vs Salary Linear Regression')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset", data)
    
    # Prepare the data
    X = data[['Experience']]
    y = data['Salary']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Train the model
    @st.cache_resource
    def create_model():
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    model = create_model()

    # Model accuracy
    accuracy = model.score(X_test, y_test)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Model coefficients
    coef = model.coef_[0]
    intercept = model.intercept_
    st.write(f"Model Coefficient (m): {coef:.2f}")
    st.write(f"Model Intercept (b): {intercept:.2f}")

    # Predict salary based on user input
    experience_input = st.slider('Select years of experience:', 1, 20, 5)
    predicted_salary = model.predict(np.array([[experience_input]]))
    st.write(f"Predicted salary for {experience_input} years of experience: ${predicted_salary[0]:,.2f}")

    # Plot the data points and the regression line
    st.subheader('Regression Line vs Data Points')

    fig, ax = plt.subplots()
    ax.scatter(data['Experience'], data['Salary'], color='r', marker='.')
    ax.plot(data['Experience'], model.predict(X), color='blue')
    ax.set_xlabel('Years of Experience')
    ax.set_ylabel('Salary')
    ax.set_title('Experience vs Salary')

    st.pyplot(fig)

    # Example usage of sys
    st.write("Python version:", sys.version)
else:
    st.write("Please upload a CSV file.")
