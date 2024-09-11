import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Title of the app
st.title('Experience vs Salary Linear Regression')

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv(r"Experience Salary.csv")
    return data

data = load_data()

# Display the dataset
st.write("Dataset", data)

# Prepare the data
X = data[['Experience']]
y = data['Salary']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

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
