Experience vs Salary Linear Regression App

Project Overview

This project demonstrates a simple linear regression model using Streamlit to predict salaries based on years of experience. The model is trained on a dataset and provides real-time predictions and visualizations through a web application.

Objective: Predict salary based on years of experience using a simple linear regression model.
Libraries Used:
pandas for data handling
numpy for numerical operations
matplotlib for plotting
scikit-learn for creating and evaluating the linear regression model
streamlit for creating the web application
Installation

To run this project, you need to install the required Python libraries. You can do this using pip:


pip install pandas numpy matplotlib scikit-learn streamlit
Dataset

The dataset used in this project is a CSV file named Experience Salary.csv containing the following columns:

Experience: Years of experience
Salary: Annual salary in USD
Example data:


Experience,Salary
1,30000
2,35000
3,40000
4,45000
5,50000
6,55000
7,60000
8,65000
9,70000
10,75000
How to Run

Clone this repository to your local machine:


git clone https://github.com/yourusername/your-repository.git
Navigate to the project directory:


cd your-repository
Run the Streamlit app:


streamlit run app.py
Open your web browser and go to http://localhost:8501 to interact with the app.

Features

Predict Salary: Input years of experience to get the predicted salary.
Visualize Data: See a scatter plot of the dataset with the regression line fitted to it.
Model Metrics: View the model's accuracy, coefficient, and intercept.
Code Explanation

Data Loading: Uses pandas to load and cache the dataset.
Model Training: Trains a LinearRegression model from scikit-learn using the training data.
Prediction: Allows users to input years of experience and get salary predictions.
Visualization: Uses matplotlib to display the regression line and data points.
Contributing

If you want to contribute to this project, feel free to fork the repository and submit a pull request. Any improvements or bug fixes are welcome!

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements

Streamlit for creating interactive web apps.
scikit-learn for machine learning tools.
matplotlib for plotting.
