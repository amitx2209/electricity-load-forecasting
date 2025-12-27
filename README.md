ELECTRICITY LOAD FORECASTING USING MACHINE LEARNING
==================================================

This project focuses on forecasting future electricity demand for a selected
Indian state using historical power consumption data and multiple machine
learning regression models. The objective of the project is to analyze past
electricity usage patterns, train different machine learning models, compare
their performance, and deploy the best-performing model using a simple web
application.

The project demonstrates a complete end-to-end machine learning pipeline,
starting from raw data preprocessing to model deployment, making it suitable
for academic evaluation as well as practical understanding.


PROJECT OBJECTIVES
------------------
• Forecast next-day electricity load using historical time-series data  
• Perform comparative analysis of multiple machine learning models  
• Select the best model based on error metrics  
• Demonstrate real-world usability through a Streamlit web application  


DATASET DESCRIPTION
-------------------
• Source: Kaggle – State-wise Power Consumption in India  
• Type: Time-series electricity consumption data  
• Scope: Single-state electricity load forecasting  
• Granularity: Daily electricity consumption  

The dataset originally contains electricity consumption values for multiple
Indian states. For this project, a single state was selected and extracted to
form a univariate time-series forecasting problem.


MACHINE LEARNING MODELS USED
---------------------------
The following regression models were trained and evaluated:

• Linear Regression (baseline model)  
• Random Forest Regressor  
• Gradient Boosting Regressor  
• Support Vector Regressor (SVR)  

Training multiple models allows fair comparison and helps in selecting the most
accurate model for electricity load forecasting.


FEATURE ENGINEERING
-------------------
To capture temporal dependencies, trends, and seasonality in electricity
consumption, the following features were engineered:

• Lag features (previous day load, previous week load)  
• Rolling mean feature (7-day moving average)  
• Calendar-based features:
  - Day of month
  - Month
  - Day of week  

These features improve the learning capability of machine learning models for
time-series forecasting tasks.


MODEL EVALUATION METRICS
-----------------------
The models were evaluated using standard regression metrics:

• Mean Absolute Error (MAE)  
• Root Mean Square Error (RMSE)  
• R² Score  

The model with the lowest RMSE was selected as the final forecasting model, as
RMSE penalizes large prediction errors and is well-suited for load forecasting
problems.


MODEL SELECTION AND SAVING
--------------------------
After evaluation, the best-performing model was retrained and saved for future
predictions. The trained model is stored as a serialized file and reused for
making new electricity load predictions without retraining.


STREAMLIT WEB APPLICATION
-------------------------
A lightweight Streamlit web application was developed to demonstrate next-day
electricity load prediction using the trained machine learning model.

The application loads the saved model, prepares the most recent feature values,
and displays the predicted electricity load in a user-friendly interface.


HOW TO RUN THE APPLICATION
--------------------------
Run the following command from the project root directory:

python -m streamlit run app.py

The application will open automatically in a web browser.


TOOLS AND TECHNOLOGIES
---------------------
• Python  
• Pandas and NumPy  
• Scikit-learn  
• Jupyter Notebook  
• Streamlit  
• Git and GitHub  


CONCLUSION
----------
This project demonstrates the effective use of machine learning techniques for
electricity load forecasting. By comparing multiple regression models and
deploying the best-performing model using a web application, the project
highlights the practical applicability of machine learning in energy demand
prediction and power management systems.


AUTHOR
------
Amit Kumar  

