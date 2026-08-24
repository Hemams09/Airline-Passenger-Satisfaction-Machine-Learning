# Airline-Passenger-Satisfaction-Machine-Learning
Machine learning project for predicting airline passenger satisfaction using 7 classification algorithms, with XGBoost achieving 96% accuracy and deployed through Streamlit.

**Airline Passenger Satisfaction Prediction Using Machine Learning**

Project Overview

This project develops a machine learning-based system to predict airline passenger satisfaction using passenger demographic, travel, service, and flight-related information.

A total of 7 supervised classification algorithms were trained and compared to identify the best-performing model. Among the tested models, XGBoost achieved the highest test accuracy of 96.00% and was selected as the final model.

The trained model is integrated into a Streamlit web application for passenger-level satisfaction prediction.

**Objectives**

The main objectives of this project are:

Analyze airline passenger satisfaction data
Clean and preprocess the dataset
Convert categorical data into numerical form
Perform exploratory data analysis
Train multiple machine learning classification algorithms
Compare model performance
Identify important factors influencing passenger satisfaction
Select the best-performing model
Deploy the final model using Streamlit

**Dataset**

The project uses an airline passenger satisfaction dataset containing:

25,974 passenger records

The dataset includes information related to:

Passenger demographics
Customer type
Type of travel
Travel class
Flight information
Service ratings
Flight delays
Passenger satisfaction
Target Variable

The target variable represents passenger satisfaction.

**Machine Learning Models**

Seven supervised classification algorithms were trained and compared:

Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)
Naïve Bayes
Support Vector Machine (SVM)
XGBoost

These models were selected to compare different classification approaches and determine which algorithm performs best on the dataset.

Model Comparison
Model	Test Accuracy
XGBoost	96.00%
Random Forest	95.80%
Support Vector Machine (SVM)	94.84%
Decision Tree	93.46%
K-Nearest Neighbors (KNN)	91.72%
Logistic Regression	86.51%
Naïve Bayes	85.10%
⭐ Best Model: XGBoost

XGBoost achieved 96.00% test accuracy, which was the highest among the seven tested algorithms.

Therefore, XGBoost was selected as the final model for the application.

**Project Workflow**
Dataset
   ↓
Data Cleaning
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Exploratory Data Analysis
   ↓
Train 7 Machine Learning Models
   ↓
Model Evaluation
   ↓
Comparative Analysis
   ↓
Select XGBoost
   ↓
Streamlit Application
   ↓
Passenger Satisfaction Prediction

**Important Features**

The analysis identified several important factors related to passenger satisfaction, including:

Online Boarding
Inflight Wi-Fi Service
Type of Travel
Travel Class
Inflight Entertainment

These features provide useful insights into factors that influence the passenger experience.

**Streamlit Application**

The final XGBoost model is integrated with a Streamlit application.

The application allows users to enter passenger-related information and obtain a satisfaction prediction.

Application Flow
Passenger Details
       ↓
Preprocessing
       ↓
XGBoost Model
       ↓
Prediction
       ↓
Passenger Satisfaction Result

**Technologies Used**

Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Seaborn
Streamlit
Jupyter Notebook
Model Evaluation

The models were compared using machine learning evaluation techniques including:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
ROC-AUC
Cross-validation

XGBoost achieved:

Test Accuracy: 96.00%
5-Fold Cross-Validation Mean: 0.96
ROC-AUC: 0.98–0.99

**Results**

The project successfully developed an end-to-end machine learning framework for airline passenger satisfaction prediction.

The comparative analysis showed that XGBoost performed the best with 96.00% test accuracy. The model was therefore selected for deployment through the Streamlit application.

**Future Scope**

Future improvements can include:

Using larger and more diverse airline passenger datasets
Integrating real-time passenger feedback
Exploring deep learning models
Exploring hybrid ensemble approaches
Adding advanced feature engineering
Implementing continuous model monitoring
Connecting predictions with airline service-management workflows
