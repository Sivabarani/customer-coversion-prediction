# Customer Conversion Analysis Using Clickstream Data

## Project Overview

This project analyzes customer browsing behavior on an e-commerce website using clickstream data. The goal is to build machine learning models that help businesses understand customer engagement and improve sales.

The system predicts:

- Whether a customer will complete a purchase
- The potential revenue from a customer session
- Customer segments based on browsing behavior

An interactive Streamlit application allows users to analyze both bulk datasets and individual customer sessions.

---

## Problem Statement

E-commerce platforms collect large amounts of clickstream data representing customer interactions with products and webpages.

This project aims to use machine learning to:

- Predict purchase conversion
- Estimate customer revenue
- Segment customers for targeted marketing

---

## Dataset

Source: UCI Machine Learning Repository

Dataset: Clickstream Data for Online Shopping

Each row represents a customer click event during a session.

### Important Features

| Feature | Description |
|------|-------------|
| session_id | Unique session identifier |
| order | Click sequence |
| country | Visitor country |
| page1_main_category | Main product category |
| page2_clothing_model | Product model |
| colour | Product color |
| location | Image location on page |
| model_photography | Photography type |
| price | Product price |
| price_2 | Price above category average |
| page | Page number within website |

---

## Feature Engineering

Since the raw dataset contains multiple clicks per session, session-level features were created:

- total_clicks
- avg_price
- total_spent
- max_price
- min_price
- unique_pages
- max_page
- unique_products
- unique_categories
- product_exploration_ratio
- page_progression_ratio

These features represent customer browsing behavior.

---

## Machine Learning Models

### Classification

Goal: Predict purchase completion.

Models tested:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

Metrics used:

- Accuracy
- Precision
- Recall
- F1 Score

---

### Regression

Goal: Predict customer revenue.

Models tested:

- Linear Regression
- Ridge
- Lasso
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Metrics used:

- RMSE
- MAE
- R² Score

---

### Clustering

Goal: Segment customers based on browsing patterns.

Models tested:

- K-Means
- DBSCAN
- Agglomerative Clustering

Evaluation metric:

- Silhouette Score

---

## ML Pipeline

Scikit-learn pipelines were used to ensure consistent preprocessing during training and prediction.
Input Features
↓
Preprocessing
(Encoding + Scaling)
↓
Machine Learning Model
↓
Prediction


---

## Experiment Tracking

MLflow was used to track model performance and store trained models.

---

## Streamlit Application

The project includes an interactive dashboard with:

### Home Page
Overview of the project.

### Bulk Customer Analyzer
Upload clickstream dataset and generate predictions.

### Single Customer Analyzer
Manually enter session data and predict:

- purchase conversion
- revenue
- customer segment

---

## Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
XGBoost  
Seaborn  
Matplotlib  
MLflow  
Streamlit  

---

## Project Structure
data/
raw dataset

src/
data_loader.py
feature_engineering.py
preprocessing_pipeline.py
train_classification.py
train_regression.py
train_clustering.py
mlflow_tracking.py
streamlit.py

models/
trained models

notebooks/
EDA analysis

---

## Future Improvements

- Deploy the application on cloud
- Add real-time prediction
- Implement recommendation systems

---

## Conclusion

This project demonstrates how machine learning can transform raw clickstream data into actionable business insights.

The combination of classification, regression, clustering, and interactive dashboards helps businesses better understand customer behavior and improve conversion strategies.





Pipeline structure:
