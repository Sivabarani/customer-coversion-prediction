"""
Train multiple regression models to predict customer revenue.

Steps
1. Load processed feature dataset
2. Split train and validation
3. Apply preprocessing pipeline
4. Train multiple regression models
5. Compare model performance
6. Save best model
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from preprocessing_pipeline import create_preprocessing_pipeline


def train_regression():

    # ---------------------------------------------------
    # Ensure models folder exists
    # ---------------------------------------------------

    os.makedirs("models", exist_ok=True)

    # ---------------------------------------------------
    # Load processed dataset
    # ---------------------------------------------------

    df = pd.read_csv("data/processed/train_session_features.csv")

    # ---------------------------------------------------
    # Define features and target
    # ---------------------------------------------------

    X = df.drop(["session_id", "conversion", "revenue"], axis=1)

    y = df["revenue"]

    # ---------------------------------------------------
    # Train validation split
    # ---------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------------------------------------------
    # Preprocessing pipeline
    # ---------------------------------------------------

    preprocessor = create_preprocessing_pipeline()

    # ---------------------------------------------------
    # Define regression models
    # ---------------------------------------------------

    models = {

        "Linear Regression":
        LinearRegression(),

        "Ridge":
        Ridge(),

        "Lasso":
        Lasso(),

        "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

        "Gradient Boosting":
        GradientBoostingRegressor(),

        "XGBoost":
        XGBRegressor()
    }

    results = []

    best_model = None
    best_score = float("inf")
    best_model_name = ""

    # ---------------------------------------------------
    # Train models
    # ---------------------------------------------------

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        results.append([name, rmse, mae, r2])

        print("\nModel:", name)
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R2:", r2)

        # lower RMSE is better
        if rmse < best_score:
            best_score = rmse
            best_model = pipeline
            best_model_name = name

    # ---------------------------------------------------
    # Save best model
    # ---------------------------------------------------

    joblib.dump(best_model, "models/best_regression_model.pkl")

    print("\nBest Model:", best_model_name)
    print("Best RMSE:", best_score)

    # ---------------------------------------------------
    # Model comparison table
    # ---------------------------------------------------

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "RMSE",
            "MAE",
            "R2 Score"
        ]
    )

    print("\nModel Comparison:")
    print(results_df)


if __name__ == "__main__":

    train_regression()