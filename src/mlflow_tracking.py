"""
MLflow experiment tracking for classification and regression models
"""

import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.linear_model import LogisticRegression, LinearRegression

from preprocessing_pipeline import create_preprocessing_pipeline


def run_mlflow():

    # --------------------------------------------
    # Load dataset
    # --------------------------------------------

    df = pd.read_csv("data/processed/train_session_features.csv")

    X = df.drop(["session_id", "conversion", "revenue"], axis=1)

    y_class = df["conversion"]

    y_reg = df["revenue"]

    # --------------------------------------------
    # Train test split
    # --------------------------------------------

    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    _, _, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    preprocessor = create_preprocessing_pipeline()

    # --------------------------------------------
    # MLflow experiment
    # --------------------------------------------

    mlflow.set_experiment("Clickstream Customer Analysis")

    with mlflow.start_run():

        # ---------------------------
        # Classification model
        # ---------------------------

        clf_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000))
        ])

        clf_pipeline.fit(X_train, y_train_class)

        preds_class = clf_pipeline.predict(X_test)

        accuracy = accuracy_score(y_test_class, preds_class)

        mlflow.log_metric("classification_accuracy", accuracy)

        mlflow.sklearn.log_model(clf_pipeline, "classification_model")

        # ---------------------------
        # Regression model
        # ---------------------------

        reg_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ])

        reg_pipeline.fit(X_train, y_train_reg)

        preds_reg = reg_pipeline.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test_reg, preds_reg))

        mlflow.log_metric("regression_rmse", rmse)

        mlflow.sklearn.log_model(reg_pipeline, "regression_model")

        print("MLflow experiment logged successfully")


if __name__ == "__main__":

    run_mlflow()




    