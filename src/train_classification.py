"""
Train multiple classification models to predict customer conversion.

Steps:
1. Load processed feature dataset
2. Split into train and validation
3. Apply preprocessing pipeline
4. Train multiple models
5. Compare model performance
6. Save the best model
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

from preprocessing_pipeline import create_preprocessing_pipeline


def train_classification():

    # --------------------------------------------------
    # Ensure models directory exists
    # --------------------------------------------------
    os.makedirs("models", exist_ok=True)

    # --------------------------------------------------
    # Load processed dataset
    # --------------------------------------------------
    df = pd.read_csv("data/processed/train_session_features.csv")

    # --------------------------------------------------
    # Define features and target
    # --------------------------------------------------

    X = df.drop(["session_id", "conversion", "revenue"], axis=1)

    y = df["conversion"]

    # --------------------------------------------------
    # Train validation split
    # --------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # Preprocessing pipeline
    # --------------------------------------------------

    preprocessor = create_preprocessing_pipeline()

    # --------------------------------------------------
    # Define multiple models
    # --------------------------------------------------

    models = {

        "Logistic Regression":
        LogisticRegression(max_iter=1000),

        "Decision Tree":
        DecisionTreeClassifier(max_depth=10),

        "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "Gradient Boosting":
        GradientBoostingClassifier(),

        "XGBoost":
        XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False
        )
    }

    results = []

    best_model = None
    best_score = 0
    best_model_name = ""

    # --------------------------------------------------
    # Train each model
    # --------------------------------------------------

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_val)

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds)
        rec = recall_score(y_val, preds)
        f1 = f1_score(y_val, preds)

        results.append([name, acc, prec, rec, f1])

        print("\nModel:", name)
        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("F1 Score:", f1)

        if acc > best_score:
            best_score = acc
            best_model = pipeline
            best_model_name = name

    # --------------------------------------------------
    # Save best model
    # --------------------------------------------------

    joblib.dump(best_model, "models/best_classification_model.pkl")

    print("\nBest Model:", best_model_name)
    print("Best Accuracy:", best_score)

    # --------------------------------------------------
    # Model comparison table
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    )

    print("\nModel Comparison:")
    print(results_df)


if __name__ == "__main__":

    train_classification()