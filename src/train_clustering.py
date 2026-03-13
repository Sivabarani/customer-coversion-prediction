"""
Train clustering model for customer segmentation
Save pipeline (preprocessing + KMeans)
"""

import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from preprocessing_pipeline import create_preprocessing_pipeline


def train_clustering():

    # ------------------------------------------------
    # Create models folder if not exists
    # ------------------------------------------------
    os.makedirs("models", exist_ok=True)

    # ------------------------------------------------
    # Load processed dataset
    # ------------------------------------------------
    df = pd.read_csv("data/processed/train_session_features.csv")

    # ------------------------------------------------
    # Remove target columns
    # ------------------------------------------------
    X = df.drop(["session_id", "conversion", "revenue"], axis=1)

    # ------------------------------------------------
    # Create preprocessing pipeline
    # ------------------------------------------------
    preprocessor = create_preprocessing_pipeline()

    # ------------------------------------------------
    # Create full pipeline (preprocessing + clustering)
    # ------------------------------------------------
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", KMeans(n_clusters=4, random_state=42))
    ])

    # ------------------------------------------------
    # Train clustering model
    # ------------------------------------------------
    pipeline.fit(X)

    # ------------------------------------------------
    # Evaluate clustering
    # ------------------------------------------------
    X_processed = preprocessor.fit_transform(X).toarray()

    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_processed)

    score = silhouette_score(X_processed, labels)

    print("Silhouette Score:", score)

    # ------------------------------------------------
    # Save pipeline
    # ------------------------------------------------
    joblib.dump(pipeline, "models/best_clustering_model.pkl")

    print("Clustering model saved successfully")


if __name__ == "__main__":
    train_clustering()