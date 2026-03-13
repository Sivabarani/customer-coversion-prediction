"""
Preprocessing pipeline using sklearn
Handles encoding and scaling
"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def create_preprocessing_pipeline():

    numeric_features = [
        "total_clicks",
        "avg_price",
        "max_price",
        "min_price",
        "unique_pages",
        "unique_products"
    ]

    categorical_features = [
        "country"
    ]

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([

        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)

    ])

    return preprocessor