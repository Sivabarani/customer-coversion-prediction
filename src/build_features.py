"""
Build session level feature dataset and save as CSV

This script:
1. Loads raw clickstream data
2. Creates session level features
3. Creates ML targets
4. Saves processed dataset
"""

import os
import pandas as pd

from data_loader import load_train_data, load_test_data
from feature_engineering import create_session_features, create_targets


def build_feature_dataset():

    # create folder if not exists
    os.makedirs("data/processed", exist_ok=True)

    # ---------------------------------------------------
    # TRAIN DATA
    # ---------------------------------------------------

    train_df = load_train_data()

    # convert click events → session features
    train_session = create_session_features(train_df)

    # create targets
    train_session = create_targets(train_session)

    # save processed train dataset
    train_session.to_csv(
        "data/processed/train_session_features.csv",
        index=False
    )

    print("Train feature dataset saved")

    # ---------------------------------------------------
    # TEST DATA
    # ---------------------------------------------------

    test_df = load_test_data()

    test_session = create_session_features(test_df)

    test_session = create_targets(test_session)

    test_session.to_csv(
        "data/processed/test_session_features.csv",
        index=False
    )

    print("Test feature dataset saved")


if __name__ == "__main__":
    build_feature_dataset()