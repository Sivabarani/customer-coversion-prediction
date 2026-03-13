"""
Data Loader Module
Loads training and testing datasets
"""

import pandas as pd


def load_train_data(path="data/train.csv"):
    """
    Load training dataset
    """
    df = pd.read_csv(path)
    return df


def load_test_data(path="data/test.csv"):
    """
    Load testing dataset
    """
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":

    train_df = load_train_data()
    test_df = load_test_data()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)