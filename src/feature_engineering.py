"""
Feature Engineering for Clickstream Data

Goal:
The raw dataset contains ONE ROW PER CLICK.
But ML models should learn from ONE ROW PER SESSION.

Therefore we aggregate click events → session level features.
"""

import pandas as pd


def create_session_features(df):
    """
    Convert clickstream events into session level dataset
    """

    # -----------------------------------------------------------
    # GROUP BY SESSION
    # -----------------------------------------------------------
    # Reason:
    # A session represents a single user visit.
    # We want to summarize all clicks of that visit into one row.
    # -----------------------------------------------------------

    session_df = df.groupby("session_id").agg({

        # -------------------------------------------------------
        # NUMBER OF CLICKS IN SESSION
        # -------------------------------------------------------
        # order column represents the sequence of clicks.
        # Counting it gives total user interactions.
        # Higher clicks → higher engagement.
        # -------------------------------------------------------
        "order": "count",

        # -------------------------------------------------------
        # PRICE STATISTICS
        # -------------------------------------------------------
        # mean → average price of viewed products
        # sum → total value of viewed items (proxy for spending intent)
        # max → most expensive item user explored
        # min → cheapest item viewed
        # -------------------------------------------------------
        "price": ["mean", "sum", "max", "min"],

        # -------------------------------------------------------
        # PAGE DEPTH FEATURES
        # -------------------------------------------------------
        # nunique → how many different pages visited
        # max → deepest page reached in the website funnel
        #
        # Page depth is important because:
        # page 1 → browsing
        # page 5 → checkout stage
        # -------------------------------------------------------
        "page": ["nunique", "max"],

        # -------------------------------------------------------
        # COUNTRY
        # -------------------------------------------------------
        # Country does not change within session
        # so taking the first value is safe
        # -------------------------------------------------------
        "country": "first",

        # -------------------------------------------------------
        # PRODUCT DIVERSITY
        # -------------------------------------------------------
        # nunique models → how many different products viewed
        # Higher diversity may indicate exploratory browsing
        # -------------------------------------------------------
        "page2_clothing_model": "nunique",

        # -------------------------------------------------------
        # CATEGORY DIVERSITY
        # -------------------------------------------------------
        # number of different clothing categories visited
        # Example: trousers, skirts, sale, etc
        # -------------------------------------------------------
        "page1_main_category": "nunique"

    })

    # -----------------------------------------------------------
    # FLATTEN MULTI-LEVEL COLUMN NAMES
    # -----------------------------------------------------------
    # After aggregation pandas creates multi-index columns.
    # We convert them into simple column names.
    # -----------------------------------------------------------

    session_df.columns = [

        "total_clicks",        # count(order)

        "avg_price",           # mean(price)
        "total_spent",         # sum(price)
        "max_price",           # max(price)
        "min_price",           # min(price)

        "unique_pages",        # nunique(page)
        "max_page",            # max(page)

        "country",

        "unique_products",     # nunique(page2_clothing_model)

        "unique_categories"    # nunique(page1_main_category)
    ]

    # -----------------------------------------------------------
    # RESET INDEX
    # -----------------------------------------------------------
    # session_id becomes index after groupby.
    # We convert it back to a column.
    # -----------------------------------------------------------

    session_df = session_df.reset_index()

    # -----------------------------------------------------------
    # BEHAVIORAL FEATURES
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # PRODUCT INTERACTION RATE
    # -----------------------------------------------------------
    # Measures how many unique products user explored
    # relative to clicks.
    # -----------------------------------------------------------
    session_df["product_exploration_ratio"] = (
        session_df["unique_products"] / session_df["total_clicks"]
    )

    # -----------------------------------------------------------
    # PAGE PROGRESSION
    # -----------------------------------------------------------
    # Measures how far user progressed in funnel.
    # -----------------------------------------------------------
    session_df["page_progression_ratio"] = (
        session_df["max_page"] / session_df["unique_pages"]
    )

    return session_df


def create_targets(df):
    """
    Create targets for ML tasks
    """
    df["conversion"] = (df["max_page"] >= 5).astype(int)

    df["revenue"] = df["total_spent"]

    return df


if __name__ == "__main__":

    from data_loader import load_train_data

    df = load_train_data()

    session_df = create_session_features(df)

    session_df = create_targets(session_df)

    print(session_df.head())

    print("\nClass Distribution:")
    print(session_df["conversion"].value_counts())