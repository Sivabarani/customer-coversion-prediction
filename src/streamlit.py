import streamlit as st
import pandas as pd
import joblib

from feature_engineering import create_session_features

st.set_page_config(page_title="Customer Behavior Analyzer", layout="wide")

# -----------------------------------------------------
# Load models
# -----------------------------------------------------

classification_model = joblib.load("models/best_classification_model.pkl")
regression_model = joblib.load("models/best_regression_model.pkl")
clustering_model = joblib.load("models/best_clustering_model.pkl")

# -----------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Bulk Customer Analyzer", "Single Customer Analyzer"]
)

# -----------------------------------------------------
# HOME PAGE
# -----------------------------------------------------

if page == "Home":

    st.title("Customer Conversion Analysis")

    st.write(
        """
        This application analyzes customer browsing behavior using machine learning.

        Features:

        • Predict customer **purchase conversion**  
        • Estimate **customer revenue**  
        • Segment customers using **clustering**

        Navigate using the sidebar.
        """
    )

# -----------------------------------------------------
# BULK CUSTOMER ANALYZER
# -----------------------------------------------------

elif page == "Bulk Customer Analyzer":

    st.title("Bulk Customer Analyzer")

    uploaded_file = st.file_uploader("Upload raw clickstream CSV", type=["csv"])

    if uploaded_file:

        raw_df = pd.read_csv(uploaded_file)

        st.subheader("Raw Uploaded Data")

        st.dataframe(raw_df.head())

        # -------------------------------------------
        # Convert raw click data → session features
        # -------------------------------------------

        session_df = create_session_features(raw_df)

        st.subheader("Session Feature Dataset")

        st.dataframe(session_df.head())

        X = session_df.drop(["session_id"], axis=1)

        # -------------------------------------------
        # Clustering
        # -------------------------------------------

        st.subheader("Customer Segmentation")

        clusters = clustering_model.predict(X)

        session_df["cluster"] = clusters

        st.bar_chart(session_df["cluster"].value_counts())

        # -------------------------------------------
        # Classification
        # -------------------------------------------

        st.subheader("Conversion Prediction")

        conversion_preds = classification_model.predict(X)

        session_df["predicted_conversion"] = conversion_preds

        st.bar_chart(session_df["predicted_conversion"].value_counts())

        # -------------------------------------------
        # Regression
        # -------------------------------------------

        st.subheader("Revenue Prediction")

        revenue_preds = regression_model.predict(X)

        session_df["predicted_revenue"] = revenue_preds

        st.write(session_df[["session_id", "predicted_revenue"]].head())

        # -------------------------------------------
        # Download results
        # -------------------------------------------

        csv = session_df.to_csv(index=False)

        st.download_button(
            "Download Results",
            csv,
            "customer_predictions.csv"
        )

# -----------------------------------------------------
# SINGLE CUSTOMER ANALYZER
# -----------------------------------------------------

elif page == "Single Customer Analyzer":

    st.title("Single Customer Analyzer")

    st.write("Enter one customer click event")

    col1, col2, col3 = st.columns(3)

    order = col1.number_input("Order", 1, 100, 5)
    country = col2.number_input("Country", 1, 50, 20)
    session_id = col3.number_input("Session ID", 1, 999999, 1001)

    page_category = col1.number_input("Main Category", 1, 4, 1)
    clothing_model = col2.number_input("Clothing Model", 1, 500, 50)

    colour = col3.number_input("Colour", 1, 14, 2)

    location = col1.number_input("Location", 1, 6, 1)
    model_photo = col2.number_input("Model Photography", 1, 2, 1)

    price = col3.number_input("Price", 1, 500, 50)

    price2 = col1.number_input("Price above average", 1, 2, 1)

    page_num = col2.number_input("Page", 1, 5, 3)

    if st.button("Run Prediction"):

        # -------------------------------------------
        # Create raw click dataframe
        # -------------------------------------------

        raw_df = pd.DataFrame([[
            2024,
            6,
            1,
            order,
            country,
            session_id,
            page_category,
            clothing_model,
            colour,
            location,
            model_photo,
            price,
            price2,
            page_num
        ]], columns=[
            "year",
            "month",
            "day",
            "order",
            "country",
            "session_id",
            "page1_main_category",
            "page2_clothing_model",
            "colour",
            "location",
            "model_photography",
            "price",
            "price_2",
            "page"
        ])

        # -------------------------------------------
        # Convert to session features
        # -------------------------------------------

        session_df = create_session_features(raw_df)

        X = session_df.drop(["session_id"], axis=1)

        # -------------------------------------------
        # Run models
        # -------------------------------------------

        conversion = classification_model.predict(X)[0]
        revenue = regression_model.predict(X)[0]
        cluster = clustering_model.predict(X)[0]

        st.subheader("Prediction Results")

        if conversion == 1:
            st.success("Customer likely to purchase")
        else:
            st.warning("Customer unlikely to purchase")

        st.write("Predicted Revenue:", round(revenue, 2))
        st.write("Customer Cluster:", cluster)