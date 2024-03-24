import streamlit as st
import pandas as pd
import joblib
import json

# Load data from JSON file
with open("../model/columns.json") as f:
    columns_data = json.load(f)
data_columns = columns_data[0]["data_columns"]

# Load model
model = joblib.load("../model/bangalore_home_prices_model.pickle")

# Define Streamlit app
def main():
    st.title("House Price Predictor")

    st.write("This tool will help you estimate the price of your home based on key features.")

    # Display feature selection options
    selected_features = select_features(data_columns)

    if selected_features:
        # Button to trigger price prediction
        if st.sidebar.button("Predict Price"):
            # Predict price based on selected features
            predicted_price = predict_price(selected_features, data_columns)
            st.success(f"Estimated price: INR {predicted_price:.1f} LAKH RS")

# Function to select features
def select_features(data_columns):
    selected_features = {}
    st.sidebar.title("Select Features")

    # Area (Sq Ft)
    selected_features["Area (Sq Ft)"] = st.sidebar.slider("Area (Sq Ft)", 0, 10000, step=50)

    # BHK (Bedroom, Hall, Kitchen)
    selected_features["BHK (Bedroom, Hall, Kitchen)"] = st.sidebar.selectbox("BHK (Bedroom, Hall, Kitchen)", [1, 2, 3, 4, 5])

    # Bath (Bathroom)
    selected_features["Bath (Bathroom)"] = st.sidebar.selectbox("Bath (Bathroom)", [1, 2, 3, 4, 5])

    # Location (Dropdown menu)
    selected_features["Location"] = st.sidebar.selectbox("Location", data_columns[3:])

    return selected_features

# Function to predict price
def predict_price(selected_features, data_columns):
    # Create input data based on selected features
    input_data = [[
        selected_features["Area (Sq Ft)"],
        selected_features["BHK (Bedroom, Hall, Kitchen)"],
        selected_features["Bath (Bathroom)"]
    ]]
    
    # Add one-hot encoded columns to input data
    for column in data_columns[3:]:
        if column == selected_features["Location"]:
            input_data[0].append(1)  # Set 1 for selected location
        else:
            input_data[0].append(0)

    # Predict price using the model
    predicted_price = model.predict(input_data)
    return predicted_price[0]

if __name__ == "__main__":
    main()
