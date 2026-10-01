import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.("carpredict_model.pkl")

# Load encoders
encoders = joblib.load("encoders.pkl")


# Page title
st.title("🚗 Car Price Prediction")

st.write("Enter the car details to predict the selling price.")


# Car Name
name = st.text_input(
    "Car Name",
    "Maruti Swift Dzire VDI"
)

# Year
year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2026,
    value=2015
)

# Kilometers Driven
km_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=50000
)

# Fuel
fuel = st.selectbox(
    "Fuel Type",
    encoders["fuel"].classes_
)

# Seller Type
seller_type = st.selectbox(
    "Seller Type",
    encoders["seller_type"].classes_
)

# Transmission
transmission = st.selectbox(
    "Transmission",
    encoders["transmission"].classes_
)

# Owner
owner = st.selectbox(
    "Owner",
    encoders["owner"].classes_
)

# Mileage
mileage = st.number_input(
    "Mileage (km/ltr/kg)",
    min_value=0.0,
    value=20.0
)

# Engine
engine = st.number_input(
    "Engine (CC)",
    min_value=0.0,
    value=1200.0
)

# Max Power
max_power = st.number_input(
    "Max Power",
    min_value=0.0,
    value=80.0
)

# Seats
seats = st.number_input(
    "Seats",
    min_value=2.0,
    max_value=10.0,
    value=5.0
)


# Prediction
if st.button("Predict Price"):

    try:

        # Convert text values into numbers
        name_encoded = encoders["name"].transform([name])[0]
        fuel_encoded = encoders["fuel"].transform([fuel])[0]
        seller_encoded = encoders["seller_type"].transform([seller_type])[0]
        transmission_encoded = encoders["transmission"].transform([transmission])[0]
        owner_encoded = encoders["owner"].transform([owner])[0]

        # Create input dataframe
        input_data = pd.DataFrame({
            "name": [name_encoded],
            "year": [year],
            "km_driven": [km_driven],
            "fuel": [fuel_encoded],
            "seller_type": [seller_encoded],
            "transmission": [transmission_encoded],
            "owner": [owner_encoded],
            "mileage(km/ltr/kg)": [mileage],
            "engine": [engine],
            "max_power": [max_power],
            "seats": [seats]
        })

        # Predict
        prediction = model.predict(input_data)

        # Display result
        st.success(
            f"Predicted Car Price: ₹ {prediction[0]:,.2f}"
        )

    except ValueError:
        st.error(
            "Car name not found in the dataset. Please enter a valid car name."
        )
