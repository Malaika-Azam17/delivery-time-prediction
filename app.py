import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression

# Load the trained model
with open("models/delivery_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Delivery Time Prediction App for Online Orders")

# File uploader for custom dataset
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV format)", type=["csv"])

# Load dataset
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Custom dataset uploaded successfully!")
else:
    data = pd.read_csv("data/delivery_data.csv")

# Input fields
st.sidebar.header("Enter Order Details")
distance = st.sidebar.number_input("Distance (km):", min_value=1, max_value=100, value=10)
traffic = st.sidebar.selectbox("Traffic Conditions:", options=[1, 2, 3], format_func=lambda x: ["Low", "Medium", "High"][x-1])
weather = st.sidebar.selectbox("Weather Conditions:", options=[0, 1], format_func=lambda x: ["Bad", "Good"][x])
processing_time = st.sidebar.number_input("Order Processing Time (minutes):", min_value=1, max_value=60, value=15)

# Initialize session state for prediction
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# Predict button with a unique key
if st.sidebar.button("Predict Delivery Time", key="predict_button"):
    # Make prediction
    input_data = np.array([[distance, traffic, weather, processing_time]])
    st.session_state.prediction = model.predict(input_data)
    st.success(f"Predicted Delivery Time: {st.session_state.prediction[0]:.2f} minutes")

    # Add user input to the dataset for visualization
    new_data = pd.DataFrame([[distance, traffic, weather, processing_time, st.session_state.prediction[0]]], columns=data.columns)
    data = pd.concat([data, new_data], ignore_index=True)

    # Save only if using default dataset
    if uploaded_file is None:
        data.to_csv("data/delivery_data.csv", index=False)
        st.success("Data saved to CSV file!")

# Visualization: Scatter plot with regression line
st.write("### Distance vs Delivery Time")
fig, ax = plt.subplots()
ax.scatter(data['distance'], data['delivery_time'], alpha=0.5, label="Existing Data")

# Show My Order button with a unique key
if st.sidebar.button("Show My Order", key="show_order_button"):
    if st.session_state.prediction is not None:  # Check if prediction exists
        ax.scatter(distance, st.session_state.prediction[0], color="red", label="Your Order")
    else:
        st.warning("Please click 'Predict Delivery Time' first to calculate the prediction.")

# Fit a regression line
X = data[['distance']]
y = data['delivery_time']
reg = LinearRegression()
reg.fit(X, y)
x_range = np.linspace(data['distance'].min(), data['distance'].max(), 100).reshape(-1, 1)
y_pred = reg.predict(x_range)
ax.plot(x_range, y_pred, color="red", linewidth=2, label="Regression Line")

ax.set_xlabel("Distance (km)")
ax.set_ylabel("Delivery Time (minutes)")
ax.legend()
st.pyplot(fig)

# Display the updated dataset in a table
st.write("### Updated Dataset with Predictions")
st.dataframe(data)  # Use st.table(data) for a static table
