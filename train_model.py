import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import os

# Load dataset
data = pd.read_csv("data/delivery_data.csv")
X = data[['distance', 'traffic', 'weather', 'processing_time']]  # Features
y = data['delivery_time']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Save the model to a file
os.makedirs("models", exist_ok=True)
with open("models/delivery_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as models/delivery_model.pkl")