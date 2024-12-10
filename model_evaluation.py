import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the trained model from the 'model.pkl' file
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the dataset again (you need to split it the same way as before)
data = fetch_california_housing(as_frame=True)
X = data.data  # Features
y = data.target  # Target variable (house prices)

# Split the dataset into training and testing sets (same as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions on the test set using the loaded model
predictions = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)

# Calculate R-squared (R²) Score
r2 = r2_score(y_test, predictions)

# Print the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
