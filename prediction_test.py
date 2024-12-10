import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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

# Show some predictions and compare them with the actual target values
print("Predictions:", predictions[:10])  # First 10 predictions
print("Actual values:", y_test[:10].values)  # First 10 actual target values
