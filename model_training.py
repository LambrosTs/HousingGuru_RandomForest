from sklearn.ensemble import RandomForestRegressor  # Or RandomForestClassifier for classification
from sklearn.model_selection import train_test_split
import pickle
from sklearn.datasets import fetch_california_housing

# Load the dataset
data = fetch_california_housing(as_frame=True)
X = data.data  # Features
y = data.target  # Target variable (house prices)

# Assuming X_train, y_train are your training data and target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Use RandomForestClassifier for classification
model.fit(X_train, y_train)

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
