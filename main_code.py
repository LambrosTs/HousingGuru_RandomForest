from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load the trained model from the file
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

print(model)

# Initialize the FastAPI app
app = FastAPI()

# Define the input data schema
class PredictionRequest(BaseModel):
    features: list  # Input should be a list of numerical features

# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert the input features to a NumPy array and reshape for the model
    features = np.array(request.features).reshape(1, -1)
    
    # Use the model to make a prediction
    prediction = model.predict(features)
    
    # Return the prediction as a JSON response
    return {"prediction": prediction.tolist()}
