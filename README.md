Overview

This project includes:
* Training a Random Forest Regressor to predict target values based on input features.
* A FastAPI application that serves predictions via an HTTP endpoint.
* A trained model evaluated using Mean Squared Error (MSE) and R² (R-squared), achieving:
  * MSE: 0.2554
  * R²: 0.8051

Features
* Accepts input features as a JSON payload.
* Returns predictions in real-time.
* Designed for easy deployment in production environments.
* Includes an evaluation report to assess model performance.

Technologies Used
* Python 3.11
* FastAPI for building the API.
* Scikit-learn for machine learning and model training.
* NumPy for numerical computations.
* Pydantic for data validation.

Actually is the same HousingGuru with the one I made with Linear Regration, but as we can see, with Random Forest we get way better MSE and R square.
