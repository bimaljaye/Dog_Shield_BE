import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI(title="Rabies Prediction API")

# --- 1. Load the Model ---
# Load the trained model pipeline from the file
try:
    with open('xgboost_rabies_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file not found. Ensure 'xgboost_rabies_model.pkl' is in the directory.")
    model = None

# --- 2. Define the Input Data Model using Pydantic ---
# This ensures that the input data for a request is valid
class DogFeatures(BaseModel):
    vaccinated: str
    grooming: str
    fever: str
    lethargy: str
    behavior_change: str
    aggression: str
    excessive_drooling: str
    difficulty_swallowing: str
    seizures: str
    staggering_gait: str
    hind_leg_paralysis: str
    jaw_dropped: str

# --- 3. Define the Prediction Endpoint ---
@app.post("/predict")
def predict_rabies(features: DogFeatures):
    """
    Receives dog features, processes them, and returns a rabies risk prediction.
    """
    if not model:
        return {"error": "Model not loaded. Please check the server logs."}

    # Convert the Pydantic model to a dictionary
    input_data = features.dict()

    # Convert the dictionary into a pandas DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Map 'yes'/'no' string values to 1/0 integers for the model
    for col in input_df.columns:
        input_df[col] = input_df[col].map({'yes': 1, 'no': 0})
    
    # Define the feature order the model was trained on (CRUCIAL)
    feature_order = [
        'vaccinated', 'grooming', 'fever', 'lethargy', 'behavior_change',
        'aggression', 'excessive_drooling', 'difficulty_swallowing', 'seizures',
        'staggering_gait', 'hind_leg_paralysis', 'jaw_dropped'
    ]
    
    # Ensure the DataFrame columns are in the correct order
    input_df = input_df[feature_order]

    # Predict the probability of rabies (class 1)
    # predict_proba returns [[prob_no, prob_yes]]
    rabies_probability = model.predict_proba(input_df)[0][1]

    # Determine the prediction text based on a threshold 0.5
    if rabies_probability > 0.5:
        prediction_text = "High Risk: Rabies Likely"
    else:
        prediction_text = "Low Risk: Rabies Unlikely"

    # Return the prediction and the probability
    return {
        "prediction": prediction_text,
        "risk_probability": float(rabies_probability)
    }

# A simple root endpoint to confirm the API is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Rabies Prediction API. Use the /predict endpoint to get a risk assessment."}