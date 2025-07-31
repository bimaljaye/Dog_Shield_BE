# frontend.py

import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Dog Rabies Risk Assessment",
    page_icon="🐕",
    layout="centered"
)

# --- FastAPI Backend URL ---
# This is where your FastAPI app is running.
API_URL = "http://127.0.0.1:8000/predict"

# --- UI Elements ---
st.title("Dog Rabies Risk Assessment Tool 🐕")
st.markdown("Select the symptoms and status of the dog to assess the risk of rabies. This tool uses a machine learning model to predict the probability of infection.")

st.sidebar.header("Dog's Status & Symptoms")

# Use a function to create the input widgets in the sidebar
def user_input_features():
    vaccinated = st.sidebar.radio("Has the dog been vaccinated?", ('yes', 'no'))
    grooming = st.sidebar.radio("Is the dog regularly groomed?", ('yes', 'no'))
    fever = st.sidebar.radio("Does the dog have a fever?", ('yes', 'no'))
    lethargy = st.sidebar.radio("Is the dog showing signs of lethargy?", ('yes', 'no'))
    behavior_change = st.sidebar.radio("Has there been a noticeable change in behavior?", ('yes', 'no'))
    aggression = st.sidebar.radio("Is the dog unusually aggressive?", ('yes', 'no'))
    excessive_drooling = st.sidebar.radio("Is the dog drooling excessively?", ('yes', 'no'))
    difficulty_swallowing = st.sidebar.radio("Does the dog have difficulty swallowing?", ('yes', 'no'))
    seizures = st.sidebar.radio("Has the dog experienced seizures?", ('yes', 'no'))
    staggering_gait = st.sidebar.radio("Does the dog have a staggering gait?", ('yes', 'no'))
    hind_leg_paralysis = st.sidebar.radio("Is there paralysis in the hind legs?", ('yes', 'no'))
    jaw_dropped = st.sidebar.radio("Is the dog's jaw dropped or slack?", ('yes', 'no'))

    # Store the inputs in a dictionary
    data = {
        'vaccinated': vaccinated, 'grooming': grooming, 'fever': fever, 
        'lethargy': lethargy, 'behavior_change': behavior_change, 'aggression': aggression,
        'excessive_drooling': excessive_drooling, 'difficulty_swallowing': difficulty_swallowing,
        'seizures': seizures, 'staggering_gait': staggering_gait,
        'hind_leg_paralysis': hind_leg_paralysis, 'jaw_dropped': jaw_dropped
    }
    return data

# Get user inputs
input_data = user_input_features()

# "Assess Risk" button
if st.sidebar.button("Assess Risk"):
    with st.spinner("Analyzing..."):
        try:
            # Send a POST request to the FastAPI backend
            response = requests.post(API_URL, json=input_data)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # Parse the JSON response
            result = response.json()
            
            prediction = result['prediction']
            probability = result['risk_probability']
            
            # --- Display the results ---
            st.subheader("Assessment Result")
            
            if "High Risk" in prediction:
                st.error(f"**Prediction:** {prediction}")
            else:
                st.success(f"**Prediction:** {prediction}")

            st.metric(label="Rabies Risk Probability", value=f"{probability:.2%}")
            
            st.progress(probability)
            
            st.info("Disclaimer: This is an AI-powered prediction and not a substitute for professional veterinary advice. Please consult a qualified veterinarian for an accurate diagnosis.", icon="⚠️")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the prediction service. Please ensure the backend is running. Error: {e}")

else:
    st.info("Please select the dog's symptoms in the sidebar and click 'Assess Risk'.")