# Rabies Prediction ML model and Web App

This project provides a machine learning-powered web app to predict the likelihood of rabies based on input features. It uses a FastAPI backend to serve an XGBoost model and a Streamlit frontend for a user-friendly web interface.

## Project Structure

/

├── .venv/

├── backend.py # FastAPI application and model serving logic

├── frontend.py # Streamlit user interface

├── xgboost_rabies_model.pkl # The pre-trained model file

└── requirements.txt # Project dependencies



---

## Prerequisites

Before you begin, ensure you have the following installed on your system:
*   [Python 3.8+](https://www.python.org/downloads/)
*   [Git](https://git-scm.com/downloads/)

---

## Setup and Installation Guide

Follow these steps to set up and run the application on your local machine.

### 1. Get the Code from GitHub

First, clone the repository from GitHub to your local machine using the following command in your terminal.

```bash
git clone https://github.com/bimaljaye/Dog_Shield_BE.git
```

Navigate into the newly created project folder

```bash
cd Rabies_Detection_Model/Rabies_Classification
```


### 2. Create a virtual environment

```bash
python -m venv .venv
```


### 3. Activating the virtual environment


On Windows (PowerShell/CMD):

```bash
.\.venv\Scripts\Activate
```


### 4. Install Required Packages

```bash
pip install -r requirements.txt
```


### 5. Run the Backend Server

The backend is a FastAPI application that serves the ML model. It must be run with an ASGI server like Uvicorn.

**➡️ Open your first terminal** and run the following command from the project's root directory:

```bash
uvicorn backend:app --reload
```


### 6. Run the Frontend Application

➡️ Open a second, new terminal (do not close the first one!). Activate the virtual environment again as you did in Step 3. Then, run the following command from the project's root directory:

```bash
streamlit run frontend.py
```


This will launch the Streamlit application. A new tab should automatically open in your web browser pointing to http://localhost:8501.
You can now interact with the web application to get rabies predictions. The frontend will communicate in the background with the API server you started in Step 5.




