import joblib
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = "AIzaSyCVURliB0WKQBge9h8VKu4sA2CIRb9D1hU"

# Load the trained model and encoders
model = joblib.load("random_forest_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LangChain with Gemini API
llm = ChatGoogleGenerativeAI(model="gemini-2.0", google_api_key=GOOGLE_API_KEY)

# Define request data model
class InputData(BaseModel):
    Age: conint(ge=0, le=120)
    BMI: confloat(ge=10, le=50)
    Smoking: conint(ge=0, le=1)
    AlcoholConsumption: conint(ge=0, le=1)
    PhysicalActivity: conint(ge=0, le=3)
    DietType: conint(ge=0, le=2)
    SleepHours: conint(ge=0, le=24)
    StressLevel: conint(ge=0, le=10)
    BloodPressure: conint(ge=0, le=2)
    Cholesterol: conint(ge=0, le=2)
    FamilyHistory: conint(ge=0, le=1)
    BloodSugar: confloat(ge=0, le=500)
    WaistCircumference: confloat(ge=0, le=200)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Health Prediction API"}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Encode categorical columns
        categorical_cols = ["PhysicalActivity", "StressLevel", "BloodPressure", "Cholesterol"]
        for col in categorical_cols:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])

        # Make predictions
        predictions = model.predict(input_df)

        # Ensure predictions are in the expected format
        if len(predictions[0]) != 4:
            raise ValueError("Model output format is incorrect")

        # Extract disease predictions
        diseases = {
            "Hypertension": bool(predictions[0][0]),
            "HeartDisease": bool(predictions[0][1]),
            "Obesity": bool(predictions[0][2]),
            "Diabetes": bool(predictions[0][3]),
        }

        # Get precautionary measures for detected diseases
        detected_diseases = [disease for disease, present in diseases.items() if present]
        precautions = {}

        for disease in detected_diseases:
            query = f"What precautions should a person take to avoid {disease} in later stages of life?"
            response = llm.invoke(query)
            precautions[disease] = response.content if hasattr(response, "content") else str(response)

        return {
            "predictions": diseases,
            "precautions": precautions,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

