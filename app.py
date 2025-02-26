import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
from fastapi.middleware.cors import CORSMiddleware

# Load the saved model and encoders
model = joblib.load("random_forest_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend requests (Android app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request data model with additional validation
class InputData(BaseModel):
    Age: conint(ge=0, le=120)  # Age between 0 and 120
    BMI: confloat(ge=10, le=50)  # BMI between 10 and 50
    Smoking: conint(ge=0, le=1)  # 0 or 1
    AlcoholConsumption: conint(ge=0, le=1)  # 0 or 1
    PhysicalActivity: conint(ge=0, le=3)  # Example: 0 to 3 levels
    DietType: conint(ge=0, le=2)  # Example: 0 to 2 types
    SleepHours: conint(ge=0, le=24)  # Sleep hours between 0 and 24
    StressLevel: conint(ge=0, le=10)  # Stress level between 0 and 10
    BloodPressure: conint(ge=0, le=2)  # Example: 0 to 2 levels
    Cholesterol: conint(ge=0, le=2)  # Example: 0 to 2 levels
    FamilyHistory: conint(ge=0, le=1)  # 0 or 1
    BloodSugar: confloat(ge=0, le=500)  # Blood sugar between 0 and 500
    WaistCircumference: confloat(ge=0, le=200)  # Waist circumference between 0 and 200 cm

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
        if predictions.shape[1] != 4:  # Check for 4 output columns
            raise ValueError("Model output format is incorrect")

        # Format the output
        output = {
            "Hypertension": int(predictions[0][0]),
            "HeartDisease": int(predictions[0][1]),
            "Obesity": int(predictions[0][2]),
            "Diabetes": int(predictions[0][3])
        }

        return output

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
