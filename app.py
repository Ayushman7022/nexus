import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load the saved model and encoders
model = joblib.load("random_forest_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend requests (Android app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request data model
class InputData(BaseModel):
    Age: int
    BMI: float
    Smoking: int
    AlcoholConsumption: int
    PhysicalActivity: int
    DietType: int
    SleepHours: int
    StressLevel: int
    BloodPressure: int
    Cholesterol: int
    FamilyHistory: int
    BloodSugar: float
    WaistCircumference: float

# Home route
@app.get("/")
def home():
    return {"message": "Hello, Railway!"}

# Define API endpoint
@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])

    # Encode categorical columns
    categorical_cols = ["PhysicalActivity", "StressLevel", "BloodPressure", "Cholesterol"]
    for col in categorical_cols:
        input_df[col] = label_encoders[col].transform([input_df[col].values[0]])  # Fix `.iloc[0]`

    # Make predictions
    predictions = model.predict(input_df)

    # Format the output
    output = {
        "Hypertension": int(predictions[0][0]),
        "HeartDisease": int(predictions[0][1]),
        "Obesity": int(predictions[0][2]),
        "Diabetes": int(predictions[0][3])
    }

    return output

# Ensure this runs only when executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
