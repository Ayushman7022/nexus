
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder  # Add this line
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # For saving/loading the model

# Load the dataset
df = pd.read_csv("nuanced_lifestyle_disease_data_50000.csv")

# Display basic info
print(df.head())
print(df.info())  # Check data types and missing values
print(df.describe())  # Summary statistics


# Create a dictionary to store encoders
label_encoders = {}

categorical_cols = ["PhysicalActivity", "StressLevel", "BloodPressure", "Cholesterol"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Encode each column separately
    label_encoders[col] = le  # Store the encoder for future transformations

print(df.head())


# Define feature variables (X) and target variables (Y)
X = df.drop(columns=["Hypertension", "HeartDisease", "Obesity", "Diabetes"])
Y = df[["Hypertension", "HeartDisease", "Obesity", "Diabetes"]]  # Multi-label classification


from sklearn.model_selection import train_test_split

# Split into train (80%) and test (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")


from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, Y_train)


from sklearn.metrics import accuracy_score

# Predict on test data
Y_pred = model.predict(X_test)

# Calculate accuracy for each disease
for i, col in enumerate(Y.columns):
    acc = accuracy_score(Y_test[col], Y_pred[:, i])
    print(f"Accuracy for {col}: {acc:.4f}")

print(Y_pred)



import numpy as np

# Define a custom input
custom_input = {
    "Age": 20,
    "BMI": 29,
    "Smoking": 1,              # 1 = Smoker, 0 = Non-smoker
    "AlcoholConsumption": 1,   # 1 = Yes, 0 = No
    "PhysicalActivity": 2,     # 0 = Low, 1 = Moderate, 2 = High (Encoded)
    "DietType": 1,             # 1 = Unhealthy, 0 = Healthy
    "SleepHours": 8,
    "StressLevel": 2,          # 0 = Low, 1 = Medium, 2 = High (Encoded)
    "BloodPressure": 1,        # 0 = Normal, 1 = Elevated, 2 = High (Encoded)
    "Cholesterol": 2,          # 0 = Normal, 1 = Borderline, 2 = High (Encoded)
    "FamilyHistory": 1,        # 0 = No, 1 = Yes
    "BloodSugar": 140.0,        # Blood sugar in mg/dL (within normal range)
    "WaistCircumference": 40.0 # Waist circumference in cm (lower range)
}


# Convert to a DataFrame (as model expects this format)
custom_input_df = pd.DataFrame([custom_input])

# Print the custom input
print("Custom Input:\n", custom_input_df)


# Predict disease probabilities
custom_prediction = model.predict(custom_input_df)

# Convert the output to a readable format
predicted_labels = dict(zip(Y.columns, custom_prediction[0]))

# Display the prediction
print( predicted_labels)


import joblib

# Save the trained model
joblib.dump(model, "random_forest_model.pkl")

# Save the encoders
joblib.dump(label_encoders, "label_encoders.pkl")
