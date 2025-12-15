from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf

app = FastAPI()

# Load saved scaler and model
scaler = joblib.load("scaler.joblib")
model = tf.keras.models.load_model("nn_energy_model.h5")

# Input schema
class EnergyInput(BaseModel):
    Relative_Compactness: float
    Surface_Area: float
    Wall_Area: float
    Roof_Area: float
    Overall_Height: float
    Orientation: int
    Glazing_Area: float
    Glazing_Area_Distribution: int

# Health check
@app.get("/")
def home():
    return {"message": "Neural Network Energy Model is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: EnergyInput):
    X = np.array([
        data.Relative_Compactness,
        data.Surface_Area,
        data.Wall_Area,
        data.Roof_Area,
        data.Overall_Height,
        data.Orientation,
        data.Glazing_Area,
        data.Glazing_Area_Distribution
    ]).reshape(1, -1)

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)

    return {
        "Heating_Load": float(prediction[0][0]),
        "Cooling_Load": float(prediction[0][1])
    }
