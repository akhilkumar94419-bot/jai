import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(page_title="Energy Prediction App", layout="centered")

st.title("Neural Network Energy Prediction")
st.write("Predict Heating Load and Cooling Load")

# Load model and scaler
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("nn_energy_model.h5")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model()

# Input fields
Relative_Compactness = st.number_input("Relative Compactness", value=0.75)
Surface_Area = st.number_input("Surface Area", value=500.0)
Wall_Area = st.number_input("Wall Area", value=300.0)
Roof_Area = st.number_input("Roof Area", value=200.0)
Overall_Height = st.number_input("Overall Height", value=7.0)
Orientation = st.selectbox("Orientation", [2, 3, 4, 5])
Glazing_Area = st.number_input("Glazing Area", value=0.2)
Glazing_Area_Distribution = st.selectbox(
    "Glazing Area Distribution", [0, 1, 2, 3, 4, 5]
)

# Predict button
if st.button("Predict"):
    X = np.array([
        Relative_Compactness,
        Surface_Area,
        Wall_Area,
        Roof_Area,
        Overall_Height,
        Orientation,
        Glazing_Area,
        Glazing_Area_Distribution
    ]).reshape(1, -1)

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)

    st.success("Prediction Complete")
    st.write(f"🔥 **Heating Load:** {prediction[0][0]:.2f}")
    st.write(f"❄️ **Cooling Load:** {prediction[0][1]:.2f}")
