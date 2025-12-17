from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI()


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Definisikan format data input (Validation)
class CancerData(BaseModel):
    features: list[float]  # Menerima list angka (fitur)

@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API is Running!"}

@app.post("/predict")
def predict(data: CancerData):
    # Ubah list menjadi numpy array 2D
    input_data = np.array(data.features).reshape(1, -1)
    
    # Lakukan prediksi
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data).max()
    
    # Return JSON
    result = "Malignant" if prediction[0] == 0 else "Benign" # Sesuaikan mapping label Anda
    
    return {
        "prediction": int(prediction[0]),
        "label": result,
        "probability": float(probability)
    }