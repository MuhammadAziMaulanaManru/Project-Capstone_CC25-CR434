from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import tensorflow_decision_forests as tfdf
import ydf

model = ydf.from_tensorflow_decision_forests("/Users/najwarwardana/Downloads/capstone_proyek/tfdf_model_laptop")
app = FastAPI(title="Laptop Price Prediction API")

class LaptopSpecs(BaseModel):
    Ram: int
    Weight: float
    SSD: int
    TypeName_enc: int
    OpSys_enc: int

@app.post("/predict")
def predict_price(specs: LaptopSpecs):
    input_df = pd.DataFrame([specs.dict()])
    print("📥 Input:", input_df)

    prediction = model.predict(input_df)[0]  # cukup ini jika 1 baris input
    print("📤 Prediction:", prediction)

    return {"predicted_price_idr": float(round(prediction, 2))}

@app.get("/health")
def health_check():
    return {"status": "ok"}