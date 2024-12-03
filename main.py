from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.preprocessing import StandardScaler
from fastapi.responses import StreamingResponse

def load_model():
    return joblib.load('model.pkl')

def load_scaler():
    return joblib.load('scaler.pkl')

def to_float(el):
  if el and (not pd.isna(el)):
    raw_d = re.findall(r'\b\d+\.\d+|\d+\b', el)
    if raw_d:
      return float(raw_d[0])
  return np.nan

def extract_torque(torque_str):
    if not isinstance(torque_str, str):
        return pd.Series([None, None])

    # для формата '114Nm@ 4000rpm' и '22.4 kgm at 1750-2750rpm'
    pattern1 = r'(\d+\.?\d*)\s*([a-zA-Z]+)?\s*(?:@|at)\s*(\d+[\d,-]*)\s*rpm'
    match1 = re.search(pattern1, torque_str)
    if match1:
        torque = float(match1.group(1))
        unit = match1.group(2)
        rpm = match1.group(3).replace(',', '')
        if unit == 'kgm':
            torque *= 9.81
        return pd.Series([torque, max(map(float, rpm.split('-')))])

    # '11.5@ 4,500(kgm@ rpm)'
    pattern2 = r'(\d+\.?\d*)\s*(?:@|at)\s*(\d+[\d,-]*)\s*\(([a-zA-Z]+)@\s*rpm\)'
    match2 = re.search(pattern2, torque_str)
    if match2:
        torque = float(match2.group(1))
        unit = match2.group(3)
        rpm = match2.group(2).replace(',', '')
        if unit.lower() == 'kgm':
            torque *= 9.81
        return pd.Series([torque, max(map(float, rpm.split('-')))])

    return pd.Series([None, None])

def transform_data(df, scaler):
    df['mileage'] = df.mileage.apply(to_float)
    df['engine'] = df.engine.apply(to_float)
    df['max_power'] = df.max_power.apply(to_float)
    df[['torque', 'max_torque_rpm']] = df['torque'].apply(extract_torque)
    train_medians = joblib.load('train_medians.pkl')
    df.update(df.fillna(train_medians))

    prediction_features = ['year', 'km_driven', 'mileage', 'max_power', 'torque', 'max_torque_rpm']
    prediction_data = df[prediction_features]

    scaled_data = scaler.transform(prediction_data)
    return pd.DataFrame(scaled_data, columns=prediction_data.columns)

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["model"] = load_model()
    ml_models["train_medians"] = joblib.load('train_medians.pkl')
    ml_models["scaler"] = load_scaler()

    yield

    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class CarFeatures(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

@app.post("/predict_item")
async def predict_item(item: CarFeatures):
    data = pd.DataFrame([item.dict()])
    transformed_data = transform_data(data, ml_models["scaler"])
    prediction = ml_models["model"].predict(transformed_data)
    return {
        "parameters": item.dict(),
        "predicted_selling_price": prediction[0],
    }

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    transformed_data = transform_data(data, ml_models["scaler"])
    predictions = ml_models["model"].predict(transformed_data)
    data['predicted_selling_price'] = predictions
    csv_buffer = data.to_csv(index=False)

    return StreamingResponse(
        iter([csv_buffer]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions.csv"}
    )