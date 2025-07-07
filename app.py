from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from typing import Optional
from preprocessing.cleaning_data import preprocess
import json


model = CatBoostRegressor()
model.load_model("model/catboost_model.cbm")

categorical_features = [
    "type", "subtype", "province", "locality", "postCode", "buildingCondition", "epcScore"
]

class PropertyInput(BaseModel):
    type: Optional[str] = "HOUSE"
    subtype: Optional[str] = "House"
    province: Optional[str] = "Antwerp"
    locality: Optional[str] = "Antwerp"
    postCode: Optional[str] = "2000"
    buildingCondition: Optional[str] = "TO_BE_DONE_UP"
    epcScore: Optional[str] = "C"
    bedroomCount: Optional[int] = 3
    toilet_and_bath: Optional[int] = 1
    habitableSurface: Optional[float] = 100.0
    facedeCount: Optional[int] = 2
    hasTerrace: Optional[int] = 0
    totalParkingCount: Optional[int] = 1

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome to the Property Price Prediction API"}

@app.post("/predict")
def predict_price(input: PropertyInput):
    try:
        # Preprocess input into DataFrame
        input_df = preprocess(input)
        
        # Run model prediction
        prediction = model.predict(input_df)

        return {
            "prediction": round(float(prediction[0]), 2),
            "status_code": 200
        }
    
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")