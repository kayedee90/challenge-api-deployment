from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoost
import pandas as pd
import numpy as np
from typing import Optional
import json


model = CatBoost()
model.load_model("catboost_model.cbm")

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
        # Convert input to DataFrame in exact feature order
        input_df = pd.DataFrame([{
            "type": input.type,
            "subtype": input.subtype,
            "province": input.province,
            "locality": input.locality,
            "postCode": input.postCode,
            "buildingCondition": input.buildingCondition,
            "epcScore": input.epcScore,
            "bedroomCount": input.bedroomCount,
            "toilet_and_bath": input.toilet_and_bath,
            "habitableSurface": input.habitableSurface,
            "facedeCount": input.facedeCount,
            "hasTerrace": input.hasTerrace,
            "totalParkingCount": input.totalParkingCount
        }])
       
        with open("model_features.json") as f:
            feature_order = json.load(f)
        input_df = input_df[feature_order]

        cat_features = [6, 7, 8, 9, 10, 11, 12] # Adjust if needed

        prediction = model.predict(input_df)

        return {"predicted_price": round(float(prediction[0]), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))