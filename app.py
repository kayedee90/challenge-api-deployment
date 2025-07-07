from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import Optional, Literal
from catboost import CatBoostRegressor
import pandas as pd
import json
# Import functions
from preprocessing.cleaning_data import preprocess
from predict.prediction import predict

# Load the trained model from file

model = CatBoostRegressor()
model.load_model("model/catboost_model.cbm")

# Define the structure of the expected input inside the "data" key

class DataInput(BaseModel):
    area: int  
    property_type: Literal["APARTMENT", "HOUSE", "OTHERS"]
    rooms_number: int 
    zip_code: int  
    land_area: Optional[int] = None
    garden: Optional[bool] = None
    garden_area: Optional[int] = None
    equipped_kitchen: Optional[bool] = None
    full_address: Optional[str] = None
    swimming_pool: Optional[bool] = None
    furnished: Optional[bool] = None
    open_fire: Optional[bool] = None
    terrace: Optional[bool] = None
    terrace_area: Optional[int] = None
    facades_number: Optional[int] = None
    building_state: Optional[
        Literal["NEW", "GOOD", "TO RENOVATE", "JUST RENOVATED", "TO REBUILD"]
    ] = None

    @field_validator("property_type", "building_state", mode="before")
    @classmethod
    def normalize_uppercase(cls, v):
        if isinstance(v, str):
            return v.upper()
        return v



# Wrap input to match the JSON format
class PropertyInput(BaseModel):
    data: DataInput

# Create the FastAPI app
app = FastAPI()

# check if the API is running
@app.get("/")
def root():
    return "Alive"

# Handles POST requests for property price prediction
@app.post("/predict")
def predict_price(input: PropertyInput):
    try:
        # Extract data from the input
        features = input.data

        # Convert input into a DataFrame
        input_df = preprocess(features)

        # Generate prediction
        price = predict(model, input_df)

        # Return the predicted price in JSON format
        return {
            "prediction": price,
            "status_code": 200
        }


    # Catch and handle value-specific issues
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))

    # Catch any other unexpected errors during prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
@app.get("/predict")
def predict_instructions():
    return JSONResponse(content={
        "message": "Please input your data in the following format:",
        "format": {
            "data": {
                "area": "Input a number, like: 105",
                "property_type": "Choose between: APARTMENT | HOUSE | OTHERS",
                "rooms_number": "Input a number, like: 3",
                "zip_code": "Input a number, like: 9300",
                "land_area": "Input a number, like: 200",
                "garden": "Enter: true or false",
                "garden_area": "Input a number, like: 50",
                "equipped_kitchen": "Enter: true or false",
                "full_address": "Input the city name, like: 'Aalst'",
                "swimming_pool": "Enter: true or false",
                "furnished": "Enter: true or false",
                "open_fire": "Enter: true or false",
                "terrace": "Enter: true or false",
                "terrace_area": "Input a number, like: 20",
                "facades_number": "Input a number, like: 2",
                "building_state": "Choose between: NEW | GOOD | TO RENOVATE | JUST RENOVATED | TO REBUILD"
            }
        }
    })

