import pandas as pd
import json


# Load defaults
defaults = {
  "type": "HOUSE", # Most common
  "subtype": "House", # Most common
  "province": "East Flanders", # Mid range prices
  "locality": "Aalst", # Mid range prices
  "postCode": "9300", # Mid range prices
  "buildingCondition": "GOOD", # Most common
  "epcScore": "C", # Most common
  "bedroomCount": 3, # Most common
  "toilet_and_bath": 1, # Most common
  "habitableSurface": 105.0, # Most common
  "facedeCount": 2, # Most common
  "hasTerrace": 0, # Most common
  "totalParkingCount": 1 # Most common
}





# Define the preprocessing function
def preprocess(input_data):
    try:
        # Convert input object to dictionary
        raw = input_data.dict()
        # Default to set values if no input
        features = {
            "type": raw.get("property_type") or defaults["type"],
            "subtype": raw.get("subtype") or defaults["subtype"],
            "province": raw.get("province") or defaults["province"],
            "locality": raw.get("full_address") or defaults["locality"],
            "postCode": str(raw.get("zip_code") or defaults["postCode"]),
            "buildingCondition": raw.get("building_state") or defaults["buildingCondition"],
            "epcScore": raw.get("epcScore") or defaults["epcScore"],
            "bedroomCount": raw.get("rooms_number") or defaults["bedroomCount"],
            "toilet_and_bath": raw.get("toilet_and_bath") or defaults["toilet_and_bath"],
            "habitableSurface": raw.get("area") or defaults["habitableSurface"],
            "facedeCount": raw.get("facades_number") or defaults["facedeCount"],
            "hasTerrace": int(raw.get("terrace") or defaults["hasTerrace"]),
            "totalParkingCount": raw.get("parking_count") or defaults["totalParkingCount"]
        }

        # Create a DataFrame for prediction
        df = pd.DataFrame([features])

        # Ensure column order matches model training layout
        with open("model_features.json") as f:
            feature_order = json.load(f)

        df = df[feature_order]

        return df

    except Exception as e:
        # Raise valueerror for the API
        raise ValueError(f"Preprocessing error: {e}")