import pandas as pd
import json

def preprocess(input_data):
    try:
        df = pd.DataFrame([{
            "type": input_data.type,
            "subtype": input_data.subtype,
            "province": input_data.province,
            "locality": input_data.locality,
            "postCode": input_data.postCode,
            "buildingCondition": input_data.buildingCondition,
            "epcScore": input_data.epcScore,
            "bedroomCount": input_data.bedroomCount,
            "toilet_and_bath": input_data.toilet_and_bath,
            "habitableSurface": input_data.habitableSurface,
            "facedeCount": input_data.facedeCount,
            "hasTerrace": input_data.hasTerrace,
            "totalParkingCount": input_data.totalParkingCount
        }])

        with open("model_features.json") as f:
            feature_order = json.load(f)

        df = df[feature_order]

        return df

    except Exception as e:
        raise ValueError(f"Preprocessing error: {e}")