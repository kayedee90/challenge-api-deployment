"""define prediction function"""
def predict(model, input_df):
    prediction = model.predict(input_df)
    return round(float(prediction[0]), 2)
