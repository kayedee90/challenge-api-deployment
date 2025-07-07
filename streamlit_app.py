import streamlit as st
import requests
import json

st.set_page_config(page_title="Property Price Predictor", page_icon="🏠")

# --- App Title ---
st.title(" Property Price Predictor")
st.markdown("Enter the property details below to estimate the price.")

# --- Required fields ---
area = st.number_input("Living Area (m²)", min_value=6, max_value=185347, value=100)
property_type = st.selectbox("Property Type", ["APARTMENT", "HOUSE", "OTHERS"])
rooms_number = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Toilets & Bathrooms", min_value=1, max_value=10, value=1)
zip_code = st.number_input("Zip Code", min_value=1000, max_value=9992, value=1000)

# --- Optional numerical fields ---
facades_number = st.number_input("Number of Facades", min_value=1, max_value=4, value=2)

# --- Optional boolean features ---
terrace = st.checkbox("Terrace")

# --- Building state ---
building_state = st.selectbox("Building Condition", [
    "NEW", "GOOD", "TO RENOVATE", "JUST RENOVATED", "TO REBUILD"
])

# --- Submit button ---
if st.button("Predict Price"):
    input_data = {
        "data": {
            "area": area,
            "property_type": property_type,
            "rooms_number": rooms_number,
            "zip_code": zip_code,
            "terrace": terrace,
            "facades_number": facades_number,
            "building_state": building_state
        }
    }

    try:
        api_url = "https://challenge-api-deployment-wlr4.onrender.com/predict"
        response = requests.post(api_url, json=input_data)
        
        # Only try to parse JSON if the request succeeded
        if response.status_code == 200:
            result = response.json()
            price = int(result['prediction'])
            st.success(f"Estimated Price: €{price:,}")
        else:
            try:
                error_detail = response.json().get('detail', 'Unknown error')
            except:
                error_detail = response.text
            st.error(f"Error: {error_detail}")
    except Exception as e:
        st.error(f"Could not reach the API: {e}")
  
