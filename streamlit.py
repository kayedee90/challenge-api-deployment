import streamlit as st
import requests
import json


# ---- Page Configuration ----
st.set_page_config(page_title="Belgium Real Estate Price Predictor", page_icon="🏠", layout="centered")

st.markdown("""
<h1 style='text-align: center; font-size: 3.2rem; color: #333333; font-weight: 800; margin-top: 1rem;'>
Belgium Real Estate Price Predictor
</h1>
<p style='text-align: center; font-size: 1.2rem; color: #333333; margin-top: -10px;'>
Instantly estimate property prices across Belgium using machine learning.
</p>
""", unsafe_allow_html=True)


st.image("house3.jpg", use_container_width=True)

st.markdown("""

<p style='text-align: center; font-size: 1.1rem; color: #333333; margin-top: 1rem; max-width: 700px; margin-left: auto; margin-right: auto;'>

Welcome to our <strong>Property Price Predictor!</strong> – your intelligent assistant for estimating real estate values across Belgium using machine learning.<br><br>

""", unsafe_allow_html=True)

with st.container():
    st.markdown("""
    <h2 style='color: #333333;'>How it works:</h2>
    <p style='color: #555555; font-size: 1.1rem;'>
    Our model analyzes key property features — like area, number of rooms, building condition, and location — to estimate the likely market value using advanced AI. Just enter the property details below to get started!
    </p>

    <h2 style='color: #333333;'>Who is it for?</h2>
    <ul style='color: #555555; font-size: 1.1rem;'>
        <li>A buyer wanting to avoid overpaying,</li>
        <li>A seller curious about your home's worth,</li>
        <li>An investor comparing market trends,</li>
        <li>Or simply curious about real estate in your neighborhood —</li>
    </ul>

    <p style='color: #555555; font-size: 1.1rem;'>
    This tool provides a fast, data-backed price estimate based on real listings and predictive algorithms.
    </p>
    """, unsafe_allow_html=True)



st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

    html, body, [class*="st-"], [data-testid="stAppViewContainer"] {
        font-family: 'Montserrat', sans-serif;
        background-color: #FAF9F6;
    }

    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(to bottom right, #ffffff, #f4f4f4);
        padding: 2rem 3rem;
    }

    .stApp {
        background-color: #FAF9F6 !important;
        background-image: linear-gradient(to bottom right, #ffffff, #f4f4f4);
        background-attachment: fixed;
    }

    h1 {
        color: #2A9D8F;
        font-weight: 700;
        font-size: 2.8rem;
    }

    .stForm {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    button[kind="primary"] {
        background-color: #2A9D8F;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# ---- Title ----
st.title("Property Price Predictor")
st.markdown("Use the form below to estimate the market value of a property in Belgium.")


# ---- Input Fields ----
with st.container():
    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            area = st.number_input("Living Area (m²)", min_value=6, max_value=200000, value=100)
            rooms_number = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
            bathrooms = st.number_input("Toilets & Bathrooms", min_value=1, max_value=10, value=1)

        with col2:
            property_type = st.selectbox("Property Type", ["APARTMENT", "HOUSE", "OTHERS"])
            zip_code = st.number_input("Zip Code", min_value=1000, max_value=9992, value=1000)
            facades_number = st.number_input("Number of Facades", min_value=1, max_value=4, value=2)

        terrace = st.checkbox("Terrace")

        building_state = st.selectbox("Building Condition", [
            "NEW", "GOOD", "TO RENOVATE", "JUST RENOVATED", "TO REBUILD"
        ])

        submitted = st.form_submit_button("Predict Price")


# ---- API Call ----
if submitted:
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

    with st.spinner("Sending data to prediction model... ⏳"):
        try:
            api_url = "https://challenge-api-deployment-wlr4.onrender.com/predict"
            response = requests.post(api_url, json=input_data)

            if response.status_code == 200:
                result = response.json()

                if 'prediction' in result:
                    price = int(result['prediction'])
                    st.success(f" Estimated Property Price: €{price:,}")
                else:
                    st.warning("No price prediction was returned by the model.")

            else:
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                except:
                    error_detail = response.text
                st.error(f"API Error: {error_detail}")

        except Exception as e:
            st.error(f"Could not reach the API: {e}")

