import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import json

# Set Streamlit page title and icon
st.set_page_config(page_title="Property Price Predictor", layout="wide")

# --- Load zip code data (cached) ---
@st.cache_data
def load_zipcode_data():
    with open("municipalities_codes.json", encoding="utf-8") as f:
        return json.load(f)

zipcode_data = load_zipcode_data()

def get_zipcode_info(zipcode, data):
    for entry in data:
        if str(entry["column_1"]) == str(zipcode):
            return {
                "lat": entry["coordonnees"]["lat"],
                "lon": entry["coordonnees"]["lon"],
                "municipality": entry["municipality_name_french"]
            }
    return None

st.markdown("""
<style>
    .main {
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .block-container {
        max-width: 100%;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("""
    <div style='text-align: center; padding: 20px 0 10px 0;'>
        <h1 style='font-size: 80px;'>IMMO ELIZA</h1>
        <h2 style='font-size: 48px; font-weight: normal; margin-bottom: 30px;'>
            Belgium Real Estate Price Predictor
        </h2>
    </div>
""", unsafe_allow_html=True)

# Image
col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
with col_img2:
    st.image("skyline.png", use_column_width=True)

# Description
st.markdown("""
    <div style='text-align: center; padding: 10px 0 40px 0;'>
        <p style='font-size: 18px; max-width: 720px; margin: auto; line-height: 1.6;'>
            Estimate the price of real estate in Belgium using machine learning. Our model was trained on real-world data, making it a reliable tool to guide pricing decisions.<br>
            This tool helps home buyers, sellers, investors, and real estate professionals make smarter buying/selling decisions.
        </p>
    </div>
""", unsafe_allow_html=True)

# Information sections
st.markdown("""
    <div style='padding-left: 10%; padding-right: 10%; margin-bottom: 30px;'>
        <h2>What data do we use?</h2>
        <p style='font-size: 16px;'>
            Our prediction model evaluates key features such as:<br>
            - Living Area<br>
            - Property Type<br>
            - Zip Code<br>
            - Number of Rooms<br>
            - Facades<br>
            - Terrace<br>
            - Building Condition<br><br>
            We aim to make the real estate market more transparent and accessible for everyone. Whether you're buying, selling, or comparing investment opportunities, this tool delivers fast and informative price estimates that help you take the next step with confidence.
        </p>
    </div>
""", unsafe_allow_html=True)


# Example dataset for visualizations
data = {
    "area": [50, 70, 100, 150, 200, 120, 80, 60, 90, 130],
    "price": [100000, 150000, 200000, 300000, 400000, 220000, 140000, 110000, 180000, 250000],
    "property_type": ["APARTMENT", "HOUSE", "HOUSE", "HOUSE", "OTHERS", "APARTMENT", "APARTMENT", "OTHERS", "HOUSE", "HOUSE"],
    "rooms_number": [1, 3, 4, 5, 6, 2, 2, 3, 4, 5],
    "building_state": ["NEW", "GOOD", "TO RENOVATE", "JUST RENOVATED", "TO REBUILD", "GOOD", "NEW", "TO RENOVATE", "GOOD", "JUST RENOVATED"]
}
df = pd.DataFrame(data)

# --- Required fields ---
col1, col2 = st.columns(2)

with col1:
    zip_code = st.number_input("Zip Code", min_value=1000, max_value=9992, value=1000)
    property_type = st.selectbox("Property Type", ["APARTMENT", "HOUSE", "OTHERS"])
    rooms_number = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
    area = st.number_input("Living Area (m²)", min_value=6, max_value=185347, value=100)

with col2:
    building_state = st.selectbox("Building Condition", [
        "NEW", "GOOD", "TO RENOVATE", "JUST RENOVATED", "TO REBUILD"
    ])
    facades_number = st.number_input("Number of Facades", min_value=1, max_value=4, value=2)
    bathrooms = st.number_input("Toilets & Bathrooms", min_value=1, max_value=10, value=1)
    terrace = st.checkbox("Terrace")

# stylize button
st.markdown("""
    <style>
    div.stButton > button:first-child {
        font-size: 1.5em !important;
        padding: 1em 1.5em !important;
        font-weight: bold !important;
        background-color: teal !important;
        color: white !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

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
    with st.spinner("Calculating..."):
        try:
            api_url = "https://challenge-api-deployment-wlr4.onrender.com/predict"
            response = requests.post(api_url, json=input_data)

            if response.status_code == 200:
                result = response.json()
                price = int(result['prediction'])

                # Get zip code info
                info = get_zipcode_info(zip_code, zipcode_data)

                # --- Card Layout (Horizontal) ---
                with st.container():
                    st.subheader("Property Summary")
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown(f"**Municipality**: {info['municipality'] if info else 'Unknown'}")
                        st.markdown(f"**Property Type**: {property_type}")
                        st.markdown(f"**Living Area**: {area} m²")
                        st.markdown(f"**Bedrooms**: {rooms_number}")
                        st.markdown(f"**Bathrooms**: {bathrooms}")
                        st.markdown(f"**Facades**: {facades_number}")
                        st.markdown(f"**Terrace**: {'Yes' if terrace else 'No'}")
                        st.markdown(f"**Building Condition**: {building_state}")
                        st.subheader(f"Predicted Price: €{price:,}")

                    with col2:
                        if info:
                            st.map(pd.DataFrame({'lat': [info["lat"]], 'lon': [info["lon"]]}))
                        else:
                            st.warning("❗ Location not found for this zip code.")

            else:
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                except:
                    error_detail = response.text
                st.error(f"Error: {error_detail}")

        except Exception as e:
            st.error(f"Could not reach the API: {e}")


# Color scheme per property type
colors = {
    "ALL": "cyan",
    "APARTMENT": "red",
    "HOUSE": "teal",
    "OTHERS": "purple"
}

# Dropdown to filter by property type (affects chart colors)
property_type_filter = st.selectbox(
    "Filter by Property Type",
    options=["ALL", "APARTMENT", "HOUSE", "OTHERS"],
    index=0
)

selected_color = colors[property_type_filter]

# Filter dataframe based on selection
if property_type_filter != "ALL":
    df_filtered = df[df["property_type"] == property_type_filter]
else:
    df_filtered = df.copy()

# Average price by area
st.subheader(f"Price Trend by Living Area ({property_type_filter})")
df_area_price = df_filtered.groupby("area")["price"].mean().reset_index()
fig_area_price = px.line(
    df_area_price,
    x="area",
    y="price",
    labels={"area": "Living Area (m²)", "price": "Average Price (€)"},
    title=f"Average Price vs Living Area ({property_type_filter})",
    color_discrete_sequence=[selected_color]
)
st.plotly_chart(fig_area_price, use_container_width=True)

# Global price distribution boxplot (always uses ALL color)
st.subheader("Price Distribution per Property Type")
fig1 = px.box(
    df,
    x="property_type",
    y="price",
    color="property_type",
    color_discrete_map={
        "APARTMENT": colors["APARTMENT"],
        "HOUSE": colors["HOUSE"],
        "OTHERS": colors["OTHERS"]
    },
    labels={"property_type": "Property Type", "price": "Price (€)"},
    title="Price Distribution by Property Type"
)
st.plotly_chart(fig1, use_container_width=True)

# Average price by property type
st.subheader("Average Price per Property Type")
df_avg_type = df.groupby("property_type")["price"].mean().reset_index()
fig_avg_type = px.bar(
    df_avg_type,
    x="property_type",
    y="price",
    color="property_type",
    color_discrete_map={
        "APARTMENT": colors["APARTMENT"],
        "HOUSE": colors["HOUSE"],
        "OTHERS": colors["OTHERS"]
    },
    labels={"property_type": "Property Type", "price": "Avg Price (€)"},
    title="Average Price per Property Type",
)
fig_avg_type.update_traces(opacity=0.5)
st.plotly_chart(fig_avg_type, use_container_width=True)

