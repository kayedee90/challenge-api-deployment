# ImmoEliza House Price Prediction API

---

## Description

The **ImmoEliza API** is a service designed to predict real estate prices based on detailed property attributes. Built for ImmoEliza’s web developers, it exposes an easy-to-use JSON API backed by a trained machine learning regression model (CatBoost).

With this API, users can send property details and receive accurate price predictions in real-time — no need for manual price estimations or offline calculations.

This project showcases modern deployment practices: built with **FastAPI**, containerized using **Docker**, and hosted on **Render.com** for seamless cloud availability.

---
This project includes:
- A **FastAPI backend** that serves price predictions through a RESTful API.
- A **Streamlit frontend** that provides an intuitive UI for users to input property data and view predictions instantly.
- A **Dockerized architecture**, ensuring easy setup and consistent deployment.
- **Cloud deployment** on [Render](https://render.com/) for the API.


## Installation

### Backend API (FastAPI with Docker)
### Description

The backend API is containerized using Docker. The Dockerfile uses Ubuntu 22.04 and installs Python 3.10. It runs the FastAPI app with Uvicorn on port 8000.

**Key points:**

- Base image: `ubuntu:22.04`
- Python 3.10 installed along with pip
- Exposes port `8000`
- Starts the API with:
   
```bash
  uvicorn app:app --host 0.0.0.0 --port 8000
```

## Installation

To run this project locally, make sure you have **Docker** installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/kayedee90/challenge-api-deployment.git
   cd challenge-api-deployment
   ```

2. Build the Docker image:
    ```bash
    docker build -t immoeliza-api .
   ```

3. Run the container locally:
    ```bash 
    docker run -p 8000:8000 immoeliza-api

   ```

4. Access the API at:
    ```bash 
    http://localhost:8000/docs
    
   ```
### Streamlit Frontend App

### Description
This is the frontend app built with Streamlit to provide a simple and interactive user interface for the ImmoEliza house price prediction API. Users can input house features and get instant price predictions via the deployed API.

---

## Running the Streamlit App

To run the Streamlit frontend locally, follow these steps:

1. Make sure your FastAPI backend is deployed and accessible on Render (or your chosen hosting platform).

2. Set the correct API URL in your Streamlit app (e.g. inside streamlit_app.py): API_URL = "https://your-api.onrender.com/predict"

3. Run the Streamlit app:  
   ```bash
   streamlit run streamlit_app.py
   ```
4. Open your browser at http://localhost:8501 to use the app.

### Usage

Once both the backend and frontend are running:

- Navigate to the Streamlit app in your browser: `http://localhost:8501`
- Enter property details like number of bedrooms, location, living area, etc.
- Click the “Predict” button to receive an estimated property price.
- For developers: you can directly interact with the API at `http://localhost:8000/docs`

### Project Structure
├── app.py # FastAPI application (entry point)

├── streamlit_app.py # Streamlit frontend UI

├── model/ # Trained ML model files

├── Dockerfile # Docker config for backend API

├── requirements.txt # Python dependencies

└── README.md # Project documentation

### Visuals
screenshot from our app

### Contributors
Kenny, Yassine, Evi
