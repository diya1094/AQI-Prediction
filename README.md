# AQI-Prediction
Streamlit AQI Analysis and Prediction App

This application provides a comprehensive platform for analyzing historical Air Quality Index (AQI) data across various Indian cities and predicting AQI based on pollutant concentrations.

---

## Features
- **Historical Analysis**: Visualize and compare AQI and pollutant trends over time for different cities.  
- **AQI Prediction**: A machine learning model to predict the AQI value based on real-time pollutant data.  
- **Health Advisor**: Get actionable health recommendations based on different AQI levels.  

---

### File Structure
 - app.py: The main Streamlit application script.
 - data_preprocessing.py: Script to clean and prepare the data.
 - model_training.py: Script to train the AQI prediction model.
 - requirements.txt: A list of all Python dependencies.
 - city_day.csv: The raw input data file (you must provide this).
 - image_9a8c07.png: The image for the health advisor page.
 - cleaned_city_day.csv: (Generated) The cleaned data used by the app.
 - aqi_predictor.pkl: (Generated) The trained machine learning model.
