#--- LOAD DATA AND MODEL ---
import pickle
import streamlit as st
import pandas as pd
def load_data(path):
    """Loads the cleaned data."""
    try:
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file at {path} was not found. Please run `data_preprocessing.py` first.")
        return None

@st.cache_resource
def load_model(path):
    """Loads the trained machine learning model."""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: The model file at {path} was not found. Please run `model_training.py` first.")
        return None

df = load_data('cleaned_city_day.csv')
model = load_model('aqi_predictor.pkl')

# --- HEALTH ADVISOR FUNCTION ---
def get_health_advisor(aqi_value):
    """Provides health advice based on AQI level."""
    if aqi_value <= 50:
        return ("Good", "Minimal impact. Enjoy outdoor activities.", "#4CAF50")
    elif aqi_value <= 100:
        return ("Satisfactory", "Minor breathing discomfort to sensitive people.", "#FFEB3B")
    elif aqi_value <= 200:
        return ("Moderate", "Breathing discomfort to people with lung disease, and children.", "#FF9800")
    elif aqi_value <= 300:
        return ("Poor", "Breathing discomfort to most people on prolonged exposure.", "#f44336")
    elif aqi_value <= 400:
        return ("Very Poor", "Respiratory illness on prolonged exposure.", "#9C27B0")
    else:
        return ("Severe", "Affects healthy people and seriously impacts those with existing diseases.", "#795548")


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🌬️ AQI India Dashboard")
st.sidebar.markdown("Navigate through the sections below.")
page = st.sidebar.radio("Choose a Page", ["🏠 Home", "📊 Historical Analysis", "🤖 AQI Prediction", "❤️ Health Advisor"])


# --- HOME PAGE ---
if page == "🏠 Home":
    st.title("Air Quality Index (AQI) Analysis and Prediction for India")
    st.markdown("""
        Welcome to the AQI India Dashboard. This application provides tools to analyze historical air quality data and predict future AQI values based on pollutant concentrations.
        
        **What is AQI?**
        The Air Quality Index is a number used by government agencies to communicate to the public how polluted the air currently is or how polluted it is forecast to become.
        
        Use the navigation on the left to explore different features:
        - **Historical Analysis:** Visualize AQI trends and pollutant levels over time for various cities.
        - **AQI Prediction:** Predict the AQI value based on real-time pollutant data.
        - **Health Advisor:** Get health recommendations based on AQI levels.
    """)
    st.image('image_9a8c07.png', caption='AQI Categories and Health Impacts', use_column_width=True)


# --- HISTORICAL ANALYSIS PAGE ---
elif page == "📊 Historical Analysis":
    st.title("Historical AQI Analysis")
    st.markdown("Explore the trends of different pollutants and AQI over time in Indian cities.")

    if df is not None:
        city = st.selectbox("Select a City", sorted(df['City'].unique()))

        if city:
            city_df = df[df['City'] == city]

            # Date Range Selector
            min_date = city_df['Date'].min().date()
            max_date = city_df['Date'].max().date()
            start_date, end_date = st.date_input(
                "Select Date Range",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date,
            )
            
            # Convert dates from date_input to datetime to filter dataframe
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            mask = (city_df['Date'] >= start_date) & (city_df['Date'] <= end_date)
            filtered_df = city_df.loc[mask]

            # AQI Trend
            st.subheader(f"AQI Trend for {city}")
            fig_aqi = px.line(filtered_df, x='Date', y='AQI', title=f'AQI Over Time in {city}',
                              labels={'AQI': 'Air Quality Index'})
            fig_aqi.update_layout(template="plotly_white")
            st.plotly_chart(fig_aqi, use_container_width=True)

            # Pollutant Analysis
            st.subheader(f"Pollutant Concentration Analysis for {city}")
            pollutant_to_plot = st.selectbox("Select a Pollutant", ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'])
            fig_pollutant = px.line(filtered_df, x='Date', y=pollutant_to_plot,
                                    title=f'{pollutant_to_plot} Levels Over Time in {city}',
                                    labels={pollutant_to_plot: f'{pollutant_to_plot} (μg/m³)'})
            fig_pollutant.update_layout(template="plotly_white")
            st.plotly_chart(fig_pollutant, use_container_width=True)


# --- AQI PREDICTION PAGE ---
elif page == "🤖 AQI Prediction":
    st.title("AQI Prediction")
    st.markdown("Enter the concentration of pollutants to predict the AQI.")

    if model is not None:
        with st.form("prediction_form"):
            st.subheader("Enter Pollutant Values:")
            cols = st.columns(3)
            # Input features for the model
            pm25 = cols[0].number_input("PM2.5 (μg/m³)", min_value=0.0, format="%.2f")
            pm10 = cols[1].number_input("PM10 (μg/m³)", min_value=0.0, format="%.2f")
            no = cols[2].number_input("NO (μg/m³)", min_value=0.0, format="%.2f")
            no2 = cols[0].number_input("NO2 (μg/m³)", min_value=0.0, format="%.2f")
            nox = cols[1].number_input("NOx (ppb)", min_value=0.0, format="%.2f")
            nh3 = cols[2].number_input("NH3 (μg/m³)", min_value=0.0, format="%.2f")
            co = cols[0].number_input("CO (mg/m³)", min_value=0.0, format="%.2f")
            so2 = cols[1].number_input("SO2 (μg/m³)", min_value=0.0, format="%.2f")
            o3 = cols[2].number_input("O3 (μg/m³)", min_value=0.0, format="%.2f")
            benzene = cols[0].number_input("Benzene (μg/m³)", min_value=0.0, format="%.2f")
            toluene = cols[1].number_input("Toluene (μg/m³)", min_value=0.0, format="%.2f")
            xylene = cols[2].number_input("Xylene (μg/m³)", min_value=0.0, format="%.2f")

            submit_button = st.form_submit_button(label="Predict AQI")

        if submit_button:
            features = [[pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene]]
            prediction = model.predict(features)
            predicted_aqi = int(prediction[0])
            
            bucket, advice, color = get_health_advisor(predicted_aqi)

            st.subheader("Prediction Result")
            st.metric(label="Predicted AQI", value=predicted_aqi)
            
            st.markdown(f"""
            <div style="padding: 1rem; border-radius: 10px; background-color: {color}; color: white; text-align: center;">
                <h3 style="color: white;">AQI Bucket: {bucket}</h3>
                <p>{advice}</p>
            </div>
            """, unsafe_allow_html=True)


# --- HEALTH ADVISOR PAGE ---
elif page == "❤️ Health Advisor":
    st.title("Health Advisor Based on AQI")
    st.markdown("Understand the health implications of different AQI levels.")
    
    aqi_level = st.slider("Select an AQI Value to see health advice:", 0, 500, 150)
    
    bucket, advice, color = get_health_advisor(aqi_level)
    
    st.subheader(f"For an AQI of {aqi_level}:")
    st.markdown(f"""
    <div style="padding: 1rem; border-radius: 10px; background-color: {color}; color: white;">
        <h3 style="color: white;">Category: {bucket}</h3>
        <p><strong>Health Advice:</strong> {advice}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.image('image_9a8c07.png', caption='AQI Categories and Health Impacts', use_column_width=True)

