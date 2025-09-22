# AQI-Prediction
Streamlit AQI Analysis and Prediction App
This application provides a comprehensive platform for analyzing historical Air Quality Index (AQI) data across various Indian cities and predicting AQI based on pollutant concentrations.

Features
Historical Analysis: Visualize and compare AQI and pollutant trends over time for different cities.

AQI Prediction: A machine learning model to predict the AQI value based on real-time pollutant data.

Health Advisor: Get actionable health recommendations based on different AQI levels.

How to Run the Application
Follow these steps to set up and run the project on your local machine.

Step 1: Install Dependencies
First, you need to install all the required Python libraries. A requirements.txt file is provided for this purpose.

pip install -r requirements.txt

Step 2: Prepare the Data
The raw data needs to be cleaned and preprocessed. Run the data_preprocessing.py script. This will read city_day.csv, clean it, and save the result as cleaned_city_day.csv.

python data_preprocessing.py

Step 3: Train the Prediction Model
Next, train the machine learning model using the cleaned data. Run the model_training.py script. This will create a file named aqi_predictor.pkl which contains the trained model.

python model_training.py

Step 4: Launch the Streamlit App
Now you are ready to run the main application. Use the following command in your terminal:

streamlit run app.py

Your web browser should open a new tab with the running application.

File Structure
app.py: The main Streamlit application script.

data_preprocessing.py: Script to clean and prepare the data.

model_training.py: Script to train the AQI prediction model.

requirements.txt: A list of all Python dependencies.

city_day.csv: The raw input data file (you must provide this).

image_9a8c07.png: The image for the health advisor page.

cleaned_city_day.csv: (Generated) The cleaned data used by the app.

aqi_predictor.pkl: (Generated) The trained machine learning model.
