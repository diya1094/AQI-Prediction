import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np

def train_model(input_path='cleaned_city_day.csv', model_path='aqi_predictor.pkl'):
    """
    Trains a RandomForestRegressor model to predict AQI.

    Args:
        input_path (str): The path to the cleaned data file.
        model_path (str): The path to save the trained model.
    """
    print("Starting model training...")

    # Load the cleaned data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found. Please run data_preprocessing.py first.")
        return

    # Define features (X) and target (y)
    features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    target = 'AQI'

    X = df[features]
    y = df[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the RandomForestRegressor model
    # --- OPTION 1: Original Model (might be > 100MB) ---
    # model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # --- OPTION 2: Smaller Model (likely < 100MB) ---
    # By reducing the number of trees (n_estimators) and limiting their depth (max_depth),
    # we can significantly reduce the final .pkl file size.
    # This might result in a slightly less accurate model, but it will be much smaller.
    model = RandomForestRegressor(
        n_estimators=50,      # Reduced from 100
        max_depth=15,         # Limit the depth of each tree
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=5    # Each leaf must have at least 5 samples
    )
    
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model training complete.")
    print(f"Root Mean Squared Error on the test set: {rmse:.2f}")

    # Save the trained model to a file using pickle
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()

