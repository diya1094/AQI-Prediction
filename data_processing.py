import pandas as pd

def preprocess_data(input_path='city_day.csv', output_path='cleaned_city_day.csv'):
    """
    Cleans the city_day.csv data and saves it to a new file.

    Args:
        input_path (str): The path to the input CSV file.
        output_path (str): The path to save the cleaned CSV file.
    """
    print("Starting data preprocessing...")

    # Read the dataset
    try:
        df = pd.read_csv(input_path, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
        return

    # Sort by City and Date to ensure correct time-series handling
    df.sort_values(['City', 'Date'], inplace=True)

    # List of pollutant columns to process
    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']

    # Impute missing values using forward fill and then backward fill
    # This is a common strategy for time-series data
    for col in pollutant_cols:
        df[col] = df.groupby('City')[col].transform(lambda x: x.ffill().bfill())

    # Drop rows that still have missing AQI values after imputation
    df.dropna(subset=['AQI', 'AQI_Bucket'], inplace=True)

    # Ensure data types are correct
    for col in pollutant_cols:
        if col != 'AQI_Bucket':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any remaining rows with NaNs in key columns
    df.dropna(inplace=True)

    # Save the cleaned data
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Cleaned data saved to {output_path}")

if __name__ == '__main__':
    preprocess_data()
