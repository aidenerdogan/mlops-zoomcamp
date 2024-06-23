import argparse
import pickle
import pandas as pd

# Function to read data and preprocess
def read_data(filename):
    df = pd.read_parquet(filename)

    # Calculate duration in minutes
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60

    # Filter out durations not in range [1, 60] minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Handle categorical columns
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

# Function to predict and calculate mean duration
def predict_mean_duration(year, month):
    # Construct filename based on year and month
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'

    # Read data
    df = read_data(filename)

    # Load model and vectorizer
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # Prepare input for prediction
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    # Predict durations
    y_pred = model.predict(X_val)

    # Calculate mean predicted duration
    mean_predicted_duration = y_pred.mean()

    return mean_predicted_duration

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Calculate mean predicted duration for a specific month and year.')
    parser.add_argument('year', type=int, help='Year (e.g., 2023)')
    parser.add_argument('month', type=int, help='Month (e.g., 4 for April)')
    args = parser.parse_args()

    # Call predict_mean_duration function with provided year and month
    mean_duration = predict_mean_duration(args.year, args.month)

    # Print the mean predicted duration
    print(f"Mean predicted duration for {args.month}/{args.year}: {mean_duration:.2f} minutes")
