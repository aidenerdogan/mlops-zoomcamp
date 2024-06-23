import pickle
import pandas as pd
import os

# Load the model and vectorizer from the pickle file within the Docker container
with open('/app/model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def predict_mean_duration(year, month):
    # Construct filename based on year and month
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'

    # Read data
    df = read_data(filename)

    # Prepare input for prediction
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    # Predict durations
    y_pred = model.predict(X_val)

    # Calculate mean predicted duration
    mean_predicted_duration = y_pred.mean()

    return mean_predicted_duration

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Calculate mean predicted duration for a specific month and year.')
    parser.add_argument('year', type=int, help='Year (e.g., 2023)')
    parser.add_argument('month', type=int, help='Month (e.g., 5 for May)')
    args = parser.parse_args()

    # Call predict_mean_duration function with provided year and month
    mean_duration = predict_mean_duration(args.year, args.month)

    # Print the mean predicted duration
    print(f"Mean predicted duration for {args.month}/{args.year}: {mean_duration:.2f} minutes")

    # Create ride_id and save predictions to Parquet file (optional)
    year = args.year
    month = args.month
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    output_file = '/app/predictions.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    # Get the size of the output file
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert bytes to megabytes
    print(f"Size of {output_file}: {file_size:.2f} MB")
