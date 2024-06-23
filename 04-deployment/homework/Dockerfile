FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Install pandas (and any other dependencies you may need)
RUN pip install pandas scikit-learn pyarrow

# Set working directory in the container
WORKDIR /app

# Copy the Python script into the container
COPY predict_duration_script.py .

# Run the script for May 2023
CMD ["python", "predict_duration_script.py", "2023", "5"]
