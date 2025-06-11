import pandas as pd
import numpy as np


"""
Load data from a CSV file and return a DataFrame.

Parameters:
file_path (str): Path to the CSV file.

Returns:
pd.DataFrame: Data loaded from the CSV file.
"""
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


"""
Preprocess the data by handling missing values and normalizing numerical features.

Parameters:
data (pd.DataFrame): The input DataFrame to preprocess.

Returns:
pd.DataFrame: The preprocessed DataFrame.
"""
def preprocess_data(data):
    # Handle missing values
    data.fillna(data.mean(), inplace=True)

    # Normalize numerical features
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].mean()) / data[numerical_cols].std()

    return data
