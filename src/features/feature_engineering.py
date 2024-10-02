
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml


# Load the test size from the parameters file
def test_params(params_path: str) -> float:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    test_size = params['feature_engineering']['test_size']
    return test_size


# Split data into X and Y
def x_y() -> tuple:
    df = pd.read_csv('./data/interim/dfprocessed_.csv')
    x = df.drop(columns='Churn')
    y = df['Churn']
    return x, y


# Save the train/test split data to CSV files
def save_train_test(data_path: str, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    os.makedirs(data_path, exist_ok=True)  # Ensure the directory exists
    pd.DataFrame(x_test).to_csv(os.path.join(data_path, 'x_test.csv'), index=False)
    pd.DataFrame(x_train).to_csv(os.path.join(data_path, 'x_train.csv'), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(data_path, 'y_test.csv'), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(data_path, 'y_train.csv'), index=False)


# Main function to load, split, and save data
def main() -> None:
    test_size = test_params('params.yaml')  # Load test size from YAML
    x, y = x_y()  # Load X and Y data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=41)
    
    data_path = os.path.join("data", "processed")
    save_train_test(data_path, x_train, x_test, y_train, y_test)  # Save train/test splits


# Ensure the script runs as expected
if __name__ == "__main__":
    main()
