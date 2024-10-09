import pandas as pd
import numpy as np
import os

# Function to load the data
def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, on_bad_lines='skip')
    df = df.drop(columns='Unnamed: 20')
    df = df.drop(columns='CustomerID')
    df = df.drop(columns='CouponUsed')
    
    return df

# Save the data locally
def save_data(data_path: str, df: pd.DataFrame) -> None:
    os.makedirs(data_path, exist_ok=True)  
    df.to_csv(os.path.join(data_path, "df.csv"), index=False)
    

# Main function to load and save data
def main() -> None:
    df = load_data('data/customer_data_model_refined.csv')
    # df = load_data(os.path.join("data", "customer_data_model_refined.csv"))

    data_path = os.path.join("data", "raw")
    
    save_data(data_path, df)

# Correct the __main__ check
if __name__ == "__main__":
    main()
