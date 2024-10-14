import pandas as pd
import os

# Function to load the data
def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, on_bad_lines='skip')
    df = df.drop(columns='Unnamed: 20', errors='ignore')  # Make this safer if column doesn't exist
    df = df.drop(columns='CustomerID', errors='ignore')
    df = df.drop(columns='CouponUsed', errors='ignore')
    
    return df

# Save the data locally
def save_data(data_path: str, df: pd.DataFrame) -> None:
    os.makedirs(data_path, exist_ok=True)  
    df.to_csv(os.path.join(data_path, "df.csv"), index=False)

# Main function to load and save data
def main() -> None:
    # Using relative path with forward slashes
    df = load_data("https://raw.githubusercontent.com/Sumitdatascience/Churn_model_development/refs/heads/master/customer_data_model_refined.csv")
    data_path = os.path.join("data", "raw")
    save_data(data_path, df)

if __name__ == "__main__":
    main()
