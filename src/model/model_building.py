import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
import yaml
import pickle


# Load parameters from yaml file
def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)['model_building']
    return params


# Load the dataset
def load_data(x_train_path: str, y_train_path: str):
    x_train = pd.read_csv(
        x_train_path,
        usecols=['Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome', 'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 'PreferredOrderCat', 'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromLastYear', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']
    )
    y_train = pd.read_csv(y_train_path)
    return x_train, y_train


# Feature transformation using ColumnTransformer
def build_transformer():
    trf1 = ColumnTransformer([
        ('Categorical_column_OHE', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), [1, 4, 5, 8])
    ], remainder='passthrough')

    trf2 = ColumnTransformer([
        ('scale', StandardScaler(), slice(0, 24))
    ])

    return trf1, trf2


# Build the model pipeline
def build_model(params: dict, trf1, trf2):
    decisiontreeclassifier = DecisionTreeClassifier(
        max_depth=params['max_depth'], 
        criterion='gini', 
        max_leaf_nodes=params['max_leaf_nodes']
    )
    
    model = make_pipeline(trf1, trf2, decisiontreeclassifier)
    return model


# Train the model
def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    print("Model trained successfully")
    return model


# Save the trained model
def save_model(model, output_path: str):
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {output_path}")


# Main function to run the model pipeline
def main():
    # Load params
    params = load_params('params.yaml')
    
    # Load data
    x_train, y_train = load_data('./data/processed/x_train.csv', './data/processed/y_train.csv')
    
    # Build transformers and model
    trf1, trf2 = build_transformer()
    model = build_model(params, trf1, trf2)
    
    # Train the model
    trained_model = train_model(model, x_train, y_train)
    
    # Save the trained model
    save_model(trained_model, 'model.pkl')


if __name__ == "__main__":
    main()
