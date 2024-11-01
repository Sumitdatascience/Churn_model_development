
# from flask import Flask, request, jsonify, render_template
# import pickle
# import numpy as np
# import pandas as pd
# import mlflow
# import dagshub
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# import os

# # Initialize MLflow and DAGsHub
# # mlflow.set_tracking_uri('https://dagshub.com/Sumitdatascience/Churn_model_development.mlflow')
# # dagshub.init(repo_owner='Sumitdatascience', repo_name='Churn_model_development', mlflow=True)

# # Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("DAGSHUB_PAT")
# if not dagshub_token:
#     raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "Sumitdatascience"
# repo_name = "Churn_model_development"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# app = Flask(__name__)

# # load model from model registry
# def get_latest_model_version(model_name):
#     client = mlflow.MlflowClient()
#     latest_version = client.get_latest_versions(model_name, stages=["Production"])
#     if not latest_version:
#         latest_version = client.get_latest_versions(model_name, stages=["None"])
#     return latest_version[0].version if latest_version else None

# model_name = "Churn_Prediction_Model"
# model_version = get_latest_model_version(model_name)

# model_uri = f'models:/{model_name}/{model_version}'
# model = mlflow.pyfunc.load_model(model_uri)

# # Initialize the Flask app
# # app = Flask(__name__)

# # Load the trained model
# # Model parameters
# # model_name = "Churn_Prediction_Model"
# # model_version = 6
# # model_uri = f'models:/{model_name}/{model_version}'

# # try:
# #     model = mlflow.pyfunc.load_model(model_uri)
# #     print('Model loaded successfully')
# # except Exception as e:
# #     print(f"Error loading model: {e}")
# #     model = None


# # Preprocess function to transform input data
# def preprocess_input(data):
#     # Convert the input data into a pandas DataFrame
#     columns = ['Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome', 'PreferredPaymentMode', 'Gender',
#                'HourSpendOnApp', 'NumberOfDeviceRegistered', 'PreferredOrderCat', 'SatisfactionScore',
#                'MaritalStatus', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromLastYear', 'OrderCount',
#                'DaySinceLastOrder', 'CashbackAmount']

#     input_df = pd.DataFrame([data], columns=columns)


#     print("input data loaded successfully")

#     return input_df


# # Define the home route (to serve the form)
# @app.route('/')
# def home():
#     return render_template('index.html')


# # Define the predict route (to make predictions)
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the data from the POST request
#         data = request.get_json()
#         print(f"data recived--{data}")

#         # Extract the input data from the form submission
#         input_data = [
#             int(data['tenure']),
#             data['preferredLoginDevice'],
#             int(data['cityTier']),
#             int(data['warehouseToHome']),
#             data['preferredPaymentMode'],
#             data['gender'],
#             int(data['hourSpendOnApp']),
#             int(data['numberOfDeviceRegistered']),
#             data['preferredOrderCat'],
#             int(data['satisfactionScore']),
#             int(data['maritalStatus']),
#             float(data['numberOfAddress']),
#             int(data['complain']),
#             float(data['orderAmountHikeFromLastYear']),
#             int(data['orderCount']),
#             float(data['daySinceLastOrder']),
#             float(data['cashbackAmount'])
#         ]

#         print(f"input data --{input_data}")

#         # Preprocess the input data to match the model's requirements
#         processed_input = preprocess_input(input_data)
#         print(f"processed_input --{processed_input}")

#         # Make prediction
#         prediction = model.predict(processed_input)
#         print(prediction)

#         # Return the prediction result as a JSON response
#         return jsonify({'prediction': str(prediction[0])})

#     except Exception as e:
#         # Handle any error that occurs during prediction
#         return jsonify({'error': str(e)}), 500


# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import mlflow
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Set up MLflow and DAGsHub
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Sumitdatascience"
repo_name = "Churn_model_development"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

app = Flask(__name__)
CORS(app)

# Function to get the latest model version
def get_latest_model_version(model_name):
    try:
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_version:
            logging.warning(f"No production version found for {model_name}, checking for any version.")
            latest_version = client.get_latest_versions(model_name, stages=["None"])
        return latest_version[0].version if latest_version else None
    except Exception as e:
        logging.error(f"Error retrieving model version: {str(e)}")
        return None

# Load the model from MLflow Model Registry
model_name = "Churn_Prediction_Model"
model_version = get_latest_model_version(model_name)

if model_version:
    try:
        model_uri = f'models:/{model_name}/{model_version}'
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info(f"Model loaded successfully: {model_name}, version {model_version}")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        model = None
else:
    logging.error(f"No model version available for {model_name}")
    model = None

# Preprocess input data
def preprocess_input(data):
    columns = ['Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome', 'PreferredPaymentMode', 'Gender',
               'HourSpendOnApp', 'NumberOfDeviceRegistered', 'PreferredOrderCat', 'SatisfactionScore',
               'MaritalStatus', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromLastYear', 'OrderCount',
               'DaySinceLastOrder', 'CashbackAmount']

    input_df = pd.DataFrame([data], columns=columns)

    # Add transformation logic here if necessary (e.g., encoding, scaling)
    logging.info("Input data loaded successfully")
    return input_df

# Define the home route
@app.route('/')
def home():
    return render_template('index1.html')

# # Define the predict route
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the data from the POST request
#         data = request.get_json()

#         # Validate the input data
#         required_fields = ['tenure', 'preferredLoginDevice', 'cityTier', 'warehouseToHome', 'preferredPaymentMode',
#                            'gender', 'hourSpendOnApp', 'numberOfDeviceRegistered', 'preferredOrderCat',
#                            'satisfactionScore', 'maritalStatus', 'numberOfAddress', 'complain', 
#                            'orderAmountHikeFromLastYear', 'orderCount', 'daySinceLastOrder', 'cashbackAmount']
#         for field in required_fields:
#             if field not in data:
#                 raise ValueError(f"Missing field in input data: {field}")

#         # Extract and process the input data
#         input_data = [
#             int(data['tenure']),
#             data['preferredLoginDevice'],
#             int(data['cityTier']),
#             int(data['warehouseToHome']),
#             data['preferredPaymentMode'],
#             data['gender'],
#             int(data['hourSpendOnApp']),
#             int(data['numberOfDeviceRegistered']),
#             data['preferredOrderCat'],
#             int(data['satisfactionScore']),
#             int(data['maritalStatus']),
#             float(data['numberOfAddress']),
#             int(data['complain']),
#             float(data['orderAmountHikeFromLastYear']),
#             int(data['orderCount']),
#             float(data['daySinceLastOrder']),
#             float(data['cashbackAmount'])
#         ]

#         processed_input = preprocess_input(input_data)

#         # Make prediction
#         if model:
#             prediction = model.predict(processed_input)
#             logging.info(f"Prediction result: {prediction[0]}")
#             return jsonify({'prediction': str(prediction[0])})
#         else:
#             raise Exception("Model is not available")

#     except Exception as e:
#         logging.error(f"Error during prediction: {str(e)}")
#         return jsonify({'error': str(e)}), 500


   ##### Work with Index1.html ###
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate the input data
        required_fields = ['tenure', 'preferredLoginDevice', 'cityTier', 'warehouseToHome', 'preferredPaymentMode',
                           'gender', 'hourSpendOnApp', 'numberOfDeviceRegistered', 'preferredOrderCat',
                           'satisfactionScore', 'maritalStatus', 'numberOfAddress', 'complain', 
                           'orderAmountHikeFromLastYear', 'orderCount', 'daySinceLastOrder', 'cashbackAmount']
        
        # Retrieve data from the form submission
        data = {}
        for field in required_fields:
            value = request.form.get(field)
            if value is None:
                raise ValueError(f"Missing field in input data: {field}")
            data[field] = value

        # Extract and process the input data without changing the data types
        input_data = [
            int(data['tenure']),
            data['preferredLoginDevice'],
            int(data['cityTier']),
            int(data['warehouseToHome']),
            data['preferredPaymentMode'],
            data['gender'],
            int(data['hourSpendOnApp']),
            int(data['numberOfDeviceRegistered']),
            data['preferredOrderCat'],
            int(data['satisfactionScore']),
            int(data['maritalStatus']),
            float(data['numberOfAddress']),
            int(data['complain']),
            float(data['orderAmountHikeFromLastYear']),
            int(data['orderCount']),
            float(data['daySinceLastOrder']),
            float(data['cashbackAmount'])
        ]

        # Preprocess the input data for the model
        processed_input = preprocess_input(input_data)

        # Make prediction
        if model:
            prediction = model.predict(processed_input)
            logging.info(f"Prediction result: {prediction[0]}")
            return jsonify({'prediction': str(prediction[0])})
        else:
            raise Exception("Model is not available")

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,host= "0.0.0.0")
