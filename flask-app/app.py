# from flask import Flask,render_template,request,jsonify
# import numpy as np
# import mlflow
# import dagshub

# mlflow.set_tracking_uri('https://dagshub.com/Sumitdatascience/Churn_model_development.mlflow')
# dagshub.init(repo_owner='Sumitdatascience', repo_name='Churn_model_development', mlflow=True)

# app = Flask(__name__)
# # load the model from mlflow(model registry)
# model_name = "Churn_Prediction_Model"
# model_version=6
# model_uri = f'models:/{model_name}/{model_version}'
# print('model_uri read successfully')

# model = mlflow.pyfunc.load_model(model_uri)
# print('model loaded successfully')


# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict',methods =['POST'])
# def predict():
#      # Get the data from the POST request
#     data = request.get_json()

#     # Extract the 17 input values
#     input_data = [
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
#             data['maritalStatus'],
#             int(data['numberOfAddress']),
#             int(data['complain']),
#             float(data['orderAmountHikeFromLastYear']),
#             int(data['orderCount']),
#             int(data['daySinceLastOrder']),
#             float(data['cashbackAmount'])
#         ] 

#     # Convert input data into a 2D array as expected by scikit-learn
#     input_array = np.array([input_data])

#     # Make the prediction using the loaded model
#     prediction = model.predict(input_array)

#     # Return the prediction result as a JSON response
#     return jsonify({'prediction': str(prediction[0])})
    


# app.run(debug=True)

# 


# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import mlflow
# import dagshub
# from flask_cors import CORS

# # Initialize MLflow with the Dagshub repository
# mlflow.set_tracking_uri('https://dagshub.com/Sumitdatascience/Churn_model_development.mlflow')
# dagshub.init(repo_owner='Sumitdatascience', repo_name='Churn_model_development', mlflow=True)

# # Create Flask app and enable CORS
# app = Flask(__name__)
# CORS(app)

# # Model parameters
# model_name = "Churn_Prediction_Model"
# model_version = 6
# model_uri = f'models:/{model_name}/{model_version}'

# try:
#     model = mlflow.pyfunc.load_model(model_uri)
#     print('Model loaded successfully')
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None:
#         return jsonify({'error': 'Model not available'})

#     try:
#         # Get data from POST request
#         data = request.get_json()

#         # Log the data to verify the request
#         print(f"Received data: {data}")

#         # Extract and validate the input data
#         input_data = [
#             int(data.get('tenure', 0)),
#             data.get('preferredLoginDevice', 'Unknown'),
#             int(data.get('cityTier', 0)),
#             int(data.get('warehouseToHome', 0)),
#             data.get('preferredPaymentMode', 'Unknown'),
#             data.get('gender', 'Unknown'),
#             int(data.get('hourSpendOnApp', 0)),
#             int(data.get('numberOfDeviceRegistered', 0)),
#             data.get('preferredOrderCat', 'Unknown'),
#             int(data.get('satisfactionScore', 0)),
#             data.get('maritalStatus', 'Unknown'),
#             int(data.get('numberOfAddress', 0)),
#             int(data.get('complain', 0)),
#             float(data.get('orderAmountHikeFromLastYear', 0.0)),
#             int(data.get('orderCount', 0)),
#             int(data.get('daySinceLastOrder', 0)),
#             float(data.get('cashbackAmount', 0.0))
#         ]
#         print(f"Input data: {input_data}")
#         # Convert input data to a 2D array as expected by the model
#         input_array = np.array([input_data])
#         print(f"Input array: {input_array}")

#         # Make prediction
#         prediction = model.predict(input_array)

#         # Return prediction as JSON response
#         return jsonify({'prediction': str(prediction[0])})

#     except KeyError as e:
#         return jsonify({'error': f"Missing key: {e}"}), 400
#     except Exception as e:
#         return jsonify({'error': f"An error occurred: {e}"}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import mlflow
import dagshub
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import os

# Initialize MLflow and DAGsHub
mlflow.set_tracking_uri('https://dagshub.com/Sumitdatascience/Churn_model_development.mlflow')
dagshub.init(repo_owner='Sumitdatascience', repo_name='Churn_model_development', mlflow=True)

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
# Model parameters
model_name = "Churn_Prediction_Model"
model_version = 6
model_uri = f'models:/{model_name}/{model_version}'

try:
    model = mlflow.pyfunc.load_model(model_uri)
    print('Model loaded successfully')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# Preprocess function to transform input data
def preprocess_input(data):
    # Convert the input data into a pandas DataFrame
    columns = ['Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome', 'PreferredPaymentMode', 'Gender',
               'HourSpendOnApp', 'NumberOfDeviceRegistered', 'PreferredOrderCat', 'SatisfactionScore',
               'MaritalStatus', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromLastYear', 'OrderCount',
               'DaySinceLastOrder', 'CashbackAmount']

    input_df = pd.DataFrame([data], columns=columns)

    # # Apply the same transformations used during model building (OneHotEncoding and Scaling)
    # trf1 = ColumnTransformer([('Categorical_column_OHE', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
    #                            [1, 4, 5, 8])], remainder='passthrough')
    # trf2 = ColumnTransformer([('scale', StandardScaler(), slice(0, 24))])

    # # First transformation (OneHotEncoding)
    # input_transformed = trf1.fit_transform(input_df)

    # # Second transformation (Scaling)
    # input_scaled = trf2.fit_transform(input_transformed)
    print("input data loaded successfully")

    return input_df


# Define the home route (to serve the form)
@app.route('/')
def home():
    return render_template('index.html')


# Define the predict route (to make predictions)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json()
        print(f"data recived--{data}")

        # Extract the input data from the form submission
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

        print(f"input data --{input_data}")

        # Preprocess the input data to match the model's requirements
        processed_input = preprocess_input(input_data)
        print(f"processed_input --{processed_input}")

        # Make prediction
        prediction = model.predict(processed_input)
        print(prediction)

        # Return the prediction result as a JSON response
        return jsonify({'prediction': str(prediction[0])})

    except Exception as e:
        # Handle any error that occurs during prediction
        return jsonify({'error': str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
