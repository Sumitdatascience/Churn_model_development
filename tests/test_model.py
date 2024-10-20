# # load test + signature test + performance test

# import unittest
# import mlflow
# import os
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import pickle

# class TestModelLoading(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         # Set up DagsHub credentials for MLflow tracking
#         dagshub_token = os.getenv("DAGSHUB_PAT")
#         if not dagshub_token:
#             raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

#         os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
#         os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

#         dagshub_url = "https://dagshub.com"
#         repo_owner = "Sumitdatascience"
#         repo_name = "Churn_model_development"

#         # Set up MLflow tracking URI
#         mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

#         # Load the new model from MLflow model registry
#         cls.new_model_name = "Churn_Prediction_Model"
#         cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
#         cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
#         cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

#         @staticmethod
#         def get_latest_model_version(model_name, stage="Staging"):
#             client = mlflow.MlflowClient()
#             latest_version = client.get_latest_versions(model_name, stages=[stage])
#             return latest_version[0].version if latest_version else None

#         def test_model_loaded_properly(self):
#             self.assertIsNotNone(self.new_model)    

# if __name__ == '__main__':
#     unittest.main()            



import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import numpy as np

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Sumitdatascience"
        repo_name = "Churn_model_development"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "Churn_Prediction_Model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load holdout test data
        cls.holdout_x_test_data = pd.read_csv('data/processed/x_test.csv')
        cls.holdout_y_test_data = pd.read_csv('data/processed/y_test.csv')


    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)   



    def test_model_signature(self):
        # Define the data and columns
        data = [12, 'desktop', 1, 2, 'credit card', 'male', 2, 3, 'fashion', 2, 0, 3.0, 0, 13.0, 10, 20.0, 160.0]
        columns = ['Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome', 'PreferredPaymentMode', 'Gender',
                   'HourSpendOnApp', 'NumberOfDeviceRegistered', 'PreferredOrderCat', 'SatisfactionScore',
                   'MaritalStatus', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromLastYear', 'OrderCount',
                   'DaySinceLastOrder', 'CashbackAmount']

        # Create a pandas DataFrame using the data and columns
        input_df = pd.DataFrame([data], columns=columns)

        # Predict using the new churn prediction model
        prediction = self.new_model.predict(input_df)

        # Verify that the input shape matches the number of columns expected by the model
        self.assertEqual(input_df.shape[1], len(columns))

        # Verify the output shape (assuming binary classification with a single output per row)
        self.assertEqual(len(prediction), input_df.shape[0])

        # Assuming the prediction is a 1D array of binary values (0 or 1 for churn prediction)
        self.assertEqual(len(prediction.shape), 1)

        # Check that the predictions are binary (either 0 or 1)
        unique_values = np.unique(prediction)
        self.assertTrue(set(unique_values).issubset({0, 1}))     

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_x_test_data
        y_holdout = self.holdout_y_test_data

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.90
        expected_precision = 0.90
        expected_recall = 0.90
        expected_f1 = 0.85

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')



if __name__ == '__main__':
    unittest.main()
