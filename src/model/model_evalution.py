

# import pickle
# import pandas as pd
# import json
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# import dagshub
# import mlflow
# import mlflow.sklearn
# from mlflow.models import infer_signature

# # Set the tracking URI to DAGsHub
# mlflow.set_tracking_uri('https://dagshub.com/Sumitdatascience/Churn_model_development.mlflow')
# dagshub.init(repo_owner='Sumitdatascience', repo_name='Churn_model_development', mlflow=True)


# # Load the trained model
# def load_model(model_path: str):
#     with open(model_path, 'rb') as file:
#         model = pickle.load(file)
#     print(f"Model loaded from {model_path}")
#     return model


# # Load test data
# def load_test_data(x_test_path: str, y_test_path: str):
#     x_test = pd.read_csv(
#         x_test_path,
#         usecols=['Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome', 'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 'PreferredOrderCat', 'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromLastYear', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']
#     )
#     y_test = pd.read_csv(y_test_path)
#     print("Test data loaded successfully")
#     return x_test, y_test


# # Make predictions using the model
# def predict(model, x_test):
#     y_pred = model.predict(x_test)
#     print("Predictions generated successfully")
#     return y_pred


# # Calculate and save metrics
# def calculate_metrics(y_test, y_pred, output_path: str):
#     acc_score = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='binary')
#     recall = recall_score(y_test, y_pred, average='binary')
#     f1 = f1_score(y_test, y_pred, average='binary')

#     metrics_dict = {
#         'accuracy': acc_score,
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1
#     }

#     # Save metrics to a JSON file
#     with open(output_path, 'w') as file:
#         json.dump(metrics_dict, file, indent=4)
    
#     print(f"Metrics saved to {output_path}")
#     return metrics_dict


# # Log metrics to MLflow
# def log_metrics_to_mlflow(metrics_dict):
#     mlflow.log_metric('accuracy', metrics_dict['accuracy'])
#     mlflow.log_metric('precision', metrics_dict['precision'])
#     mlflow.log_metric('recall', metrics_dict['recall'])
#     mlflow.log_metric('f1_score', metrics_dict['f1_score'])


# # Log artifacts to MLflow
# def log_artifacts_to_mlflow(metrics_file, model_file):
#     mlflow.log_artifact(metrics_file)  # Log the metrics file
#     mlflow.log_artifact(model_file)    # Log the model pickle file


# # Main function to load model, test data, and calculate metrics
# def main():
#     mlflow.set_experiment("dvc-pipeline")
#     with mlflow.start_run() as run:  # Start an MLflow run
#         # Load model
#         model = load_model('model.pkl')

#         # Load test data
#         x_test, y_test = load_test_data(
#             r'.\data\processed\x_test.csv',
#             r'.\data\processed\y_test.csv'
#         )

#         # Create an input example (using the first row of the test data)
#         input_example = x_test.head(1)

#         # Generate predictions
#         y_pred = predict(model, x_test)

#         # Infer the model signature
#         signature = infer_signature(x_test, y_pred)

#         # Log the model with signature and input example to MLflow
#         mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

#         # Calculate and save metrics
#         metrics_dict = calculate_metrics(y_test, y_pred, "metrics.json")

#         # Log metrics to MLflow
#         log_metrics_to_mlflow(metrics_dict)

#         # Log artifacts to MLflow (metrics and model)
#         log_artifacts_to_mlflow("metrics.json", "model.pkl")

#         print("Metrics and model logged to MLflow with signature and input example")


# if __name__ == "__main__":
#     main()



import os
import pickle
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import dagshub
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Set the tracking URI to DAGsHub
# mlflow.set_tracking_uri('https://dagshub.com/Sumitdatascience/Churn_model_development.mlflow')
# dagshub.init(repo_owner='Sumitdatascience', repo_name='Churn_model_development', mlflow=True)

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


# Load the trained model
def load_model(model_path: str):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {model_path}")
    return model


# Load test data
def load_test_data(x_test_path: str, y_test_path: str):
    x_test = pd.read_csv(
        x_test_path,
        usecols=['Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome', 'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 'PreferredOrderCat', 'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromLastYear', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']
    )
    y_test = pd.read_csv(y_test_path)
    print("Test data loaded successfully")
    return x_test, y_test


# Make predictions using the model
def predict(model, x_test):
    y_pred = model.predict(x_test)
    print("Predictions generated successfully")
    return y_pred


# Calculate and save metrics
def calculate_metrics(y_test, y_pred, output_path: str):
    acc_score = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    metrics_dict = {
        'accuracy': acc_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    # Save metrics to a JSON file
    with open(output_path, 'w') as file:
        json.dump(metrics_dict, file, indent=4)
    
    print(f"Metrics saved to {output_path}")
    return metrics_dict


# Log metrics to MLflow
def log_metrics_to_mlflow(metrics_dict):
    mlflow.log_metric('accuracy', metrics_dict['accuracy'])
    mlflow.log_metric('precision', metrics_dict['precision'])
    mlflow.log_metric('recall', metrics_dict['recall'])
    mlflow.log_metric('f1_score', metrics_dict['f1_score'])


# Log artifacts to MLflow
def log_artifacts_to_mlflow(metrics_file, model_file):
    mlflow.log_artifact(metrics_file)  # Log the metrics file
    mlflow.log_artifact(model_file)    # Log the model pickle file


def save_experiment_info(run_id, model_path, output_path: str):
    experiment_info = {
        'run_id': run_id,
        'model_uri': model_path  # Ensure model_uri is converted to a string
    }

    # Ensure the reports directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the experiment info as JSON
    with open(output_path, 'w') as file:
        json.dump(experiment_info, file, indent=4)
    
    print(f"Experiment info saved to {output_path}")

def save_metrics(metrics, file_path):
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)
        

# Main function to load model, test data, and calculate metrics
def main():
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:  # Start an MLflow run
        # Capture the run ID
        run_id = run.info.run_id

        # Load model
        model = load_model('models/model.pkl')

        # Load test data
        x_test, y_test = load_test_data(
            r'.\data\processed\x_test.csv',
            r'.\data\processed\y_test.csv'
        )

        # Create an input example (using the first row of the test data)
        input_example = x_test.head(1)
        print(input_example)

        # Generate predictions
        y_pred = predict(model, x_test)
        # y_pred1 = predict(model,input_example)
        # print(y_pred1)

        # Infer the model signature
        signature = infer_signature(x_test, y_pred)

        # Log the model with signature and input example to MLflow
        model_uri = mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

        # Calculate and save metrics
        metrics_dict = calculate_metrics(y_test, y_pred, "metrics.json")

        # save the metrics at reports
        save_metrics(metrics_dict, 'reports/metrics.json')

        # Log metrics to MLflow
        log_metrics_to_mlflow(metrics_dict)

        # Log artifacts to MLflow (metrics and model)
        # log_artifacts_to_mlflow("metrics.json", "model.pkl")
        log_artifacts_to_mlflow("metrics.json", "models/model.pkl")


        # Save experiment info (run ID and model path) to the reports folder
        save_experiment_info(run_id, "model", "./reports/experiment_info.json")

        print("Metrics, model, and experiment info logged to MLflow")


if __name__ == "__main__":
    main()

