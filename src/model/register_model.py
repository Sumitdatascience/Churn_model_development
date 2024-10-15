import json
import mlflow
import os
import dagshub

from mlflow.tracking import MlflowClient


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



# Load experiment information (run ID and model path) from JSON file
def load_model_info(file_path):
    with open(file_path, 'r') as file:
        model_info = json.load(file)
    print('model_info loaded successfully')    
    return model_info
    


# Register the model in MLflow and transition it to the 'staging' stage
def register_model(model_name, model_info):
    print("entre into register_model")
    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_uri']}"
    print("model_uri loaded")   
    # Register the model
    model_version = mlflow.register_model(model_uri, model_name)
        
    # Transition the model to "Staging" stage
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
    

        



def main():
    try:
    
        
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        # Name of the  model in the MLflow Model Registry
        model_name = "Churn_Prediction_Model"  

        # Register the model and transition it to the staging stage
        register_model(model_name, model_info)
        

    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
