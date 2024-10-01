import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/Sumitdatascience/Churn_model_development.mlflow')
dagshub.init(repo_owner='Sumitdatascience', repo_name='Churn_model_development', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)