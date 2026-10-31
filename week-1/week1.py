import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameters for the experiment
params = {
    "solver": "liblinear",
    "max_iter": 100,
    "random_state": 42
}

# Start an MLflow run
# If you have a tracking server running, you can set the tracking URI
# mlflow.set_tracking_uri("http://localhost:5000")
# By default, MLflow logs to an 'mlruns' directory in the current folder

with mlflow.start_run(run_name="Iris Logistic Regression"):
    # Log hyperparameters
    mlflow.log_params(params)

    # Initialize and train the model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log the trained model
    # This saves the model in a format that MLflow understands
    mlflow.sklearn.log_model(model, "model") # "model" is the artifact path

    print(f"MLflow Run completed.")
    print(f"  Run ID: {mlflow.active_run().info.run_id}")
    print(f"  Accuracy: {accuracy}")
    print(f"  Model logged to artifact path: 'model'")

# You can run this script multiple times with different parameters to see multiple runs in the UI
# Example:
# params_v2 = {"solver": "lbfgs", "max_iter": 200, "random_state": 42}
# with mlflow.start_run(run_name="Iris Logistic Regression V2"):
#     mlflow.log_params(params_v2)
#     model_v2 = LogisticRegression(**params_v2)
#     model_v2.fit(X_train, y_train)
#     y_pred_v2 = model_v2.predict(X_test)
#     accuracy_v2 = accuracy_score(y_test, y_pred_v2)
#     mlflow.log_metric("accuracy", accuracy_v2)
#     mlflow.sklearn.log_model(model_v2, "model")
#     print(f"MLflow Run V2 completed.")
#     print(f"  Run ID: {mlflow.active_run().info.run_id}")
#     print(f"  Accuracy: {accuracy_v2}")


import mlflow

# Replace with your actual run ID
run_id = 'your_run_id_here'
model_artifact_path = "model" # The path where the model was logged in the run

# Register the model
# This creates a new model in the registry with version 1
model_uri = f"runs:/{run_id}/{model_artifact_path}"
registered_model_name = "IrisLogisticRegressionModel"

try:
    mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    print(f"Model '{registered_model_name}' version {mv.version} registered successfully.")

    # You can then transition the model stage (e.g., to Staging)
    # This requires a client connection to a registry server
    # For a simple local demo, we'll just show the registration.
    # In a real scenario, you'd use:
    # client = mlflow.tracking.MlflowClient()
    # client.transition_model_version_stage(
    #     name=registered_model_name,
    #     version=mv.version,
    #     stage="Staging"
    # )
    # print(f"Model version {mv.version} transitioned to Staging.")

except Exception as e:
    print(f"Error registering model: {e}")
    print("Ensure MLflow is configured with a backend store and model registry.")