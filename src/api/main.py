import mlflow
mlflow.set_tracking_uri("file:///C:/Users/tsega/OneDrive/Documents/credit-scoring/mlruns")




from fastapi import FastAPI, HTTPException
import os
import sys

# Add src folder to path
sys.path.append('..')
from pydantic_models import CustomerData, PredictionResponse
import mlflow.pyfunc

app = FastAPI()

# Load the latest production model from MLflow
model_name = "CreditRisk_RandomForest"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Production")

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    try:
        input_df = [data.features]  # shape: (1, n_features)
        prediction = model.predict(input_df)[0]
        return PredictionResponse(risk_probability=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
