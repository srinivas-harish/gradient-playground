# linreg/api.py

from fastapi import FastAPI
from pydantic import BaseModel
from logreg.model import LogisticRegressionMiniBatchGD
import torch
from typing import List

app = FastAPI()
model = LogisticRegressionMiniBatchGD()

class Features(BaseModel):
    feature1: float
    feature2: float

class TrainingExample(BaseModel):
    feature1: float
    feature2: float
    label: float

@app.post("/train")
def train_model(data: List[TrainingExample]):
    X = torch.tensor([[d.feature1, d.feature2] for d in data])
    y = torch.tensor([d.label for d in data])
    model.train(X, y)
    return {"message": "Model trained successfully."}

@app.post("/predict")
def predict(features: Features):
    X_input = torch.tensor([[features.feature1, features.feature2]])
    prob = model.predict(X_input).item()
    return {
        "probability": prob,
        "predicted_class": int(prob > 0.5)
    }

@app.get("/")
def root():
    return {"message": "Logistic Regression API is running"}
