from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from .model import SoftmaxRegression

app = FastAPI()

# ----- Pydantic Schemas -----
class TrainRequest(BaseModel):
    features: List[List[float]]   
    labels: List[int]            
    lr: float
    epochs: int
    num_classes: int

class Prediction(BaseModel):
    predictions: List[int]
    probability_vectors: List[List[float]]

# ----- API Endpoint -----
@app.post("/train-and-predict", response_model=Prediction)
def train_and_predict(data: TrainRequest):
    input_dim = len(data.features[0])
    model = SoftmaxRegression(input_dim=input_dim, num_classes=data.num_classes)

    # Convert input data to tensors
    X = [torch.tensor(vec, dtype=torch.float32) for vec in data.features]
    Y = data.labels
 
    model.train(X, Y, lr=data.lr, epochs=data.epochs)
 
    predictions = []
    prob_vectors = []
    for x in X:
        prob = model.predict_proba(x).tolist()
        pred = int(torch.argmax(torch.tensor(prob)))
        predictions.append(pred)
        prob_vectors.append(prob)

    return Prediction(predictions=predictions, probability_vectors=prob_vectors)
