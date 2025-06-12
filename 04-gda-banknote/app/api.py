from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from .gda import GDAClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

app = FastAPI(title="GDA Banknote Classifier API")

model = GDAClassifier()
classes = []


class PredictRequest(BaseModel):
    samples: List[List[float]]


class PredictResponse(BaseModel):
    predictions: List[int]


@app.on_event("startup")
def train_model():
    global model, classes

    print("🔧 Loading and training GDA on Banknote dataset...")

    X, y = fetch_openml("banknote-authentication", version=1, as_frame=False, return_X_y=True)
    y = y.astype(int)
    classes = np.unique(y).tolist()

    # Normalize
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    print(f"✅ Model trained. Classes: {classes}")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        X_new = np.array(request.samples)
        preds = model.predict(X_new)
        return PredictResponse(predictions=preds.tolist())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "GDA Banknote Classifier API",
        "classes": classes,
        "input_shape": [4]
    }
