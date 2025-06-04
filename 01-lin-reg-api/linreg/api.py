# linreg/api.py

from fastapi import FastAPI
from pydantic import BaseModel
from linreg.model import DummyModel

# Instantiate FastAPI app and model
app = FastAPI()
model = DummyModel()

# Define input schema
class Features(BaseModel):
    square_footage: float
    bedrooms: int

# Define endpoint
@app.post("/predict")
def predict(features: Features):
    price = model.predict(features.square_footage, features.bedrooms)
    return {"predicted_price": price}
