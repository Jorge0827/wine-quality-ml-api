from contextlib import asynccontextmanager # Web
from typing import Dict # Colocar tipado
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field # Realizar validaciones
from service.service import ModelService


FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

FIELD_NAME_MAP: Dict[str, str] = {
    "fixed_acidity": "fixed acidity",
    "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid",
    "residual_sugar": "residual sugar",
    "chlorides": "chlorides",
    "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide": "total sulfur dioxide",
    "density": "density",
    "pH": "pH",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
}

class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., gt= 0) #gt quiere decir que el valor debe ser mayor a 0, ge es mayor o igual a 0
    volatile_acidity: float = Field(..., gt= 0)
    citric_acid: float = Field(..., ge= 0) # ge quiere decir que el valor debe ser mayor o igual a 0
    residual_sugar: float = Field(..., ge= 0) 
    chlorides: float = Field(..., ge= 0)
    free_sulfur_dioxide: float = Field(..., ge= 0)
    total_sulfur_dioxide: float = Field(..., ge= 0)
    density: float = Field(..., gt= 0)
    pH: float = Field(..., gt= 0)
    sulphates: float = Field(..., ge= 0)
    alcohol: float = Field(..., gt= 0) # los ... indican que el campo es obligatorio, si no se coloca un valor se lanzará un error

service = ModelService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    try:
        service.load()
    except FileNotFoundError:
        service.pipeline = None
    yield
    
app = FastAPI(
    title="Predictor de calidad de vinos",
    version="1.0.0",
    lifespan=lifespan
)
    
@app.post("/predict")
def predict(features: WineFeatures) -> dict:
    if service.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Servicio no disponible"
        )
    
    payload = features.model_dump() #Hagarra la features y las transfomra a un json
    
    row = {
        FIELD_NAME_MAP[key]: value
        for key, value in payload.items()
    }
    
    dataframe = pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS,
    )
    
    try:
        prediction = service.predict(dataframe)[0]
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )
        
    return{
        "predicted_quality": round(float(prediction), 3)
    }
    
