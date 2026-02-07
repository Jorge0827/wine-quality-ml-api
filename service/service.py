from pathlib import Path
from typing import Optional #Tipado fuerte, estructura un dato
import joblib  #Guarda el modelo en un capa para llamarlo
import pandas as pd
from sklearn.pipeline import Pipeline #Establece el flujo de trabajo

class ModelService:
    def __init__(self, model_path: str = "model/pipeline.joblib") -> None:
        self.model_path = Path(model_path)
        self.pipeline: Optional[Pipeline] = None
    
    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError("No se encontró el modelo entrenado. Ejecutar train.py primero")
        
        self.pipeline = joblib.load(self.model_path)
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        
        if self.pipeline is None:
            raise RuntimeError("El modelo no está cargado.")
        
        return self.pipeline.predict(features)