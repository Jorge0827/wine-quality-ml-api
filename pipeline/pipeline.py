from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer #Manejar valores faltante
from sklearn.linear_model import ElasticNet # Para regresión lineal
from sklearn.pipeline import Pipeline #flujo de trabajo
from sklearn.preprocessing import StandardScaler 

# Declaración de constantes globales

# Data URL

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

#Función de carga y caching de datos
def load_data(
    data_url: str = DATA_URL,
    cache_path: Path = Path("data/winequality-red.csv")
    ) -> pd.DataFrame:
    if cache_path.exists(): # Si el archivo ya existe no se descarga
        return pd.read_csv(cache_path, sep=";")
    
    df = pd.read_csv(data_url, sep=";")
    
    cache_path.parent.mkdir(parents=True, exist_ok=True) #Si la carpeta data no existiera la crea por defecto
    
    df.to_csv(cache_path, index=False, sep=";")
    
    return df #Retorna finalmente el dataframe


# Función de separacion de caracteristicas y el target osea la "y"
def split_features_target(
    df: pd.DataFrame,
    target: str = "quality"
    ) -> Tuple[pd.DataFrame, pd.Series]:
    
    x = df.drop(columns=[target])
    
    y = df[target]
    
    return x, y


# Funcion de construccion del pipeline
def built_pipeline(numeric_features: list[str]) -> Pipeline:
    
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer", SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            (
            "num",
            numeric_transformer,
            numeric_features
            )
        ],
        remainder="drop" #Columnas qeu no tienen importancia, eliminarlas
    )
    
    model = ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        random_state=42
    )
    
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ]
    )
    
    return pipeline
    
