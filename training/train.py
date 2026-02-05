from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pipeline.pipeline import load_data, split_features_target, built_pipeline

#Constantes Globales
MODEL_PATH = Path("model/pipeline.joblib")

# Función de evaluacion

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    
    mse = mean_squared_error(y_true, y_pred)
    
    return{
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse2": np.sqrt(mse),
        "r2": r2_score(y_true, y_pred)
    }

def main() -> None:
    df = load_data()
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )
    
    pipeline = built_pipeline(
        numeric_features=list(x.columns)
    )
    
    pipeline.fit(x_train, y_train)
    
    y_pred = pipeline.predict(x_test)
    
    metrics = evaluate_regression(y_test, y_pred)
    
    print("Metricas de evaluación")
    print(f"  -MAE: {metrics['mae']:.4f}")
    print(f"  -MSE: {metrics['mse']:.4f}")
    print(f"  -RMSE: {metrics['rmse2']:.4f}")
    print(f"  -R2: {metrics['r2']:.4f}")
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(pipeline, MODEL_PATH)
    
    print(f"Modelo guardado en: {MODEL_PATH}")
    
if __name__ == "__main__":
        main()
