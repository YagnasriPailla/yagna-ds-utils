from fastapi import FastAPI
from yagna_ds_utils import predict_from_joblib

app = FastAPI(title="Solana Price Prediction API")

MODEL_PATH = "models/solana_rf.joblib"           # match your filename
CSV_PATH   = "models/last_row_features_solana.csv"

@app.get("/")
def root():
    return {
        "project": "Solana Price Prediction",
        "endpoints": ["/", "/health", "/predict/solana"],
        "package_used": "yagna-ds-utils==0.1.0 (TestPyPI)"
    }

@app.get("/health")
def health():
    return {"status": "API is running 👌"}

@app.get("/predict/solana")
def predict_solana():
    try:
        y = predict_from_joblib(MODEL_PATH, CSV_PATH)
        return {"predicted_high_next_day": round(y, 2)}
    except Exception as e:
        return {"error": str(e)}
