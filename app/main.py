from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from .features import extract_features
from .schemas import PredictRequest, PredictResponse

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

MODEL_PATH = "model.pkl"


model_bundle: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_bundle.update(joblib.load(MODEL_PATH))
        print(f"Loaded model '{model_bundle['model_name']}' from {MODEL_PATH}")
    except FileNotFoundError:
        print(
            f"WARNING: {MODEL_PATH} not found. Train one first with:\n"
            f"    python -m app.train --data data.csv --out {MODEL_PATH}"
        )
    yield


app = FastAPI(
    title="Phishing URL Detector API",
    description="Extracts lexical/structural features from a URL and classifies it as phishing or legitimate.",
    version="2.0.0",
    lifespan=lifespan,
)

# this will be needed if we point to another client
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(model_bundle)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not model_bundle:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train it and place model.pkl next to the app before predicting.",
        )

    lookup_tables = model_bundle.get("lookup_tables", {})
    features = extract_features(request.url, lookup_tables)
    row = pd.DataFrame([features])[model_bundle["feature_names"]]

    model = model_bundle["model"]
    scaler = model_bundle.get("scaler")
    row_for_model = scaler.transform(row) if scaler is not None else row

    prediction = int(model.predict(row_for_model)[0])
    probability = float(model.predict_proba(row_for_model)[0][1]) 

    return PredictResponse(
        url=request.url,
        status="phishing" if prediction == 1 else "legitimate",
        probability=round(probability, 4),
        model_used=model_bundle["model_name"],
        features=features,
    )


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


# mounted after routes because if we will add file in static then it will not shadow the routes
app.mount("/static", StaticFiles(directory="static"), name="static")
