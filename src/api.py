import json
import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_preprocessing import preprocess_input as transform_input


class CustomerInput(BaseModel):
    """Representa os dados de entrada de um cliente para predicao de churn."""

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Optional[float] = None


class PredictionResponse(BaseModel):
    """Representa a resposta da API com a predicao de churn."""

    churn: bool
    probability: float
    model_used: str = "best_model"


def preprocess_customer_input(data: CustomerInput, scaler, columns: list):
    """Prepara uma entrada individual com o mesmo pipeline do treino."""
    frame = pd.DataFrame([data.model_dump()])
    return transform_input(frame, scaler=scaler, columns=columns)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelo, scaler e colunas no inicio da aplicacao."""
    models_dir = "models"
    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    columns_path = os.path.join(models_dir, "columns.pkl")
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    app.state.model = joblib.load(model_path) if os.path.exists(model_path) else None
    app.state.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    app.state.columns = joblib.load(columns_path) if os.path.exists(columns_path) else None
    app.state.model_name = "best_model"

    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        app.state.model_name = metadata.get("best_model_name", "best_model")

    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="API para predicao de churn de clientes de telecomunicacoes",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    """Retorna uma mensagem simples de status da API."""
    return {"message": "Telco Churn API esta online", "status": "ok"}


@app.get("/health")
def health_check():
    """Verifica se os artefatos principais foram carregados."""
    return {
        "status": "healthy",
        "model_loaded": getattr(app.state, "model", None) is not None,
        "scaler_loaded": getattr(app.state, "scaler", None) is not None,
        "columns_loaded": getattr(app.state, "columns", None) is not None,
        "model_used": getattr(app.state, "model_name", "best_model"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    """Gera a predicao de churn para um cliente individual."""
    model = getattr(app.state, "model", None)
    scaler = getattr(app.state, "scaler", None)
    columns = getattr(app.state, "columns", None)

    if model is None or scaler is None or columns is None:
        raise HTTPException(
            status_code=500,
            detail="Artefatos do modelo nao carregados. Verifique a pasta models/.",
        )

    try:
        processed_input = preprocess_customer_input(customer, scaler, columns)
        probability = float(model.predict_proba(processed_input)[0][1])
        churn = bool(model.predict(processed_input)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao realizar a predicao: {exc}") from exc

    return PredictionResponse(
        churn=churn,
        probability=probability,
        model_used=getattr(app.state, "model_name", "best_model"),
    )
