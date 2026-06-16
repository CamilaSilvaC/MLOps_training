import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa strings e converte TotalCharges para numerico."""
    data = df.copy()

    object_columns = data.select_dtypes(include=["object"]).columns
    for column in object_columns:
        data[column] = data[column].apply(lambda value: value.strip() if isinstance(value, str) else value)

    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        if "tenure" in data.columns and "MonthlyCharges" in data.columns:
            fallback_total = data["tenure"] * data["MonthlyCharges"]
            data["TotalCharges"] = data["TotalCharges"].fillna(fallback_total)
        data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

    return data


def load_data(filepath: str) -> pd.DataFrame:
    """Carrega um CSV, limpa strings e trata a coluna TotalCharges."""
    df = pd.read_csv(filepath)
    return clean_telco_data(df)


def preprocess_data(df: pd.DataFrame, fit: bool = True, scaler=None, columns: list = None):
    """Prepara dados de treino ou inferencia usando o mesmo encoding e escala."""
    data = clean_telco_data(df)

    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    y = None
    if "Churn" in data.columns:
        y = data["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})
        y = pd.Series(y, name="Churn")
        data = data.drop(columns=["Churn"])

    X = data
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=False)

    if fit:
        scaler = StandardScaler()
        columns = X.columns.tolist()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("O parametro scaler deve ser informado quando fit=False.")
        if columns is None:
            raise ValueError("O parametro columns deve ser informado quando fit=False.")

        X = X.reindex(columns=columns, fill_value=0)
        X_scaled = scaler.transform(X)

    return np.asarray(X_scaled), y, scaler, columns


def preprocess_input(df: pd.DataFrame, scaler, columns: list) -> np.ndarray:
    """Transforma entradas sem Churn para inferencia alinhadas ao treino."""
    X, _, _, _ = preprocess_data(df, fit=False, scaler=scaler, columns=columns)
    return X


def save_artifacts(scaler, columns: list, output_dir: str = "models/"):
    """Salva o scaler e a lista de colunas processadas em disco."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(columns, os.path.join(output_dir, "columns.pkl"))


def load_artifacts(models_dir: str = "models/"):
    """Carrega o scaler e a lista de colunas processadas a partir dos artefatos."""
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    columns = joblib.load(os.path.join(models_dir, "columns.pkl"))
    return scaler, columns


def log_processed_data(df, output_path="data/processed/churn-processed.csv"):
    """Salva o DataFrame processado em CSV para rastreabilidade."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data logged to {output_path}")
