import json
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.data_preprocessing import load_data, preprocess_data, save_artifacts

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
}


def split_data(X, y):
    """Divide os dados em treino e teste preservando a proporcao das classes."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def _get_probability_scores(model, X_test):
    """Obtém scores de probabilidade ou decisão para calcular ROC-AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    raise ValueError("O modelo nao possui predict_proba nem decision_function.")


def evaluate_model(model, X_test, y_test, model_name):
    """Avalia um modelo em um conjunto de teste e retorna as metricas principais."""
    y_pred = model.predict(X_test)
    y_score = _get_probability_scores(model, X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="binary"),
        "roc_auc": roc_auc_score(y_test, y_score),
    }

    print(f"\nResumo das metricas - {model_name}")
    print(classification_report(y_test, y_pred))
    print(
        f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}"
    )

    return metrics


def train_all_models(X_train, X_test, y_train, y_test, feature_names=None) -> tuple:
    """Treina todos os modelos, registra os experimentos no MLflow e retorna o melhor."""
    mlflow.set_experiment("telco_churn")

    best_model_name = None
    best_model = None
    best_roc_auc = -np.inf
    best_metrics = None
    trained_models = {}

    if feature_names is not None:
        X_train_log = pd.DataFrame(X_train, columns=feature_names)
        X_test_log = pd.DataFrame(X_test, columns=feature_names)
    else:
        X_train_log = X_train
        X_test_log = X_test

    for model_name, model in MODELS.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = _get_probability_scores(model, X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="binary")
            roc_auc = roc_auc_score(y_test, y_score)

            metrics = {
                "accuracy": accuracy,
                "f1": f1,
                "roc_auc": roc_auc,
            }

            trained_models[model_name] = model

            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            signature = infer_signature(X_train_log, model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_test_log[:3] if feature_names is not None else X_test[:3],
            )

            print(f"\nModelo: {model_name}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1: {f1:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(classification_report(y_test, y_pred))

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model_name = model_name
                best_model = model
                best_metrics = metrics

    return best_model_name, best_model, best_roc_auc, best_metrics, trained_models


def plot_roc_curves(models, X_test, y_test):
    """Plota curvas ROC para um conjunto de modelos treinados."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        try:
            y_score = _get_probability_scores(model, X_test)
        except ValueError:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.title("ROC Curve Comparison of Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def main():
    """Executa o fluxo completo de treino, rastreamento e salvamento do melhor modelo."""
    DATA_PATH = "data/raw/telco_churn.csv"
    if not os.path.exists(DATA_PATH):
        fallback = "data/churn-processed.csv"
        if os.path.exists(fallback):
            DATA_PATH = fallback
            print(f"[INFO] Usando fallback: {DATA_PATH}")
        else:
            raise FileNotFoundError(
                "Dataset nao encontrado. Esperado em data/raw/telco_churn.csv"
            )
    print(f"[INFO] Carregando dataset: {DATA_PATH}")

    df = load_data(DATA_PATH)
    X, y, scaler, columns = preprocess_data(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(X, y)

    best_model_name, best_model, best_roc_auc, best_metrics, trained_models = train_all_models(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names=columns,
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(best_model_name, "models/best_model_name.pkl")
    save_artifacts(scaler, columns)

    metadata = {
        "best_model_name": best_model_name,
        "selection_metric": "roc_auc",
        "best_roc_auc": best_roc_auc,
        "metrics": best_metrics,
        "feature_count": len(columns),
        "trained_models": list(trained_models.keys()),
    }
    with open("models/model_metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    with mlflow.start_run(run_name="best_model_artifacts"):
        mlflow.log_params(
            {
                "best_model_name": best_model_name,
                "selection_metric": "roc_auc",
                "feature_count": len(columns),
            }
        )
        mlflow.log_metrics(best_metrics)
        mlflow.log_artifacts("models", artifact_path="deployment_artifacts")
        mlflow.sklearn.log_model(best_model, artifact_path="best_model")

    print(f"Melhor modelo: {best_model_name} com ROC-AUC: {best_roc_auc:.4f}")
    print("Artefatos salvos em models/")


if __name__ == "__main__":
    main()
