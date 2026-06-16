import json
import os

import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(
    page_title="Telco Churn Predictor",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")


def check_api_health(api_url: str) -> dict:
    """Consulta o endpoint de health da API."""
    response = requests.get(f"{api_url}/health", timeout=10)
    response.raise_for_status()
    return response.json()


def build_payload(form_data: dict) -> dict:
    """Monta o JSON esperado pelo endpoint POST /predict."""
    return {
        "gender": form_data["gender"],
        "SeniorCitizen": form_data["SeniorCitizen"],
        "Partner": form_data["Partner"],
        "Dependents": form_data["Dependents"],
        "tenure": form_data["tenure"],
        "PhoneService": form_data["PhoneService"],
        "MultipleLines": form_data["MultipleLines"],
        "InternetService": form_data["InternetService"],
        "OnlineSecurity": form_data["OnlineSecurity"],
        "OnlineBackup": form_data["OnlineBackup"],
        "DeviceProtection": form_data["DeviceProtection"],
        "TechSupport": form_data["TechSupport"],
        "StreamingTV": form_data["StreamingTV"],
        "StreamingMovies": form_data["StreamingMovies"],
        "Contract": form_data["Contract"],
        "PaperlessBilling": form_data["PaperlessBilling"],
        "PaymentMethod": form_data["PaymentMethod"],
        "MonthlyCharges": form_data["MonthlyCharges"],
        "TotalCharges": form_data["TotalCharges"],
    }


def create_gauge_chart(probability: float) -> go.Figure:
    """Cria um grafico de gauge para a probabilidade de churn."""
    bar_color = "#c0392b" if probability >= 0.5 else "#1f8a70"
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Probabilidade de churn"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 50], "color": "#e9f5f1"},
                    {"range": [50, 100], "color": "#f9e7e4"},
                ],
            },
        )
    )
    figure.update_layout(height=320, margin={"t": 40, "b": 20, "l": 20, "r": 20})
    return figure


st.sidebar.title("Configuracao")
api_url = st.sidebar.text_input("API URL", value=API_URL)

if st.sidebar.button("Verificar API"):
    try:
        health_data = check_api_health(api_url)
        st.sidebar.success(
            f"Status: {health_data.get('status', 'unknown')} | "
            f"Modelo: {health_data.get('model_used', 'best_model')}"
        )
    except requests.exceptions.RequestException:
        st.sidebar.warning("Nao foi possivel acessar a API FastAPI.")
    except Exception as exc:
        st.sidebar.error(f"Erro ao verificar a API: {exc}")


st.title("Telco Customer Churn Predictor")
st.caption("Preencha os dados do cliente para estimar a probabilidade de churn.")

with st.form("churn_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("gender", ["Female", "Male"])
        senior_citizen = st.selectbox("SeniorCitizen", [0, 1])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("tenure", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("PhoneService", ["Yes", "No"])
        multiple_lines = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("InternetService", ["Fiber optic", "DSL", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        online_security = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("OnlineBackup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("TechSupport", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"])
        paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"])
        payment_method = st.selectbox(
            "PaymentMethod",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.slider(
            "MonthlyCharges",
            min_value=0.0,
            max_value=150.0,
            value=65.0,
            step=0.5,
        )
        total_charges = st.number_input(
            "TotalCharges",
            min_value=0.0,
            max_value=10000.0,
            value=None,
            placeholder="Opcional",
        )

    submitted = st.form_submit_button("Prever churn")


if submitted:
    payload = build_payload(
        {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
    )

    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()

        churn = bool(result.get("churn", False))
        probability = float(result.get("probability", 0.0))
        model_used = result.get("model_used", "best_model")

        metric_col, model_col = st.columns([1, 1])
        metric_col.metric("Probabilidade", f"{probability:.2%}")
        model_col.metric("Modelo", model_used)

        if churn:
            st.error("Predicao: cliente com maior risco de churn.")
        else:
            st.success("Predicao: cliente com menor risco de churn.")

        st.plotly_chart(create_gauge_chart(probability), use_container_width=True)

        with st.expander("JSON enviado"):
            st.code(json.dumps(payload, indent=2), language="json")

        with st.expander("Resposta da API"):
            st.json(result)

    except requests.exceptions.ConnectionError:
        st.warning("Nao foi possivel conectar a API em http://localhost:8000.")
    except requests.exceptions.Timeout:
        st.warning("A API demorou para responder.")
    except requests.exceptions.RequestException as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = f" Detalhe: {exc.response.json().get('detail', '')}"
            except Exception:
                detail = f" Detalhe: {exc.response.text}"
        st.error(f"Erro ao chamar a API: {exc}.{detail}")
    except Exception as exc:
        st.error(f"Erro inesperado ao gerar a predicao: {exc}")
