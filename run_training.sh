#!/bin/bash
set -e

echo "========================================"
echo " Telco Churn MLOps - Pipeline de Treino"
echo "========================================"

# Ativa o ambiente virtual se existir
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "[OK] Ambiente virtual ativado"
fi

# Verifica se o dataset existe
DATA_PATH="data/raw/telco_churn.csv"
if [ ! -f "$DATA_PATH" ]; then
    echo "[ERRO] Dataset não encontrado em $DATA_PATH"
    echo "Procurando alternativas..."
    FOUND=$(find data/ -name "*.csv" 2>/dev/null | head -1)
    if [ -z "$FOUND" ]; then
        echo "[ERRO] Nenhum CSV encontrado na pasta data/"
        exit 1
    fi
    echo "[INFO] Usando: $FOUND"
fi

# Cria pasta de modelos se não existir
mkdir -p models/

# Executa o treino
echo ""
echo "[INICIANDO] Treinamento dos modelos..."
uv run python -m src.model

# Valida artefatos
echo ""
echo "[VALIDANDO] Artefatos gerados..."
for artifact in "models/best_model.pkl" "models/scaler.pkl" "models/columns.pkl"; do
    if [ -f "$artifact" ]; then
        echo "[OK] $artifact"
    else
        echo "[ERRO] $artifact não foi gerado!"
        exit 1
    fi
done

echo ""
echo "[SUCESSO] Pipeline de treino concluído!"
echo "Execute 'mlflow ui' para visualizar os experimentos."
