# Sistema de Previsão de Churn de Clientes

Este projeto é uma solução de machine learning de ponta a ponta para prever o churn (rotatividade) de clientes em uma empresa de telecomunicações. Ele contém um pipeline completo que inclui exploração de dados, pré-processamento e treinamento de modelos.

Melhorias essenciais são: rastreamento de experimentos com MLflow, disponibilização do modelo via uma API de inferência FastAPI e uma interface gráfica via Streamlit.

---

## 📁 Estrutura do Projeto

```bash
MLOps-training/
│
├── data/                   # Arquivos de dados brutos e processados
│   ├── processed/
│   └── raw/
├── models/                 # Artefatos de modelos pré-treinados e salvos
├── notebooks/              # Notebooks Jupyter para EDA, treinamento e avaliação
├── outputs/                # Relatórios, métricas de avaliação ou plots gerados pelo mlflow
├── src/                    # Módulos Python principais
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── model.py
│   └── visualization.py
├── main.py                 # Função principal do programa
└── README.md
```

---

## Componentes & Visão Geral do Pipeline

1. **Exploração de Dados (notebooks/)**
   - Entender distribuições de features, correlações e valores ausentes.
   - Visualizações comparando clientes que churnaram e os que não churnaram.

2. **Pré-processamento (src/data_preprocessing.py)**
   - Tratamento de valores ausentes
   - Codificação de variáveis categóricas
   - Escalonamento/normalização de features

3. **Treinamento de Modelos (src/model.py + notebook)**
   - Modelos treinados:
     - Regressão Logística
     - Random Forest
     - XGBoost
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
     - Multi-layer Perceptron (MLP)

4. **Avaliação (src/visualization.py + notebook)**
   - Acurácia, F1 score, ROC AUC
   - Plots: Matriz de Confusão, curvas ROC, etc.

---

## Ferramentas & Bibliotecas Utilizadas

- **Python 3.12**
- **Pandas**, **NumPy**, **scikit-learn** – Processamento de dados e modelagem
- **XGBoost** – Modelo de boosting avançado
- **MLflow** – Rastreamento de experimentos e registro de modelos
- **FastAPI** – API de inferência
- **Streamlit** – Frontend opcional
- **Docker** – Contêinerização para desenvolvimento e deploy
- **matplotlib**, **seaborn** – Visualizações

---

## Como Usar o Projeto

### 1. Configurar o Ambiente

```bash
uv init
uv sync
source .venv/bin/activate  # No Windows: source .venv\Scripts\activate
```

### 2. Executar o Notebook Jupyter

```bash
jupyter notebook notebooks/EDA.ipynb
```

### 3. TODOs:

**Rastreamento de Experimentos com MLflow**
   - Treina múltiplos modelos
   - Registra métricas, modelos e artefatos.
   - Compara desempenho entre diferentes modelos.
   - Salva o modelo com melhor desempenho

**API de Inferência (FastAPI - `src/api.py`)**
   - Aceita entrada em JSON
   - Pré-processa a requisição
   - Carrega o melhor modelo
   - Retorna a predição

**Interface do modelo com Streamlit**
   - Permite interagir com o modelo via interface gráfica

### Trabalhos Futuros

- Adicionar **testes** unitários e de integração
- Automatizar todo o pipeline com **CI/CD**
- Adicionar **monitoramento** e logging em tempo real
- Incorporar **re-treinamento automático** de modelos

## Licença

Este projeto é para **fins educacionais**.
