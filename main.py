import argparse
import sys
import os


def run_training():
    """Executa o pipeline de treinamento com MLflow."""
    from src.model import main as train_main

    print("Iniciando pipeline de treinamento...")
    train_main()


def run_api():
    """Inicia a API FastAPI."""
    import uvicorn

    print("Iniciando API FastAPI em http://localhost:8000")
    print("Documentação: http://localhost:8000/docs")
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)


def run_app():
    """Inicia a interface Streamlit."""
    import subprocess

    print("Iniciando interface Streamlit em http://localhost:8501")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "src/app.py",
        "--server.port=8501"
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Telco Churn MLOps Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  uv run python main.py train    # Treina modelos e registra no MLflow
  uv run python main.py api      # Inicia a API de inferência
  uv run python main.py app      # Inicia a interface Streamlit
        """
    )
    parser.add_argument(
        "command",
        choices=["train", "api", "app"],
        help="Comando a executar"
    )
    args = parser.parse_args()

    commands = {
        "train": run_training,
        "api": run_api,
        "app": run_app,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
