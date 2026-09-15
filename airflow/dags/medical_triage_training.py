from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator


BASE_DIR = Path("/opt/airflow/project")

TRAIN_PATH = BASE_DIR / "data" / "medical_tc_train.csv"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"


def read_dataset():
    """
    Task 1:
    Carrega e valida o dataset de treinamento.
    """
    df = pd.read_csv(TRAIN_PATH)

    print(f"Dataset carregado: {df.shape}")
    print(f"Colunas: {list(df.columns)}")

    if df.empty:
        raise ValueError("Dataset de treinamento está vazio.")

    required_columns = {"condition_label", "medical_abstract"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset deve possuir as colunas: {required_columns}"
        )

    print("Dataset validado com sucesso.")


def train_model():
    """
    Task 2:
    Treina o modelo e salva o artefato.
    """
    import joblib

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    df = pd.read_csv(TRAIN_PATH)

    X = df["medical_abstract"]
    y = df["condition_label"]

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                max_features=10000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    print("Iniciando treinamento...")

    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"Modelo salvo em: {MODEL_PATH}")


with DAG(
    dag_id="medical_triage_training",
    description="Pipeline de treinamento do modelo NLP de classificação médica",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["medical-triage", "ml", "training"],
) as dag:

    read_dataset_task = PythonOperator(
        task_id="read_dataset",
        python_callable=read_dataset,
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    read_dataset_task >> train_model_task