from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "medical_tc_train.csv"
TEST_PATH = BASE_DIR / "data" / "medical_tc_test.csv"
LABELS_PATH = BASE_DIR / "data" / "medical_tc_labels.csv"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("MEDICAL TRIAGE NLP - MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------------

    print("\n[1/6] Carregando datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    labels_df = pd.read_csv(LABELS_PATH)

    print(f"Dataset de treino: {train_df.shape}")
    print(f"Dataset de teste:  {test_df.shape}")

    # --------------------------------------------------------
    # 2. Load labels
    # --------------------------------------------------------

    print("\n[2/6] Classes:")

    label_map = dict(
        zip(
            labels_df["condition_label"],
            labels_df["condition_name"]
        )
    )

    for label, name in label_map.items():
        print(f"  {label} -> {name}")

    # --------------------------------------------------------
    # 3. Prepare data
    # --------------------------------------------------------

    print("\n[3/6] Preparando dados...")

    X_train = train_df["medical_abstract"]
    y_train = train_df["condition_label"]

    X_test = test_df["medical_abstract"]
    y_test = test_df["condition_label"]

    print("\nDistribuição das classes no treino:")

    print(
        y_train
        .value_counts()
        .sort_index()
        .rename(index=label_map)
    )

    # --------------------------------------------------------
    # 4. Create model
    # --------------------------------------------------------

    print("\n[4/6] Criando modelo NLP...")

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

    print("\nModelo:")
    print("  TF-IDF + Logistic Regression")

    # --------------------------------------------------------
    # 5. Train
    # --------------------------------------------------------

    print("\n[5/6] Treinando modelo...")

    model.fit(X_train, y_train)

    print("Treinamento concluído.")

    # --------------------------------------------------------
    # 6. Evaluate
    # --------------------------------------------------------

    print("\n[6/6] Avaliando modelo...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                label_map[i]
                for i in sorted(label_map.keys())
            ],
            zero_division=0,
        )
    )

    print("Matriz de Confusão:")

    cm = confusion_matrix(y_test, predictions)

    print(cm)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print("\n" + "=" * 70)
    print("MODELO SALVO")
    print("=" * 70)

    print(f"\nArquivo:")
    print(MODEL_PATH)

    print("\nTreinamento finalizado com sucesso!")


if __name__ == "__main__":
    main()