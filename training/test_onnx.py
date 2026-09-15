from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "model.pkl"
ONNX_PATH = BASE_DIR / "model" / "model.onnx"


TEXT = "Patient with cardiovascular disease and severe chest pain"


def main():
    print("=" * 70)
    print("TESTE DO MODELO ONNX RUNTIME")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Modelo original
    # ---------------------------------------------------------
    print("\n[1/3] Testando modelo original...")

    model = joblib.load(MODEL_PATH)

    prediction_original = int(model.predict([TEXT])[0])

    print(f"Previsão Scikit-Learn: {prediction_original}")

    # ---------------------------------------------------------
    # 2. Modelo ONNX Runtime
    # ---------------------------------------------------------
    print("\n[2/3] Testando ONNX Runtime...")

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    print(f"Input ONNX: {input_name}")

    input_data = np.array([[TEXT]], dtype=object)

    outputs = session.run(
        None,
        {input_name: input_data},
    )

    print(f"Outputs: {outputs}")

    # O primeiro output normalmente contém as classes
    prediction_onnx = int(outputs[0][0])

    print(f"Previsão ONNX Runtime: {prediction_onnx}")

    # ---------------------------------------------------------
    # 3. Comparação
    # ---------------------------------------------------------
    print("\n[3/3] Comparando resultados...")

    if prediction_original == prediction_onnx:
        print("✅ As previsões são iguais!")
    else:
        print("❌ As previsões são diferentes!")

    print("\n" + "=" * 70)
    print("TESTE CONCLUÍDO")
    print("=" * 70)


if __name__ == "__main__":
    main()