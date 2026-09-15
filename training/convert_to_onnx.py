from pathlib import Path

import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "model.pkl"
ONNX_PATH = BASE_DIR / "model" / "model.onnx"


def main():
    print("=" * 70)
    print("CONVERSÃO DO MODELO PARA ONNX")
    print("=" * 70)

    # 1. Carregar modelo original
    print("\n[1/4] Carregando modelo Scikit-Learn...")
    model = joblib.load(MODEL_PATH)

    print(f"Modelo carregado: {MODEL_PATH}")

    # 2. Definir entrada do modelo
    print("\n[2/4] Configurando entrada...")
    initial_type = [
        (
            "text",
            StringTensorType([None, 1]),
        )
    ]

    # 3. Converter para ONNX
    print("\n[3/4] Convertendo para ONNX...")

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=17,
    )

    # 4. Salvar modelo
    print("\n[4/4] Salvando modelo ONNX...")

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("\nModelo ONNX salvo em:")
    print(ONNX_PATH)

    # Validar arquivo
    print("\nValidando modelo ONNX...")

    loaded_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(loaded_model)

    print("Modelo ONNX validado com sucesso!")

    print("\n" + "=" * 70)
    print("CONVERSÃO CONCLUÍDA")
    print("=" * 70)


if __name__ == "__main__":
    main()