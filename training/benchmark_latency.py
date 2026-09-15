from pathlib import Path
import time

import joblib
import numpy as np
import onnxruntime as ort


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "model.pkl"
ONNX_PATH = BASE_DIR / "model" / "model.onnx"

TEXT = "Patient with cardiovascular disease and severe chest pain"

ITERATIONS = 1000
WARMUP = 50


def benchmark_sklearn(model):
    for _ in range(WARMUP):
        model.predict([TEXT])

    start = time.perf_counter()

    for _ in range(ITERATIONS):
        model.predict([TEXT])

    elapsed = time.perf_counter() - start

    return (elapsed / ITERATIONS) * 1000


def benchmark_onnx(session, input_name):
    input_data = np.array([[TEXT]], dtype=object)

    for _ in range(WARMUP):
        session.run(None, {input_name: input_data})

    start = time.perf_counter()

    for _ in range(ITERATIONS):
        session.run(None, {input_name: input_data})

    elapsed = time.perf_counter() - start

    return (elapsed / ITERATIONS) * 1000


def main():
    print("=" * 70)
    print("BENCHMARK DE LATÊNCIA")
    print("=" * 70)

    print(f"\nIterações: {ITERATIONS}")
    print(f"Warm-up:   {WARMUP}")

    # ---------------------------------------------------------
    # Scikit-Learn
    # ---------------------------------------------------------

    print("\n[1/2] Benchmark Scikit-Learn...")

    model = joblib.load(MODEL_PATH)

    sklearn_latency = benchmark_sklearn(model)

    print(f"Latência média Scikit-Learn: {sklearn_latency:.4f} ms")

    # ---------------------------------------------------------
    # ONNX Runtime
    # ---------------------------------------------------------

    print("\n[2/2] Benchmark ONNX Runtime...")

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    onnx_latency = benchmark_onnx(
        session,
        input_name,
    )

    print(f"Latência média ONNX Runtime: {onnx_latency:.4f} ms")

    # ---------------------------------------------------------
    # Comparação
    # ---------------------------------------------------------

    difference = sklearn_latency - onnx_latency

    if sklearn_latency > 0:
        improvement = (
            (sklearn_latency - onnx_latency)
            / sklearn_latency
        ) * 100
    else:
        improvement = 0

    print("\n" + "=" * 70)
    print("RESULTADO")
    print("=" * 70)

    print(f"\nScikit-Learn : {sklearn_latency:.4f} ms")
    print(f"ONNX Runtime : {onnx_latency:.4f} ms")

    print(f"\nDiferença: {difference:.4f} ms")
    print(f"Variação:  {improvement:.2f}%")

    if onnx_latency < sklearn_latency:
        print("\n✅ ONNX Runtime apresentou menor latência.")
    elif onnx_latency > sklearn_latency:
        print("\n⚠️ Scikit-Learn apresentou menor latência.")
    else:
        print("\n⚖️ As latências foram equivalentes.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()