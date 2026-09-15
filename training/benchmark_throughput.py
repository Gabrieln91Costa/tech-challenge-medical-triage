from pathlib import Path
import time

import joblib
import numpy as np
import onnxruntime as ort


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "model.pkl"
ONNX_PATH = BASE_DIR / "model" / "model.onnx"

TEXT = "Patient with cardiovascular disease and severe chest pain"

ITERATIONS = 10000
WARMUP = 100


def benchmark_sklearn(model):
    # Warm-up
    for _ in range(WARMUP):
        model.predict([TEXT])

    start = time.perf_counter()

    for _ in range(ITERATIONS):
        model.predict([TEXT])

    elapsed = time.perf_counter() - start

    throughput = ITERATIONS / elapsed

    return elapsed, throughput


def benchmark_onnx(session, input_name):
    input_data = np.array([[TEXT]], dtype=object)

    # Warm-up
    for _ in range(WARMUP):
        session.run(None, {input_name: input_data})

    start = time.perf_counter()

    for _ in range(ITERATIONS):
        session.run(None, {input_name: input_data})

    elapsed = time.perf_counter() - start

    throughput = ITERATIONS / elapsed

    return elapsed, throughput


def main():
    print("=" * 70)
    print("BENCHMARK DE THROUGHPUT")
    print("=" * 70)

    print(f"\nInferências: {ITERATIONS}")
    print(f"Warm-up:     {WARMUP}")

    # ---------------------------------------------------------
    # Scikit-Learn
    # ---------------------------------------------------------

    print("\n[1/2] Medindo Scikit-Learn...")

    model = joblib.load(MODEL_PATH)

    sklearn_time, sklearn_throughput = benchmark_sklearn(model)

    print(f"Tempo total:  {sklearn_time:.4f} segundos")
    print(f"Throughput:   {sklearn_throughput:.2f} inferências/segundo")

    # ---------------------------------------------------------
    # ONNX Runtime
    # ---------------------------------------------------------

    print("\n[2/2] Medindo ONNX Runtime...")

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    onnx_time, onnx_throughput = benchmark_onnx(
        session,
        input_name,
    )

    print(f"Tempo total:  {onnx_time:.4f} segundos")
    print(f"Throughput:   {onnx_throughput:.2f} inferências/segundo")

    # ---------------------------------------------------------
    # Comparação
    # ---------------------------------------------------------

    throughput_gain = (
        (onnx_throughput - sklearn_throughput)
        / sklearn_throughput
    ) * 100

    print("\n" + "=" * 70)
    print("RESULTADO")
    print("=" * 70)

    print(
        f"\nScikit-Learn : "
        f"{sklearn_throughput:.2f} inferências/s"
    )

    print(
        f"ONNX Runtime : "
        f"{onnx_throughput:.2f} inferências/s"
    )

    print(
        f"\nAumento de throughput: "
        f"{throughput_gain:.2f}%"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()