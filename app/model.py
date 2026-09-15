from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

model = joblib.load(MODEL_PATH)


LABEL_MAP = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}


def predict(text: str):
    prediction = model.predict([text])[0]

    prediction = int(prediction)

    return {
        "label": prediction,
        "class_name": LABEL_MAP[prediction],
    }